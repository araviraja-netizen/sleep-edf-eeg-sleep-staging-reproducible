"""M4 checkpoint export/status and saved-validation comparison; no trainer import/call."""
import argparse
import json
from pathlib import Path
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from .analysis import analyze_export,load_export,contiguous_blocks
from .checkpoints import CheckpointManager
from .common import digest,fingerprint,json_write,now
from .experiments import Experiment,frozen_dataset_contract
from .metrics import classification_metrics,CLASS_NAMES
from .cnn_inference import model_for
from .inference import load_model,export_inference
from .explanations import export_explanations
from .real_data import ResearchEpochDataset

PROJECT=Path(__file__).resolve().parents[2]
QUALITY=['raw_rail_contact_samples','filter_boundary_flag','quality_flagged']


def registration(project=PROJECT):
    project=Path(project);envelope=json.loads((project/'results/phase3d_registry.json').read_text());body=envelope['body']
    if fingerprint(body)!=envelope['sha256']:raise ValueError('Registry checksum mismatch')
    if frozen_dataset_contract(project,'A')!=body['dataset_contract']:raise ValueError('Frozen data changed')
    if yaml.safe_load((project/'configs/phase3d_m4.yaml').read_text())!=body['configuration']:raise ValueError('M4 configuration changed')
    for relative,sha in body['protected_checkpoints'].items():
        if digest(project/relative)!=sha:raise ValueError('Previous checkpoint changed')
    return body


def status(project=PROJECT):
    project=Path(project);body=registration(project);root=project/'experiments/cnn'/body['run_id']
    result=dict(run_id=body['run_id'],experiment=str(root),full_training_launched=False,status='registered_unlaunched',
                completed_epochs=0,latest_completed_epoch_1_based=None,best_validation_macro_f1=None,early_stopping=None,
                log=str(root/'logs/training.log'),frozen_test_evaluated=False)
    if not root.exists():return result
    exp=Experiment(project,'cnn',run_id=body['run_id']);result.update(status=exp.metadata['status'])
    if 'resume_fingerprints' not in exp.metadata:return result
    manager=CheckpointManager(root/'reproducibility/checkpoints',exp.metadata['resume_fingerprints'],exp.run_id)
    state=manager.recover('auto') # Read-only discovery; never bootstraps or repairs history.
    if state is not None:
        result.update(full_training_launched=True,completed_epochs=state['next_epoch'],
                      latest_completed_epoch_1_based=state['next_epoch'] if state['next_epoch'] else None,
                      best_validation_macro_f1=state['early_stopping']['best_score'],early_stopping=state['early_stopping'],
                      recoverable_checkpoint=str(manager.recovered_path),checkpoints=manager.discover(),
                      recovery_notes=manager.recovery_notes)
    result['note']='Checkpoint history is authoritative. A stale running status after power failure does not prove a process is alive. Incomplete epochs repeat.'
    return result


def completed_experiment(project,body):
    root=Path(project)/'experiments/cnn'/body['run_id']
    if not root.exists():raise RuntimeError('M4 has not been trained; no measured results can be generated')
    exp=Experiment(project,'cnn',run_id=body['run_id'])
    if exp.metadata['status'] not in ['completed','early_stopped']:raise RuntimeError('M4 training must finish before full validation analysis')
    cfg=exp.metadata['configuration'];expected=dict(body['configuration'],training_class_counts=body['training_counts'],loss_alpha=body['weights_float32'])
    if cfg!=expected or exp.metadata['dataset_contract']!=body['dataset_contract']:raise ValueError('M4 experiment differs from registration')
    return exp


def export_validation(project=PROJECT):
    project=Path(project);body=registration(project);exp=completed_experiment(project,body);cfg=exp.metadata['configuration']
    if not torch.backends.mps.is_available():raise RuntimeError('Validation export requires local MPS')
    refpath=project/'results/phase3d_m4_export.json'
    if refpath.exists():
        reference=json.loads(refpath.read_text())
        if reference['run_id']!=body['run_id']:raise ValueError('Export run differs')
        load_export(project/reference['export'])
        explanation=project/reference['explanations'];manifest=json.loads((explanation/'manifest.json').read_text())
        for filename,sha in manifest['files'].items():
            if digest(explanation/filename)!=sha:raise ValueError('Explanation changed')
        return reference
    dataset=ResearchEpochDataset(project,'validation',cfg['max_open_recordings'])
    try:
        if len(dataset)!=30266:raise ValueError('Incomplete validation set')
        loader=DataLoader(dataset,batch_size=16,shuffle=False,num_workers=0)
        model=model_for(exp)
        export=export_inference(model,loader,exp,checkpoint='best',device='mps',penultimate_layer='embedding',
            intermediate_layers=cfg['analysis']['intermediate_layers'],max_feature_dimension=512,metadata_fields=QUALITY)
        model,state,path=load_model(model,exp,'best','mps')
        indices=[int(np.flatnonzero(dataset.index.label.to_numpy()==label)[0]) for label in range(5)]
        batch=torch.utils.data.default_collate([dataset[i] for i in indices])
        explanations=export_explanations(model,batch,exp.path/'explanations'/export.name,path,
                        conv_layer='blocks.3.0',max_samples=5)
        reference=dict(run_id=exp.run_id,export=str(export.relative_to(project)),explanations=str(explanations.relative_to(project)),
                       checkpoint=str(path.relative_to(project)),completed_epoch=state['completed_epoch'],synthetic=False,split='validation',frozen_test_evaluated=False)
        json_write(refpath,reference);return reference
    finally:dataset.close()


def verify_validation_rows(manifest,frame,features,canonical,flags):
    if manifest['splits']!=['validation'] or manifest['synthetic'] or len(frame)!=30266 or features.shape!=(30266,64):
        raise ValueError('Expected all30266 real validation epochs and64-value features')
    for key in ['epoch_id','subject_id','recording_id','night','original_epoch_index','onset_seconds','split']:
        np.testing.assert_array_equal(frame[key],canonical[key])
    np.testing.assert_array_equal(frame.true_label,canonical.label)
    quality=(frame.raw_rail_contact_samples>0)|(frame.filter_boundary_flag>0)
    if int(quality.sum())!=206 or set(frame.loc[quality,'epoch_id'])!=set(flags.epoch_id):
        raise ValueError('Validation quality flags differ from frozen inventory')
    np.testing.assert_array_equal(frame.quality_flagged,quality.astype(int))


def compare(project=PROJECT):
    project=Path(project);results=project/'results'
    body=registration(project)
    completed_experiment(project,body)
    registry=dict(shared_dataset_contract=body['dataset_contract'],models={'M1':dict(run_id=body['baseline_run_id']),'M4':dict(run_id=body['run_id'])})
    reference=json.loads((results/'phase3d_m4_export.json').read_text())
    if reference['run_id']!=registry['models']['M4']['run_id']:raise ValueError('Export is not the registered M4')
    exports={'M1':json.loads((results/'phase3_validation_export.json').read_text())['export'],'M4':reference['export']}
    frames={};histories={};metrics={};rows=[];checkpoints={};experiments={}
    canonical=pd.read_csv(results/'protocol_a_index.csv');canonical=canonical[canonical.split=='validation'].reset_index(drop=True)
    flags=pd.read_csv(results/'quality_flagged_epochs.csv');flags=flags[(flags.split=='validation')&(flags.wake_trimmed_inclusion==1)]
    for name,relative in exports.items():
        manifest,frame,features,root=load_export(project/relative)
        verify_validation_rows(manifest,frame,features,canonical,flags)
        experiment=Experiment(project,'cnn',run_id=registry['models'][name]['run_id']);experiments[name]=experiment
        manager=CheckpointManager(experiment.path/'reproducibility/checkpoints',experiment.metadata['resume_fingerprints'],experiment.run_id)
        roles=manager.discover();best=manager.read(roles['best']);latest=manager.read(roles['latest']);checkpoints[name]=roles
        if experiment.metadata['status'] not in ['completed','early_stopped']:raise ValueError('Do not report an unfinished full run')
        if digest(roles['best'])!=manifest['checkpoint_sha256']:raise ValueError('Export does not bind the selected best checkpoint')
        history=latest['history'];histories[name]=history;frames[name]=frame
        metric=classification_metrics(frame.true_label.to_numpy(),frame[[f'probability_{i}' for i in range(5)]].to_numpy());metrics[name]=metric
        np.testing.assert_allclose(metric['macro_f1'],history[best['completed_epoch']]['validation_macro_f1'],atol=1e-12,rtol=0)
        np.testing.assert_allclose(metric['accuracy'],history[best['completed_epoch']]['validation_accuracy'],atol=1e-12,rtol=0)
        if latest['global_step']!=len(history)*8378:raise ValueError('Incomplete full-dataset epoch or invalid step count')
        if experiment.metadata['dataset_contract']!=registry['shared_dataset_contract']:raise ValueError('M1/M4 data contracts differ')
        row=dict(model_id=name,run_id=experiment.run_id,optimizer=experiment.metadata['configuration']['optimizer']['name'],loss='weighted_cross_entropy' if name=='M4' else 'ordinary_cross_entropy',
                 seed=42,training_examples=134047,validation_examples=len(frame),training_subjects=54,validation_subjects=12,
                 parameters=40501,status=experiment.metadata['status'],completed_epochs=len(history),best_epoch_1_based=best['completed_epoch']+1,
                 training_attempt_seconds=experiment.metadata['total_training_attempt_seconds'],
                 completed_epoch_seconds=sum(r['seconds'] for r in history),time_to_best_seconds=sum(r['seconds'] for r in history[:best['completed_epoch']+1]),
                 mean_epoch_seconds=float(np.mean([r['seconds'] for r in history])),
                 accuracy=metric['accuracy'],macro_f1=metric['macro_f1'],balanced_accuracy=metric['balanced_accuracy'],cohen_kappa=metric['cohen_kappa'],
                 checkpoint_sha256=manifest['checkpoint_sha256'],export=relative,validation_only=True)
        for stage,value in metric['classes'].items():
            for key in ['precision','recall','f1','support']:row[stage.lower()+'_'+key]=value[key]
        rows.append(row)
    metric_keys=['accuracy','macro_f1','balanced_accuracy','cohen_kappa']+[stage.lower()+'_'+key for stage in CLASS_NAMES for key in ['precision','recall','f1']]
    for key in metric_keys:rows[1]['delta_'+key+'_M4_minus_M1']=rows[1][key]-rows[0][key]
    frame=pd.DataFrame(rows);frame.to_csv(results/'phase3d_m1_m4_comparison.csv',index=False)
    per_class=[]
    for name,metric in metrics.items():
        for stage,value in metric['classes'].items():per_class.append(dict(model_id=name,stage=stage,**value))
    pd.DataFrame(per_class).to_csv(results/'phase3d_classwise_scores.csv',index=False)
    m4=experiments['M4'];config=m4.metadata['configuration'];params=config['analysis']
    analysis=analyze_export(project/exports['M4'],tsne_max_samples=params['tsne_max_samples'],tsne_perplexity=params['tsne_perplexity'],
                            tsne_iterations=params['tsne_iterations'],seed=params['seed'])
    figures=results/'phase3d_figures';figures.mkdir(exist_ok=True)
    for source in analysis.glob('*.png'):shutil.copyfile(source,figures/('M4_'+source.name))
    fig,axes=plt.subplots(1,2,figsize=(12,5),constrained_layout=True)
    for name,ax in zip(['M1','M4'],axes):
        matrix=np.array(metrics[name]['confusion_matrix']);ax.imshow(matrix,cmap='Blues')
        for i in range(5):
            for j in range(5):ax.text(j,i,str(matrix[i,j]),ha='center',va='center')
        ax.set(title=name+' — complete validation',xticks=range(5),yticks=range(5),xticklabels=CLASS_NAMES,yticklabels=CLASS_NAMES,xlabel='Predicted',ylabel='True')
    fig.savefig(figures/'M1_M4_confusion.png',dpi=140);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(14,4),constrained_layout=True)
    for name,history in histories.items():
        x=[r['epoch']+1 for r in history]
        axes[0].plot(x,[r['train_loss'] for r in history],label=name+' train')
        axes[0].plot(x,[r['validation_loss'] for r in history],linestyle='--',label=name+' validation')
        axes[1].plot(x,[r['validation_macro_f1'] for r in history],label=name)
        axes[2].plot(x,[r['validation_accuracy'] for r in history],label=name)
    for ax,label in zip(axes,['Loss (weighted M4 train; ordinary validation CE)','Validation macro-F1','Validation accuracy']):ax.set(xlabel='Completed epoch',ylabel=label);ax.legend(fontsize=8)
    fig.savefig(figures/'M1_M4_learning_curves.png',dpi=140);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4),constrained_layout=True);x=np.arange(5)
    for offset,name in [(-.18,'M1'),(.18,'M4')]:ax.bar(x+offset,[metrics[name]['classes'][stage]['f1'] for stage in CLASS_NAMES],width=.36,label=name)
    ax.set(xticks=x,xticklabels=CLASS_NAMES,ylabel='Validation F1',ylim=(0,1),title='Selected best checkpoints — seed42');ax.legend()
    fig.savefig(figures/'M1_M4_class_f1.png',dpi=140);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4),constrained_layout=True);h=histories['M4']
    for key,label in [('validation_macro_f1','Macro-F1'),('validation_n1_f1','N1 F1'),('validation_n3_f1','N3 F1')]:
        ax.plot([r['epoch']+1 for r in h],[r[key] for r in h],label=label)
    ax.set(xlabel='Completed epoch',ylabel='M4 validation F1',ylim=(0,1));ax.legend();fig.savefig(figures/'M4_class_learning_curves.png',dpi=140);plt.close(fig)
    # Original epoch grid retains excluded-annotation gaps rather than connecting them.
    for recording,group in frames['M1'].groupby('recording_id',observed=True):
        indices=group.index.to_numpy();original=group.original_epoch_index.to_numpy(dtype=int)
        fig,axes=plt.subplots(3,1,figsize=(12,6),sharex=True,constrained_layout=True)
        for ax,label,values in zip(axes,['True','M1','M4'],[group.true_label.to_numpy(),group.predicted_label.to_numpy(),frames['M4'].loc[indices,'predicted_label'].to_numpy()]):
            for block in contiguous_blocks(original):
                edges=np.r_[original[block],original[block[-1]]+1]*30/3600
                ax.stairs(values[block],edges)
            ax.set(ylabel=label,yticks=range(5),yticklabels=CLASS_NAMES);ax.invert_yaxis()
        axes[0].set_title(str(recording)+' — identical complete validation cohort');axes[-1].set_xlabel('Hours from PSG start')
        fig.savefig(figures/('M1_M4_hypnogram_'+str(recording)+'.png'),dpi=140);plt.close(fig)
    delta={key:metrics['M4'][key]-metrics['M1'][key] for key in ['accuracy','macro_f1','balanced_accuracy','cohen_kappa']}
    for stage in CLASS_NAMES:
        for key in ['precision','recall','f1']:delta[stage.lower()+'_'+key]=metrics['M4']['classes'][stage][key]-metrics['M1']['classes'][stage][key]
    summary=dict(successful=True,generated_at=now(),M4_run_id=m4.run_id,models=rows,delta_M4_minus_M1=delta,
                 M4_analysis=str(analysis.relative_to(project)),M4_export=exports['M4'],training_called=False,
                 frozen_test_evaluated=False,M3_launched=False,M4_launched=True,additional_seeds_launched=False,
                 same_validation_ids=True,retained_flagged_validation_epochs=206)
    json_write(results/'phase3d_analysis_summary.json',summary)
    write_report(project,registry,rows,metrics,histories,delta,summary,checkpoints)
    return summary


def write_report(project,registry,rows,metrics,histories,delta,summary,checkpoints):
    results=project/'results';m4=rows[1];root=project/'experiments/cnn'/m4['run_id']
    resources=json.loads((root/'reproducibility/phase3d_resources.json').read_text())
    architecture=json.loads((root/'reproducibility/architecture.json').read_text())
    lines=['# Phase 3D — measured M4 validation results',f'Generated {now()}.',
        f"M4 `{m4['run_id']}`: {m4['status']}, {m4['completed_epochs']} completed epochs, best epoch {m4['best_epoch_1_based']} (1-based).",
        'Exact40501-parameter M1 CNN, AdamW lr0.001 decay0.01, batch16,float32,MPS,seed42,max25,patience8,no scheduler; only intended experimental change is weighted training CE. Shared Protocol A/train-only normalization/frozen54/12/12 subjects;134047 training,30266 complete validation epochs. All206 quality-flagged validation epochs retained. Test31156 remains locked.',
        'Weights use N_train/(5*n_c) from training only, ordered Wake,N1,N2,N3,REM. Optimization is sum(w_y*CE)/sum(w_y). Validation loss is ordinary CE; validation_objective_loss is weighted CE. Training losses across objectives are not numerically comparable.',
        '', '| Model | Accuracy | Macro-F1 | Balanced accuracy | Kappa | Best epoch | Duration s |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for row in rows:
        lines.append(f"| {row['model_id']} | {row['accuracy']:.6f} | {row['macro_f1']:.6f} | {row['balanced_accuracy']:.6f} | {row['cohen_kappa']:.6f} | {row['best_epoch_1_based']} | {row['training_attempt_seconds']:.2f} |")
    lines+=['','Absolute differences M4−M1: '+json.dumps(delta,sort_keys=True),'',
        '| Class | M1 precision | M1 recall | M1 F1 | M4 precision | M4 recall | M4 F1 | Support |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for stage in CLASS_NAMES:
        a=metrics['M1']['classes'][stage];b=metrics['M4']['classes'][stage]
        lines.append(f"| {stage} | {a['precision']:.6f} | {a['recall']:.6f} | {a['f1']:.6f} | {b['precision']:.6f} | {b['recall']:.6f} | {b['f1']:.6f} | {b['support']} |")
    lines+=['','Confusions: rows=true, columns=predicted, order Wake,N1,N2,N3,REM.']
    for name in ['M1','M4']:
        lines+=['',name,'','```text',*[' '.join(map(str,row)) for row in metrics[name]['confusion_matrix']],'```']
    lines+=['', '| Model | Mean epoch s | Time through best s | Final macro-F1 |', '| --- | ---: | ---: | ---: |']
    for row in rows:
        lines.append(f"| {row['model_id']} | {row['mean_epoch_seconds']:.2f} | {row['time_to_best_seconds']:.2f} | {histories[row['model_id']][-1]['validation_macro_f1']:.6f} |")
    lines+=['',f"M4 process peak RSS {resources['process_peak_rss_bytes']/2**30:.3f}GiB; end MPS readings are not peak memory. Architecture {architecture['parameters']} parameters, {architecture['conv_linear_macs_per_epoch']} Conv/Linear forward MACs per EEG epoch. No normalized dataset duplication.",
        'Training accuracy/F1 are online with changing weights/dropout; validation is eval mode. Timing across separate runs includes thermal/cache/software effects. M1 historical per-epoch N1/N3 scores are unavailable; its selected checkpoint scores are compared. All differences are selected-validation, single-seed observations; no significance, seed stability or final test generalization claim.',
        f"Saved full validation export `{summary['M4_export']}`; analysis `{summary['M4_analysis']}`. These include five logits/probabilities, stable original epoch/subject/recording IDs and flags;64-value penultimate/pooled intermediate features; full PCA and seeded500-row t-SNE; class/subject metrics, confusion and original-index-respecting hypnograms. Checkpoint-bound saliency/Grad-CAM use the first epoch of each true validation class, selected without prediction confidence. Explanations describe predictions and do not establish causal physiology.",
        'Comparison figures: `results/phase3d_figures/`; exact metrics and absolute differences: `phase3d_m1_m4_comparison.csv`; class rows: `phase3d_classwise_scores.csv`. Source/software/config/fingerprints/weights, optimizer/RNG/early-stop/authoritative histories and best/latest checkpoints remain in the permanent M4 experiment. Incomplete epochs repeat; bitwise MPS reproducibility is not promised.',
        '**Stopped after M4 validation analysis. M1/M2/pilots preserved. No M3, additional seeds, advanced model or frozen test evaluation.**']
    (results/'phase3d_m4_results.md').write_text('\n'.join(lines)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['status','export','analyze','complete'])
    parser.add_argument('--project',type=Path,default=PROJECT);args=parser.parse_args()
    if args.command=='status':result=status(args.project)
    elif args.command=='export':result=export_validation(args.project)
    elif args.command=='analyze':result=compare(args.project)
    else:export_validation(args.project);result=compare(args.project)
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
