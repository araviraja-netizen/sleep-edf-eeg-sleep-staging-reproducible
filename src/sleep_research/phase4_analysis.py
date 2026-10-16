"""Verified exports, per-run and consolidated Phase4 reports; never imports trainer."""
import argparse
import json
from pathlib import Path
import shutil
import math

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from models.advanced_sleep import make_advanced
from .phase4_registry import PROJECT,read_registry,envelope,active_keys,source_identity
from .phase4_data import ContextDataset
from .real_data import ResearchEpochDataset
from .common import digest,fingerprint,json_write,atomic_file,now
from .experiments import Experiment
from .checkpoints import CheckpointManager
from .inference import export_inference,load_model
from .analysis import analyze_export,load_export
from .metrics import classification_metrics,CLASS_NAMES
from .phase4_explanations import save_explanations

EXTRAS=['raw_rail_contact_samples','filter_boundary_flag','quality_flagged','context_epoch_ids','context_mask','missing_context_count']


def run_root(project,row):return Path(project)/'experiments'/row['configuration']['family']/row['run_id']


def aligned(frame,canonical):
    if len(frame)!=30266 or frame.epoch_id.duplicated().any():raise ValueError('Incomplete/duplicate validation targets')
    for key in ['epoch_id','subject_id','recording_id','night','original_epoch_index','onset_seconds','split']:
        np.testing.assert_array_equal(frame[key].to_numpy(),canonical[key].to_numpy())
    np.testing.assert_array_equal(frame.true_label.to_numpy(),canonical.label.to_numpy())
    if set(frame.split)!= {'validation'} or int(frame.quality_flagged.sum())!=206:raise ValueError('Validation split/quality mismatch')


def transition_metrics(frame):
    rows=[]
    for _,group in frame.groupby(['subject_id','recording_id','night'],observed=True):
        group=group.sort_values('original_epoch_index');index=group.original_epoch_index.to_numpy();true=group.true_label.to_numpy()
        adjacent=np.r_[False,np.diff(index)==1];change=np.r_[False,true[1:]!=true[:-1]]
        for i in np.flatnonzero(adjacent):rows.append((group.index[i],bool(change[i])))
    result={}
    for label,flag in [('transition',True),('stable',False)]:
        selected=[i for i,value in rows if value==flag];population=frame.loc[selected]
        result[label]=dict(epochs=len(population),metrics=classification_metrics(population.true_label.to_numpy(),population[[f'probability_{i}' for i in range(5)]].to_numpy()) if len(population) else None)
    result['definition']='Previous consecutive original epoch in same subject/recording/night has a different true label; gaps/boundaries excluded. Analysis only, never model input.'
    return result


def training_complete(exp,manager,cfg):
    state=manager.recover('auto')
    if state is None:return False
    if state['next_epoch']!=len(state['history']):raise ValueError('Invalid committed history')
    return state['next_epoch']>=cfg['max_epochs'] or state['early_stopping']['stopped']


def verify_run(project,row,canonical=None):
    root=run_root(project,row);path=root/'performance/phase4_result.json'
    marker=root/'reproducibility/phase4_complete.json'
    if not path.exists() or not marker.exists():raise FileNotFoundError('No committed verified result: '+row['key'])
    committed=json.loads(marker.read_text())
    if digest(path)!=committed['result_sha256'] or digest(root/'reproducibility/phase4_artifacts.json')!=committed['artifact_manifest_sha256']:raise ValueError('Completion commit checksum mismatch')
    checksum=json.loads((root/'reproducibility/phase4_artifacts.json').read_text())
    for relative,sha in checksum['files'].items():
        target=(root/relative).resolve()
        if not target.is_relative_to(root.resolve()) or digest(target)!=sha:raise ValueError('Completed artifact corrupt: '+relative)
    result=json.loads(path.read_text());cfg=row['configuration']
    if result['key']!=row['key'] or result['configuration_sha256']!=row['configuration_sha256']:raise ValueError('Result registration mismatch')
    exp=Experiment(project,cfg['family'],run_id=row['run_id'])
    if exp.metadata['configuration']!=cfg or exp.metadata['dataset_contract']!=read_registry(project)['dataset_contract']:raise ValueError('Experiment changed')
    manager=CheckpointManager(root/'reproducibility/checkpoints',exp.metadata['resume_fingerprints'],row['run_id'])
    if not training_complete(exp,manager,cfg):raise ValueError('Incomplete training reported completed')
    roles=manager.discover();latest=manager.read(roles['latest']);best=manager.read(roles['best'])
    if latest['global_step']!=len(latest['history'])*math.ceil(134047/cfg['batch_size']):raise ValueError('Incomplete full-dataset epoch')
    manifest,frame,features,_=load_export(Path(project)/result['export'])
    if manifest['synthetic'] or manifest['splits']!=['validation'] or features.shape!=(30266,64):raise ValueError('Expected full real validation export')
    if manifest['checkpoint_sha256']!=digest(roles['best']):raise ValueError('Export is not selected best')
    if canonical is None:canonical=pd.read_csv(Path(project)/'results/protocol_a_index.csv').query("split == 'validation'").reset_index(drop=True)
    aligned(frame,canonical)
    metric=classification_metrics(frame.true_label.to_numpy(),frame[[f'probability_{i}' for i in range(5)]].to_numpy())
    np.testing.assert_allclose(result['macro_f1'],metric['macro_f1'],atol=1e-12,rtol=0)
    np.testing.assert_allclose(result['macro_f1'],latest['history'][best['completed_epoch']]['validation_macro_f1'],atol=1e-12,rtol=0)
    return result


def complete_run(project,body,row):
    project=Path(project);root=run_root(project,row);cfg=row['configuration']
    if (root/'reproducibility/phase4_complete.json').exists():return verify_run(project,row)
    exp=Experiment(project,cfg['family'],run_id=row['run_id']);manager=CheckpointManager(root/'reproducibility/checkpoints',exp.metadata['resume_fingerprints'],row['run_id'])
    if not training_complete(exp,manager,cfg):raise RuntimeError('Cannot export unfinished training')
    if not torch.backends.mps.is_available():raise RuntimeError('Real validation export requires MPS')
    ds=ContextDataset(ResearchEpochDataset(project,'validation',cfg['max_open_recordings']),cfg['architecture']['window'])
    try:
        model=make_advanced(cfg['architecture']);loader=DataLoader(ds,batch_size=cfg['batch_size'],shuffle=False,num_workers=0)
        reference=root/'performance/phase4_export.json'
        if reference.exists():
            ref=json.loads(reference.read_text());export=project/ref['export'];load_export(export)
        else:
            export=export_inference(model,loader,exp,device='mps',penultimate_layer='embedding',
                intermediate_layers=[dict(name='intermediate',pooling='flatten')],metadata_fields=EXTRAS,streaming=True)
            json_write(reference,dict(export=str(export.relative_to(project))))
        params=cfg['analysis'];analysis=analyze_export(export,tsne_max_samples=params['tsne_max_samples'],
            tsne_perplexity=params['tsne_perplexity'],tsne_iterations=params['tsne_iterations'],seed=params['seed'])
        model,state,checkpoint=load_model(model,exp,'best','mps')
        chosen=[int(np.flatnonzero(ds.index.label.to_numpy()==i)[0]) for i in range(5)]
        batch=torch.utils.data.default_collate([ds[i] for i in chosen])
        explanations=root/'explanations'/export.name
        save_explanations(model,batch,checkpoint,explanations)
        manifest,frame,features,_=load_export(export)
        canonical=pd.read_csv(project/'results/protocol_a_index.csv').query("split == 'validation'").reset_index(drop=True);aligned(frame,canonical)
        metric=classification_metrics(frame.true_label.to_numpy(),frame[[f'probability_{i}' for i in range(5)]].to_numpy())
        transitions=transition_metrics(frame);json_write(analysis/'transition_metrics.json',transitions)
        latest=manager.recover('auto');roles=manager.discover();best=manager.read(roles['best'])
        row_result=dict(key=row['key'],run_id=row['run_id'],model_id=cfg['model_id'],loss=cfg['loss']['name'],seed=cfg['seed'],
            kind=row['kind'],ablation=cfg['ablation'],parameters=row['parameters'],configuration_sha256=row['configuration_sha256'],
            export=str(export.relative_to(project)),analysis=str(analysis.relative_to(project)),explanations=str(explanations.relative_to(project)),
            completed_epochs=latest['next_epoch'],best_epoch_1_based=best['completed_epoch']+1,training_seconds=exp.metadata['total_training_attempt_seconds'],
            batch_size=cfg['batch_size'],validation_epochs=30266,source_identity=body['source_identity'],
            **{key:metric[key] for key in ['accuracy','macro_f1','balanced_accuracy','cohen_kappa']},
            n1_f1=metric['classes']['N1']['f1'],n3_f1=metric['classes']['N3']['f1'],metrics=metric,
            transition_metrics=transitions,training_called_by_analysis=False,frozen_test_evaluated=False)
        json_write(root/'performance/phase4_result.json',row_result)
        (analysis/'run_report.md').write_text('# '+row['key']+' — completed validation\n\n'+
            f"Best epoch{row_result['best_epoch_1_based']}; macro-F1{row_result['macro_f1']:.6f}, N1F1{row_result['n1_f1']:.6f}, N3F1{row_result['n3_f1']:.6f}; parameters{row['parameters']}. Full30266 central targets; frozen test not evaluated. Seed42/43/44 as registered; selected validation findings, not final test generalization.\n")
        protected={str(f.relative_to(root)):digest(f) for f in root.rglob('*') if f.is_file() and not any(x.startswith('.') or x in ['__pycache__','logs'] for x in f.relative_to(root).parts) and f.name not in ['phase4_artifacts.json','phase4_complete.json']}
        json_write(root/'reproducibility/phase4_artifacts.json',dict(created_at=now(),files=protected))
        json_write(root/'reproducibility/phase4_complete.json',dict(result_sha256=digest(root/'performance/phase4_result.json'),artifact_manifest_sha256=digest(root/'reproducibility/phase4_artifacts.json'),committed_at=now()))
        return verify_run(project,row,canonical)
    finally:ds.close()


def normalized_confusion(matrix):
    a=np.asarray(matrix,dtype=float);return np.divide(a,a.sum(1,keepdims=True),out=np.zeros_like(a),where=a.sum(1,keepdims=True)>0)


def progress(project=PROJECT):
    project=Path(project);body=read_registry(project);rows=[]
    for key in body['initial_keys']+body['conditional_keys']+body['ablation_keys']:
        row=body['runs'][key]
        if (run_root(project,row)/'reproducibility/phase4_complete.json').exists():rows.append(verify_run(project,row))
    if rows:pd.DataFrame([{k:v for k,v in r.items() if not isinstance(v,dict)} for r in rows]).to_csv(project/'results/phase4_progress_comparison.csv',index=False)
    return rows


def regenerate(project=PROJECT):
    project=Path(project);body=read_registry(project);selection=envelope(project/'results/phase4_selection.json')
    keys=active_keys(body,selection);rows=[verify_run(project,body['runs'][key]) for key in keys]
    if len(rows)!=15:raise ValueError('All15 registered active runs required for final reports')
    canonical=pd.read_csv(project/'results/protocol_a_index.csv').query("split == 'validation'").reset_index(drop=True)
    baseline=[]
    for name,file in [('M1','phase3_validation_export.json'),('M4','phase3d_m4_export.json')]:
        ref=json.loads((project/'results'/file).read_text());manifest,frame,features,root=load_export(project/ref['export']);aligned(frame,canonical)
        metric=classification_metrics(frame.true_label.to_numpy(),frame[[f'probability_{i}' for i in range(5)]].to_numpy())
        meta=json.loads((root/'reproducibility/experiment.json').read_text())
        baseline.append(dict(key=name,model_id=name,kind='reference',seed=42,loss='weighted_cross_entropy' if name=='M4' else 'cross_entropy',parameters=40501,
            training_seconds=meta['total_training_attempt_seconds'],export=ref['export'],metrics=metric,**{k:metric[k] for k in ['accuracy','macro_f1','balanced_accuracy','cohen_kappa']},
            n1_f1=metric['classes']['N1']['f1'],n3_f1=metric['classes']['N3']['f1'],transition_metrics=transition_metrics(frame)))
    allrows=baseline+rows;out=project/'results';figures=out/'phase4_figures';figures.mkdir(exist_ok=True)
    flat=[{k:v for k,v in row.items() if not isinstance(v,dict)} for row in allrows]
    pd.DataFrame(flat).to_csv(out/'phase4_models_comparison.csv',index=False)
    classes=[];transitions=[]
    for row in allrows:
        for stage,scores in row['metrics']['classes'].items():classes.append(dict(key=row['key'],model_id=row['model_id'],seed=row['seed'],stage=stage,**scores))
        for group in ['transition','stable']:
            d=row['transition_metrics'][group];transitions.append(dict(key=row['key'],subset=group,epochs=d['epochs'],macro_f1=d['metrics']['macro_f1'] if d['metrics'] else None,accuracy=d['metrics']['accuracy'] if d['metrics'] else None))
        matrix=row['metrics']['confusion_matrix'];fig,axes=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
        for ax,array,title in zip(axes,[np.array(matrix),normalized_confusion(matrix)],['Raw','Row normalized']):
            ax.imshow(array,cmap='Blues',vmin=0);ax.set(title=title,xticks=range(5),yticks=range(5),xticklabels=CLASS_NAMES,yticklabels=CLASS_NAMES)
            for i in range(5):
                for j in range(5):ax.text(j,i,f'{array[i,j]:.2f}' if title!='Raw' else str(array[i,j]),ha='center',va='center',fontsize=7)
        fig.suptitle(row['key']);fig.savefig(figures/(row['key']+'_confusions.png'),dpi=140);plt.close(fig)
        if row.get('analysis'):
            analysis=project/row['analysis']
            for name in ['learning_curves.png','roc_pr.png','pca.png','tsne.png','subject_metrics.png']:
                shutil.copyfile(analysis/name,figures/(row['key']+'_'+name))
    pd.DataFrame(classes).to_csv(out/'phase4_classwise_results.csv',index=False)
    pd.DataFrame(transitions).to_csv(out/'phase4_transition_results.csv',index=False)
    initial=[r for r in rows if r['kind']=='initial'];fig,axes=plt.subplots(1,2,figsize=(13,5),constrained_layout=True)
    for loss,marker in [('cross_entropy','o'),('weighted_cross_entropy','^')]:
        pop=[r for r in initial if r['loss']==loss]
        for r in pop:
            axes[0].scatter(r['parameters'],r['macro_f1'],marker=marker);axes[0].annotate(r['key'],(r['parameters'],r['macro_f1']),fontsize=7)
            axes[1].scatter(r['training_seconds']/60,r['macro_f1'],marker=marker);axes[1].annotate(r['key'],(r['training_seconds']/60,r['macro_f1']),fontsize=7)
    axes[0].set(xlabel='Parameters',ylabel='Validation macro-F1');axes[1].set(xlabel='Measured training minutes',ylabel='Validation macro-F1')
    fig.savefig(figures/'cost_vs_performance.png',dpi=150);plt.close(fig)
    metrics=['accuracy','macro_f1','balanced_accuracy','cohen_kappa','n1_f1','n3_f1'];seedrows=[]
    for picked in selection['selected']:
        prefix=picked['key'].rsplit('_s',1)[0];population=[r for r in rows if r['key'].rsplit('_s',1)[0]==prefix]
        if sorted(r['seed'] for r in population)!=[42,43,44]:raise ValueError('Require three distinct seeds')
        stats=dict(configuration=prefix,seeds=[42,43,44])
        for k in metrics:
            x=np.array([r[k] for r in population]);stats.update({k+'_mean':float(x.mean()),k+'_sample_sd':float(x.std(ddof=1)),k+'_min':float(x.min()),k+'_max':float(x.max())})
        seedrows.append(stats)
    pd.DataFrame(seedrows).to_csv(out/'phase4_multiseed_summary.csv',index=False)
    with atomic_file(out/'phase4_multiseed_analysis.md') as f:f.write('# Phase4 three-seed validation\n\n'+pd.DataFrame(seedrows).to_string(index=False)+'\n\nShortlist selected on seed42 validation; sample SD(ddof1) describes seed variability, not an unbiased test estimate or confidence interval. Repeated runs do not include retraining M1/M4.\n')
    ablation=[r for r in rows if r['kind']=='ablation'];control=next(r for r in ablation if r['ablation']=='control')
    with atomic_file(out/'phase4_ablation_results.md') as f:
        f.write('# Phase4 bounded M8 ablation\n\nEqual3-epoch full-dataset budgets, seed42, CE, fresh initialization. Reduced capacity and computation are confounds, not solely module efficacy.\n\n')
        for r in ablation:f.write(f"{r['ablation']}: macro-F1 {r['macro_f1']:.6f}, N1F1 {r['n1_f1']:.6f}, delta macro-F1 vs bounded control {r['macro_f1']-control['macro_f1']:+.6f}.\n\n")
    with atomic_file(out/'phase4_research_contribution_analysis.md') as f:
        f.write('# Phase4 contributions and limitations\n\nM8 combines established multi-scale EEG encoding, context attention and gated fusion. No novelty or state-of-the-art claim is established. The differentiating hypothesis is whether masked five-epoch fusion and explicit class weighting improve N1/transition recognition on this frozen protocol without reducing macro-F1. See actual comparisons, transition CSV and equal-budget ablations; components are not assumed effective.\n\nRelated methods: [DeepSleepNet](https://arxiv.org/abs/1703.04046), [SeqSleepNet](https://arxiv.org/abs/1809.10932), [SleepTransformer](https://arxiv.org/abs/2105.11043), [AttnSleep](https://ieeexplore.ieee.org/document/9417097). These candidates are compact research variants, not reproductions. Novelty remains unresolved pending a broader literature review.\n\nSelected-validation results, one frozen cohort and validation-selected repeats cannot establish significance or final-test generalization. Future EEG context, noncausal filtering and label-based Wake trimming limit deployment. Retained quality flags, R&K merging and filename-derived subjects remain limitations. No final test evaluation.\n')
    lines=['# Phase4 consolidated measured validation campaign',f'Completed analysis {now()}.','',
           'Eight initial configurations, four conditional repeat runs, three equal-budget ablation/control runs; all artifacts verified. Same30266 validation target IDs; no test evaluation. M1/M4 original trained baselines reused.','',
           '| Key | Parameters | Accuracy | Macro-F1 | N1F1 | N3F1 | Training min |','| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for r in allrows:lines.append(f"| {r['key']} | {r['parameters']} | {r['accuracy']:.6f} | {r['macro_f1']:.6f} | {r['n1_f1']:.6f} | {r['n3_f1']:.6f} | {r['training_seconds']/60:.2f} |")
    lines+=['','CE/weighted CE rows use matched architecture/batch/budget; examine N1 gains alongside primary macro-F1 and N3 errors. Class precision/recall/F1 and raw/normalized confusion figures expose tradeoffs. Per-subject, ROC/PR, learning curves and saved embeddings remain available per run. Transition errors exclude recording boundaries and invalid-index gaps.','',
            'See phase4_models_comparison.csv,phase4_classwise_results.csv,phase4_transition_results.csv,phase4_multiseed_analysis.md,phase4_ablation_results.md and phase4_research_contribution_analysis.md. Figures include cost versus validation performance, confusions and per-run PCA/t-SNE/subject/ROC/learning curves. Separate model PCA/t-SNE axes are not shared geometric spaces.',
            'The best initial configuration is '+sorted(initial,key=lambda r:(-r['macro_f1'],r['key']))[0]['key']+'. This is validation selection, not statistically proven improvement or test generalization. No Phase5 or new training is invoked by report generation.']
    with atomic_file(out/'phase4_final_comparison.md') as f:f.write('\n'.join(lines)+'\n')
    json_write(out/'phase4_analysis_summary.json',dict(successful=True,generated_at=now(),campaign_id=body['campaign_id'],
        active_keys=keys,measured_runs=15,models=flat,training_called=False,frozen_test_evaluated=False,validation_ids_identical=True))
    return dict(successful=True,measured_runs=15,training_called=False,frozen_test_evaluated=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=['progress','regenerate']);parser.add_argument('--project',type=Path,default=PROJECT);args=parser.parse_args()
    print(json.dumps(regenerate(args.project) if args.command=='regenerate' else progress(args.project),indent=2))
