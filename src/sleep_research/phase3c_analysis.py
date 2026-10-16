"""Analysis of verified complete validation exports; never imports the trainer."""

import argparse
import json
from pathlib import Path
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .analysis import analyze_export,load_export
from .checkpoints import CheckpointManager
from .common import digest,fingerprint,json_write,now
from .experiments import Experiment,frozen_dataset_contract
from .metrics import classification_metrics,CLASS_NAMES

PROJECT=Path(__file__).resolve().parents[2]


def compare(project=PROJECT):
    project=Path(project);results=project/'results'
    envelope=json.loads((results/'phase3c_registry.json').read_text());registry=envelope['body']
    if fingerprint(registry)!=envelope['sha256'] or frozen_dataset_contract(project,'A')!=registry['shared_dataset_contract']:
        raise ValueError('Registered data/config contract changed')
    reference=json.loads((results/'phase3c_m2_export.json').read_text())
    if reference['run_id']!=registry['models']['M2']['run_id']:raise ValueError('Export is not the registered M2')
    exports={'M1':json.loads((results/'phase3_validation_export.json').read_text())['export'],'M2':reference['export']}
    frames={};histories={};metrics={};rows=[];checkpoints={};experiments={}
    canonical=pd.read_csv(results/'protocol_a_index.csv');canonical=canonical[canonical.split=='validation'].reset_index(drop=True)
    flags=pd.read_csv(results/'quality_flagged_epochs.csv');flags=flags[(flags.split=='validation')&(flags.wake_trimmed_inclusion==1)]
    for name,relative in exports.items():
        manifest,frame,features,root=load_export(project/relative)
        if manifest['splits']!=['validation'] or manifest['synthetic'] or len(frame)!=30266 or features.shape!=(30266,64):
            raise ValueError('Expected all30266 real validation epochs and64-value features')
        for key in ['epoch_id','subject_id','recording_id','night','original_epoch_index','onset_seconds','split']:
            np.testing.assert_array_equal(frame[key],canonical[key])
        np.testing.assert_array_equal(frame.true_label,canonical.label)
        quality=(frame.raw_rail_contact_samples>0)|(frame.filter_boundary_flag>0)
        assert int(quality.sum())==206 and set(frame.loc[quality,'epoch_id'])==set(flags.epoch_id)
        np.testing.assert_array_equal(frame.quality_flagged,quality.astype(int))
        experiment=Experiment(project,'cnn',run_id=registry['models'][name]['run_id']);experiments[name]=experiment
        manager=CheckpointManager(root/'reproducibility/checkpoints',experiment.metadata['resume_fingerprints'],experiment.run_id)
        roles=manager.discover();best=manager.read(roles['best']);latest=manager.read(roles['latest']);checkpoints[name]=roles
        if experiment.metadata['status'] not in ['completed','early_stopped']:raise ValueError('Do not report an unfinished full run')
        if digest(roles['best'])!=manifest['checkpoint_sha256']:raise ValueError('Export does not bind the selected best checkpoint')
        history=latest['history'];histories[name]=history;frames[name]=frame
        metric=classification_metrics(frame.true_label.to_numpy(),frame[[f'probability_{i}' for i in range(5)]].to_numpy());metrics[name]=metric
        np.testing.assert_allclose(metric['macro_f1'],history[best['completed_epoch']]['validation_macro_f1'],atol=1e-12,rtol=0)
        np.testing.assert_allclose(metric['accuracy'],history[best['completed_epoch']]['validation_accuracy'],atol=1e-12,rtol=0)
        if latest['global_step']!=len(history)*8378:raise ValueError('Incomplete full-dataset epoch or invalid step count')
        if experiment.metadata['dataset_contract']!=registry['shared_dataset_contract']:raise ValueError('M1/M2 data contracts differ')
        row=dict(model_id=name,run_id=experiment.run_id,optimizer=experiment.metadata['configuration']['optimizer']['name'],loss='ordinary_cross_entropy',
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
    frame=pd.DataFrame(rows);frame.to_csv(results/'phase3c_validation_comparison.csv',index=False)
    per_class=[]
    for name,metric in metrics.items():
        for stage,value in metric['classes'].items():per_class.append(dict(model_id=name,stage=stage,**value))
    pd.DataFrame(per_class).to_csv(results/'phase3c_classwise_scores.csv',index=False)
    m2=experiments['M2'];config=m2.metadata['configuration'];params=config['analysis']
    analysis=analyze_export(project/exports['M2'],tsne_max_samples=params['tsne_max_samples'],tsne_perplexity=params['tsne_perplexity'],
                            tsne_iterations=params['tsne_iterations'],seed=params['seed'])
    figures=results/'phase3c_figures';figures.mkdir(exist_ok=True)
    for source in analysis.glob('*.png'):shutil.copyfile(source,figures/('M2_'+source.name))
    fig,axes=plt.subplots(1,2,figsize=(12,5),constrained_layout=True)
    for name,ax in zip(['M1','M2'],axes):
        matrix=np.array(metrics[name]['confusion_matrix']);ax.imshow(matrix,cmap='Blues')
        for i in range(5):
            for j in range(5):ax.text(j,i,str(matrix[i,j]),ha='center',va='center')
        ax.set(title=name+' — complete validation',xticks=range(5),yticks=range(5),xticklabels=CLASS_NAMES,yticklabels=CLASS_NAMES,xlabel='Predicted',ylabel='True')
    fig.savefig(figures/'M1_M2_confusion.png',dpi=140);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(14,4),constrained_layout=True)
    for name,history in histories.items():
        x=[r['epoch']+1 for r in history]
        axes[0].plot(x,[r['train_loss'] for r in history],label=name+' train')
        axes[0].plot(x,[r['validation_loss'] for r in history],linestyle='--',label=name+' validation')
        axes[1].plot(x,[r['validation_macro_f1'] for r in history],label=name)
        axes[2].plot(x,[r['validation_accuracy'] for r in history],label=name)
    for ax,label in zip(axes,['Ordinary CE','Validation macro-F1','Validation accuracy']):ax.set(xlabel='Completed epoch',ylabel=label);ax.legend(fontsize=8)
    fig.savefig(figures/'M1_M2_learning_curves.png',dpi=140);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4),constrained_layout=True);x=np.arange(5)
    for offset,name in [(-.18,'M1'),(.18,'M2')]:ax.bar(x+offset,[metrics[name]['classes'][stage]['f1'] for stage in CLASS_NAMES],width=.36,label=name)
    ax.set(xticks=x,xticklabels=CLASS_NAMES,ylabel='Validation F1',ylim=(0,1),title='Selected best checkpoints — seed42');ax.legend()
    fig.savefig(figures/'M1_M2_class_f1.png',dpi=140);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4),constrained_layout=True);h=histories['M2']
    for key,label in [('validation_macro_f1','Macro-F1'),('validation_n1_f1','N1 F1'),('validation_n3_f1','N3 F1')]:
        ax.plot([r['epoch']+1 for r in h],[r[key] for r in h],label=label)
    ax.set(xlabel='Completed epoch',ylabel='M2 validation F1',ylim=(0,1));ax.legend();fig.savefig(figures/'M2_class_learning_curves.png',dpi=140);plt.close(fig)
    for name in ['M3','M4']:
        if (project/'experiments/cnn'/registry['models'][name]['run_id']).exists():raise ValueError('Unauthorized M3/M4 experiment directory exists')
    delta={key:metrics['M2'][key]-metrics['M1'][key] for key in ['accuracy','macro_f1','balanced_accuracy','cohen_kappa']}
    for stage in ['N1','N3']:
        for key in ['precision','recall','f1']:delta[stage.lower()+'_'+key]=metrics['M2']['classes'][stage][key]-metrics['M1']['classes'][stage][key]
    summary=dict(successful=True,generated_at=now(),M2_run_id=m2.run_id,models=rows,delta_M2_minus_M1=delta,
                 M2_analysis=str(analysis.relative_to(project)),M2_export=exports['M2'],training_called=False,
                 frozen_test_evaluated=False,M3_launched=False,M4_launched=False,additional_seeds_launched=False,
                 same_validation_ids=True,retained_flagged_validation_epochs=206)
    json_write(results/'phase3c_analysis_summary.json',summary)
    write_report(project,registry,rows,metrics,histories,delta,summary,checkpoints)
    return summary


def write_report(project,registry,rows,metrics,histories,delta,summary,checkpoints):
    results=project/'results';m2=rows[1];root=project/'experiments/cnn'/m2['run_id']
    resources=json.loads((root/'reproducibility/phase3c_resources.json').read_text())
    architecture=json.loads((root/'reproducibility/architecture.json').read_text())
    lines=['# Phase 3C — full-budget M2 confirmation',f'Generated: {now()}.','',
           f"**M2 {m2['run_id']} completed on MPS float32**, status `{m2['status']}`, {m2['completed_epochs']} completed epochs; best epoch {m2['best_epoch_1_based']} (1-based). M1 is preserved; M3/M4 and seeds43/44 were not launched.",'',
           '## Registered comparison and full validation outcomes','',
           'Both models use the same134047 training epochs/54subjects and30266 validation epochs/12subjects/24recordings. Test31156/12subjects stays locked. Protocol A, Fpz-Cz, class mapping, normalization and all quality flags are unchanged; all206 flagged validation epochs are included. M1 used its original Torch CE; M2 uses tested equivalent ordinary CE through the shared criterion interface.',
           'Both use LR0.001, batch16, decay0.01, max25, patience8, seed42, no scheduler, validation macro-F1 selection. Adam uses coupled L2; AdamW decouples decay. The contrast includes optimizer/regularization semantics. No preprocessing, weighting, architecture or subject allocation was changed.','',
           '| Model | Optimizer/loss | Accuracy | Macro-F1 | Balanced accuracy | Kappa | Best epoch | Completed epochs | Train+val attempt s |',
           '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for row in rows:lines.append(f"| {row['model_id']} | {row['optimizer']} / CE | {row['accuracy']:.6f} | {row['macro_f1']:.6f} | {row['balanced_accuracy']:.6f} | {row['cohen_kappa']:.6f} | {row['best_epoch_1_based']} | {row['completed_epochs']} | {row['training_attempt_seconds']:.2f} |")
    lines+=['',f"M2−M1 absolute changes: accuracy {delta['accuracy']:+.6f}, macro-F1 {delta['macro_f1']:+.6f}, balanced accuracy {delta['balanced_accuracy']:+.6f}, kappa {delta['cohen_kappa']:+.6f}. These are selected validation differences from one optimization seed, not final test performance or statistical significance.",'',
            '| Class | M1 precision | M1 recall | M1 F1 | M2 precision | M2 recall | M2 F1 | Support |',
            '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for stage in CLASS_NAMES:
        a=metrics['M1']['classes'][stage];b=metrics['M2']['classes'][stage]
        lines.append(f"| {stage} | {a['precision']:.6f} | {a['recall']:.6f} | {a['f1']:.6f} | {b['precision']:.6f} | {b['recall']:.6f} | {b['f1']:.6f} | {b['support']} |")
    lines+=['',f"N1 F1 change {delta['n1_f1']:+.6f}; N3 F1 change {delta['n3_f1']:+.6f}. N1/N3 precision and recall are retained in the table and comparison CSV. No class-specific threshold or model adjustment was made.",'',
            '## Confusion matrices','', 'Rows=true; columns=predicted. Order: Wake,N1,N2,N3,REM.']
    for name in ['M1','M2']:
        lines+=['',name,'','```text',*[' '.join(map(str,row)) for row in metrics[name]['confusion_matrix']],'```']
    lines+=['','## Convergence and computational requirements','',
            f"Parameter count {architecture['parameters']}; Conv/Linear forward MACs {architecture['conv_linear_macs_per_epoch']} per EEG epoch, approximate FLOPs {architecture['approximate_conv_linear_flops_per_epoch']} excluding norm/activation/pooling/backward/optimizer. Same architecture and64-value penultimate representation as M1.",
            f"M2 peak process RSS {resources['process_peak_rss_bytes']/2**30:.3f}GiB. End-of-run MPS allocator/driver values are stored separately and are not peak readings. Dataset access uses the existing bounded read-only recording memmaps; no normalized EEG copy is created.",
            'Training accuracy/F1 are accumulated online with dropout and changing weights; validation is eval mode after each epoch. Timing is descriptive across separate executions: cache/thermal/system/software effects prevent a controlled optimizer speed claim. M1 historical per-epoch N1/N3 scores were not retained; only its selected-checkpoint class scores are compared.','',
            '| Model | Mean epoch s | Completed epoch total s | Time through best epoch s | Final validation F1 | Best validation F1 |',
            '| --- | ---: | ---: | ---: | ---: | ---: |']
    for row in rows:
        h=histories[row['model_id']]
        lines.append(f"| {row['model_id']} | {row['mean_epoch_seconds']:.2f} | {row['completed_epoch_seconds']:.2f} | {row['time_to_best_seconds']:.2f} | {h[-1]['validation_macro_f1']:.6f} | {row['macro_f1']:.6f} |")
    h=histories['M2']
    lines+=['',f"M2 first-epoch validation macro-F1 {h[0]['validation_macro_f1']:.6f}, final {h[-1]['validation_macro_f1']:.6f}; realized best epoch {m2['best_epoch_1_based']}. Best and latest weights have independent roles. Equal maximum budgets can stop at different realized epochs.",'',
            '## Artifacts and verification','',
            f"M2 run: `experiments/cnn/{m2['run_id']}`. Best: `{Path(checkpoints['M2']['best']).relative_to(project)}`; latest: `{Path(checkpoints['M2']['latest']).relative_to(project)}`. Immutable generations/checksums, optimizer, null scheduler, criterion, early-stop, CPU/MPS RNG, sampler/generators, source/software/config/fingerprints and authoritative epoch history are retained.",
            f"Validation export `{summary['M2_export']}` includes all30266 IDs, labels, five logits/probabilities, subject/recording/night/original index/onset/split and quality flags. Penultimate and pooled intermediate features are each[30266,64], with exact row mappings and checkpoint-bound hashes. Analysis: `{summary['M2_analysis']}`; PCA uses all rows and t-SNE a reproducibly sampled500-row subset(seed42,perplexity30,500iterations). Saved class/subject metrics, calibration/ROC/PR and original-index-respecting hypnograms remain available.",
            'Reviewable copies and M1/M2 class/confusion/learning plots: `results/phase3c_figures/`. `phase3c_validation_comparison.csv` and `phase3c_classwise_scores.csv` preserve machine-readable results. Analysis never imports/calls the trainer; original M1 plots/exports are read only.',
            'Real-data MPS smoke checked exact normalization, finite forward/loss/gradients, CE equivalence and interrupted-epoch recovery. Automated tests include exact-CNN CPU optimizer/history equality and corrupted-latest recovery. Bitwise MPS reproducibility is not promised. Separate test and final integrity evidence are in `phase3c_test_results.*` and `phase3c_final_validation.json`.',
            'The registry predates full training. All previous source/config/data/reports/pilot/M1 artifacts are checked against `phase3c_preservation_snapshot.json`; earlier test outputs are not overwritten. No packages/environments/downloads/raw EDF writes occurred.','',
            '## Commands and stopping boundary','', '```sh',
            'PYTHONPATH=src /Volumes/Mac/envs/dl-m4/bin/python scripts/run_phase3c_tests.py',
            'PYTHONPATH=src /Volumes/Mac/envs/dl-m4/bin/python -m sleep_research.phase3c register',
            'PYTHONPATH=src /Volumes/Mac/envs/dl-m4/bin/python -m sleep_research.phase3c smoke',
            'PYTHONPATH=src /Volumes/Mac/envs/dl-m4/bin/python -m sleep_research.phase3c train-m2 --resume auto',
            'PYTHONPATH=src /Volumes/Mac/envs/dl-m4/bin/python -m sleep_research.phase3c_analysis','```','',
            'M3/M4 and the predefined seeds42/43/44 plan are prepared in `configs/phase3c/`; resource discussion is in `phase3c_multiseed_plan.md`. M1 seed42 is reused, never retrained. One-seed validation selection cannot establish seed variability, population significance or test generalization. Historical R&K merging, offline Wake trimming/filtering, retained artifacts and filename-derived identities remain limitations.',
            '**Stopped after M2 full training and validation analysis. No M3/M4, additional seed, advanced architecture or frozen final test evaluation.**']
    (results/'phase3c_m2_results.md').write_text('\n'.join(lines)+'\n')
    plan=results/'phase3c_multiseed_plan.md';text=plan.read_text().split('\n## Resource update after measured M2')[0]
    m1mean=rows[0]['mean_epoch_seconds'];m2mean=rows[1]['mean_epoch_seconds'];low=min(m1mean,m2mean)*25/60*.8;high=max(m1mean,m2mean)*25/60*1.3
    text+=f'\n## Resource update after measured M2\n\nM2 measured {m2["completed_epochs"]} completed epochs in {m2["training_attempt_seconds"]:.2f}s of training/validation attempts, mean {m2mean:.2f}s per completed epoch, peak process RSS {resources["process_peak_rss_bytes"]/2**30:.3f}GiB. For remaining full25-epoch budgets, use a cautious {low:.1f}–{high:.1f}minutes per run based on M1/M2 epoch timing with0.8/1.3 uncertainty multipliers; ten future runs imply {low*10/60:.2f}–{high*10/60:.2f}GPU-hours plus inference/analysis. Early stopping may shorten this. Keep the100MiB/run storage allowance and3–4GiB process-memory allowance; shared EEG is not duplicated. All future timing remains estimated, not measured.\n'
    plan.write_text(text)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--project',type=Path,default=PROJECT);args=parser.parse_args()
    result=compare(args.project.resolve());print(json.dumps({k:v for k,v in result.items() if k not in ('models',)},indent=2))
