"""Compare verified saved exports; no imports or calls to the trainer."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from .analysis import analyze_export,load_export
from .common import digest,fingerprint,json_write,now
from .metrics import classification_metrics,CLASS_NAMES

PROJECT=Path(__file__).resolve().parents[2]


def measured_row(result,project):
    manifest,frame,features,root=load_export(project/result['export'])
    if manifest['splits']!=['validation'] or manifest['synthetic']:
        raise ValueError('Only real validation exports are eligible')
    cfg=json.loads((root/'reproducibility/configuration.json').read_text())
    history=json.loads((root/manifest['history_snapshot']).read_text())
    metrics=classification_metrics(frame.true_label.to_numpy(),frame[[f'probability_{i}' for i in range(5)]].to_numpy())
    best=manifest['completed_epoch']
    selected=history[best]
    np.testing.assert_allclose(metrics['macro_f1'],selected['validation_macro_f1'],atol=1e-12,rtol=0)
    if cfg['stage']!='full':
        saved=json.loads((project/'results/phase3b_pilot_subsets.json').read_text())
        if frame.epoch_id.tolist()!=saved['epoch_ids']['validation']:
            raise ValueError('Pilot comparison populations differ')
    row=dict(run_id=result['run_id'],candidate_id=result['candidate_id'],stage=result['stage'],scope='pilot_subset' if cfg['stage']!='full' else 'full_validation',
             optimizer=cfg['optimizer']['name'],learning_rate=cfg['optimizer']['learning_rate'],loss=cfg['loss']['name'],
             seed=cfg['seed'],training_epochs=result['training_count'],validation_epochs=len(frame),
             training_subjects=None,
             validation_subjects=frame.subject_id.nunique(),completed_epochs=len(history),best_epoch_1_based=best+1,
             accuracy=metrics['accuracy'],macro_f1=metrics['macro_f1'],balanced_accuracy=metrics['balanced_accuracy'],cohen_kappa=metrics['cohen_kappa'],
             training_seconds=result['wall_training_seconds'],time_to_best_seconds=sum(r['seconds'] for r in history[:best+1]),
             epoch_seconds_mean=float(np.mean([r['seconds'] for r in history])),
             epoch_macro_f1_sd=float(np.std([r['validation_macro_f1'] for r in history],ddof=1)) if len(history)>1 else None,
             finite_completed_run=True,peak_process_rss_gib=result['process_high_water_rss_bytes']/2**30,
             unique_predicted_classes=frame.predicted_label.nunique(),single_class_collapse=frame.predicted_label.nunique()==1,
             checkpoint_sha256=manifest['checkpoint_sha256'],export=result['export'],synthetic=False)
    for stage,value in metrics['classes'].items():
        for key in ['precision','recall','f1','support']:row[f'{stage.lower()}_{key}']=value[key]
        row[f'{stage.lower()}_predicted_count']=int((frame.predicted_label==value['label']).sum())
    return row,manifest,frame,features,root,history,metrics


def compare(project=PROJECT):
    project=Path(project);results=project/'results'
    search=yaml.safe_load((project/'configs/phase3b_optimizer_search.yaml').read_text())
    measured=json.loads((results/'phase3b_pilot_runs.json').read_text())
    if measured['status']!='complete' or len(measured['runs'])!=8:
        raise ValueError('All eight registered pilot trials must complete before the final comparison')
    output=results/'phase3b_figures';output.mkdir(exist_ok=True)
    rows=[];histories={};analyses={}
    for result in measured['runs']:
        row,manifest,frame,features,root,history,metrics=measured_row(result,project)
        subset=json.loads((results/'phase3b_pilot_subsets.json').read_text())
        row['training_subjects']=len({epoch.split(':')[0][:5] for epoch in subset['epoch_ids']['train']})
        rows.append(row);histories[row['run_id']]=history
        params=search['analysis']
        analysis=analyze_export(project/result['export'],tsne_max_samples=params['tsne_max_samples'],
                    tsne_perplexity=params['tsne_perplexity'],tsne_iterations=params['tsne_iterations'],seed=params['seed'],hypnograms=False)
        analyses[row['run_id']]=str(analysis.relative_to(project))
        # Explicit one-vs-rest 2x2 confusion matrices, alongside the standard 5x5.
        matrix=np.array(metrics['confusion_matrix']);fig,axes=plt.subplots(1,5,figsize=(13,3),constrained_layout=True)
        for i,ax in enumerate(axes):
            tp=matrix[i,i];fn=matrix[i].sum()-tp;fp=matrix[:,i].sum()-tp;tn=matrix.sum()-tp-fn-fp
            binary=np.array([[tn,fp],[fn,tp]]);ax.imshow(binary,cmap='Blues')
            for y in range(2):
                for x in range(2):ax.text(x,y,str(binary[y,x]),ha='center',va='center')
            ax.set(title=CLASS_NAMES[i],xticks=[0,1],yticks=[0,1],xlabel='Predicted',ylabel='True')
        fig.savefig(analysis/'classwise_confusion.png',dpi=140);plt.close(fig)
        fig,axes=plt.subplots(1,3,figsize=(12,4),constrained_layout=True)
        epochs=[r['epoch']+1 for r in history]
        axes[0].plot(epochs,[r['train_loss'] for r in history],label='Train objective')
        axes[0].plot(epochs,[r['validation_objective_loss'] for r in history],label='Validation objective')
        axes[0].set(xlabel='Completed epoch',ylabel=cfg_loss_label(row['loss']));axes[0].legend(fontsize=7)
        axes[1].plot(epochs,[r['validation_loss'] for r in history]);axes[1].set(xlabel='Completed epoch',ylabel='Unweighted validation CE')
        for key,label in [('validation_macro_f1','Macro-F1'),('validation_n1_f1','N1 F1'),('validation_n3_f1','N3 F1')]:
            axes[2].plot(epochs,[r[key] for r in history],label=label)
        axes[2].set(xlabel='Completed epoch',ylabel='Validation F1',ylim=(0,1));axes[2].legend(fontsize=7)
        fig.savefig(analysis/'objective_learning_curves.png',dpi=140);plt.close(fig)
        saved=json.loads((analysis/'analysis_manifest.json').read_text())
        saved['files']={p.name:digest(p) for p in analysis.iterdir() if p.is_file() and p.name!='analysis_manifest.json'}
        json_write(analysis/'analysis_manifest.json',saved)
    frame=pd.DataFrame(rows)
    frame.to_csv(results/'phase3b_validation_comparison.csv',index=False)
    optimizer=frame[frame.stage=='optimizer_pilot'].sort_values('macro_f1',ascending=False,kind='stable')
    selection=json.loads((results/'phase3b_optimizer_selection.json').read_text())
    assert optimizer.iloc[0].run_id==selection['selected_run_id']
    chosen=optimizer.iloc[0]
    losses=frame[(frame.candidate_id==chosen.candidate_id)]
    for name,population in [('optimizer',optimizer),('loss',losses)]:
        fig,axes=plt.subplots(1,2,figsize=(12,4),constrained_layout=True)
        x=np.arange(len(population));labels=[f'{r.optimizer}\n{r.learning_rate:g}\n{r.loss}' for r in population.itertuples()]
        for offset,key,label in [(-.24,'macro_f1','Macro-F1'),(0,'n1_f1','N1 F1'),(.24,'n3_f1','N3 F1')]:
            axes[0].bar(x+offset,population[key],width=.24,label=label)
        axes[0].set(xticks=x,xticklabels=labels,ylabel='Validation F1',ylim=(0,1),title='Seed42 subset pilot');axes[0].tick_params(axis='x',labelsize=7);axes[0].legend(fontsize=7)
        for r in population.itertuples():
            h=histories[r.run_id];axes[1].plot([v['epoch']+1 for v in h],[v['validation_macro_f1'] for v in h],marker='o',label=f'{r.candidate_id}/{r.loss}')
        axes[1].set(xlabel='Completed epoch',ylabel='Validation macro-F1',ylim=(0,1));axes[1].legend(fontsize=6)
        fig.savefig(output/f'{name}_comparison.png',dpi=140);plt.close(fig)
    # M1 on the same validation IDs is a reference with a substantially larger training budget.
    old=json.loads((results/'phase3_validation_export.json').read_text())
    old_manifest,old_frame,_,_=load_export(project/old['export'])
    subset=json.loads((results/'phase3b_pilot_subsets.json').read_text())
    old_subset=old_frame.set_index('epoch_id').loc[subset['epoch_ids']['validation']]
    m1=classification_metrics(old_subset.true_label.to_numpy(),old_subset[[f'probability_{i}' for i in range(5)]].to_numpy())
    json_write(results/'phase3b_m1_subset_reference.json',dict(metrics=m1,source_export=old['export'],
               source_checkpoint_sha256=old_manifest['checkpoint_sha256'],sample_count=len(old_subset),
               reference_only=True,training_budget_not_matched=True,baseline_artifacts_modified=False))
    candidates=[]
    # Promote the two strongest CE optimizers, then compare losses under the first.
    for row in optimizer.head(2).itertuples():candidates.append(dict(candidate_id=row.candidate_id,loss=dict(name='cross_entropy'),role='optimizer_confirmation'))
    for loss in search['loss_candidates']:
        role='loss_confirmation_after_optimizer_confirmation' if loss['name']=='weighted_cross_entropy' else 'optional_deferred_focal_confirmation'
        candidates.append(dict(candidate_id=chosen.candidate_id,loss=loss,role=role))
    proposals=[]
    for candidate in candidates:
        opt=next(c['optimizer'] for c in search['optimizer_candidates'] if c['id']==candidate['candidate_id'])
        matching=frame[(frame.candidate_id==candidate['candidate_id']) & (frame.loss==candidate['loss']['name'])].iloc[0]
        # Small pilot projections are noisy: use both M1 full-epoch evidence and the pilot ratio.
        m1_history=json.loads((project/'experiments/cnn'/search['baseline_run_id']/'performance/history.json').read_text())
        ratio=(134047+30266)/(8192+4096)
        projected=float(matching.epoch_seconds_mean)*ratio
        m1_mean=float(np.mean([r['seconds'] for r in m1_history]))
        low=min(projected,m1_mean)*25/60*.8;high=max(projected,m1_mean)*25/60*1.3
        proposals.append(dict(**candidate,optimizer=opt,seeds=search['repeat_seeds'],max_epochs=25,
                    training_examples=134047,validation_examples=30266,batch_size=16,precision='float32',device='mps',
                    scheduler=search['full_run_proposal']['scheduler'],early_stopping=search['full_run_proposal']['early_stopping'],
                    seed_varies_initialization_and_shuffle_only=True,split_normalization_fixed=True,
                    estimated_minutes_per_seed=[low,high],estimated_three_seed_gpu_hours=[low*3/60,high*3/60],
                    estimated_process_rss_gib=[3,4],estimated_storage_mib_per_seed=100,
                    measured=False,auto_launch=False))
    json_write(results/'phase3b_full_run_proposals.json',dict(configurations=proposals,full_runs_launched=0,
               estimation='25*(pilot mean epoch seconds * population ratio and measured M1 mean epoch seconds), range min*.8 to max*1.3; thermal/caching/optimizer effects uncertain; early stop may shorten.'))
    summary=dict(successful=True,generated_at=now(),pilot_trials=len(frame),seed=42,full_runs_launched=0,
                 winning_ce_candidate=chosen.candidate_id,winning_ce_macro_f1=float(chosen.macro_f1),
                 measured_rows=rows,analyses=analyses,training_called=False,frozen_test_evaluated=False)
    json_write(results/'phase3b_analysis_summary.json',summary)
    write_reports(project,frame,optimizer,losses,m1,proposals)
    return summary


def cfg_loss_label(name):return f'{name} objective (own reduction)'


def write_reports(project,frame,optimizer,losses,m1,proposals):
    results=project/'results';best=optimizer.iloc[0]
    lines=['# Phase 3B — measured optimizer/loss pilot comparison',f'Generated: {now()}.','',
           '**Eight real MPS subset trials completed, seed42 only; zero new full runs.**',
           'Identical frozen8192train/4096validation subsets, three epochs/trial, same40501-parameter M1 architecture, initialization/shuffle seed, batch16, float32, normalization and splits. All flags retained. Test remains locked.',
           'These short subset trials cannot establish improvement over the fully trained M1 or estimate between-seed variability. RMSprop and class-balanced focal were tested/smoked but were not trained as pilot trials.','',
           '| Stage | Optimizer/LR | Loss | Macro-F1 | N1 F1 | N3 F1 | Accuracy | Bal. accuracy | Kappa | Best epoch | Seconds to best | Total train+val s |',
           '| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for r in frame.itertuples():
        lines.append(f'| {r.stage} | {r.optimizer}/{r.learning_rate:g} | {r.loss} | {r.macro_f1:.4f} | {r.n1_f1:.4f} | {r.n3_f1:.4f} | {r.accuracy:.4f} | {r.balanced_accuracy:.4f} | {r.cohen_kappa:.4f} | {r.best_epoch_1_based} | {r.time_to_best_seconds:.2f} | {r.training_seconds:.2f} |')
    lines+=['',f'Optimizer-stage selection: **{best.candidate_id}**, macro-F1 **{best.macro_f1:.6f}**. This selection was recorded before the two loss trials. Same metric chooses each run’s retained best epoch; exact ties use registered order.',
            'Full CE optimizer ranking is in the CSV. The selected optimizer’s CE trial is the loss-stage control; loss trials change only the objective. Training loss reductions differ; validation unweighted CE is stored for every trial and objective-specific validation loss is separate.','',
            '## Baseline context and limitations','',
            f'M1, evaluated from its preserved saved predictions on these same4096 validation IDs: macro-F1 {m1["macro_f1"]:.6f}, N1 F1 {m1["classes"]["N1"]["f1"]:.6f}, N3 F1 {m1["classes"]["N3"]["f1"]:.6f}. M1 used all134047 training examples and17 epochs, so this is a descriptive reference, not a matched optimizer comparison.',
            'Across-epoch F1 sample SD and seconds/epoch are saved in the CSV. They describe short-run fluctuations, not random-seed stability. All trials completed with finite losses/logits; the16real smoke updates also had finite gradients/parameters. Finite arithmetic does not guarantee useful classification: SGD at0.05 collapsed to one predicted class at its selected epoch (kappa0). Predicted class counts and a collapse indicator are retained. No failed trial was silently excluded. Learning curves, class-wise one-vs-rest/5x5 confusions, per-subject metrics, PCA and seeded200-row t-SNE are saved under each run’s analysis folder.',
            'All subsets span the frozen subject groups; sampling did not reallocate subjects. No flag removal, class resampling, test use or M1 fine-tuning. All initializations are fresh. The optimizer-specific LR probes are sparse; a slowly converging SGD configuration may be underestimated by three epochs. Shared numeric weight_decay0.01 does not imply identical coupled/decoupled regularization.',
            'Validation selection across candidates/epochs is optimistic; confirm a shortlist with matched full budgets and seeds42/43/44 before drawing performance conclusions. No test generalization estimate, confidence interval or significance claim is made.','',
            '## Artifacts and commands','',
            '`phase3b_validation_comparison.csv` contains all measured trial metrics, class support/precision/recall/F1, resource/time information and exact checkpoint/export identities. `phase3b_pilot_runs.json` maps permanent run IDs; `phase3b_trial_registry.json` permits safe campaign continuation. Each run retains config/source/software/fingerprints, optimizer/null-scheduler/criterion/RNG/sampler/history state, best/latest/previous checkpoints, all five logits/probabilities/IDs/quality flags and64-value penultimate/pooled intermediate features.',
            '`phase3b_analysis_summary.json` maps analysis directories. `phase3b_figures/` contains comparison figures. `03b_optimizer_analysis.ipynb` reads saved artifacts only; it never launches training.',
            '```sh','PYTHONPATH=src /Volumes/Mac/envs/dl-m4/bin/python scripts/run_phase3b_tests.py',
            'PYTHONPATH=src /Volumes/Mac/envs/dl-m4/bin/python -m sleep_research.phase3b pilot',
            'PYTHONPATH=src /Volumes/Mac/envs/dl-m4/bin/python -m sleep_research.phase3b resume --run-id <EXISTING_PHASE3B_RUN_ID>',
            'PYTHONPATH=src /Volumes/Mac/envs/dl-m4/bin/python -m sleep_research.phase3b_analysis','```',
            'Use `phase3b_experiment_plan.md` and `phase3b_full_run_proposals.json` for proposed full configurations and resource estimates. **Stopped after the bounded pilot; zero full runs launched.**']
    weighted=losses[losses.loss=='weighted_cross_entropy'].iloc[0]
    lines.insert(lines.index('## Baseline context and limitations'),
                 f'For the selected optimizer, weighted CE changed macro-F1 by {weighted.macro_f1-best.macro_f1:+.6f}, N1 F1 by {weighted.n1_f1-best.n1_f1:+.6f}, N3 F1 by {weighted.n3_f1-best.n3_f1:+.6f}, and accuracy by {weighted.accuracy-best.accuracy:+.6f}. This supports a matched full-budget confirmation, not an improvement claim over M1. Focal is deferred because this pilot did not improve the primary metric.')
    (results/'phase3b_optimizer_comparison.md').write_text('\n\n'.join(lines[:6])+'\n\n'+'\n'.join(lines[6:])+'\n')
    plan=results/'phase3b_experiment_plan.md';text=plan.read_text().split('\n## Pilot-informed full-run proposals')[0]
    text+='\n## Pilot-informed full-run proposals (not launched)\n\n'
    text+='Confirm the strongest CE optimizers first. If one optimizer is confirmed, compare its CE/weighted CE/focal objectives. The order below is a staged proposal, not a full-grid launch. Keep all fixed controls above.\n\n'
    text+='| Proposal | Optimizer/LR | Loss | Seeds | Max epochs | Minutes/seed estimate | Three-seed GPU-hours estimate |\n| --- | --- | --- | --- | ---: | --- | --- |\n'
    for p in proposals:
        lo,hi=p['estimated_minutes_per_seed'];a,b=p['estimated_three_seed_gpu_hours']
        text+=f'| {p["role"]} | {p["optimizer"]["name"]}/{p["optimizer"]["learning_rate"]:g} | {p["loss"]["name"]} | 42,43,44 | 25 | {lo:.1f}–{hi:.1f} | {a:.2f}–{b:.2f} |\n'
    text+='\nEach proposal uses batch16, MPS float32, fixed LR/no scheduler, patience8, ordinary weight_decay0.01 and the exact frozen architecture/data/normalization. SGD momentum settings remain as registered. Focal usesgamma2. Full settings and estimates are in `phase3b_full_run_proposals.json`; process memory3–4GiB and storage allowance100MiB/seed excluding shared EEG. Repeats reuse the same subject allocation; only model/order RNG seed changes.\n'
    text+='\nTiming estimates combine pilot mean epoch time scaled by total train+validation population with M1 measured full-epoch time, using an uncertainty range of0.8×minimum to1.3×maximum over25epochs. Train/validation cost differs, thermal throttling/caching vary, and early stopping may shorten runs. Do not interpret estimates as measured full-run timing.\n'
    first=proposals[0]
    text+=f'\nExample command for one proposed full run, **not executed**:\n\n```sh\nPYTHONPATH=src /Volumes/Mac/envs/dl-m4/bin/python -m sleep_research.phase3b train-one --candidate {first["candidate_id"]} --loss cross_entropy --seed 42\n```\n'
    text+='\nThe CLI supports one explicit full run per call and seeds42/43/44; there is no automatic multi-seed/full-grid launcher. No additional full runs are launched in this phase.\n'
    plan.write_text(text)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--project',type=Path,default=PROJECT)
    args=parser.parse_args();result=compare(args.project.resolve())
    print(json.dumps({k:v for k,v in result.items() if k not in ('measured_rows','analyses')},indent=2))


if __name__=='__main__':main()
