"""Publication analysis of committed frozen exports only; never imports a trainer."""
import json
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .common import digest,fingerprint,json_write,now
from .phase5_registry import PROJECT,plan,align
from .phase5_evaluation import authorization,workspace,checked_export
from .analysis import analyze_export,load_export
from .metrics import CLASS_NAMES
from .phase4_analysis import transition_metrics
from .phase5_statistics import detailed_metrics,subject_bootstrap
from .phase5_data import validate_saved_context


def savefig(fig,path):
    fig.savefig(path.with_suffix('.png'),dpi=300,bbox_inches='tight')
    fig.savefig(path.with_suffix('.pdf'),bbox_inches='tight');plt.close(fig)


def representation_summary(features,labels):
    x=np.asarray(features,dtype=np.float64);center=x.mean(0);within=0.;between=0.;classes=[]
    for label in range(5):
        subset=x[labels==label]
        if not len(subset):continue
        mean=subset.mean(0);within+=float(((subset-mean)**2).sum());between+=float(len(subset)*((mean-center)**2).sum())
        classes.append(dict(label=label,count=len(subset),centroid=mean.tolist()))
    return dict(classes=classes,between_scatter=between,within_scatter=within,
                between_within_ratio=between/within if within else None,
                note='Descriptive centroid scatter ratio in each model embedding; not a validated separability test or directly comparable feature geometry')


def regenerate(project=PROJECT,explicit=False,sha=None):
    body=plan(project);authorization(project,body,explicit,sha)
    project=Path(project);home=workspace(project,body);reference=json.loads((home/'exports.json').read_text())
    if reference['plan_sha256']!=fingerprint(body) or set(reference['models'])!={r['key'] for r in body['models']}:
        raise ValueError('All nine registered models must complete before final result reports')
    canonical=pd.read_csv(project/'results/protocol_a_index.csv').query("split == 'test'").reset_index(drop=True)
    frames={};exports={}
    for row in body['models']:
        ref=reference['models'][row['key']];path=project/ref['export']
        if digest(path/'manifest.json')!=ref['manifest_sha256'] or ref['checkpoint_sha256']!=row['checkpoint_sha256']:
            raise ValueError('Frozen export reference differs')
        exports[row['key']]=checked_export(path,canonical,row['checkpoint_sha256'])
        frames[row['key']]=exports[row['key']][1]
        validate_saved_context(frames[row['key']],canonical,1 if row['model_id'] in ['M1','M4'] else 5)
    figures=project/'results/phase5_figures';figures.mkdir(exist_ok=True)
    out=home/'analysis';out.mkdir(exist_ok=True);rows=[];classes=[];subjectrows=[];recordingrows=[];subsets=[];transitions=[];confusions=[]
    for row in body['models']:
        key=row['key'];manifest,frame,features,root=exports[key]
        tick=__import__('time').monotonic()
        params=body['analysis'];analysis=analyze_export(project/reference['models'][key]['export'],final_evaluation=True,
            tsne_max_samples=params['tsne_max_samples'],tsne_perplexity=params['tsne_perplexity'],tsne_iterations=params['tsne_iterations'],seed=params['seed'])
        metrics,curves=detailed_metrics(frame);json_write(analysis/'phase5_detailed_metrics.json',metrics)
        json_write(analysis/'phase5_representation_summary.json',representation_summary(features,frame.true_label.to_numpy(dtype=int)))
        rows.append(dict(key=key,model_id=row['model_id'],role=row['role'],seed=row['seed'],parameters=row['parameters'],
            best_epoch_1_based=row['best_epoch_1_based'],training_seconds=row['training_seconds'],inference_export_seconds=reference['models'][key].get('inference_export_seconds'),inference_timing_complete=reference['models'][key].get('timing_complete'),epochs=len(frame),subjects=frame.subject_id.nunique(),
            checkpoint_sha256=row['checkpoint_sha256'],**{k:metrics[k] for k in ['accuracy','macro_f1','micro_f1','weighted_f1','balanced_accuracy','cohen_kappa','multiclass_brier_score','expected_calibration_error']},
            n1_f1=metrics['classes']['N1']['f1'],n3_f1=metrics['classes']['N3']['f1']))
        for stage,c in metrics['classes'].items():classes.append(dict(key=key,seed=row['seed'],stage=stage,**c))
        for column,destination in [('subject_id',subjectrows),('recording_id',recordingrows)]:
            for name,group in frame.groupby(column,observed=True):
                m,_=detailed_metrics(group);destination.append(dict(key=key,identifier=str(name),epochs=len(group),
                    **{k:m[k] for k in ['accuracy','macro_f1','balanced_accuracy','cohen_kappa']},
                    **{f'support_{i}':m['classes'][stage]['support'] for i,stage in enumerate(CLASS_NAMES)}))
        for name,mask in [('flagged',frame.quality_flagged==1),('flag_free',frame.quality_flagged==0)]:
            population=frame[mask];m=detailed_metrics(population)[0] if len(population) else None
            subsets.append(dict(key=key,subset=name,epochs=len(population),subjects=population.subject_id.nunique(),
                accuracy=m['accuracy'] if m else None,macro_f1=m['macro_f1'] if m else None,
                note='Descriptive retained quality subset; no exclusion-based model tuning'))
        transition=transition_metrics(frame);json_write(analysis/'phase5_transition_metrics.json',transition)
        for name in ['transition','stable']:
            m=transition[name];transitions.append(dict(key=key,subset=name,epochs=m['epochs'],accuracy=m['metrics']['accuracy'] if m['metrics'] else None,macro_f1=m['metrics']['macro_f1'] if m['metrics'] else None))
        matrix=np.asarray(metrics['confusion_matrix']);normal=np.divide(matrix,matrix.sum(1,keepdims=True),out=np.zeros((5,5)),where=matrix.sum(1,keepdims=True)>0)
        fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
        for ax,a,title in zip(axes,[matrix,normal],['Counts','Row-normalized']):
            ax.imshow(a,cmap='Blues');ax.set(xticks=range(5),yticks=range(5),xticklabels=CLASS_NAMES,yticklabels=CLASS_NAMES,xlabel='Predicted',ylabel='True',title=key+' '+title)
            for i in range(5):
                for j in range(5):ax.text(j,i,f'{a[i,j]:.2f}' if title!='Counts' else str(a[i,j]),ha='center',va='center',fontsize=7)
        savefig(fig,figures/(key+'_confusion'))
        fig,axes=plt.subplots(1,3,figsize=(15,4),constrained_layout=True)
        for stage,c in curves.items():
            if c['defined']:
                axes[0].plot(c['false_positive_rate'],c['true_positive_rate'],label=f"{stage}: {c['roc_auc']:.3f}")
                axes[1].plot(c['recall'],c['precision'],label=f"{stage}: {c['average_precision']:.3f}")
        axes[0].plot([0,1],[0,1],'k--',lw=.5);axes[0].set(xlabel='False-positive rate',ylabel='True-positive rate',title='ROC')
        axes[1].set(xlabel='Recall',ylabel='Precision',title='Precision-recall')
        cal=[x for x in metrics['calibration_bins'] if x['count']];axes[2].plot([0,1],[0,1],'k--',lw=.5)
        axes[2].plot([x['confidence'] for x in cal],[x['accuracy'] for x in cal],marker='o');axes[2].set(xlabel='Mean confidence',ylabel='Observed accuracy',title=f"ECE {metrics['expected_calibration_error']:.3f}")
        axes[0].legend(fontsize=7);axes[1].legend(fontsize=7);savefig(fig,figures/(key+'_roc_pr_calibration'))
        for name in ['pca','tsne']:
            coords=pd.read_csv(analysis/(name+'_coordinates.csv'));fig,ax=plt.subplots(figsize=(6,5))
            for i,stage in enumerate(CLASS_NAMES):
                mask=coords.true_label==i;ax.scatter(coords.iloc[mask.to_numpy(),2],coords.iloc[mask.to_numpy(),3],s=5,alpha=.45,label=stage)
            ax.set(title=key+' '+name.upper(),xlabel=name+' 1',ylabel=name+' 2');ax.legend();savefig(fig,figures/(key+'_'+name))
        # Predefined first five original-order confusion examples per true/predicted W,N1,N2 pair.
        candidates=frame[frame.true_label.isin([0,1,2]) & frame.predicted_label.isin([0,1,2]) & (frame.true_label!=frame.predicted_label)]
        chosen=candidates.groupby(['true_label','predicted_label'],sort=True,observed=True).head(5).copy()
        chosen['checkpoint_sha256']=row['checkpoint_sha256'];chosen.to_csv(analysis/'n1_n2_wake_confusion_examples.csv',index=False)
        confusions.extend(chosen.assign(key=key).to_dict('records'))
        if len(chosen):
            fig,ax=plt.subplots(figsize=(9,max(3,.24*len(chosen))))
            ax.imshow(chosen[[f'probability_{i}' for i in range(5)]].to_numpy(),aspect='auto',vmin=0,vmax=1,cmap='viridis')
            ax.set(xticks=range(5),xticklabels=CLASS_NAMES,yticks=range(len(chosen)),
                yticklabels=[f'{r.epoch_id} true:{CLASS_NAMES[int(r.true_label)]} pred:{CLASS_NAMES[int(r.predicted_label)]}' for r in chosen.itertuples()],
                title=key+' — predefined first W/N1/N2 confusion examples',xlabel='Saved class probability')
            ax.tick_params(axis='y',labelsize=6);savefig(fig,figures/(key+'_confusion_examples'))
        # Reuse linked development explanations; no test EEG/gradient extraction for figure choice.
        evidence=project/row['validation_explanations'];description=json.loads((evidence/'manifest.json').read_text())
        if description['checkpoint_sha256']!=row['checkpoint_sha256']:raise ValueError('Explanation checkpoint differs')
        for name,value in description['files'].items():
            if digest(evidence/name)!=value:raise ValueError('Explanation evidence changed')
        for image in evidence.glob('*.png'):shutil.copyfile(image,figures/(key+'_validation_'+image.name))
        json_write(analysis/'reused_validation_explanations.json',dict(source=str(evidence.relative_to(project)),manifest_sha256=digest(evidence/'manifest.json'),split='validation',checkpoint_sha256=row['checkpoint_sha256'],note='Descriptive gradients/attention; not causal physiology; no test-driven example selection'))
        print(now(),'ANALYZED',key,'seconds',round(__import__('time').monotonic()-tick,2),flush=True)
    comparison=pd.DataFrame(rows);comparison.to_csv(project/'results/phase5_model_comparison.csv',index=False)
    pd.DataFrame(classes).to_csv(project/'results/phase5_classwise_metrics.csv',index=False)
    for name,data in [('subject_metrics',subjectrows),('recording_metrics',recordingrows),('quality_subsets',subsets),('transition_metrics',transitions),('confusion_examples',confusions)]:pd.DataFrame(data).to_csv(project/f'results/phase5_{name}.csv',index=False)
    stats=body['statistics'];groups={k+'_fixed_seeds':v for k,v in body['primary_groups'].items()}
    comparisons=[[a+'_fixed_seeds',b+'_fixed_seeds'] for a,b in body['primary_comparisons']]
    uncertainty=subject_bootstrap(frames,groups,comparisons+body['secondary_comparisons'],resamples=stats['resamples'],seed=stats['seed'],confidence=stats['confidence'],simultaneous_confidence=stats['primary_simultaneous_confidence'])
    json_write(project/'results/phase5_subject_bootstrap.json',uncertainty)
    seeds=[]
    for name,keys in body['primary_groups'].items():
        population=comparison.set_index('key').loc[keys]
        for metric in ['accuracy','macro_f1','n1_f1','n3_f1','balanced_accuracy','cohen_kappa']:
            x=population[metric].to_numpy(dtype=float);seeds.append(dict(model_id=name,metric=metric,seeds=len(x),mean=float(x.mean()),sample_sd=float(x.std(ddof=1)) if len(x)>1 else None,min=float(x.min()),max=float(x.max())))
    pd.DataFrame(seeds).to_csv(project/'results/phase5_seed_variability.csv',index=False)
    for x,label,name in [('parameters','Trainable parameters','parameters_vs_performance'),('training_seconds','Measured training seconds','cost_vs_performance')]:
        fig,ax=plt.subplots(figsize=(8,5))
        for r in rows:ax.scatter(r[x],r['macro_f1']);ax.annotate(r['key'],(r[x],r['macro_f1']),fontsize=7)
        ax.set(xlabel=label,ylabel='Frozen test macro-F1');savefig(fig,figures/name)
    table=comparison[['key','parameters','accuracy','macro_f1','n1_f1','n3_f1','balanced_accuracy','cohen_kappa']].to_markdown(index=False) if __import__('importlib').util.find_spec('tabulate') else comparison[['key','accuracy','macro_f1','n1_f1','n3_f1']].to_string(index=False)
    (project/'results/phase5_final_test_results.md').write_text('# Frozen final-test results\n\n'+f"Plan `{body['plan_id']}`; SHA-256 `{fingerprint(body)}`. All nine preselected checkpoint exports verified on the identical {len(canonical):,} epochs, 12 subjects, 22 recordings. No tuning/training or lucky-seed selection.\n\n"+table+'\n\nM1/M4/weighted M5 have one seed; primary M5/M7 have three. Primary model-level estimates average metrics across all predefined seeds without ensembling. See class-wise, subject/recording, quality, transition and uncertainty artifacts. No new model selection is permitted from these outcomes.\n')
    for d in uncertainty['paired_differences']:
        d['primary_macro_f1_comparison']=[d['a'],d['b']] in comparisons and d['metric']=='macro_f1'
    json_write(project/'results/phase5_subject_bootstrap.json',uncertainty)
    diffs=pd.DataFrame([dict(primary_macro_f1_comparison=r['primary_macro_f1_comparison'],a=r['a'],b=r['b'],metric=r['metric'],estimate=r['estimate'],lower95=r['percentile_interval']['lower'],upper95=r['percentile_interval']['upper'],lower_simultaneous=r['simultaneous_interval']['lower'] if r['primary_macro_f1_comparison'] else None,upper_simultaneous=r['simultaneous_interval']['upper'] if r['primary_macro_f1_comparison'] else None) for r in uncertainty['paired_differences']])
    diffs.to_csv(project/'results/phase5_paired_comparisons.csv',index=False)
    (project/'results/phase5_statistical_analysis.md').write_text('# Subject-aware uncertainty\n\n'+stats['procedure']+'\n\n10,000 paired resamples, seed20261016; 95% percentile intervals. The three predefined primary comparisons additionally show Bonferroni98.333% marginal percentile intervals (approximate simultaneous family coverage); secondary comparisons remain descriptive. No null-hypothesis p-values or superiority declarations. Only12 independent subjects; percentile coverage and rare-class intervals can be unstable. Subject resampling conditions on the trained seeds; seed sample SD is reported separately and is null for M1. Five-class F1 uses zero for undefined classes; balanced accuracy averages classes present; undefined kappa is null and valid-replicate denominators are reported. See phase5_subject_bootstrap.json and phase5_paired_comparisons.csv.\n\n'+diffs.to_string(index=False)+'\n')
    # Only finalized measured files receive the final completion commit.
    json_write(home/'analysis_complete.json',dict(plan_sha256=fingerprint(body),completed_at=now(),models=list(frames),training_called=False,
        files={str(p.relative_to(project)):digest(p) for p in list(figures.rglob('*'))+list(project.glob('results/phase5_*.csv'))+list(project.glob('results/phase5_*results.md'))+[project/'results/phase5_statistical_analysis.md',project/'results/phase5_subject_bootstrap.json'] if p.is_file()}))
    from .phase5_publication import research_report
    research_report(project,body,final=True)
    reproducibility=project/'results/phase5_reproducibility_report.md'
    initial=reproducibility.read_text().split('\n## Authorized final evaluation completed')[0]
    reproducibility.write_text(initial+'\n## Authorized final evaluation completed\n\nAll nine preselected checkpoint exports passed canonical target/checksum/embedding verification. See experiments/final_evaluation/'+body['plan_id']+'/exports.json and analysis_complete.json for permanent evidence. No training or model selection was performed by final analysis. Paired subject-bootstrap draws and source/model/data fingerprints are retained. Inference/export times are lower bounds if an orphan commit was adopted without its timing record.\n')
    return dict(successful=True,models=9,training_called=False,final_test_evaluated=True)
