"""Build Phase6 tables/figures from frozen saved evidence, without model forward/training.

Run with the existing interpreter from the project root. This file is deliberately
outside the locked Phase1–5 src/scripts/tests trees. Historical artifacts are read-only.
"""
from pathlib import Path
import csv,json,sys,math,shutil,re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
from scipy.signal import sosfreqz
P=Path(__file__).resolve().parents[2];sys.path.insert(0,str(P/'src'))
from sleep_research.common import digest,fingerprint,json_write,now
from sleep_research.phase5_registry import plan,verify_files
from sleep_research.phase5_data import validate_saved_context
from sleep_research.phase5_evaluation import checked_export,workspace
from sleep_research.phase5_statistics import detailed_metrics,subject_bootstrap
from models.cnn1d import CNN1D,architecture_details
from models.advanced_sleep import make_advanced

T=P/'results/publication_tables';F=P/'results/publication_figures';R=P/'reports'
for directory in [T,F]:directory.mkdir(parents=True,exist_ok=True)
GROUPS={'CNN':['M1'],'CNN-BiLSTM':['m5_ce_s42','m5_ce_s43','m5_ce_s44'],'CNN-Transformer':['m7_ce_s42','m7_ce_s43','m7_ce_s44']}
STAGES=['Wake','N1','N2','N3','REM'];COLORS=['#466a95','#ca8326','#3f8f76'];STAGE_COLORS=['#777777','#da9d30','#4a84b7','#584a93','#c15b72']
figures=[];inputs={};ledger={};facts={}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'savefig.dpi':300})


def readcsv(relative):
    inputs[relative]=digest(P/relative);return pd.read_csv(P/relative)


def readjson(relative):
    inputs[relative]=digest(P/relative);return json.loads((P/relative).read_text())


def number(key,value,source,fmt='.4f'):
    if isinstance(value,(np.integer,np.floating)):value=value.item()
    formatted=format(value,fmt)
    facts[key]=formatted;ledger[key]=dict(value=value,formatted=formatted,source=source,format=fmt)
    return formatted


def table(name,frame,caption):
    frame.to_csv(T/(name+'.csv'),index=False)
    def cell(v):
        if pd.isna(v):return '—'
        if isinstance(v,(float,np.floating)):return f'{v:.4f}'
        return str(v)
    columns=list(frame.columns)
    md='| '+' | '.join(columns)+' |\n| '+' | '.join(['---']*len(columns))+' |\n'+'\n'.join('| '+' | '.join(cell(v) for v in row)+' |' for row in frame.itertuples(index=False,name=None))
    (T/(name+'.md')).write_text(md+'\n\n'+caption+'\n')
    return md


def save(fig,name,caption,source):
    fig.savefig(F/(name+'.png'),dpi=300,bbox_inches='tight');fig.savefig(F/(name+'.pdf'),bbox_inches='tight');plt.close(fig)
    figures.append(dict(name=name,caption=caption,sources=source,png_sha256=digest(F/(name+'.png')),pdf_sha256=digest(F/(name+'.pdf'))))


def main():
    body=plan(P);sha=fingerprint(body);home=workspace(P,body)
    commit=readjson(str((home/'analysis_complete.json').relative_to(P)));verify_files(P,commit['files'])
    refs=readjson(str((home/'exports.json').relative_to(P)))
    assert sha==commit['plan_sha256']==refs['plan_sha256']
    index=readcsv('results/protocol_a_index.csv');splits=readcsv('results/subject_splits.csv');canonical=index.query("split == 'test'").reset_index(drop=True)
    test=readcsv('results/phase5_model_comparison.csv').set_index('key');val=readcsv('results/phase4_models_comparison.csv').set_index('key')
    oldclasses=readcsv('results/phase5_classwise_metrics.csv');oldseed=readcsv('results/phase5_seed_variability.csv');oldci=readjson('results/phase5_subject_bootstrap.json')
    frames={};features={};roots={};metrics={};histories={};analysis={}
    for row in body['models']:
        key=row['key'];ref=refs['models'][key];export=P/ref['export'];assert digest(export/'manifest.json')==ref['manifest_sha256']
        inputs[str((export/'manifest.json').relative_to(P))]=ref['manifest_sha256']
        manifest,frame,feature,root=checked_export(export,canonical,row['checkpoint_sha256'])
        assert manifest['dataset_contract']==body['dataset_contract'] and manifest['experiment_id']==row['run_id']
        validate_saved_context(frame,canonical,1 if row['model_id'] in ['M1','M4'] else 5)
        m,_=detailed_metrics(frame)
        for metric in ['accuracy','macro_f1','weighted_f1','balanced_accuracy','cohen_kappa','multiclass_brier_score','expected_calibration_error']:
            np.testing.assert_allclose(m[metric],test.loc[key,metric],atol=1e-12,rtol=0)
        for stage,scores in m['classes'].items():
            saved=oldclasses[(oldclasses.key==key)&(oldclasses.stage==stage)].iloc[0]
            for metric in ['precision','recall','f1','specificity','roc_auc','average_precision']:
                np.testing.assert_allclose(scores[metric],saved[metric],atol=1e-12,rtol=0)
        model=CNN1D() if row['model_id'] in ['M1','M4'] else make_advanced(row['configuration']['architecture'])
        assert sum(p.numel() for p in model.parameters())==row['parameters']
        del model
        frames[key]=frame;features[key]=feature;roots[key]=root;metrics[key]=m;analysis[key]=root/'analysis'/manifest['export_id']
        histories[key]=readjson(str((root/manifest['history_snapshot']).relative_to(P)))
        assert manifest['checkpoint_sha256']==digest(P/row['checkpoint'])
    args=body['statistics'];groups={k+'_fixed_seeds':v for k,v in body['primary_groups'].items()}
    comparisons=[[a+'_fixed_seeds',b+'_fixed_seeds'] for a,b in body['primary_comparisons']]
    bootstrap=subject_bootstrap(frames,groups,comparisons+body['secondary_comparisons'],resamples=args['resamples'],seed=args['seed'],confidence=args['confidence'],simultaneous_confidence=args['primary_simultaneous_confidence'])
    assert bootstrap['bootstrap_weights_sha256']==oldci['bootstrap_weights_sha256']
    assert bootstrap['estimates']==oldci['estimates']
    for a,b in zip(bootstrap['paired_differences'],oldci['paired_differences']):
        assert all(a[k]==b[k] for k in a)
    json_write(T/'bootstrap_recalculation.json',bootstrap)
    number('bootstrap_draws',args['resamples'],'results/phase5_subject_bootstrap.json','d');number('test_epochs',len(canonical),'results/protocol_a_index.csv',',d')
    number('test_subjects',canonical.subject_id.nunique(),'results/subject_splits.csv','d');number('test_recordings',canonical.recording_id.nunique(),'results/protocol_a_index.csv','d')
    number('selected_checkpoints',len(body['models']),'results/phase5_evaluation_plan.json','d');number('protocol_A_epochs',len(index),'results/protocol_a_index.csv',',d')
    distribution=readcsv('results/stage_distribution_comparison.csv')
    full=distribution.query("protocol == 'B' and split == 'all'").sort_values('label');trimmed=distribution.query("protocol == 'A' and split == 'all'").sort_values('label')
    assert len(full)==len(trimmed)==5
    number('full_epochs',int(full.epoch_count.sum()),'results/stage_distribution_comparison.csv',',d')
    number('wake_removed',int(full.epoch_count.sum()-trimmed.epoch_count.sum()),'results/stage_distribution_comparison.csv',',d')
    number('wake_full_pct',full.query('label == 0').percent_of_protocol_split.iloc[0],'results/stage_distribution_comparison.csv','.2f')
    number('wake_A_pct',trimmed.query('label == 0').percent_of_protocol_split.iloc[0],'results/stage_distribution_comparison.csv','.2f')
    inventory=readcsv('results/dataset_inventory.csv')
    for study in ['SC','ST']:
        sample=inventory[inventory.study==study]
        number(study+'_recordings',len(sample),'results/dataset_inventory.csv','d');number(study+'_subjects',sample.subject_id.nunique(),'results/dataset_inventory.csv','d')
    for name in ['movement_epochs','unknown_epochs','missing_epochs','overlap_epochs','invalid_timing_epochs']:
        number('SC_'+name,int(inventory.query("study == 'SC'")[name].sum()),'results/dataset_inventory.csv',',d')
    ds=[]
    for split in ['train','validation','test']:
        a=index[index.split==split];b=distribution.query('protocol == "B" and split == @split')
        ds.append(dict(split=split,subjects=a.subject_id.nunique(),recordings=a.recording_id.nunique(),A_epochs=len(a),B_epochs=int(b.epoch_count.sum())))
        number(split+'_epochs',len(a),'results/protocol_a_index.csv',',d');number(split+'_subjects',a.subject_id.nunique(),'results/subject_splits.csv','d')
    table('01_dataset_split',pd.DataFrame(ds),'Fixed subject-independent allocation; nights never cross splits. Protocol B is descriptive only; all models use A.')
    table('02_stage_distribution',pd.DataFrame(dict(stage=STAGES,B_epochs=full.epoch_count.to_numpy(),B_percent=full.percent_of_protocol_split.to_numpy(),A_epochs=trimmed.epoch_count.to_numpy(),A_percent=trimmed.percent_of_protocol_split.to_numpy())),'SC only; denominator is the valid epoch count within the indicated protocol, all subject splits combined.')
    summary=[];classrows=[];cost=[];subjectrows=[];transitionrows=[]
    per_subject=readcsv('results/phase5_subject_metrics.csv');transition=readcsv('results/phase5_transition_metrics.csv')
    for label,keys in GROUPS.items():
        model=body['models'][next(i for i,r in enumerate(body['models']) if r['key']==keys[0])];code=model['model_id']
        rec=dict(model=label,seeds=len(keys),parameters=model['parameters'])
        for metric in ['accuracy','macro_f1','n1_f1','n3_f1','balanced_accuracy','cohen_kappa']:
            x=test.loc[keys,metric].to_numpy(dtype=float);mean=float(x.mean());sd=float(x.std(ddof=1)) if len(keys)>1 else None
            rec[metric+'_mean']=mean;rec[metric+'_seed_sd']=sd
            number('test_'+code+'_'+metric,mean,'results/phase5_model_comparison.csv grouped fixed seeds')
            if sd is not None:number('test_'+code+'_'+metric+'_sd',sd,'results/phase5_seed_variability.csv')
            ci=oldci['estimates'][code+'_fixed_seeds'][metric]
            if metric=='macro_f1':
                rec['macro_f1_subject_CI_low']=ci['lower'];rec['macro_f1_subject_CI_high']=ci['upper']
                number(code+'_CI_low',ci['lower'],'results/phase5_subject_bootstrap.json');number(code+'_CI_high',ci['upper'],'results/phase5_subject_bootstrap.json')
            if metric=='accuracy':number('test_'+code+'_accuracy_percent',100*mean,'results/phase5_model_comparison.csv','.2f')
            saved=oldseed[(oldseed.model_id==code)&(oldseed.metric==metric)].iloc[0];np.testing.assert_allclose(mean,saved['mean'],atol=1e-12)
        summary.append(rec)
        for stage in STAGES:
            pop=oldclasses[(oldclasses.key.isin(keys))&(oldclasses.stage==stage)]
            classrows.append(dict(model=label,stage=stage,support_per_seed=int(pop.support.iloc[0]),**{m:float(pop[m].mean()) for m in ['precision','recall','f1','specificity','roc_auc','average_precision']}))
        number('parameters_'+code,model['parameters'],'model source and final plan',',d')
        number('validation_'+code+'_macro',float(val.loc[keys,'macro_f1'].mean()),'results/phase4_models_comparison.csv')
        sub=per_subject[per_subject.key.isin(keys)].groupby('identifier',sort=True,observed=True)[['macro_f1','accuracy']].mean()
        for subject,r in sub.iterrows():subjectrows.append(dict(model=label,subject_id=subject,macro_f1=float(r.macro_f1),accuracy=float(r.accuracy)))
        for subset in ['transition','stable']:
            pop=transition[(transition.key.isin(keys))&(transition.subset==subset)]
            transitionrows.append(dict(model=label,subset=subset,epochs_per_seed=int(pop.epochs.iloc[0]),macro_f1=float(pop.macro_f1.mean()),accuracy=float(pop.accuracy.mean())))
            number(code+'_'+subset+'_f1',float(pop.macro_f1.mean()),'results/phase5_transition_metrics.csv fixed-seed mean')
        conv_macs=architecture_details(CNN1D())['conv_linear_macs_per_epoch']
        macs=conv_macs if code=='M1' else (5*(conv_macs-320)+2*5*4*32*(64+32)+320 if code=='M5' else 5*(conv_macs-320)+2*(4*5*64*64+2*5*5*64+2*5*64*128)+320)
        cost.append(dict(model=label,seeds=len(keys),parameters=model['parameters'],float32_weight_KiB=model['parameters']*4/1024,
            approximate_forward_MACs_per_target=macs,training_minutes_mean=float(test.loc[keys,'training_seconds'].mean()/60),
            inference_export_seconds_mean=float(test.loc[keys,'inference_export_seconds'].mean()),best_checkpoint_bytes=int((P/model['checkpoint']).stat().st_size)))
        number(code+'_MAC_millions',macs/1e6,'Analytic counts from saved model source; multiply-accumulate convention excludes nonlinearity/norm/pool/bias','.3f')
        number(code+'_train_minutes',cost[-1]['training_minutes_mean'],'results/phase5_model_comparison.csv','.2f')
        number(code+'_inference_seconds',cost[-1]['inference_export_seconds_mean'],'results/phase5_model_comparison.csv','.2f')
    table('03_primary_test_summary',pd.DataFrame(summary),'Scores are 0–1. Means/seed SD are computed across predefined fixed training seeds; M1 SD is undefined. Subject bootstrap intervals condition on those fitted seeds, and are not seed-variability intervals.')
    display_rows=[]
    for rec in summary:
        display_rows.append(dict(model=rec['model'],seeds=rec['seeds'],parameters=rec['parameters'],accuracy=f"{rec['accuracy_mean']:.4f}",macro_F1=f"{rec['macro_f1_mean']:.4f}"+(f" ± {rec['macro_f1_seed_sd']:.4f}" if rec['macro_f1_seed_sd'] is not None else ' (single seed)'),N1_F1=f"{rec['n1_f1_mean']:.4f}",subject_95_CI=f"[{rec['macro_f1_subject_CI_low']:.4f}, {rec['macro_f1_subject_CI_high']:.4f}]"))
    table('03a_primary_display',pd.DataFrame(display_rows),'Macro-F1 ± sample SD across fixed training seeds; the interval column is separately computed paired-subject uncertainty conditional on those seeds. Scores are ratios0–1.')
    table('04a_classwise_display',pd.DataFrame(classrows)[['model','stage','support_per_seed','precision','recall','f1']],'P/R/F1 are fixed-seed arithmetic means; each seed evaluates the same true-class supports.')
    table('04_primary_classwise',pd.DataFrame(classrows),'Arithmetic mean of per-seed class scores, not scores derived from an averaged confusion matrix. Each seed has the same class support.')
    table('05_paired_comparisons',readcsv('results/phase5_paired_comparisons.csv'),'Differences a−b. 95% subject-paired intervals; simultaneous intervals apply only to the three primary macro-F1 comparisons, not the exploratory secondary metrics.')
    table('06_computational_cost',pd.DataFrame(cost),'Measured training-attempt and chunk inference/export wall times are from separate executions on M4/MPS, not controlled kernel benchmarks. Analytic MACs exclude normalization, activation, pooling, biases, softmax, and elementwise recurrent operations; two FLOPs per MAC is only a partial operation convention. Checkpoint bytes include metadata/state, whereas weight KiB counts trainable float32 parameters only.')
    table('07_subjectwise_primary',pd.DataFrame(subjectrows),'Each subject value averages fixed-seed metrics; all five classes enter macro-F1, with undefined class F1=0. This unweighted subject description differs from primary epoch-pooled macro-F1.')
    table('08_transition_primary',pd.DataFrame(transitionrows),'Target is a transition if the previous consecutive original epoch within subject/recording/night differs in true label. Boundaries/gaps excluded. Fixed-seed means, descriptive only.')
    table('09_all_nine_checkpoints',pd.DataFrame([{k:r[k] for k in ['key','model_id','role','seed','parameters','best_epoch_1_based','checkpoint','checkpoint_sha256']} for r in body['models']]),'Exact preselected best-validation checkpoints. No post-test seed or model selection.')
    table('10_all_seed_test_metrics',test.reset_index(),'Original nine-run final metrics re-exported without changes to original files.')
    table('11_validation_metrics',val.reset_index(),'Historical validation comparisons including M8 and bounded ablations; never treated as final test evidence.')
    table('12_quality_subsets',readcsv('results/phase5_quality_subsets.csv'),'Predefined retained quality descriptions; small flagged subsets are class-composition confounded and do not establish artifact robustness.')
    table('13_all_classwise_metrics',oldclasses,'All nine individual seed/checkpoint class results including support, specificity, AUC and AP.')
    table('14_recording_metrics',readcsv('results/phase5_recording_metrics.csv'),'Saved per-recording values; recordings from the same subject are not independent replicates.')
    for name,a,b in [('M5_minus_M1','M5_fixed_seeds','M1_fixed_seeds'),('M7_minus_M1','M7_fixed_seeds','M1_fixed_seeds'),('M7_minus_M5','M7_fixed_seeds','M5_fixed_seeds')]:
        d=next(x for x in oldci['paired_differences'] if x['a']==a and x['b']==b and x['metric']=='macro_f1')
        for suffix,value in [('delta',d['estimate']),('low',d['percentile_interval']['lower']),('high',d['percentile_interval']['upper']),('sim_low',d['simultaneous_interval']['lower']),('sim_high',d['simultaneous_interval']['upper'])]:number(name+'_'+suffix,value,'results/phase5_subject_bootstrap.json')
    for key in ['M4','m5_wce_s42']:
        for metric in ['macro_f1','n1_f1','accuracy']:number(key+'_'+metric,float(test.loc[key,metric]),'results/phase5_model_comparison.csv')
    for key in ['M4','m5_wce_s42']:
        b='M1' if key=='M4' else 'm5_ce_s42'
        for metric in ['macro_f1','n1_f1']:
            d=next(x for x in oldci['paired_differences'] if x['a']==key and x['b']==b and x['metric']==metric)
            for part,value in [('delta',d['estimate']),('low',d['percentile_interval']['lower']),('high',d['percentile_interval']['upper'])]:number(key+'_'+metric+'_'+part,value,'results/phase5_subject_bootstrap.json')
    number('m5_ce_s42_N1',float(test.loc['m5_ce_s42','n1_f1']),'results/phase5_model_comparison.csv')
    number('m5_ce_s42_F1',float(test.loc['m5_ce_s42','macro_f1']),'results/phase5_model_comparison.csv')
    number('transition_epochs',3498,'results/phase5_transition_metrics.csv',',d');number('stable_epochs',27628,'results/phase5_transition_metrics.csv',',d')
    assert set(transition.query("subset == 'transition'").epochs)=={3498} and set(transition.query("subset == 'stable'").epochs)=={27628}
    number('excluded_transition_boundary_epochs',len(canonical)-3498-27628,'Canonical epochs minus valid transition comparison targets','d')
    number('flagged_test',int(frames['M1'].quality_flagged.sum()),'Saved test export quality flags','d')
    norm=readjson('results/normalization.json')['protocols']['A'];number('norm_mean',norm['mean_microvolts'],'results/normalization.json','.8f');number('norm_std',norm['std_microvolts'],'results/normalization.json','.6f')
    for key in ['m8_ce_s42','m8_ablation_control_s42','m8_ablation_no_multiscale_s42','m8_ablation_no_context_s42']:
        number(key+'_macro',float(val.loc[key,'macro_f1']),'results/phase4_models_comparison.csv')
    number('N1_test_support',int(metrics['M1']['classes']['N1']['support']),'Saved test labels',',d')
    # Publication figures: all representations and sample predictions are seed42 by prior convention.
    fig,ax=plt.subplots(figsize=(12,6));ax.set(xlim=(0,12),ylim=(0,6));ax.axis('off')
    def box(x,y,w,h,text,color='#edf2f7'):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.08',facecolor=color,edgecolor='#425466'));ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=8)
    def arrow(a,b):ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=12,color='#425466'))
    box(.1,4.5,2.4,1,'EEG Fpz-Cz, 100 Hz\nContinuous filter → 30-s epochs\nFixed train-only z-score')
    box(3.1,4.5,3.1,1,'Epoch CNN encoder\nchannels 16/32/64/64\nkernels 25/9/7/5; GroupNorm(4)\nReLU + MaxPool(2); average pool')
    arrow((2.6,5),(3,5));box(7,4.5,4.7,1,'M1: central 64-D embedding\nDropout 0.3 → Linear(64,5)\n40,501 parameters; one EEG epoch')
    arrow((6.3,5),(6.9,5))
    box(.1,2.3,2.4,1.2,'Five centered epochs\nOriginal indices k−2…k+2\nContiguous within recording\nMasks + zero padding')
    box(3.1,2.3,3.1,1.2,'Shared epoch encoder\n(fresh weights per run)\nFive 64-D tokens\nNo neighbor labels')
    arrow((2.6,2.9),(3,2.9));box(7,2.5,4.7,1,'M5: masked BiLSTM, hidden32/direction\nConcatenate central states → LayerNorm\nDropout0.3 → Linear(64,5); 65,717 params')
    box(7,.5,4.7,1.3,'M7: learned 5×64 positions\n2 attention/residual blocks: 4 heads, width64\nFFN64→128→64, dropout0.1; central token\nLayerNorm → Dropout0.3 → Linear(64,5)\n107,893 parameters')
    arrow((6.3,2.9),(6.9,3));arrow((6.3,2.9),(6.9,1.1))
    ax.text(.1,.7,'All models: one five-class prediction per central target.\nM5/M7 use future EEG: offline protocol.\nMissing keys are masked; padded LSTM updates skipped.',fontsize=9)
    save(fig,'01_architectures','Implemented compact architectures, not reproductions of published networks. Context and architecture both differ from M1; all targets and normalization are shared.','src/models/cnn1d.py; src/models/advanced_sleep.py')
    fig,axes=plt.subplots(1,2,figsize=(10,4),constrained_layout=True);x=np.arange(5)
    for ax,col,title in zip(axes,['epoch_count','percent_of_protocol_split'],['Epoch counts','Within-protocol percentage']):
        ax.bar(x-.18,full[col],.36,label='B: all valid',color='#70889f');ax.bar(x+.18,trimmed[col],.36,label='A: Wake-trimmed',color='#3f8f76');ax.set(xticks=x,xticklabels=STAGES,title=title,ylabel='30-s epochs' if col=='epoch_count' else 'Percent of valid epochs');ax.legend(fontsize=8)
    save(fig,'02_stage_distribution','All SC subjects combined for descriptive eligibility counts. Only external Wake is removed in A; no non-Wake class changes.','results/stage_distribution_comparison.csv')
    fig,axes=plt.subplots(2,1,figsize=(10,7),gridspec_kw={'height_ratios':[2,1]},constrained_layout=True)
    original=P/'results/phase2_figures/SC4001E_eeg_psd.png';inputs[str(original.relative_to(P))]=digest(original)
    axes[0].imshow(plt.imread(original));axes[0].axis('off');axes[0].set_title('Historical training example: SC4001E, original epoch1021 (N1)',fontsize=10)
    response=readjson('results/filter_response.json');f,h=sosfreqz(np.asarray(response['sos']),worN=4096,fs=100)
    axes[1].plot(f,20*np.log10(np.maximum(np.abs(h)**2,1e-12)),color=COLORS[0]);axes[1].set(xlim=(0,50),ylim=(-80,3),xlabel='Frequency (Hz)',ylabel='Forward/backward gain (dB)');axes[1].axvline(.3,ls='--',c='grey');axes[1].axvline(35,ls='--',c='grey')
    save(fig,'03_preprocessing','Top: unchanged saved training-only raster illustration of raw/continuous-filtered EEG and PSD, embedded at native raster resolution; no new EDF read or preprocessing. Bottom: analytic squared-amplitude response from saved SOS coefficients. The training example is illustrative, not an estimate of artifact prevalence.','results/phase2_figures/SC4001E_eeg_psd.png; results/filter_response.json')
    # Primary mean, subject CI and separate seed variation.
    fig,axes=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
    for i,(label,keys) in enumerate(GROUPS.items()):
        code='M1' if i==0 else 'M5' if i==1 else 'M7';xv=test.loc[keys,'macro_f1'].to_numpy();ci=oldci['estimates'][code+'_fixed_seeds']['macro_f1'];mean=xv.mean()
        axes[0].errorbar(i,mean,yerr=[[mean-ci['lower']],[ci['upper']-mean]],fmt='o',color=COLORS[i],capsize=5)
        axes[1].scatter(np.full(len(xv),i)+np.linspace(-.08,.08,len(xv)),xv,c=COLORS[i]);axes[1].plot([i-.2,i+.2],[mean,mean],c=COLORS[i])
    for ax in axes:ax.set(xticks=range(3),xticklabels=list(GROUPS),ylabel='Frozen-test five-class macro-F1',ylim=(.65,.84));ax.tick_params(axis='x',rotation=12)
    axes[0].set_title('95% paired-subject bootstrap intervals');axes[1].set_title('All predefined seeds; line = arithmetic mean')
    save(fig,'04_primary_comparison','Left intervals resample12 test subjects, conditional on the saved checkpoints. Right points display training-seed variability; M1 has one seed, M5/M7 three. Neither panel chooses a test-winning seed.','results/phase5_subject_bootstrap.json; results/phase5_model_comparison.csv')
    fig,axes=plt.subplots(3,2,figsize=(10,12),constrained_layout=True)
    for i,(label,key) in enumerate(zip(GROUPS,['M1','m5_ce_s42','m7_ce_s42'])):
        matrix=np.asarray(metrics[key]['confusion_matrix']);normed=matrix/matrix.sum(1,keepdims=True)
        for j,a in enumerate([matrix,normed]):
            ax=axes[i,j];ax.imshow(a,cmap='Blues');ax.set(xticks=range(5),yticks=range(5),xticklabels=STAGES,yticklabels=STAGES,xlabel='Predicted stage',ylabel='True stage',title=label+' seed42 — '+('counts' if j==0 else 'row normalized'))
            for r in range(5):
                for c in range(5):ax.text(c,r,str(a[r,c]) if j==0 else f'{a[r,c]:.2f}',ha='center',va='center',fontsize=8,color='white' if a[r,c]>a.max()*.6 else 'black')
    save(fig,'05_confusion_matrices','Predefined seed42 illustrative matrices; rows are true classes, columns predicted classes. Each raw matrix sums to31,156. Model-level results use all fixed seeds and are reported separately.','Frozen saved prediction exports for M1, M5 seed42 and M7 seed42')
    fig,axes=plt.subplots(1,3,figsize=(12,4),constrained_layout=True);cl=pd.DataFrame(classrows)
    for j,metric in enumerate(['precision','recall','f1']):
        for i,label in enumerate(GROUPS):axes[j].bar(x+(i-1)*.25,cl[cl.model==label][metric],.25,label=label,color=COLORS[i])
        axes[j].set(xticks=x,xticklabels=STAGES,ylim=(0,1),ylabel='Fixed-seed mean score',title=metric.capitalize())
    axes[0].legend(fontsize=7)
    save(fig,'06_classwise','Arithmetic means of per-seed class precision/recall/F1, not a pooled seed ensemble. Denominator supports are shared and recorded in Table04.','results/phase5_classwise_metrics.csv')
    fig,ax=plt.subplots(figsize=(8,3.6))
    for i,(a,b) in enumerate(comparisons):
        d=next(r for r in oldci['paired_differences'] if r['a']==a and r['b']==b and r['metric']=='macro_f1');m=d['estimate'];ci=d['percentile_interval'];sim=d['simultaneous_interval']
        ax.errorbar(m,i,xerr=[[m-sim['lower']],[sim['upper']-m]],fmt='none',color='#b9c5d1',lw=5)
        ax.errorbar(m,i,xerr=[[m-ci['lower']],[ci['upper']-m]],fmt='o',color='#234e70',capsize=4)
    ax.axvline(0,color='black',ls='--',lw=.8);ax.set(yticks=range(3),yticklabels=['CNN-BiLSTM − CNN','CNN-Transformer − CNN','CNN-Transformer − CNN-BiLSTM'],xlabel='Difference in frozen-test macro-F1');ax.invert_yaxis()
    save(fig,'07_paired_intervals','Dark95% paired-subject percentile intervals; pale98.333% intervals for the three predeclared primary comparisons. M7−M5 crosses zero; reliable superiority between M5/M7 is not established.','results/phase5_subject_bootstrap.json')
    fig,axes=plt.subplots(1,3,figsize=(12,4),constrained_layout=True)
    for ax,(label,keys) in zip(axes,GROUPS.items()):
        for key in keys:
            h=histories[key];epochs=[r['epoch']+1 for r in h];f1=[r['validation_macro_f1'] for r in h];ax.plot(epochs,f1,marker='.',label='seed'+str(int(test.loc[key,'seed'])));best=int(test.loc[key,'best_epoch_1_based']);ax.scatter(best,f1[best-1],marker='*',s=80)
        ax.set(title=label,xlabel='Completed training epoch (1-based)',ylabel='Validation macro-F1',ylim=(.4,.85));ax.legend(fontsize=7)
    save(fig,'08_training_convergence','Validation-only curves for every predefined primary checkpoint seed. Stars indicate best-validation epochs, never test-selected epochs.','Immutable checkpoint-derived saved export histories')
    fig,axes=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
    for i,r in enumerate(summary):
        axes[0].scatter(r['parameters']/1000,r['macro_f1_mean'],color=COLORS[i],s=65);axes[0].annotate(r['model'],(r['parameters']/1000,r['macro_f1_mean']),xytext=(4,4),textcoords='offset points',fontsize=8)
        axes[1].scatter(cost[i]['inference_export_seconds_mean'],r['macro_f1_mean'],color=COLORS[i],s=65);axes[1].annotate(r['model'],(cost[i]['inference_export_seconds_mean'],r['macro_f1_mean']),xytext=(4,4),textcoords='offset points',fontsize=8)
    axes[0].set(xlabel='Trainable parameters (thousands)',ylabel='Fixed-seed mean test macro-F1',xlim=(25,145),ylim=(.68,.82));axes[1].set(xlabel='Mean wall seconds:31,156-target inference/export',ylabel='Fixed-seed mean test macro-F1',xlim=(0,48),ylim=(.68,.82))
    save(fig,'09_parameter_cost','Compactness versus measured task performance. Timing includes chunk loading/export on one Mac and is not a controlled latency or energy-efficiency comparison.','results/phase5_model_comparison.csv; analytic model counts')
    fig,ax=plt.subplots(figsize=(8,4));errrows=[]
    for i,(label,keys) in enumerate(GROUPS.items()):
        distribution_n1=[]
        for key in keys:
            group=frames[key].query('true_label == 1');distribution_n1.append(np.bincount(group.predicted_label,minlength=5)/len(group))
        means=np.mean(distribution_n1,axis=0);ax.bar(x+(i-1)*.25,means,.25,color=COLORS[i],label=label)
        for stage,p in zip(STAGES,means):errrows.append(dict(model=label,true_stage='N1',predicted_stage=stage,fraction=float(p),true_N1_support=len(group)))
    ax.set(xticks=x,xticklabels=STAGES,ylabel='Fraction of true N1 epochs',ylim=(0,.65),xlabel='Predicted stage');ax.legend(fontsize=8)
    save(fig,'10_N1_errors','Distribution of predictions conditional on true N1, averaged over fixed seeds. The N1 column is recall; other columns are error fractions.','Saved final predictions, true N1 support3111 per seed')
    table('15_N1_error_distribution',pd.DataFrame(errrows),'Each model distribution sums to1; includes correct N1 predictions. No confusion example selects a new model.')
    fig,axes=plt.subplots(3,1,figsize=(11,7),sharex=True,constrained_layout=True);recording=sorted(canonical.recording_id.unique())[0];positions=np.array([0,2,3,4,1])
    from sleep_research.analysis import contiguous_blocks
    for ax,label,key in zip(axes,GROUPS,['M1','m5_ce_s42','m7_ce_s42']):
        g=frames[key].query('recording_id == @recording').sort_values('original_epoch_index');indices=g.original_epoch_index.to_numpy(dtype=int)
        for j,block in enumerate(contiguous_blocks(indices)):
            edges=np.r_[indices[block],indices[block[-1]]+1]*30/3600
            for column,color,legend in [('true_label','black','True'),('predicted_label','#397abd','Predicted')]:ax.stairs(positions[g.iloc[block][column].to_numpy(dtype=int)],edges,color=color,lw=.7,label=legend if j==0 else None)
        ax.set(yticks=range(5),yticklabels=['Wake','REM','N1','N2','N3'],ylim=(4.5,-.5),ylabel=label);ax.legend(fontsize=7,loc='upper right')
    axes[-1].set_xlabel('Hours from original PSG onset');fig.suptitle(recording+' — deterministic first recording, seed42')
    save(fig,'11_hypnograms','Lexicographically first test recording, not chosen for favorable performance. Original time and gaps retained; all three illustrative checkpoints use seed42.','Saved final export metadata and predictions')
    selections=[]
    fig,axes=plt.subplots(2,3,figsize=(12,7),constrained_layout=True)
    for j,(label,key) in enumerate(zip(GROUPS,['M1','m5_ce_s42','m7_ce_s42'])):
        for i,method in enumerate(['pca','tsne']):
            path=analysis[key]/(method+'_coordinates.csv');inputs[str(path.relative_to(P))]=digest(path);coords=pd.read_csv(path)
            joined=coords.merge(frames[key][['epoch_id','subject_id','recording_id','original_epoch_index']],on='epoch_id',validate='one_to_one');joined['checkpoint_sha256']=test.loc[key,'checkpoint_sha256'];joined.to_csv(T/(key+'_'+method+'_linked_coordinates.csv'),index=False)
            selections.append(dict(model=label,method=method,source=str(path.relative_to(P)),checkpoint_sha256=test.loc[key,'checkpoint_sha256'],rows=len(coords)))
            for stage,name,color in zip(range(5),STAGES,STAGE_COLORS):
                mask=coords.true_label==stage;axes[i,j].scatter(coords.iloc[mask.to_numpy(),2],coords.iloc[mask.to_numpy(),3],s=2 if method=='pca' else 9,alpha=.3 if method=='pca' else .65,c=color,label=name,rasterized=True)
            axes[i,j].set(title=label+' seed42 — '+method.upper(),xlabel=method+' coordinate1',ylabel=method+' coordinate2')
    axes[0,0].legend(markerscale=3,fontsize=7)
    save(fig,'12_embeddings','Saved embeddings: PCA all31,156 targets, t-SNE uniform seeded500-target subset, seed42/perplexity30/500 iterations. Separate models have independent coordinate systems; visual separation is descriptive.','Original saved PCA/t-SNE coordinates and parameter files')
    json_write(T/'representation_lineage.json',selections)
    # Re-render saved bounded validation explanations; no new model gradient calculation.
    fig,axes=plt.subplots(2,3,figsize=(12,6),constrained_layout=True)
    for j,(label,key) in enumerate(zip(GROUPS,['M1','m5_ce_s42','m7_ce_s42'])):
        row=next(r for r in body['models'] if r['key']==key);folder=P/row['validation_explanations'];meta=readjson(str((folder/'manifest.json').relative_to(P)))
        verify_files(folder,meta['files']);assert meta['checkpoint_sha256']==row['checkpoint_sha256'];z=np.load(folder/'explanations.npz',allow_pickle=False)
        idx=1;grad=z['absolute_saliency'][idx];cam=z['grad_cam'][idx] if 'grad_cam' in z else z['grad_cam_1d'][idx]
        if grad.ndim==2 and len(grad)==1:axes[0,j].plot(np.arange(3000)/100,grad[0],lw=.5);axes[0,j].set(xlabel='Seconds within epoch',ylabel='Absolute logit gradient')
        else:
            im=axes[0,j].imshow(grad,aspect='auto',extent=[0,30,4.5,-.5]);axes[0,j].set(xlabel='Seconds within context epoch',ylabel='Context slot');fig.colorbar(im,ax=axes[0,j],shrink=.8,label='Absolute gradient')
        axes[0,j].set_title(label+' validation N1 saliency')
        if cam.ndim==2 and cam.shape[0]==1:cam=cam[0]
        if cam.ndim==1:axes[1,j].plot(np.arange(3000)/100,cam,lw=.5);axes[1,j].set(ylim=(0,1),xlabel='Seconds within epoch',ylabel='Grad-CAM')
        else:
            im=axes[1,j].imshow(cam,aspect='auto',vmin=0,vmax=1,extent=[0,30,4.5,-.5]);axes[1,j].set(xlabel='Seconds within context epoch',ylabel='Context slot');fig.colorbar(im,ax=axes[1,j],shrink=.8,label='Grad-CAM')
        axes[1,j].set_title(str(z['epoch_ids'][idx]))
    save(fig,'13_explanations','Previously saved first validation example of true N1 from each seed42 checkpoint; predicted-logit saliency and compatible1D Grad-CAM. Selected without test outcomes. Scales are model dependent; not causal physiology. Exact IDs/checkpoint hashes remain in source manifests.','Checkpoint-linked original validation explanations.npz')
    row=next(r for r in body['models'] if r['key']=='m7_ce_s42');folder=P/row['validation_explanations'];z=np.load(folder/'explanations.npz',allow_pickle=False)
    weights=z['attention'][1];fig,axes=plt.subplots(1,4,figsize=(11,3),constrained_layout=True)
    for h,ax in enumerate(axes):im=ax.imshow(weights[h],vmin=0,vmax=1,cmap='viridis');ax.set(xticks=range(5),yticks=range(5),xticklabels=['−2','−1','0','+1','+2'],yticklabels=['−2','−1','0','+1','+2'],xlabel='Key context slot',ylabel='Query context slot',title='Head'+str(h+1))
    save(fig,'14_attention','M7 seed42 last-block attention for the same saved validation N1 example. Four heads, masked keys; attention is descriptive and not a causal explanation.','Saved validation explanations.npz attention tensor')
    fig,ax=plt.subplots(figsize=(10,4));sr=pd.DataFrame(subjectrows)
    for i,label in enumerate(GROUPS):g=sr[sr.model==label];ax.plot(np.arange(len(g)),g.macro_f1,marker='o',label=label,color=COLORS[i],lw=1)
    ax.set(xticks=range(len(g)),xticklabels=g.subject_id,ylabel='Subject five-class macro-F1 (fixed-seed mean)',ylim=(0,1));ax.tick_params(axis='x',rotation=45);ax.legend(fontsize=8)
    save(fig,'15_subjectwise','All12 test subjects in identifier order, not ranked by outcome. Subject-level F1 retains all five classes; omitted true classes receive zero F1 by the registered convention.','results/phase5_subject_metrics.csv')
    fig,ax=plt.subplots(figsize=(8,4));tr=pd.DataFrame(transitionrows)
    for i,label in enumerate(GROUPS):g=tr[tr.model==label];ax.bar(np.arange(2)+(i-1)*.23,g.macro_f1,.23,label=label,color=COLORS[i])
    ax.set(xticks=[0,1],xticklabels=['Transition target','Stable target'],ylabel='Subset five-class macro-F1',ylim=(0,1));ax.legend(fontsize=8)
    save(fig,'16_transition_errors','Descriptive fixed-seed means. Better overall classification does not imply better transition-target performance. Starts and gaps do not count as transitions or stable pairs.','results/phase5_transition_metrics.csv')
    json_write(T/'manuscript_numbers.json',ledger);json_write(T/'facts.json',facts);json_write(T/'input_manifest.json',dict(created_at=now(),plan_sha256=sha,files=inputs,training_called=False,new_inference_called=False))
    manuscript_order=['02_stage_distribution','03_preprocessing','01_architectures','04_primary_comparison','07_paired_intervals','05_confusion_matrices','06_classwise','10_N1_errors','15_subjectwise','16_transition_errors','11_hypnograms','09_parameter_cost','08_training_convergence','12_embeddings','13_explanations','14_attention']
    figures.sort(key=lambda f:manuscript_order.index(f['name']))
    for i,f in enumerate(figures,1):f['manuscript_figure_number']=i
    json_write(F/'figure_manifest.json',dict(figures=figures,filename_note='Numeric filename prefixes are stable artifact identifiers; manuscript numbers follow first appearance.'))
    (F/'captions.md').write_text('# Publication figure captions\n\n'+'\n\n'.join(f"**Figure {i}. {f['name']}** — {f['caption']}\n\nSource: {f['sources']}" for i,f in enumerate(figures,1))+'\n')
    render_documents()
    print(json.dumps(dict(successful=True,figures=len(figures),tables=len(list(T.glob('*.csv'))),numeric_facts=len(ledger),test_bootstrap_matches=True,training_called=False,new_inference_called=False),indent=2))


def render_documents():
    values=json.loads((T/'facts.json').read_text())
    for template in (R/'publication_package').glob('*.md.in'):
        text=template.read_text()
        def replace(match):
            key=match.group(1)
            if key.startswith('table:'):return (T/(key[6:]+'.md')).read_text()
            if key.startswith('include:'):return re.sub(r'\{\{([^{}]+)\}\}',replace,(R/'publication_package'/key[8:]).read_text())
            if key not in values:raise ValueError('Undefined numeric token: '+key)
            return values[key]
        rendered=re.sub(r'\{\{([^{}]+)\}\}',replace,text)
        if '{{' in rendered:raise ValueError('Unexpanded template token')
        target=R/template.name.removesuffix('.in') if not template.name.startswith('publication_reproducibility') else P/'results'/template.name.removesuffix('.in')
        target.write_text(rendered)

if __name__=='__main__':main()
