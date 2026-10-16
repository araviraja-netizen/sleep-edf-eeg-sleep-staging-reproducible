"""Create separate analysis-only 03c/03d notebooks; preserve the 03b pilot notebook."""
import json
from pathlib import Path
import nbformat as nbf

PROJECT=Path(__file__).resolve().parents[1]
md=nbf.v4.new_markdown_cell
code=nbf.v4.new_code_cell


def create(phase,model_id,filename):
    target=PROJECT/'notebooks'/filename
    if target.exists():raise FileExistsError('Preserve existing notebook: '+str(target))
    summaryfile='phase3c_analysis_summary.json' if phase=='3C' else 'phase3d_analysis_summary.json'
    tablefile='phase3c_validation_comparison.csv' if phase=='3C' else 'phase3d_m1_m4_comparison.csv'
    figurefolder='phase3c_figures' if phase=='3C' else 'phase3d_figures'
    description=('M2: Adam + ordinary CE. M1: AdamW + ordinary CE. Numeric weight decay is0.01 for both; Adam couples L2 regularization while AdamW decouples weight decay.' if phase=='3C' else 'M4: AdamW + training-class-weighted CE. M1: AdamW + ordinary CE. The only intended change is the training objective; the frozen dataset and CNN are unchanged.')
    interpretation=('M2 did not improve the selected M1 baseline on this validation comparison. Macro-F1 and N1 F1 were lower.' if phase=='3C' else 'M4 increased N1 F1 and recall, and balanced accuracy, while overall macro-F1, accuracy and N3 F1 decreased. This is a class-recall tradeoff, not an improvement in the primary selection metric. M1 retains the highest macro-F1 among the completed single-seed configurations.')
    cells=[md(f'''# Phase {phase} — {'Adam optimizer confirmation' if phase=='3C' else 'Weighted AdamW confirmation'}

{description}

**Actual completed full-budget runs, seed42.** Same134047 training and30266 validation epochs,54/12/12 frozen subjects,40501-parameter CNN, Fpz-Cz100Hz/3000 samples, ProtocolA and training-only normalization. Test31156 epochs remain locked.

This notebook reads saved validation artifacts only. It does not import or call the trainer, run inference, change preprocessing, or update03b. Selection and early stopping used validation macro-F1; these are not final-test estimates. {interpretation}
'''),code(f'''from pathlib import Path
import sys,json
PROJECT=Path('/Volumes/Mac/projects/sleep-stage-classification')
sys.path.insert(0,str(PROJECT/'src'))
import numpy as np
import pandas as pd
from IPython.display import display,Image
from sleep_research.analysis import load_export
from sleep_research.common import digest
from sleep_research.experiments import frozen_dataset_contract
from sleep_research.metrics import classification_metrics
summary=json.loads((PROJECT/'results/{summaryfile}').read_text())
comparison=pd.read_csv(PROJECT/'results/{tablefile}')
audit=json.loads((PROJECT/'results/phase3d_completion_validation.json').read_text())
assert audit['successful'] and not audit['training_called'] and not audit['frozen_test_evaluated']
assert not summary['training_called'] and not summary['frozen_test_evaluated']
assert comparison.model_id.tolist()==['M1','{model_id}']
assert comparison.validation_examples.tolist()==[30266,30266]
contract=frozen_dataset_contract(PROJECT,'A')
canonical=pd.read_csv(PROJECT/'results/protocol_a_index.csv').query("split == 'validation'").reset_index(drop=True)
exports={{}}
for row in comparison.to_dict('records'):
    manifest,frame,features,root=load_export(PROJECT/row['export'])
    assert manifest['splits']==['validation'] and not manifest['synthetic'] and manifest['dataset_contract']==contract
    assert len(frame)==30266 and features.shape==(30266,64) and frame.subject_id.nunique()==12
    for key in ['epoch_id','subject_id','recording_id','original_epoch_index','onset_seconds','split']:
        np.testing.assert_array_equal(frame[key],canonical[key])
    np.testing.assert_array_equal(frame.true_label,canonical.label)
    assert int(frame.quality_flagged.sum())==206
    score=classification_metrics(frame.true_label.to_numpy(),frame[[f'probability_{{i}}' for i in range(5)]].to_numpy())
    np.testing.assert_allclose(score['macro_f1'],row['macro_f1'],atol=1e-12,rtol=0)
    assert digest(root/manifest['checkpoint'])==manifest['checkpoint_sha256']
    exports[row['model_id']]=(manifest,frame,features,root,score)
analysis=PROJECT/summary['{model_id}_analysis']
figures=PROJECT/'results/{figurefolder}'
print('Verified complete validation IDs, labels, quality flags and checkpoint checksums; frozen test stays locked.')
display(comparison[['model_id','optimizer','loss','accuracy','macro_f1','balanced_accuracy','cohen_kappa','n1_f1','n3_f1','best_epoch_1_based','completed_epochs']])
'''),md('''## Overall and class-wise validation outcomes

Absolute differences below use the candidate minus M1, on identical ordered validation epochs. Epoch observations within a subject are correlated. These selected one-seed results do not establish statistical significance, seed variability or generalization.
'''),code(f'''rows=[]
for name,(_,frame,features,root,score) in exports.items():
    for stage,values in score['classes'].items():rows.append(dict(model=name,stage=stage,**values))
display(pd.DataFrame(rows))
keys=['accuracy','macro_f1','balanced_accuracy','cohen_kappa','n1_precision','n1_recall','n1_f1','n3_precision','n3_recall','n3_f1']
a=comparison.iloc[0];b=comparison.iloc[1]
display(pd.DataFrame({{'metric':keys,'candidate_minus_M1':[float(b[k]-a[k]) for k in keys]}}))
display(Image(filename=str(figures/'M1_{model_id}_confusion.png')))
display(Image(filename=str(figures/'M1_{model_id}_class_f1.png')))
'''),md('''## Convergence and computational cost

Best and latest checkpoints have different roles. Budgets were25 epochs with patience8 and strict validation macro-F1 selection, so realized epochs can differ. Online training accuracy/F1 include dropout and changing weights. Validation uses eval mode. Ordinary validation CE remains comparable; M4 weighted training loss has a different reduction and scale. Timing across separate executions is descriptive because thermal/cache/software conditions differ.
'''),code(f'''display(comparison[['model_id','completed_epochs','best_epoch_1_based','training_attempt_seconds','mean_epoch_seconds','time_to_best_seconds']])
for name,(manifest,frame,features,root,score) in exports.items():
    history=json.loads((root/manifest['history_snapshot']).read_text())
    assert len(history)==int(comparison.loc[comparison.model_id==name,'completed_epochs'].item())
    assert [r['epoch'] for r in history]==list(range(len(history)))
    print(name,'first/final/best validation macro-F1:',history[0]['validation_macro_f1'],history[-1]['validation_macro_f1'],max(r['validation_macro_f1'] for r in history))
    if name=='{model_id}':display(pd.DataFrame(history).tail())
display(Image(filename=str(figures/'M1_{model_id}_learning_curves.png')))
display(Image(filename=str(figures/'{model_id}_class_learning_curves.png')))
'''),md('''## Subject-wise variation

Per-subject metrics describe this frozen validation cohort. Small class counts and absent stages can affect subject macro-F1; these plots are not confidence intervals or independent held-out test performance.
'''),code('''subjects=json.loads((analysis/'subject_metrics.json').read_text())
assert len(subjects)==12
display(pd.DataFrame.from_dict(subjects,orient='index')[['sample_count','accuracy','macro_f1','balanced_accuracy','cohen_kappa']])
display(Image(filename=str(analysis/'subject_metrics.png')))
'''),md('''## Saved representations: PCA and bounded t-SNE

Penultimate and pooled intermediate features each have64 dimensions and preserve original epoch mappings. PCA uses all30266 validation rows. t-SNE uses a saved uniform500-row sample(seed42,perplexity30,500iterations). Embedding plots are descriptive; separately fitted model projections do not share coordinates and should not be interpreted as direct geometric comparisons.
'''),code(f'''manifest,predictions,embeddings,root,score=exports['{model_id}']
for name,path in manifest['feature_files'].items():
    values=np.load(root/path,mmap_mode='r',allow_pickle=False)
    assert values.shape==(30266,64) and np.isfinite(values).all()
    print(name,values.shape)
pca=pd.read_csv(analysis/'pca_coordinates.csv')
np.testing.assert_array_equal(pca.epoch_id,predictions.epoch_id)
parameters=json.loads((analysis/'tsne_parameters.json').read_text())
selected=np.sort(np.random.default_rng(42).choice(30266,500,replace=False))
assert parameters['epoch_ids']==predictions.iloc[selected].epoch_id.tolist()
display({{k:v for k,v in parameters.items() if k!='epoch_ids'}})
display(Image(filename=str(analysis/'pca.png')))
display(Image(filename=str(analysis/'tsne.png')))
'''),md('''## Recording hypnogram

The first validation recording is displayed for reproducibility. Original30-second epoch indices and annotation gaps remain explicit. No surrounding-epoch sequence model was trained.
'''),code(f'''recording=sorted(predictions.recording_id.unique())[0]
assert len(list(analysis.glob('hypnogram_*.png')))==24
path=figures/('M1_M4_hypnogram_'+recording+'.png') if '{model_id}'=='M4' else analysis/('hypnogram_'+recording+'.png')
display(Image(filename=str(path)))
print('Recording:',recording,'; all24 recording figures are saved on SSD.')
''')]
    if model_id=='M4':
        cells += [md('''## Checkpoint-bound gradient saliency and1D Grad-CAM

Representative epochs are the first validation example of each true stage, selected without scores or confidence. Explanations target predicted logits and remain linked to exact checkpoint and epoch IDs. They describe model predictions and do not establish causal physiology.
'''),code('''ref=json.loads((PROJECT/'results/phase3d_m4_export.json').read_text())
explanations=PROJECT/ref['explanations']
ex=json.loads((explanations/'manifest.json').read_text())
assert ex['checkpoint_sha256']==manifest['checkpoint_sha256'] and ex['sample_count']==5
for relative,sha in ex['files'].items():assert digest(explanations/relative)==sha
with np.load(explanations/'explanations.npz',allow_pickle=False) as arrays:
    assert arrays['epoch_ids'].tolist()==ex['epoch_ids']
    print({name:arrays[name].shape for name in arrays.files})
display(pd.DataFrame({'epoch_id':ex['epoch_ids'],'true_stage':['Wake','N1','N2','N3','REM']}))
display(Image(filename=str(explanations/'saliency_grad_cam.png')))
''')]
    cells += [md('''## Reproducibility, saved evidence and research boundary

Config/software/source snapshots, optimizer/null scheduler, RNG/sampler state, early-stop, completed-epoch history and best/latest/previous checkpoints are retained. MPS bitwise reproducibility is not promised. Existing162-test preparation evidence is historical; this notebook execution and the completion audit provide separate post-training evidence. Prior notebooks and research artifacts remain preserved.

Historical R&K merging, offline zero-phase filtering and label-based Wake trimming, filename-derived subject IDs and retained quality flags constrain interpretation. No M3, additional seed, advanced architecture or frozen test evaluation is run by this notebook.
'''),code(f'''detail=audit['artifacts']['{model_id}']
display(detail)
experiment=json.loads((root/'reproducibility/experiment.json').read_text())
assert experiment['dataset_contract']==contract
assert experiment['status'] in ['completed','early_stopped']
display(experiment['configuration'])
print('Retained experiment storage MiB:',detail['retained_run_bytes']/2**20)
print('Preparation tests:',json.loads((PROJECT/'results/phase3d_test_results.json').read_text())['tests_run'])
if '{model_id}'=='M4':
    completed=pd.read_csv(PROJECT/'results/phase3d_completed_models_validation_comparison.csv')
    display(completed[['model_id','optimizer','loss','macro_f1','n1_f1','n3_f1','accuracy','balanced_accuracy']])
print('Analysis only. No new training or frozen test evaluation. 03b remains the historical subset pilot.')
''')]
    notebook=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python (existing dl-m4)','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.11.17'}})
    nbf.validate(notebook);nbf.write(notebook,target)
    print('Created',target)

if __name__=='__main__':
    create('3C','M2','03c_adam_confirmation_analysis.ipynb')
    create('3D','M4','03d_weighted_adamw_analysis.ipynb')
