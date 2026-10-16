"""Create the optimizer/loss notebook from saved actual pilot artifacts only."""

from pathlib import Path
import nbformat as nbf

PROJECT=Path(__file__).resolve().parents[1]
md=nbf.v4.new_markdown_cell;code=nbf.v4.new_code_cell
cells=[md('''# Phase 3B — optimizer and loss comparison

**Measured:** eight real MPS pilot trials, seed42,8192 training /4096 validation examples,
three epochs each. **Planned only:** full25-epoch budgets and seeds42/43/44.
This notebook reads saved artifacts; it never trains, fits normalization, modifies M1,
or loads/evaluates frozen test EEG. Short subset results do not establish improvement over M1.
'''),code('''from pathlib import Path
import sys,json
PROJECT=Path('/Volumes/Mac/projects/sleep-stage-classification')
sys.path.insert(0,str(PROJECT/'src'))
import numpy as np
import pandas as pd
from IPython.display import display,Image
from sleep_research.analysis import load_export
from sleep_research.metrics import classification_metrics
summary=json.loads((PROJECT/'results/phase3b_analysis_summary.json').read_text())
trials=pd.read_csv(PROJECT/'results/phase3b_validation_comparison.csv')
assert summary['pilot_trials']==8 and summary['full_runs_launched']==0
assert not summary['training_called'] and not summary['frozen_test_evaluated']
assert set(trials.seed)=={42} and set(trials.completed_epochs)=={3}
print('Actual pilot trials:',len(trials),'full runs:',summary['full_runs_launched'])
print('Shared M1 architecture:40501 trainable parameters; frozen test:locked')
display(trials[['stage','optimizer','learning_rate','loss','macro_f1','n1_f1','n3_f1','accuracy','balanced_accuracy','cohen_kappa','best_epoch_1_based']])
'''),md('''## Stage 1 — optimizers with ordinary CE

Two LR probes per optimizer; the primary selection metric is macro-F1. Same numeric
weight decay does not give identical coupled/decoupled regularization. Three epochs may
underestimate SGD. Class collapse is a stability warning even when arithmetic remains finite.
'''),code('''optimizer=trials[trials.stage=='optimizer_pilot'].sort_values('macro_f1',ascending=False,kind='stable')
selection=json.loads((PROJECT/'results/phase3b_optimizer_selection.json').read_text())
assert optimizer.iloc[0].run_id==selection['selected_run_id']
display(optimizer[['optimizer','learning_rate','macro_f1','n1_f1','n3_f1','epoch_macro_f1_sd','training_seconds','unique_predicted_classes','single_class_collapse']])
display(Image(filename=str(PROJECT/'results/phase3b_figures/optimizer_comparison.png')))
print('Selected before loss trials:',selection['selected_candidate'])
'''),md('''## Stage 2 — losses with the selected optimizer fixed

Weighted CE uses N_train/(5*n_class) from all134047 training labels and divides the
batch sum by the sum of target weights. Focal uses gamma2,alpha1 and an arithmetic
mean. Unweighted validation CE is stored separately from each training objective.
N1 gains must be considered alongside macro-F1,N3 and overall accuracy.
'''),code('''losses=trials[trials.candidate_id==optimizer.iloc[0].candidate_id]
display(losses[['loss','macro_f1','n1_f1','n3_f1','accuracy','balanced_accuracy','cohen_kappa']])
display(Image(filename=str(PROJECT/'results/phase3b_figures/loss_comparison.png')))
weighted=losses[losses.loss=='weighted_cross_entropy'].iloc[0]
control=losses[losses.loss=='cross_entropy'].iloc[0]
display({key:float(weighted[key]-control[key]) for key in ['macro_f1','n1_f1','n3_f1','accuracy']})
reference=json.loads((PROJECT/'results/phase3b_m1_subset_reference.json').read_text())
assert reference['training_budget_not_matched'] and not reference['baseline_artifacts_modified']
print('M1 on the same validation subset, with a larger training budget:',reference['metrics']['macro_f1'])
'''),md('''## Verified best-checkpoint predictions and class-wise errors

Every run retains all five logits/probabilities, original IDs and quality flags.
The loss candidate with highest pilot macro-F1 is shown below; this is model selection
on validation, not an unbiased final evaluation. All eight run results remain in the table.
'''),code('''chosen=losses.sort_values('macro_f1',ascending=False,kind='stable').iloc[0]
manifest,predictions,embeddings,root=load_export(PROJECT/chosen['export'])
assert manifest['splits']==['validation'] and not manifest['synthetic']
assert len(predictions)==4096 and embeddings.shape==(4096,64)
np.testing.assert_allclose(predictions[[f'probability_{i}' for i in range(5)]].sum(axis=1),1,atol=1e-6)
metrics=classification_metrics(predictions.true_label.to_numpy(),predictions[[f'probability_{i}' for i in range(5)]].to_numpy())
np.testing.assert_allclose(metrics['macro_f1'],chosen.macro_f1,atol=1e-12)
analysis=PROJECT/summary['analyses'][chosen.run_id]
print('Selected checkpoint SHA-256:',manifest['checkpoint_sha256'])
display(pd.DataFrame(metrics['classes']).T)
display(Image(filename=str(analysis/'confusion_matrix.png')))
display(Image(filename=str(analysis/'classwise_confusion.png')))
'''),md('''## Convergence and saved representations

Losses with different reductions have different scales. The curves distinguish each
objective from ordinary validation CE. Across-epoch variability is not random-seed variability.
PCA uses saved validation embeddings; t-SNE uses200 uniformly sampled rows with recorded
seed/perplexity/iterations and IDs. Clusters are descriptive, not model performance.
'''),code('''display(Image(filename=str(analysis/'objective_learning_curves.png')))
display(Image(filename=str(analysis/'pca.png')))
display(Image(filename=str(analysis/'tsne.png')))
parameters=json.loads((analysis/'tsne_parameters.json').read_text())
assert len(parameters['epoch_ids'])==200
display({key:value for key,value in parameters.items() if key!='epoch_ids'})
for name,path in manifest['feature_files'].items():print(name,np.load(root/path,mmap_mode='r',allow_pickle=False).shape)
print('Retained flagged validation subset examples:',int(predictions.quality_flagged.sum()))
'''),md('''## Full-run proposals and repeated-seed support — not measured

Confirm promising optimizers with ordinary CE before full loss comparisons. Weighted CE
is promising for N1 in this pilot; focal is lower priority. No full run or additional seed
was launched. One-seed results cannot supply a seed-level confidence interval.
'''),code('''proposals=json.loads((PROJECT/'results/phase3b_full_run_proposals.json').read_text())
assert proposals['full_runs_launched']==0
display(pd.DataFrame(proposals['configurations'])[['role','optimizer','loss','seeds','max_epochs','estimated_minutes_per_seed','estimated_three_seed_gpu_hours','measured']])
print(proposals['estimation'])
repeated=pd.read_csv(PROJECT/'results/phase3b_repeated_seed_summary.csv')
assert len(repeated)==0
print('No full repeated-seed measurements exist; summary CSV currently has headers only.')
print('Future command: python -m sleep_research.repeated_seeds (saved full validation artifacts only)')
'''),md('''## Verification and stopping boundary

See the experiment plan for exact commands and budgets. All raw EDFs, processed arrays,
normalization and splits remain unchanged. M1 research artifacts are preserved. The shared
trainer gained an optional objective; shared analysis gained bounded hypnogram control.
'''),code('''tests=json.loads((PROJECT/'results/phase3b_test_results.json').read_text())
preserved=json.loads((PROJECT/'results/phase3b_preservation_validation.json').read_text())
assert tests['successful'] and tests['skipped']==0 and preserved['successful']
display(tests)
display(preserved)
print('Stopped after the bounded pilot. No full training, frozen test evaluation or later architecture was started.')
''')]
notebook=nbf.v4.new_notebook(cells=cells)
notebook.metadata={'kernelspec':{'display_name':'Python (existing dl-m4)','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.11.17'}}
nbf.validate(notebook);nbf.write(notebook,PROJECT/'notebooks/03b_optimizer_analysis.ipynb')
