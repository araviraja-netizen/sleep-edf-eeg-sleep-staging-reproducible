"""Create the Phase4 analysis-only notebook; measured results appear only when available."""
from pathlib import Path
import nbformat as nbf
P=Path(__file__).resolve().parents[1]
md=nbf.v4.new_markdown_cell;c=nbf.v4.new_code_cell
cells=[md('''# Phase4 — consolidated advanced model analysis

This notebook never starts training, calls a trainer or evaluates frozen test subjects.
During preparation it displays the immutable campaign plan and measured short-batch resource
benchmarks only. Final model metrics appear only after all15 eligible experiments are verified.
Future five-epoch EEG context makes these models offline, not causal online detectors.
'''),c('''from pathlib import Path
import sys,json
PROJECT=Path('/Volumes/Mac/projects/sleep-stage-classification')
sys.path.insert(0,str(PROJECT/'src'))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display,Image
from sleep_research.phase4_registry import read_registry,envelope
from sleep_research.analysis import load_export
from sleep_research.metrics import classification_metrics
registry=read_registry(PROJECT)
benchmark=json.loads((PROJECT/'results/phase4_benchmarks.json').read_text())
assert benchmark['successful'] and not benchmark['full_training_launched']
print('Campaign:',registry['campaign_id'],';15 eligible runs,27 reserved configurations; frozen test locked.')
display(pd.DataFrame([dict(key=k,model=v['configuration']['model_id'],loss=v['configuration']['loss']['name'],seed=v['configuration']['seed'],budget=v['configuration']['max_epochs'],parameters=v['parameters'],role=v['kind']) for k,v in registry['runs'].items()]))
'''),md('''## Measured short benchmarks and planned computational budget

These numbers are finite forward/backward and data-loading checks, not full training measurements
or model performance. Warm-step extrapolations can underestimate thermal and filesystem effects.
'''),c('''resources=pd.DataFrame.from_dict(benchmark['models'],orient='index')
display(resources[['parameters','batch_size','train_step_seconds','warm_load_seconds','estimated_25epoch_minutes']])
fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
resources.parameters.plot.bar(ax=axes[0],title='Measured parameter counts')
resources.estimated_25epoch_minutes.plot.bar(ax=axes[1],title='Estimated25-epoch minutes (not measured training)')
plt.show()
'''),md('''## Actual completed campaign results — gated

CE and weighted CE share architecture, targets, initialization seed and budget. Weighting can
improve N1 recall while reducing global macro-F1. All valid central targets remain, including
boundaries and gaps with explicit padding/masks. Never compare subset smoke scores with full runs.
'''),c('''summary_path=PROJECT/'results/phase4_analysis_summary.json'
finished=summary_path.exists()
if finished:
    summary=json.loads(summary_path.read_text())
    assert summary['successful'] and summary['measured_runs']==15 and not summary['training_called'] and not summary['frozen_test_evaluated']
    assert summary['campaign_id']==registry['campaign_id']
    comparison=pd.read_csv(PROJECT/'results/phase4_models_comparison.csv')
    classes=pd.read_csv(PROJECT/'results/phase4_classwise_results.csv')
    display(comparison[['key','parameters','accuracy','macro_f1','n1_f1','n3_f1','balanced_accuracy','cohen_kappa','training_seconds']])
    display(classes)
else:
    print('No completed full campaign comparison exists. All final model performance remains unmeasured.')
'''),md('''## Verified central-target predictions, confusions and representations

Whole-validation PCA and sampled500-row t-SNE are descriptive. Different independently fitted
model projections do not share coordinates. Attention and saliency describe predictions, not causal physiology.
'''),c('''if finished:
    canonical=pd.read_csv(PROJECT/'results/protocol_a_index.csv').query("split == 'validation'").reset_index(drop=True)
    for row in comparison.to_dict('records'):
        manifest,frame,features,root=load_export(PROJECT/row['export'])
        assert manifest['splits']==['validation'] and not manifest['synthetic']
        np.testing.assert_array_equal(frame.epoch_id,canonical.epoch_id)
        np.testing.assert_array_equal(frame.true_label,canonical.label)
        assert features.shape==(30266,64)
        scores=classification_metrics(frame.true_label.to_numpy(),frame[[f'probability_{i}' for i in range(5)]].to_numpy())
        np.testing.assert_allclose(scores['macro_f1'],row['macro_f1'],atol=1e-12,rtol=0)
    best=comparison[comparison.kind=='initial'].sort_values('macro_f1',ascending=False).iloc[0]
    figures=PROJECT/'results/phase4_figures'
    for suffix in ['confusions.png','learning_curves.png','roc_pr.png','subject_metrics.png','pca.png','tsne.png']:
        display(Image(filename=str(figures/(best['key']+'_'+suffix))))
    display(Image(filename=str(figures/'cost_vs_performance.png')))
else:print('Figures and representations will be read here after verified full training.')
'''),md('''## Selection, three-seed variation, transitions and bounded ablations

Top2 configurations are selected on seed42 validation macro-F1 (exact ties:N1F1,parameter count,
training time,key). Seeds43/44 estimate variability after selection; they are not unbiased final-test estimates.
The M8 ablations each use3epochs and a separate3-epoch control, all fresh seed42 CE.
'''),c('''if finished:
    selection=envelope(PROJECT/'results/phase4_selection.json')
    display(pd.DataFrame(selection['selected'])[['key','macro_f1','n1_f1','parameters','training_seconds']])
    display(pd.read_csv(PROJECT/'results/phase4_multiseed_summary.csv'))
    display(pd.read_csv(PROJECT/'results/phase4_transition_results.csv'))
    display(comparison[comparison.kind=='ablation'][['key','completed_epochs','macro_f1','n1_f1','n3_f1']])
    print((PROJECT/'results/phase4_research_contribution_analysis.md').read_text())
else:print('No shortlist, repeated-seed or ablation outcomes have been fabricated.')
print('Frozen test not evaluated. No Phase5. Historical M1/M2/M4 and03/03b/03c/03d artifacts preserved.')
''')]
notebook=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python (existing dl-m4)','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.11.17'}})
nbf.validate(notebook);nbf.write(notebook,P/'notebooks/04_advanced_models_analysis.ipynb')
