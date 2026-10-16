"""Create an analysis-only notebook with final results explicitly gated."""
from pathlib import Path
import nbformat as nbf
P=Path(__file__).resolve().parents[1]
nb=nbf.v4.new_notebook();nb.metadata.update(kernelspec=dict(display_name='Python (existing dl-m4)',language='python',name='sleep-edf-phase5'))
nb.cells=[nbf.v4.new_markdown_cell('''# Phase 5 — final research analysis

Preparation and validation evidence are available now. Frozen final results require the reviewed plan and explicit Terminal authorization. This notebook never trains or runs model inference. The manuscript distinguishes measured validation findings from pending final evaluation.'''),
nbf.v4.new_code_cell('''from pathlib import Path
import os, sys, json
PROJECT = Path('/Volumes/Mac/projects/sleep-stage-classification')
sys.path.insert(0, str(PROJECT / 'src'))
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Markdown, Image
from sleep_research.phase5_registry import plan
from sleep_research.common import fingerprint, digest
body = plan(PROJECT)
print('Plan:', body['plan_id'], '\\nSHA-256:', fingerprint(body))
print('Preparation only: no training or inference is called by this notebook.')'''),
nbf.v4.new_code_cell('''display(pd.DataFrame([{k:r[k] for k in ['key','role','seed','parameters','best_epoch_1_based','checkpoint_sha256']} for r in body['models']]))
validation = pd.read_csv(PROJECT / 'results/phase4_models_comparison.csv')
validation = validation[validation.key.isin([r['key'] for r in body['models']])]
display(Markdown('## Measured validation findings — not final test performance'))
display(validation[['key','accuracy','macro_f1','n1_f1','n3_f1']])
fig, ax = plt.subplots(figsize=(10,4))
ax.bar(validation.key, validation.macro_f1)
ax.set(ylabel='Validation five-class macro-F1', ylim=(0,1))
ax.tick_params(axis='x', rotation=45)
plt.tight_layout(); plt.show()'''),
nbf.v4.new_code_cell('''display(Markdown((PROJECT / 'results/phase5_model_selection.md').read_text()))
display(Markdown('## Predefined uncertainty and interpretation'))
print(json.dumps(body['statistics'], indent=2))
print('Existing validation PCA, t-SNE and explanations remain linked to exact checkpoints; attention/gradients are not causal physiology.')'''),
nbf.v4.new_code_cell('''show_final = os.environ.get('SLEEP_PHASE5_SHOW_FINAL') == 'explicitly-authorized'
if not show_final:
    display(Markdown('## Frozen final evaluation pending\\nNo test predictions or result files are read. Review `results/phase5_execution_instructions.md` and run the separate authorized Terminal command.'))
else:
    from sleep_research.phase5_evaluation import authorization, workspace
    authorization(PROJECT, body, True, os.environ.get('SLEEP_PHASE5_PLAN_SHA256'))
    complete = json.loads((workspace(PROJECT, body) / 'analysis_complete.json').read_text())
    assert complete['plan_sha256'] == fingerprint(body)
    for relative, expected in complete['files'].items():
        assert digest(PROJECT / relative) == expected, relative
    display(Markdown((PROJECT / 'results/phase5_final_test_results.md').read_text()))
    display(pd.read_csv(PROJECT / 'results/phase5_classwise_metrics.csv'))
    display(pd.read_csv(PROJECT / 'results/phase5_seed_variability.csv'))
    display(Markdown((PROJECT / 'results/phase5_statistical_analysis.md').read_text()))
    display(Image(filename=str(PROJECT / 'results/phase5_figures/parameters_vs_performance.png')))
    display(Image(filename=str(PROJECT / 'results/phase5_figures/cost_vs_performance.png')))
    for key in ['M1', 'm5_ce_s42', 'm7_ce_s42']:
        display(Image(filename=str(PROJECT / f'results/phase5_figures/{key}_confusion.png')))
        display(Image(filename=str(PROJECT / f'results/phase5_figures/{key}_roc_pr_calibration.png')))
        display(Image(filename=str(PROJECT / f'results/phase5_figures/{key}_pca.png')))
        display(Image(filename=str(PROJECT / f'results/phase5_figures/{key}_tsne.png')))''')]
for cell in nb.cells:
    if cell.cell_type=='code':compile(cell.source,'phase5_notebook_cell','exec')
nbf.write(nb,P/'notebooks/05_final_research_analysis.ipynb')
