"""Create a synthetic-only infrastructure notebook; prior notebooks stay unchanged."""

from pathlib import Path
import nbformat as nbf

PROJECT=Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    notebook=nbf.v4.new_notebook()
    notebook.metadata["kernelspec"]={"name":"sleep-edf-phase2-5","display_name":"Python (existing dl-m4)","language":"python"}
    notebook.cells=[
        nbf.v4.new_markdown_cell("""# Phase 2.5 — checkpoint, resume and analysis infrastructure

**Every model operation in this notebook uses synthetic signals, not Sleep-EDF EEG.**
The frozen dataset/split/normalization hashes are referenced for future compatibility;
no processed EEG is duplicated, regenerated or used for training. Real-data final test
performance is gated. Synthetic signals have 128 samples and arbitrary units; their
30-second-grid metadata tests the API and does not describe measured PSG duration.

The notebook demonstrates immutable checkpoint generations, an interrupted epoch,
automatic recovery, unique committed history, inference feature exports, analysis-only
metrics/plots, and bounded saliency/Grad-CAM. CPU equivalence tests do not imply
bitwise-identical resumption on Apple MPS."""),
        nbf.v4.new_code_cell("""%matplotlib inline
from pathlib import Path
from copy import deepcopy
import json, sys, uuid, yaml
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Image

PROJECT=Path.cwd()
if not (PROJECT/'configs/infrastructure_demo.yaml').exists():
    PROJECT=PROJECT.parent
sys.path.insert(0,str(PROJECT/'src'))
assert Path(sys.executable).resolve()==Path('/Volumes/Mac/envs/dl-m4/bin/python').resolve()
from sleep_research.checkpoints import CheckpointManager
from sleep_research.experiments import Experiment, frozen_dataset_contract
from sleep_research.synthetic import build_demo_trainer, SyntheticTinyCNN
from sleep_research.trainer import SimulatedInterruption
from sleep_research.common import json_write, guard_evaluation
from sleep_research.analysis import analyze_export, plot_history
from sleep_research.inference import load_model, export_inference
from sleep_research.explanations import export_explanations

config=yaml.safe_load((PROJECT/'configs/infrastructure_demo.yaml').read_text())
config['max_epochs']=3
print('Existing interpreter:',sys.executable)
contract=frozen_dataset_contract(PROJECT,'A')
print('Frozen SC split SHA-256:',contract['subject_split'])
print('Frozen normalization SHA-256:',contract['normalization'])
terminal=Experiment(PROJECT,'cnn',run_id='terminal-synthetic-demo')
manager=CheckpointManager(terminal.path/'reproducibility/checkpoints',terminal.metadata['resume_fingerprints'],terminal.run_id)
display(pd.DataFrame([{'role':role,'path':path} for role,path in manager.discover().items()]))"""),
        nbf.v4.new_markdown_cell("""## Interrupt midway through the second synthetic epoch

Epoch numbers in state/history are zero-based. A bootstrap checkpoint records
completed_epoch=-1, next_epoch=0. It permits recovery even inside the first epoch.
Only a fully trained and validated epoch advances checkpoint/history state."""),
        nbf.v4.new_code_cell("""run_id='notebook-synthetic-'+uuid.uuid4().hex[:8]
trainer=build_demo_trainer(PROJECT,config,run_id)
def interrupt(epoch,batch,step):
    if epoch==1 and batch==1:
        raise SimulatedInterruption('Planned synthetic interruption')
try:
    trainer.fit(resume='none',interrupt_callback=interrupt)
except SimulatedInterruption as exc:
    print(exc)
saved=trainer.manager.recover('auto')
assert saved['completed_epoch']==0 and saved['next_epoch']==1
assert [row['epoch'] for row in saved['history']]==[0]
print('Recoverable completed epoch:',saved['completed_epoch'],'next:',saved['next_epoch'],'global step:',saved['global_step'])
resumed=build_demo_trainer(PROJECT,config,run_id)
resumed.fit(resume='auto')
assert [row['epoch'] for row in resumed.history]==[0,1,2]
print('Synthetic run:',resumed.experiment.path)
print('Committed history epochs:',[row['epoch'] for row in resumed.history])
display(pd.DataFrame(resumed.history))
fig=plot_history(resumed.experiment.path/'performance/history.json')
display(fig);plt.close(fig)
display(pd.DataFrame([{'role':role,'path':path} for role,path in resumed.manager.discover().items()]))"""),
        nbf.v4.new_markdown_cell("""## Reload retained weights and export predictions/features

Model loading and inference do not call training. Each synthetic validation epoch
exports its true/predicted labels, five raw logits, five probabilities and stable
subject/recording/night/original-index metadata. Penultimate features are default;
this example additionally requests pooled Conv1d features rather than full tensors."""),
        nbf.v4.new_code_cell("""model,checkpoint,checkpoint_path=load_model(SyntheticTinyCNN(),resumed.experiment,'best')
export=export_inference(model,resumed.validation_loader,resumed.experiment,'best',
                        intermediate_layers=[{'name':'features.0','pooling':'mean_last'}])
manifest=json.loads((export/'manifest.json').read_text())
assert manifest['synthetic'] and manifest['splits']==['validation']
predictions=pd.read_csv(resumed.experiment.path/manifest['predictions'])
display(predictions.head())
print('All five logits/probabilities retained; epoch rows:',len(predictions))
explanations=export_explanations(model,next(iter(resumed.validation_loader)),
                               resumed.experiment.path/'explanations'/export.name,checkpoint_path,
                               conv_layer='features.0',max_samples=2)
display(Image(filename=str(explanations/'saliency_grad_cam.png')))"""),
        nbf.v4.new_markdown_cell("""## Analysis-only operations on saved artifacts

The analysis module has no trainer import. It recalculates confusion matrices, ROC/PR,
calibration, subject summaries and gap-aware hypnograms from the saved predictions,
and PCA/t-SNE from saved embeddings. Metrics here describe the synthetic fixture only.
Exact t-SNE is bounded to a documented seeded subsample; UMAP is absent and not installed."""),
        nbf.v4.new_code_cell("""analysis=analyze_export(export,tsne_max_samples=40,tsne_perplexity=10,tsne_iterations=300,seed=42)
metrics=json.loads((analysis/'metrics.json').read_text())
assert metrics['synthetic'] and metrics['training_called'] is False
display(pd.DataFrame([{key:metrics[key] for key in ['accuracy','macro_f1','balanced_accuracy','cohen_kappa',
    'multiclass_brier_score','expected_calibration_error']}]))
for name in ['confusion_matrix.png','roc_pr.png','calibration.png','pca.png','tsne.png']:
    display(Image(filename=str(analysis/name)))
print('Analysis artifacts:',analysis)
print('Saved hypnograms:',[path.name for path in analysis.glob('hypnogram_*.png')])"""),
        nbf.v4.new_markdown_cell("""## Frozen final evaluation remains closed

The test guard applies to saved exports and model inference, including mixed split
artifacts. This demo never reads or evaluates real test signals. Flag counts by test
split in the separate quality review are descriptive dataset statistics only."""),
        nbf.v4.new_code_cell("""blocked=False
try:
    guard_evaluation(['test'])
except PermissionError as exc:
    blocked=True
    print('Frozen test gate:',exc)
assert blocked
summary={'synthetic':True,'run_id':run_id,'run_path':str(resumed.experiment.path),
         'interrupted_epoch':1,'recovered_next_epoch':saved['next_epoch'],'final_next_epoch':resumed.next_epoch,
         'history_epochs':[row['epoch'] for row in resumed.history],'export':str(export),'analysis':str(analysis),
         'final_test_guard_verified':blocked,'real_eeg_training':False,'real_test_performance_computed':False}
json_write(PROJECT/'results/phase2_5_notebook_demo.json',summary)
print('Stopped after Phase 2.5 infrastructure. No full Sleep-EDF CNN training.')"""),
    ]
    nbf.write(notebook,PROJECT/'notebooks/02_5_checkpoint_resume_demo.ipynb')
    print(PROJECT/'notebooks/02_5_checkpoint_resume_demo.ipynb')
