"""Create a saved-artifact analysis notebook; never starts training."""

from pathlib import Path
import nbformat as nbf

PROJECT=Path(__file__).resolve().parents[1]
markdown=nbf.v4.new_markdown_cell
code=nbf.v4.new_code_cell
cells=[markdown('''# Phase 3 — real 1D CNN validation analysis

This notebook reads the completed real Protocol A CNN run and its verified saved
validation artifacts. Training is Terminal-driven. It does not train, refit normalization,
load frozen-test EEG or invoke final evaluation. Earlier synthetic fixtures and the bounded
real smoke are infrastructure checks, not the model results shown here.

Validation selects the best epoch; these results are not a final test estimate.
Wake trimming uses annotations and zero-phase filtering uses future samples: offline research.
'''),code('''from pathlib import Path
import sys, json
PROJECT = Path('/Volumes/Mac/projects/sleep-stage-classification')
sys.path.insert(0,str(PROJECT/'src'))
import numpy as np
import pandas as pd
from IPython.display import display, Image
from sleep_research.analysis import load_export
ref=json.loads((PROJECT/'results/phase3_validation_export.json').read_text())
summary=json.loads((PROJECT/'results/phase3_analysis_summary.json').read_text())
run=PROJECT/'experiments/cnn'/ref['run_id']
analysis=PROJECT/summary['analysis']
experiment=json.loads((run/'reproducibility/experiment.json').read_text())
assert not experiment['synthetic'] and experiment['status'] in ('completed','early_stopped')
manifest, predictions, embeddings, root = load_export(PROJECT/ref['export'])
assert manifest['splits']==['validation'] and len(predictions)==30266
assert not summary['frozen_test_evaluated']
print('Actual real run:',ref['run_id'])
print('Selected epoch (1-based):',manifest['completed_epoch']+1)
print('Checkpoint SHA-256:',manifest['checkpoint_sha256'])
print('Frozen final test: locked; 31156 epochs, 12 subjects')
print('Existing interpreter:',sys.executable)
'''),markdown('''## Architecture and fixed training settings

Four Conv1d/GroupNorm/ReLU/MaxPool blocks; 64-value global-pooled embedding,
dropout and five logits. Local convolutional support is 5.15s; GroupNorm time
statistics and global pooling span the full 30 s input. MAC estimates exclude
normalization/pooling/activations, backward pass and optimizer.
'''),code('''architecture=json.loads((run/'reproducibility/architecture.json').read_text())
display(pd.DataFrame(architecture['layers']))
print('Parameters:',architecture['parameters'],'Conv/Linear MACs/epoch:',architecture['conv_linear_macs_per_epoch'])
config=experiment['configuration']
display({k:config[k] for k in ['batch_size','seed','device','precision','optimizer','scheduler','early_stopping','loss']})
display(json.loads((run/'reproducibility/training_class_weights.json').read_text()))
normalization=json.loads((run/'reproducibility/protocol_and_normalization.json').read_text())['normalization']
print('Unchanged train-only population mean/std (µV):',normalization['mean_microvolts'],normalization['std_microvolts'])
'''),markdown('''## Committed training history

Training accuracy and macro-F1 accumulate online using evolving weights and active
dropout. Validation is in eval mode after each epoch. Checkpoint history is authoritative;
incomplete epochs repeat on recovery. CPU synthetic resume equality does not promise
bitwise MPS reproducibility. This baseline uses no scheduler (null checkpoint state).
'''),code('''history=pd.DataFrame(json.loads((root/manifest['history_snapshot']).read_text()))
display(history.assign(display_epoch=history.epoch+1))
display(Image(filename=str(analysis/'learning_curves.png')))
display(Image(filename=str(analysis/'accuracy_f1_learning_rate.png')))
display(experiment['training_attempts'])
'''),markdown('''## Full validation cohort

Five-class macro metrics use zero for absent classes; balanced accuracy averages
recall over present classes. Subjects, rather than correlated epochs, are the independent
cohort units. All five logits and probabilities remain available for further inference analysis.
'''),code('''metrics=json.loads((analysis/'metrics.json').read_text())
display({k:metrics[k] for k in ['sample_count','accuracy','macro_f1','balanced_accuracy','cohen_kappa','log_loss','expected_calibration_error']})
display(pd.DataFrame(metrics['classes']).T)
assert metrics['sample_count']==30266 and predictions.subject_id.nunique()==12
assert embeddings.shape==(30266,64)
np.testing.assert_allclose(predictions[[f'probability_{i}' for i in range(5)]].sum(axis=1),1,atol=1e-6)
display(Image(filename=str(analysis/'confusion_matrix.png')))
display(Image(filename=str(analysis/'subject_metrics.png')))
display(Image(filename=str(analysis/'roc_pr.png')))
display(Image(filename=str(analysis/'calibration.png')))
'''),markdown('''## Retained quality flags — descriptive only

All 2235 Protocol A flagged epochs remain in the frozen data. Validation includes 206,
with no new rejection rule or sensitivity training run. Different subset class/subject mixes
prevent attributing performance differences solely to artifact flags.
'''),code('''quality=json.loads((analysis/'quality_subset_metrics.json').read_text())
display(pd.DataFrame([{**{'subset':name},**{key:quality[name][key] for key in ['sample_count','accuracy','macro_f1','balanced_accuracy','class_counts']}} for name in ['all','flagged','flag_free']]))
assert quality['flagged']['sample_count']==206
assert int(predictions.quality_flagged.sum())==206
display(Image(filename=str(analysis/'quality_subsets.png')))
print(quality['note'])
'''),markdown('''## Saved embeddings: PCA and bounded t-SNE

PCA uses all validation embeddings. t-SNE uses a seeded uniform sample of500 rows,
without class balancing; seed/perplexity/iterations/IDs and coordinates are saved.
Visual clusters are descriptive and are not performance metrics or causal evidence.
Pooled intermediate features are bounded to64 values/epoch; full activation maps are not saved.
'''),code('''print('Penultimate embedding shape:',embeddings.shape)
for name,path in manifest['feature_files'].items():
    print(name,np.load(root/path,mmap_mode='r',allow_pickle=False).shape)
tsne=json.loads((analysis/'tsne_parameters.json').read_text())
display({k:v for k,v in tsne.items() if k!='epoch_ids'})
print('Saved t-SNE row IDs:',len(tsne['epoch_ids']))
display(Image(filename=str(analysis/'pca.png')))
display(Image(filename=str(analysis/'tsne.png')))
'''),markdown('''## Representative checkpoint-bound explanations and hypnogram

First validation epoch of each true stage, selected independently of confidence/error.
Input-gradient saliency and final Conv1d Grad-CAM refer to the exported best checkpoint.
Explanations are post-hoc model sensitivity and do not establish physiological causality.
Hypnograms preserve recording boundaries, original indices and annotation-exclusion gaps.
'''),code('''explanations=PROJECT/ref['explanations']
explanation_manifest=json.loads((explanations/'manifest.json').read_text())
assert explanation_manifest['checkpoint_sha256']==manifest['checkpoint_sha256']
assert not explanation_manifest['synthetic']
display(explanation_manifest)
display(Image(filename=str(explanations/'saliency_grad_cam.png')))
first_recording=str(predictions.recording_id.iloc[0])
display(Image(filename=str(analysis/f'hypnogram_{first_recording}.png')))
print('CNN baseline complete. Final test remains locked. No later architecture was started.')
''')]
notebook=nbf.v4.new_notebook(cells=cells)
notebook.metadata={'kernelspec':{'display_name':'Python (existing dl-m4)','language':'python','name':'python3'},
                   'language_info':{'name':'python','version':'3.11.17'}}
nbf.validate(notebook)
nbf.write(notebook,PROJECT/'notebooks/03_cnn_training_analysis.ipynb')
