"""Generate the Phase 2 notebook; preserves the Phase 1 notebook."""

from pathlib import Path
import nbformat as nbf

PROJECT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    notebook = nbf.v4.new_notebook()
    notebook.metadata["kernelspec"] = {"name": "sleep-edf-phase2", "display_name": "Python (existing dl-m4)", "language": "python"}
    notebook.cells = [
        nbf.v4.new_markdown_cell("""# Phase 2 — EEG preprocessing and dataset preparation

SC only. Protocol A (primary): 30-minute Wake context around first/last valid sleep.
Protocol B (secondary): full valid recording. Both use the same frozen subject split.
This label-based trimming is an offline evaluation protocol, **not a deployable
sleep-boundary detector**. No models, predictions, or model evaluation results are created.

Filtering uses a fourth-order Butterworth prototype (`butter(N=4)`), which yields an
eighth-order band-pass in one direction. SOS forward/backward filtering has squared
amplitude response (about −6.02 dB at 0.3 and 35 Hz) and zero phase. The full finite
recording is filtered before epoch extraction; no independent per-epoch filters.
Odd reflection uses 27 synthetic padding samples only inside filtering; they are not
stored as recorded samples. Edge effects are unavoidable; first/last 30 seconds are
flagged and retained. Historical acquisition filtering cannot be undone."""),
        nbf.v4.new_code_cell("""%matplotlib inline
from pathlib import Path
import csv, json, sys, yaml
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

PROJECT = Path.cwd()
if not (PROJECT / 'configs/preprocessing.yaml').exists():
    PROJECT = PROJECT.parent
sys.path.insert(0, str(PROJECT / 'src'))
config = yaml.safe_load((PROJECT / 'configs/preprocessing.yaml').read_text())
assert Path(sys.executable).resolve() == Path(config['python_executable']).resolve()
RESULTS = PROJECT / 'results'
state = json.loads((RESULTS / 'phase2_state.json').read_text())
assert state['status'] == 'complete' and state['processed_recordings'] == 153
print('Interpreter:', sys.executable)
print('Pipeline fingerprint:', state['pipeline_fingerprint'])
print('Split manifest SHA-256:', state['split_manifest_sha256'])
splits = pd.read_csv(RESULTS / 'subject_splits.csv')
display(splits.groupby('split').subject_id.nunique().rename('subjects').to_frame())
display(pd.DataFrame([state['leakage']]).drop(columns=['frozen_test_subjects']))"""),
        nbf.v4.new_markdown_cell("""## Original, full-valid, and trimmed distributions

Unknown/movement annotations are excluded. Exact constant raw EEG epochs are rejected;
rail contacts and edge effects are flagged without an arbitrary amplitude threshold.
No artificial class balancing is applied."""),
        nbf.v4.new_code_cell("""distribution = pd.read_csv(RESULTS / 'stage_distribution_comparison.csv')
all_counts = distribution[distribution['split'] == 'all']
display(all_counts.pivot(index='stage', columns='protocol', values='epoch_count').reindex(['Wake','N1','N2','N3','REM']))
fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
for ax, protocol in zip(axes, ['A', 'B']):
    group = all_counts[all_counts.protocol == protocol].sort_values('label')
    bars = ax.bar(group.stage, group.epoch_count)
    ax.bar_label(bars, padding=3)
    ax.set(title=f'Protocol {protocol}', ylabel='30-second epochs')
display(fig); plt.close(fig)
records = pd.read_csv(RESULTS / 'recording_preprocessing_summary.csv')
display(records[['recording_id','subject_id','split','original_annotation_valid_epochs','usable_full_epochs',
                 'usable_trimmed_epochs','wake_epochs_removed','constant_raw_supervised_epochs_rejected']].head(12))
print('Total full:', int(records.usable_full_epochs.sum()))
print('Total trimmed:', int(records.usable_trimmed_epochs.sum()))
print('Wake removed:', int(records.wake_epochs_removed.sum()))"""),
        nbf.v4.new_markdown_cell("""## Filter response and pilot EEG/PSD/hypnograms

Plots use unnormalized microvolts. Filtered epochs come from continuous-recording
preprocessing. Raw EEG is read only for the matching original sample interval.
Original and trimmed hypnograms preserve the original PSG timeline; excluded epochs
and discarded exterior Wake appear as gaps, without shifting remaining timestamps."""),
        nbf.v4.new_code_cell("""from sleep_edf.phase2_plotting import plot_filter_response, plot_preprocessing_recording
figure_root = RESULTS / 'phase2_figures'
figure_root.mkdir(exist_ok=True)
fig = plot_filter_response(config)
fig.savefig(figure_root / 'filter_response.png', dpi=140)
display(fig); plt.close(fig)
for recording in config['pilot_recordings']:
    eeg_fig, hyp_fig = plot_preprocessing_recording(PROJECT, recording)
    for fig, suffix in [(eeg_fig, 'eeg_psd'), (hyp_fig, 'hypnograms')]:
        fig.savefig(figure_root / f'{recording}_{suffix}.png', dpi=140)
        display(fig); plt.close(fig)"""),
        nbf.v4.new_markdown_cell("""## Normalization, quality and SSD storage

Each protocol fits one sample-weighted population mean and standard deviation from
its **training epochs only**. Validation/test use these unchanged constants.
Filtered float32 microvolt arrays are stored once. Dataset loading applies z-scoring
on demand, returning float32 dimensionless EEG; no duplicate normalized arrays.
ADC rail counts are numeric flags rather than an expert artifact assessment."""),
        nbf.v4.new_code_cell("""normalization = json.loads((RESULTS / 'normalization.json').read_text())
display(pd.DataFrame({p: {k: v for k, v in normalization['protocols'][p].items()
    if k not in ['training_subjects','training_recordings']} for p in ['A','B']}).T)
display(pd.DataFrame(state['quality'], index=[0]))
print('Arrays, GiB:', state['array_storage_bytes'] / 1024**3)
print('Processed directory, GiB:', state['processed_directory_bytes'] / 1024**3)
print('Preprocessing process peak RSS, GiB:', state['process_peak_rss_bytes'] / 1024**3)
print('Recording failures:', json.loads((RESULTS / 'phase2_exceptions.json').read_text()))
display(records[['recording_id','raw_min_uv','raw_max_uv','raw_std_uv','filtered_std_uv',
                 'raw_rail_contact_samples','usable_epochs_with_rail_contacts']].head())"""),
        nbf.v4.new_markdown_cell("""## Reloading a sample for future training

The loader returns EEG, true label, stable epoch ID, subject, night, original epoch
index, onset, and split. This metadata supports later prediction/logit/embedding
joins and subject/transition analysis. No training or model checkpoint is produced.
Use the recorded split hash, pipeline hash, normalization hash and epoch IDs in every
future experiment; model/optimizer/RNG/sampler checkpoint state remains a later task."""),
        nbf.v4.new_code_cell("""from sleep_edf.dataset import EpochDataset
for protocol in ['A', 'B']:
    dataset = EpochDataset(PROJECT, protocol=protocol, split='train')
    sample = dataset[0]
    assert sample['eeg'].shape == (1, 3000) and str(sample['eeg'].dtype) == 'float32'
    print(protocol, 'training epochs:', len(dataset),
          {key: value for key,value in sample.items() if key != 'eeg'})
    print('EEG shape/dtype:', sample['eeg'].shape, sample['eeg'].dtype)
    dataset.close()
print('Stopped after Phase 2. No CNN training.')"""),
    ]
    nbf.write(notebook, PROJECT / "notebooks/02_eeg_preprocessing.ipynb")
    print(PROJECT / "notebooks/02_eeg_preprocessing.ipynb")
