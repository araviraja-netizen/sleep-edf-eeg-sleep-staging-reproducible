"""Create the small, executable Phase 1 notebook without loading recordings."""

from pathlib import Path
import nbformat as nbf

PROJECT = Path(__file__).resolve().parents[1]


def main():
    notebook = nbf.v4.new_notebook()
    notebook.metadata["kernelspec"] = {
        "name": "sleep-edf-phase1", "display_name": "Python (existing dl-m4)", "language": "python"}
    notebook.metadata["language_info"] = {"name": "python", "version": "3.11.17"}
    notebook.cells = [
        nbf.v4.new_markdown_cell("""# Phase 1 — Sleep-EDF Expanded dataset inspection

Automatic Sleep Stage Classification Using Deep Learning on PhysioNet Sleep-EDF Expanded.
This notebook displays the verified inventory and two SC nights from subject SC400.
All valid Wake is retained. No filtering, normalization, trimming, splitting, or training.
Run the inventory command in README first; use `scripts/validate_phase1.py` to execute
with the exact existing interpreter and a temporary kernel specification.

Source: [PhysioNet Sleep-EDF Expanded](https://physionet.org/content/sleep-edfx/1.0.0/).
Label mapping: Wake=0, N1=1, N2=2, N3=3 (R&K 3+4), REM=4. Movement/unknown are excluded.
Epochs use `[30*k, 30*(k+1))` relative to PSG start; only complete, uniquely covered epochs
receive labels. Outside-signal annotations are reported without creating extra epochs."""),
        nbf.v4.new_code_cell("""%matplotlib inline
from pathlib import Path
import json, sys, tomllib
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

PROJECT = Path.cwd()
if not (PROJECT / 'configs/phase1.toml').exists():
    PROJECT = PROJECT.parent
config = tomllib.loads((PROJECT / 'configs/phase1.toml').read_text())
assert Path(sys.executable).resolve() == Path(config['python_executable']).resolve(), sys.executable
sys.path.insert(0, str(PROJECT / 'src'))
from sleep_edf.plotting import plot_recording

DATASET = Path(config['dataset_root'])
RESULTS = PROJECT / 'results'
assert DATASET.is_dir()
inventory = pd.read_csv(RESULTS / 'dataset_inventory.csv')
summary = json.loads((RESULTS / 'dataset_summary.json').read_text())
provenance = json.loads((RESULTS / 'provenance.json').read_text())
print('Interpreter:', sys.executable)
print('Inventory completed:', provenance['completed_at_Asia_Kolkata'])
print('Verified checksums:', provenance['checksum_verified_files'])
print('Inventory process MPS:', provenance['mps'])
import torch
kernel_mps = {'torch_version': torch.__version__, 'built': torch.backends.mps.is_built(),
              'available': torch.backends.mps.is_available(), 'context': 'executed notebook kernel',
              'python_executable': sys.executable}
(RESULTS / 'mps_check.json').write_text(json.dumps(kernel_mps, indent=2) + '\\n')
print('Notebook kernel MPS:', kernel_mps)
display(pd.DataFrame({s: {k: summary[s][k] for k in ['recordings', 'subjects', 'paired_recordings',
    'fpz_cz_available_recordings', 'pz_oz_available_recordings', 'valid_epochs', 'missing_epochs',
    'movement_epochs', 'unknown_epochs', 'invalid_timing_epochs']} for s in ['SC', 'ST']}).T)"""),
        nbf.v4.new_markdown_cell("""## Separate study distributions

SC is the cohort for future initial experiments. ST is inventoried separately and is
not combined with SC. Percentages use only valid supervised epochs as the denominator;
large Wake counts reflect the entire untrimmed recordings."""),
        nbf.v4.new_code_cell("""from sleep_edf.annotations import STAGE_NAMES
fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
for ax, study in zip(axes, ['SC', 'ST']):
    counts = [summary[study]['class_epoch_counts'][name] for name in STAGE_NAMES]
    bars = ax.bar(STAGE_NAMES, counts)
    ax.bar_label(bars, padding=3)
    ax.set(title=f'{study}: all valid Wake retained', ylabel='30-second epochs')
display(fig)
plt.close(fig)
distribution = pd.DataFrame({s: summary[s]['class_percent_of_valid'] for s in ['SC', 'ST']})
display(distribution.rename(columns={'SC': 'SC % of valid', 'ST': 'ST % of valid'}).round(3))"""),
        nbf.v4.new_markdown_cell("""## Subject SC400 — nights 1 and 2

Each plot reads just one native, calibrated EEG epoch at the first valid non-Wake
annotation. The complete hypnogram uses audited labels; gaps and excluded epochs are
grey. EDF acquisition filters are recorded in the inventory; this project applies none.
The numeric EEG summary scans each selected recording in 600-second chunks."""),
        nbf.v4.new_code_cell("""selected = inventory[inventory.recording_id.isin(config['inspection_recordings'])]
assert len(selected) == 2 and selected.subject_id.nunique() == 1
assert set(selected.night) == {1, 2} and set(selected.study) == {'SC'}
columns = ['recording_id', 'subject_id', 'night', 'signal_duration_seconds',
           'fpz_cz_sample_rate_hz', 'pz_oz_available', 'valid_epochs', 'Wake_epochs',
           'N1_epochs', 'N2_epochs', 'N3_epochs', 'REM_epochs', 'movement_epochs',
           'unknown_epochs', 'missing_epochs', 'invalid_timing_annotation_segments',
           'out_of_signal_annotation_segments', 'out_of_signal_annotation_seconds']
display(selected[columns].reset_index(drop=True))
for recording_id in config['inspection_recordings']:
    row = selected[selected.recording_id == recording_id].iloc[0]
    fig, audit = plot_recording(DATASET / row.psg_path, DATASET / row.hypnogram_path)
    assert audit.metrics['valid_epochs'] == int(row.valid_epochs)
    (RESULTS / 'figures').mkdir(exist_ok=True)
    fig.savefig(RESULTS / 'figures' / f'{recording_id}_inspection.png', dpi=140)
    display(fig)
    plt.close(fig)
display(pd.DataFrame(json.loads((RESULTS / 'detailed_inspection.json').read_text())).T)"""),
        nbf.v4.new_markdown_cell("""## Quality and limits

Annotation coverage is the union of all finite, positive annotation intervals intersected
with the signal, including excluded movement/unknown. Supervised coverage is stricter.
Missing seconds include incomplete trailing signal; missing epoch counts include only
complete 30-second PSG-grid epochs. Segment counts and epoch counts have different units.
Annotations extending past PSG duration do not imply additional recorded data.
No physiological artifact screening has been performed, and demographic/drug metadata
in the subject spreadsheets have not been interpreted. Review the generated Phase 1
report and automated checks before a separately authorized Phase 2."""),
    ]
    target = PROJECT / "notebooks/01_dataset_inspection.ipynb"
    target.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, target)
    print(target)


if __name__ == "__main__":
    main()
