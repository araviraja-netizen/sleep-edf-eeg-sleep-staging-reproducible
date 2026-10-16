> Historical measured report, curated export copy. Statements about unstarted later phases are time-specific history. Machine paths and links are adapted; excluded detailed assets are not bundled. Use the portable commands in docs/EXPERIMENTS.md rather than these historical local-archive commands.

# Phase 1 — Sleep-EDF Expanded dataset inspection

Automatic Sleep Stage Classification Using Deep Learning on PhysioNet Sleep-EDF Expanded.

Inventory completed: 2026-10-14T16:06:58.848579+05:30 (Asia/Kolkata).
Report generated: 2026-10-14T16:11:49.168121+05:30.

## Scope and environment

Dataset: `${DATASET_ROOT}`. Interpreter: `${PYTHON}`.
Python: `3.11.17 (main, Oct  6 2026, 23:54:59) [Clang 21.1.8 ]`. Platform: `macOS-27.0.1-arm64-arm-64bit`.
Dataset/interpreter existence was verified before project creation. Raw EDFs were read only.
No environments, installations, downloads, wake trimming, filtering, normalization, splitting, or training.
MPS observed in inventory process: `{'torch_version': '2.14.1', 'built': True, 'available': False}`.
MPS observed in executed notebook kernel: `{'torch_version': '2.14.1', 'built': True, 'available': True, 'context': 'executed notebook kernel', 'python_executable': '${PYTHON}'}`. Phase 1 uses CPU inspection only.
Hardware discovery can differ with sandbox access. No GPU computation or training was performed.

## Separate study inventories

| Study | PSGs | Hypnograms | Subjects | Paired | Inspected | Two-night subjects | One-night subjects |
| --- | --- | --- | --- | --- | --- | --- | --- |
| SC | 153 | 153 | 78 | 153 | 153 | 75 | 3 |
| ST | 44 | 44 | 22 | 44 | 44 | 22 | 0 |

SC is reserved for the initial future training experiments. ST remains a separate inventory.
Study-qualified subject IDs group both nights and distinguish SC and ST subjects with the same numeric suffix.
SC filenames include acquisition letters E, F, and G, as verified in the official RECORDS list; ST uses J.

One-night SC groups: `SC413` (SC4131E), `SC436` (SC4362F), `SC452` (SC4522E).
The documented lost SC recordings are subject 13 night 2, subject 36 night 1, and subject 52 night 1.

## Pairing, channels, duration, and integrity

Discovery/pairing issues: **0**. Inspection errors: **0**.
SHA-256 matches against supplied manifest: **398 files**; mismatches: **0**; missing referenced files: **0**; unverified non-manifest files: **0**.
The manifest itself is hashed for provenance, not checked against itself.
RECORDS missing/unlisted PSGs: **0/0**.

- SC: Fpz-Cz present in 153 recordings; Pz-Oz present in 153. Fpz-Cz native rates: [100.0] Hz. Duration range: 17.0000–24.0000 hours.
- ST: Fpz-Cz present in 44 recordings; Pz-Oz present in 44. Fpz-Cz native rates: [100.0] Hz. Duration range: 7.4944–10.6972 hours.

Native sample rates and acquisition prefilters for every channel are recorded as JSON in the CSV.
EDF file sizes are compared with declared header/data-record sizes. Shared local header times and
zero EDF+ annotation TAL origin are checked before label alignment; timestamps are never shifted.

## Five-class counts and distribution

Mapping: Wake=0, N1=1, N2=2, N3=3 (original stages 3+4), REM=4.
All valid Wake is retained. Percentages use valid supervised epochs only.

| Class | Label | SC epochs | SC % valid | ST epochs | ST % valid |
| --- | --- | --- | --- | --- | --- |
| Wake | 0 | 285433 | 68.786 | 4423 | 10.360 |
| N1 | 1 | 21522 | 5.187 | 3653 | 8.557 |
| N2 | 2 | 69132 | 16.660 | 19851 | 46.499 |
| N3 | 3 | 13039 | 3.142 | 6415 | 15.027 |
| REM | 4 | 25835 | 6.226 | 8349 | 19.557 |

## Annotation quality and exact epoch boundaries

Epochs are half-open `[30*k, 30*(k+1))` intervals on the PSG sample grid, with 3000 native EEG
samples at 100 Hz. No epoch extends beyond the signal. A label requires one grid-aligned segment
covering the entire epoch. Movement/unknown/unrecognized text has no supervised label; overlaps
and malformed timing reject affected epochs. Segment counts and epoch counts are distinct.

| Metric | SC | ST |
| --- | --- | --- |
| complete_signal_epochs | 416424 | 45446 |
| valid_epochs | 414961 | 42691 |
| movement_epochs | 128 | 83 |
| unknown_epochs | 1335 | 0 |
| unrecognized_epochs | 0 | 0 |
| missing_epochs | 0 | 2672 |
| overlap_epochs | 0 | 0 |
| invalid_timing_epochs | 0 | 0 |
| partial_annotation_epochs | 0 | 0 |
| movement_annotation_segments | 123 | 73 |
| unknown_annotation_segments | 174 | 0 |
| unrecognized_annotation_segments | 0 | 0 |
| invalid_timing_annotation_segments | 0 | 0 |
| out_of_signal_annotation_segments | 151 | 5 |
| out_of_signal_supervised_annotation_segments | 1 | 5 |
| out_of_signal_annotation_seconds | 726480.0 | 70.0 |
| annotation_missing_seconds | 0 | 80510.0 |
| annotation_overlap_seconds | 0.0 | 0.0 |
| trailing_signal_seconds | 0.0 | 430.0 |

Annotation coverage is the union of finite positive annotation intervals clipped to actual signal
duration, including excluded stages. Missing seconds include partial tails; missing epoch counts
include only complete grid epochs. Outside-signal annotation time is reported but never epochized.

Recordings requiring timing/coverage review (unknown/movement-only exclusions and unknown overhang alone are omitted below):

| Recording | Missing epochs | Overlap epochs | Malformed segments | Unrecognized segments | Supervised overhang segments |
| --- | --- | --- | --- | --- | --- |
| SC4362F | 0 | 0 | 0 | 0 | 1 |
| ST7011J | 104 | 0 | 0 | 0 | 0 |
| ST7012J | 4 | 0 | 0 | 0 | 0 |
| ST7021J | 100 | 0 | 0 | 0 | 0 |
| ST7022J | 81 | 0 | 0 | 0 | 0 |
| ST7041J | 30 | 0 | 0 | 0 | 0 |
| ST7042J | 26 | 0 | 0 | 0 | 0 |
| ST7051J | 55 | 0 | 0 | 0 | 0 |
| ST7052J | 60 | 0 | 0 | 0 | 0 |
| ST7061J | 77 | 0 | 0 | 0 | 0 |
| ST7062J | 140 | 0 | 0 | 0 | 0 |
| ST7071J | 94 | 0 | 0 | 0 | 0 |
| ST7072J | 1 | 0 | 0 | 0 | 1 |
| ST7081J | 0 | 0 | 0 | 0 | 1 |
| ST7082J | 20 | 0 | 0 | 0 | 0 |
| ST7091J | 52 | 0 | 0 | 0 | 0 |
| ST7092J | 39 | 0 | 0 | 0 | 0 |
| ST7101J | 49 | 0 | 0 | 0 | 0 |
| ST7102J | 8 | 0 | 0 | 0 | 0 |
| ST7111J | 23 | 0 | 0 | 0 | 0 |
| ST7112J | 36 | 0 | 0 | 0 | 0 |
| ST7121J | 5 | 0 | 0 | 0 | 0 |
| ST7122J | 42 | 0 | 0 | 0 | 0 |
| ST7131J | 63 | 0 | 0 | 0 | 0 |
| ST7132J | 47 | 0 | 0 | 0 | 0 |
| ST7141J | 58 | 0 | 0 | 0 | 0 |
| ST7142J | 35 | 0 | 0 | 0 | 0 |
| ST7151J | 386 | 0 | 0 | 0 | 0 |
| ST7152J | 30 | 0 | 0 | 0 | 0 |
| ST7161J | 53 | 0 | 0 | 0 | 0 |
| ST7162J | 13 | 0 | 0 | 0 | 0 |
| ST7171J | 0 | 0 | 0 | 0 | 1 |
| ST7172J | 1 | 0 | 0 | 0 | 1 |
| ST7181J | 70 | 0 | 0 | 0 | 0 |
| ST7182J | 164 | 0 | 0 | 0 | 0 |
| ST7191J | 171 | 0 | 0 | 0 | 0 |
| ST7192J | 165 | 0 | 0 | 0 | 0 |
| ST7201J | 47 | 0 | 0 | 0 | 1 |
| ST7202J | 100 | 0 | 0 | 0 | 0 |
| ST7211J | 2 | 0 | 0 | 0 | 0 |
| ST7212J | 44 | 0 | 0 | 0 | 0 |
| ST7221J | 65 | 0 | 0 | 0 | 0 |
| ST7222J | 7 | 0 | 0 | 0 | 0 |
| ST7241J | 63 | 0 | 0 | 0 | 0 |
| ST7242J | 42 | 0 | 0 | 0 | 0 |

SC4362F has a Wake annotation from 52,380 to 83,100 seconds, while its PSG ends at
67,980 seconds. The 15,120 seconds of Wake beyond the signal and subsequent unknown tail
are not converted to epochs. Its uniquely annotated complete in-signal Wake remains valid.
This is annotation/PSG duration disagreement in checksum-matching source files; it does not
justify shifting annotation times or assuming additional recorded EEG.

## Two detailed SC recordings

SC4001E and SC4002E are nights 1 and 2 of subject SC400. The notebook shows a native EEG epoch,
the full untrimmed hypnogram, and stage distribution for each. Numeric scans read 600-second chunks.

| Metric | SC4001E | SC4002E |
| --- | --- | --- |
| signal_duration_seconds | 79500.0 | 84900.0 |
| complete_signal_epochs | 2650 | 2830 |
| valid_epochs | 2650 | 2829 |
| Wake_epochs | 1997 | 1885 |
| N1_epochs | 58 | 59 |
| N2_epochs | 250 | 373 |
| N3_epochs | 220 | 297 |
| REM_epochs | 125 | 215 |
| movement_epochs | 0 | 1 |
| unknown_epochs | 0 | 0 |
| missing_epochs | 0 | 0 |
| out_of_signal_annotation_segments | 1 | 1 |
| out_of_signal_annotation_seconds | 6900.0 | 1500.0 |

| Recording | Samples scanned | Nonfinite | Min µV | Max µV | Mean µV | Std µV | Equal adjacent pairs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| SC4001E | 7950000 | 0 | -192.000000 | 170.619780 | 0.165506 | 25.909382 | 53713 |
| SC4002E | 8490000 | 0 | -208.000000 | 209.000000 | 0.670152 | 29.040892 | 43088 |

Amplitude and equal-adjacent counts are descriptive, with no artifact rejection threshold imposed.
Finite/amplitude checks and two plots do not certify physiological quality of the full dataset.

## Automated tests and notebook execution

Tests: **33 run, 0 failures, 0 errors, 0 skipped**; success: **True**. Interpreter: `${PYTHON}`. Runner: standard-library unittest.
Coverage includes strict filename parsing, missing/duplicate pairs, scorer changes, both SC acquisition
variants, mapping and exclusions, long Wake preservation, exact sample grids, half-open boundaries,
gaps, overlaps, off-grid timing, negative/nonfinite durations/onsets, overhang, subject grouping, and
actual-dataset header/channel/origin checks. Full logs: `results/test_results.txt`.

Notebook success: **True**; executed code cells: **3**; PNG outputs: **3**; error outputs: **0**.
The exact existing Python ran the kernel through a temporary local kernel spec; no kernel was installed.
Executed notebook: `notebooks/01_dataset_inspection.ipynb`; saved figures: `results/figures/`.

## Remaining issues and Phase 2 readiness

The SC inventory and validation are ready to inform a separately authorized Phase 2.

- SC missing complete epochs: 0; unknown exclusions: 1335; movement exclusions: 128. Their eligibility must remain explicit in future processing.
- Outside-signal annotations are retained as inspection evidence and cannot supply training epochs.
- ST coverage gaps/partial tails are documented separately and do not enter the first SC experiments.
- MPS is available in the executed notebook kernel; the sandboxed inventory reported it unavailable.
- Artifact review is limited to two SC recordings. Subject grouping is filename-derived; anonymized headers
  do not independently authenticate biological identity. Spreadsheet demographics and ST drug assignments
  have not been interpreted.
- No preprocessing policy or train/validation/test split is implemented. Future evaluation must group
  both nights of a subject to avoid leakage, and document all new choices and fitted transformations.
- The starting directory was not a Git repository. Source hashes provide a snapshot; no commit was made.

**Stopped after Phase 1. Phase 2 has not begun.**

## Sources and reproducibility artifacts

- [PhysioNet Sleep-EDF Expanded](https://physionet.org/content/sleep-edfx/1.0.0/), DOI 10.13026/C2X676.
- [Official recording names](https://physionet.org/content/sleep-edfx/1.0.0/RECORDS).
- [EDF layout](https://www.edfplus.info/specs/edf.html) and [EDF+ timekeeping](https://www.edfplus.info/specs/edfplus.html).
- `results/dataset_inventory.csv`, `dataset_summary.json`, `detailed_inspection.json`, `subject_groups.json`.
- `results/raw_file_manifest.json`, `provenance.json`, `environment_observed.txt`, `artifact_manifest.json`.
- `results/test_results.txt`, `test_results.json`, `notebook_execution.json`, `mps_check.json`.
- Configuration: `configs/phase1.toml`; reproduction commands and metric semantics: `README.md`.
