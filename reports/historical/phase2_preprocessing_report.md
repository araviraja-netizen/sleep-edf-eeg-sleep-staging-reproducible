> Historical measured report, curated export copy. Statements about unstarted later phases are time-specific history. Machine paths and links are adapted; excluded detailed assets are not bundled. Use the portable commands in docs/EXPERIMENTS.md rather than these historical local-archive commands.

# Phase 2 — Sleep-EDF EEG preprocessing and dataset preparation

Report generated: 2026-10-14T16:44:48.777941+05:30 (Asia/Kolkata).

## Result and scope

**153/153 SC recordings successfully processed; 0 failures.**
Protocol B retains **414,961** usable epochs. Primary Protocol A retains **195,469**,
removing **219,492 Wake epochs**. All non-Wake classes have identical counts between A and B.
All Phase 1 reports, inventories, test logs, provenance, figures and the executed 01 notebook remain
byte-for-byte unchanged against `results/phase2_phase1_snapshot.json`. ST remains separate and unprocessed.
The existing interpreter/dataset were verified; no installation, environment creation, download, or raw EDF modification.
No CNN, model training, model checkpoint, prediction, logit, embedding or model-evaluation output was created.

## Frozen subject-independent split

| Split | Subjects | % subjects | Recordings | A epochs | B epochs |
| --- | --- | --- | --- | --- | --- |
| train | 54 | 69.231 | 107 | 134047 | 289952 |
| validation | 12 | 15.385 | 24 | 30266 | 65338 |
| test | 12 | 15.385 | 22 | 31156 | 59671 |

Seed **42**. Sorted SC subject IDs were permuted with `numpy.random.default_rng(42)` and assigned
54/12/12 in order. Every night of a subject has one split; subject overlap and recording conflicts are zero.
No stratification or test-performance optimization was applied. The test cohort is frozen for future models.

Subject split manifest SHA-256: `54281aa1e316bde7b2d6373ee110ffc862855178b1a3fd1a5c95a31c7b39818a`.
Recording assignment manifest SHA-256: `e2fd97b0447ddd372f9cde37fb8b212dc0b28c88f3704ddb7914c6ce8db77623`.

**Train subjects:** SC400, SC401, SC402, SC403, SC404, SC405, SC407, SC409, SC415, SC416, SC417, SC418, SC420, SC421, SC423, SC424, SC425, SC426, SC427, SC428, SC429, SC430, SC431, SC432, SC433, SC434, SC437, SC440, SC441, SC443, SC444, SC445, SC446, SC447, SC449, SC451, SC452, SC453, SC456, SC457, SC458, SC459, SC460, SC461, SC462, SC463, SC464, SC473, SC474, SC475, SC477, SC480, SC481, SC482.

**Validation subjects:** SC406, SC410, SC411, SC422, SC438, SC442, SC450, SC455, SC470, SC471, SC472, SC476.

**Test subjects:** SC408, SC412, SC413, SC414, SC419, SC435, SC436, SC448, SC454, SC465, SC466, SC467.

## Continuous EEG preprocessing

Native **EEG Fpz-Cz, 100 Hz**. MNE applies EDF calibration to volts; the pipeline converts to microvolts.
Only this channel is read, in 600-second chunks, into one recording buffer. Continuous finite EEG is
filtered before extracting complete half-open `[30*k,30*(k+1))` epochs of exactly **3000 samples**.
Unknown/movement intervals stay in the continuous signal during filtering but supply no supervised labels.
There is no resampling, timestamp shift, per-epoch filter, invented sample or beyond-signal epoch.

Filter: `butter(4, [0.3,35], fs=100, btype='bandpass', output='sos')`, followed by
`sosfiltfilt(..., padtype='odd', padlen=27)` in float64. N=4 specifies the fourth-order Butterworth
prototype; band-pass transformation yields an eighth-order one-pass response and forward/backward
application an effective order 16. The zero-phase amplitude response is squared, with approximately
−6.02 dB at both specified cutoffs. No cutoff compensation was silently applied.
Actual SOS coefficients and measured response: `results/filter_response.json`; figure and notebook show both responses.

Odd reflection padding is synthetic filter support only. It is never exported as EEG samples. Low-frequency
settling can extend beyond 27 samples, and reflection cannot eliminate all edge effects. Epochs touching the
first/last 30 seconds are flagged and retained; this caution window does not certify exact transient extent.
Historical acquisition filters listed in the EDF/Phase 1 inventory cannot be reversed. Zero-phase filtering
uses future samples, so this preprocessing protocol is offline rather than a causal streaming detector.

## Wake protocols and class distributions

**A (primary):** locate first/last annotation-and-signal-valid sleep labels 1–4; retain up to 60 original
epochs before and after, capped by the signal. Preserve every valid epoch between them, including internal
Wake. Invalid/movement/unknown positions remain gaps in original epoch indices. No-sleep records are flagged
and excluded from A. **B (secondary):** retain every valid epoch, including all Wake, with the same splits.
No class balancing was performed. Brief early/late sleep annotations can retain long internal Wake stretches.
**Label-based trimming is an offline evaluation protocol, not a deployable sleep-boundary detector.**

| Stage | Label | Original valid annotations | B full epochs | A trimmed epochs | B % | A % |
| --- | --- | --- | --- | --- | --- | --- |
| Wake | 0 | 285433 | 285433 | 65941 | 68.786 | 33.735 |
| N1 | 1 | 21522 | 21522 | 21522 | 5.187 | 11.010 |
| N2 | 2 | 69132 | 69132 | 69132 | 16.660 | 35.367 |
| N3 | 3 | 13039 | 13039 | 13039 | 3.142 | 6.671 |
| REM | 4 | 25835 | 25835 | 25835 | 6.226 | 13.217 |

Original complete signal epochs: **416,424**; annotation-valid epochs: **414,961**;
signal-valid full epochs: **414,961**; trimmed epochs: **195,469**.
Per-recording counts and boundaries: `results/recording_preprocessing_summary.csv` and `wake_trimming_report.md`.
Per-subject totals/classes: `results/per_subject_epoch_counts.csv` and `per_subject_class_counts.csv`.
Separate class counts for all/train/validation/test: `results/stage_distribution_comparison.csv`.

## Training-only normalization and leakage verification

One sample-weighted global population mean/std (ddof=0) is fitted independently for each protocol using
only its training epochs. No validation/test signal enters either fit. The fixed parameters apply unchanged
to validation/test. Filtered unnormalized float32 microvolt EEG is stored once; the loader applies z-scoring
on demand and returns float32 dimensionless samples. There are no duplicate normalized A/B arrays.

| Protocol | Training subjects | Training epochs | Training samples | Mean µV | Population std µV | Normalized train mean | Normalized train std |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A | 54 | 134047 | 402141000 | 2.42190633376e-05 | 17.6643519158 | -5.5050593222e-13 | 1 |
| B | 54 | 289952 | 869856000 | 2.36716584639e-05 | 23.2471465207 | -4.43983778891e-13 | 1 |

Independent post-normalization scans validate population mean ≈0 and std ≈1 over every stored training
sample in each protocol. Validation/test populations are not forced to zero mean/unit variance.
Evidence: `normalization.json`, `phase2_normalization_validation.json`, and `phase2_leakage_checks.json`.

## Signal-quality issues and exceptions

| Metric | Count |
| --- | --- |
| constant_raw_supervised_epochs_rejected | 0 |
| constant_raw_epochs_all | 0 |
| raw_rail_contact_samples | 133917 |
| usable_epochs_with_rail_contacts | 12020 |
| movement_epochs | 128 |
| unknown_epochs | 1335 |
| missing_epochs | 0 |
| overlap_epochs | 0 |
| invalid_timing_epochs | 0 |
| unrecognized_epochs | 0 |
| out_of_signal_annotation_segments | 151 |
| out_of_signal_supervised_annotation_segments | 1 |

All processed raw and filtered samples are finite. Nonfinite raw/filtered samples would reject a recording
without interpolation; exact constant raw epochs would be excluded while retaining original indices.
**12,020 usable epochs** have ADC rail contacts; these remain
flagged rather than being removed by an unapproved amplitude threshold. Numeric sanity checks do not establish
physiological artifact-free data. Review the saved per-recording ranges/variances and these flags in later research.
Recording-edge flags: **301 epochs**.
SC4362F's final Wake annotation extends 15,120 seconds beyond its 67,980-second PSG. Only supported
in-signal epochs are stored, with no timestamp shift; its unknown tail and other annotation overhang remain evidence.
No-valid-sleep recordings: **0**. Failed recordings: **0**.
Detailed failure files: `phase2_exceptions.json`, `phase2_pilot_exceptions.json`, `phase2_all_exceptions.json`.

## Memory-mappable storage, atomic publication and resume

Processed root: `data/processed/phase2/ff0bf1ce053af203`.
Pipeline fingerprint: `ff0bf1ce053af203dc024cd9abb634ae5961cc516509d7ab9ac1a6b008ea3c7b`.
Each of the 153 recording directories has one `epochs.npy` array, `epoch_metadata.csv`, and `completion.json`.
Arrays have shape `(usable_full_epochs,3000)`, dtype float32, filtered microvolt units, and no stored normalization.
Completion files bind raw SHA-256, relevant config/code/dependency versions, array/metadata hashes and sizes.
A validated temporary directory is atomically renamed. Resume rechecks raw hashes and completed artifact
hashes, sample finiteness, dtype/shape and metadata; incomplete/corrupt outputs are never silently reused.
Uncommitted temporary directories are ignored; a lock prevents concurrent writers to the same fingerprint.
The full run reused **2** validated pilot outputs and created **151** new recordings.

The master epoch manifest carries unique ID, dataset/study, subject, night/recording, original index/onset,
true label, split, both inclusion flags, array path/row, units, preprocessing version/fingerprint and quality flags.
A/B index manifests refer to the same stored rows. `phase2_state.json` is the final consolidation commit
with hashes protecting all indices, normalization and frozen split artifacts. The loader checks metadata
against stored sample rows and opens only a bounded number of read-only memmaps per worker.

| Resource | Measured usage |
| --- | --- |
| One copy of epoch arrays | 4,979,551,584 bytes (4.637569 GiB) |
| Processed directory including per-recording metadata | 4,990,912,414 bytes (4.648149 GiB) |
| Phase 2 result metadata/logs/figures at report time | 135,871,375 bytes (0.126540 GiB) |
| Preprocessing process high-water RSS | 1.223709 GiB |
| Recording processing before consolidation | 110.883 seconds |
| Full pass including consolidation/normalization | 131.383 seconds |
| Allocated processed + Phase 2 result file blocks at report time | 4.778233 GiB |
| SSD free at report time | 446.319580 GiB |

Logical file bytes and allocated blocks differ; filesystem page cache and notebook/test processes are not
included in the preprocessing process RSS. Figure/report artifact sizes can change slightly when rerendered.

## Pilot, automated tests and executed notebook

Two-recording pilot: **successful=True**, B=5,479 epochs, A=1,968.
The pilot stored samples match independently filtered full-recording references, including rows on either side
of annotation exclusions. Pilot gating and tests passed before processing the remaining SC recordings.
Final suite: **72 tests; 0 failures, 0 errors, 0 skipped**.
It covers Phase 1 invariants; first/last sleep boundaries and internal Wake; stage exclusions; stable filtering
and zero phase; training-only moments; frozen subject assignment; corruption/interruption detection; every
recording's hashes/metadata/labels/trimming; full manifest uniqueness; normalization across training populations;
and loader behavior for both protocols and all three splits. Logs: `results/phase2_test_results.txt`.
Notebook execution: **success=True**, 5 executed code cells,
6 PNG outputs, 0 error outputs.
The exact existing interpreter runs through a temporary SSD-local kernel specification; no global kernel is installed.
Notebook: `notebooks/02_eeg_preprocessing.ipynb`. Saved raw/filtered EEG, Welch PSD, original/trimmed hypnograms,
and response figures: `results/phase2_figures/`. PSD comparisons use matching epochs, not falsely concatenated gaps.

## Readiness for later deep-learning experiments

**The processed A/B datasets are technically ready for later deep-learning training.**
Frozen cohorts, stable epoch IDs, true labels, onsets, source hashes, quality flags and normalization hashes
support reproducible experiments, prediction/logit/probability joins, embeddings, subject analyses and
sleep-transition analyses. Sequence/transition analysis must respect gaps in original epoch indices.
Future training must save model/optimizer/scheduler, epoch, RNG and sampler state for exact resume and record
dataset/split/config/normalization hashes. That training state is intentionally not fabricated during preprocessing.
Remaining scientific limitations: historical R&K-to-five-class mapping, offline zero-phase/label-based protocols,
ADC rail contacts requiring artifact review, retained recording-edge transients, and filename-derived identities.
ST is not eligible for these initial experiments. No test-performance-driven tuning has occurred.

**Stopped after Phase 2. CNN training has not begun.**

## Reproducibility references

- [Sleep-EDF Expanded](https://physionet.org/content/sleep-edfx/1.0.0/), DOI 10.13026/C2X676; raw attribution in README.
- [SciPy Butterworth](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.butter.html) and
  [SOS forward/backward filtering](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.sosfiltfilt.html).
- `configs/preprocessing.yaml`, `configs/wake_trimming.yaml`, source modules and commands in README.
- Relevant source/dependency snapshot: `phase2_state.json` and processed `pipeline_identity.json`.
- Final small-artifact/code snapshot: `results/phase2_artifact_manifest.json`; array hashes in completion manifests.
