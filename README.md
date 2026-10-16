# Reproducible single-channel EEG sleep staging on Sleep-EDF Expanded

An auditable comparison of compact CNN, CNN–BiLSTM and CNN–Transformer configurations on PhysioNet Sleep-EDF Expanded. This repository exposes the actual research implementation, frozen experiment settings, verified numerical evidence and publication documentation from Phases 1–6. It does not assert a novel architecture, state-of-the-art superiority or clinical validation.

**Release status:** prepared locally; final inventory review and publication approval are pending. Ravi Raja Akurathi confirmed the author/copyright details, MIT for authorized original code, and release of pseudonymized subject confusion counts and frozen split identifiers. No EEG, checkpoint weights, epoch predictions or embeddings are included. See [release policy](docs/RELEASE_ARTIFACT_POLICY.md) and the [exact proposed file inventory](provenance/PUBLIC_FILE_INVENTORY.csv).

## Main findings

| Primary configuration | Temporal input | Parameters | Completed seeds | Final accuracy | Final macro-F1 |
| --- | --- | ---: | --- | ---: | ---: |
| M1 CNN | One 30-s EEG epoch | 40,501 | 42 only | 80.18% | 0.7198 |
| M5 CNN–BiLSTM | Five centered epoch slots, up to 150 s | 65,717 | 42, 43, 44 | 84.71% | 0.7771 ± 0.0066 |
| M7 CNN–Transformer | Five centered epoch slots, up to 150 s | 107,893 | 42, 43, 44 | 84.48% | 0.7758 ± 0.0118 |

Contextual rows are arithmetic means of all three predefined seeds; ± denotes sample SD of seed-specific macro-F1, not subject confidence intervals or an ensemble. M1 has no estimable seed SD. Every checkpoint evaluates the same 31,156 original targets from 12 test subjects and 22 recordings. Source: [full-precision primary table](results/publication_tables/03_primary_test_summary.csv); independently checked by the [count-based verifier](release_tools/verify.py).

The aggregate improvement is not uniform: transition-target macro-F1 is **0.5821 for M1, 0.5636 for M5 and 0.5479 for M7**, on the defined 3,498 true-label-change targets. Stable-target values are 0.7289, 0.8044 and 0.8042. This descriptive tradeoff does not establish smoothing or another causal mechanism. M7 minus M5 aggregate macro-F1 is −0.0013, with 95% paired-subject interval [−0.0078, 0.0039]; reliable superiority between contextual families is not established. Input context, capacity and seed replication differ, so improvements over M1 cannot be assigned exclusively to recurrence or attention.

## Verify the results without EEG or retraining

From this directory, choose an existing interpreter with the reviewer dependencies installed:

```sh
PYTHON=python3
"$PYTHON" release_tools/verify.py --scope full --output verification_outputs/reviewer
```

Only Python and NumPy are required for this command. It independently recalculates accuracy, five-class macro-F1, class precision/recall/F1, balanced accuracy, kappa, confusion matrices, fixed-seed means/sample SD, registered subject-bootstrap intervals and transition/stable scores. No PyTorch import, EEG download, checkpoint loading or inference occurs. See [reviewer guide](docs/REVIEWER_VERIFICATION.md) for provenance, privacy review and analyses that require excluded probability/physiological artifacts.

A reduced verification route is also available: `"$PYTHON" release_tools/verify.py --scope aggregate --output verification_outputs/aggregate`. It matches 162 saved metric checks without reading subject matrices; it cannot reconstruct subject-paired confidence intervals. If all three split files are absent, it reports allocation verification unavailable. The default `--scope auto` uses available evidence and reports exactly which checks ran; `--scope full` fails if required subject/split evidence is missing. See the [approval report](results/github_publication_approval_report.md) for the final release review.

## Dataset and fixed protocol

The [Sleep-EDF Expanded dataset](https://physionet.org/content/sleep-edfx/1.0.0/) is obtained separately from PhysioNet under the [ODC-By v1.0 terms](https://physionet.org/content/sleep-edfx/view-license/1.0.0/). Dataset terms are separate from any approved license for this code. SC contains 78 subjects / 153 paired recordings; ST contains 22 / 44 and is inventoried separately, with no ST model evaluation here.

Five classes: Wake=0, N1=1, N2=2, N3=3 (merge original stages 3/4), REM=4. Movement and unknown are excluded from supervised labels. Fpz-Cz at native 100 Hz is filtered continuously at 0.3–35 Hz using the saved fourth-order Butterworth SOS prototype and forward/backward filtering, then segmented on the original 30-s/3,000-sample grid. Gaps are preserved. Protocol A keeps all valid internal epochs with 30-minute Wake margins around valid sleep; eligibility uses labels and is offline. Normalization uses one population mean/std fitted only on training samples.

Frozen subject allocation: **54/12/12** train/validation/test; **134,047 / 30,266 / 31,156** Protocol A epochs; all nights grouped by subject. [Split manifests and original hashes](evidence/splits/subject_split_freeze.json) are included as public-dataset identifier metadata with explicit author release approval. Context never crosses subject/recording/night/split boundaries or jumps invalid indices; masks/zero padding mark missing neighbors, and no neighbor label enters a model.

## Implementation and repository layout

- `src/sleep_edf/`: actual inspection, annotations, filtering, trimming, frozen splits, streaming normalization and memory-map loaders.
- `src/models/`: actual M1 CNN and M5/M6/M7/M8 implementations, including masked BiLSTM, attention, Transformer and exploratory ablations.
- `src/sleep_research/`: actual objectives/optimizers, resumable trainer/checkpoints, export interfaces, analysis, registered campaigns and frozen evaluation.
- `scripts/` and `tests/`: unchanged historical entry points and tests. Some require excluded historical artifacts; use the portable runner for lightweight checks.
- `release_tools/`: portable orchestration and independent reviewer verification added for this release; no replacement model or trainer.
- `configs/`: scientific settings and actual registered model hyperparameters. Local path fields are configurable templates.
- `evidence/`, `provenance/`, `results/`: author-approved count evidence, source/checkpoint hashes, saved aggregate tables and selected figures.
- `notebooks/`: executed lightweight reviewer notebook and code-only historical notebook adaptations; original detailed outputs are not redistributed.
- `reports/`, `docs/`: curated revised manuscript, bibliography, release decisions and reproduction instructions.

M2 is the Adam/CE optimizer comparison; M4 is the identical CNN with training-only weighted CE. M6 attention, M8 and its ablations remain validation-only. M1/M5/M7 are primary final comparators; M4 and weighted M5 seed 42 are secondary. M3 was prepared but not executed and is not presented as a result.

## Reproduce with user-controlled paths

Use Python 3.11.17 and the [observed version pins](requirements.txt) for the closest software match. [Installation instructions](docs/INSTALLATION.md) describe the smaller reviewer environment and the full research environment. No package installation was performed during release preparation. Bitwise equivalence across devices/software is not promised.

```sh
PYTHON=python3
WORKSPACE=/path/to/new/reproduction-workspace
DATASET_ROOT=/path/to/separately-obtained/sleep-edfx
"$PYTHON" release_tools/pipeline.py init --workspace "$WORKSPACE" --dataset-root "$DATASET_ROOT"
"$PYTHON" release_tools/pipeline.py inspect --workspace "$WORKSPACE"
"$PYTHON" release_tools/pipeline.py preprocess-pilot --workspace "$WORKSPACE"
"$PYTHON" release_tools/pipeline.py preprocess-all --workspace "$WORKSPACE"
"$PYTHON" release_tools/pipeline.py verify-splits --workspace "$WORKSPACE"
"$PYTHON" release_tools/pipeline.py normalization --workspace "$WORKSPACE"
```

The initialized workspace is a new experiment archive. Raw data stay read-only. Processed data, results and checkpoints are located under that chosen workspace; changing its root relocates all outputs. Internal relative paths stay fixed to preserve original loader/checkpoint semantics. The public source tree remains untouched.

Long training is **Terminal-first, MPS-only**, never automatically launched by the commands above:

```sh
"$PYTHON" release_tools/pipeline.py train --workspace "$WORKSPACE" --key M1 --resume auto
"$PYTHON" release_tools/pipeline.py train --workspace "$WORKSPACE" --key M2 --resume auto
"$PYTHON" release_tools/pipeline.py train --workspace "$WORKSPACE" --key M4 --resume auto
"$PYTHON" release_tools/pipeline.py campaign --workspace "$WORKSPACE" --resume auto
```

Repeating the same command resumes from validated completed epochs; incomplete epochs repeat. The advanced campaign replays the original fifteen-run roster and original pre-test M5/M7 shortlist, with new run identities. It does not select a new shortlist from test outcomes. See [full pipeline and final-evaluation gate](docs/EXPERIMENTS.md) before performing final evaluation.

## Checks, analysis and resource needs

```sh
"$PYTHON" release_tools/run_core_tests.py --output verification_outputs/core
"$PYTHON" release_tools/smoke.py --output verification_outputs/smoke
"$PYTHON" release_tools/figures.py --output verification_outputs/figures
```

The smoke is synthetic CPU data only. Full research code requires PyTorch and the other pinned dependencies; full real training/evaluation requires Apple MPS. CPU reviewer verification is platform-light; Windows research execution is not supported by POSIX `fcntl` usage. CUDA is not a validated full reproduction route in this release.

Measured primary training means were 25.46 / 63.93 / 68.53 min for M1/M5/M7 on the original M4 MacBook Air (16 GB RAM). They include different early-stop trajectories, not equal-budget speed tests. The complete fifteen-run advanced campaign's saved measured durations and conservative full-budget estimates are explained in [experiments](docs/EXPERIMENTS.md). Allow roughly 20 GiB workspace space beyond the separately obtained raw data, with at least 8 GiB free headroom; exact needs vary with retained exports and recovery copies. Raw data size depends on the separately obtained dataset contents. No speed or memory result is promised on another computer.

## Scientific interpretation, citation and release

The contribution is preserved configuration-level comparison, explicit eligibility/context boundaries and traceable metrics. Limitations include one historical cohort/split, 12 final subjects, uneven seed counts, validation selection, unmatched context/capacity, offline label-based trimming/noncausal filtering/future context, and descriptive quality/explanation analyses. **The original final-test cohort is consumed and cannot be reused for model development.** See [research integrity](docs/REPRODUCIBILITY.md), [limitations](docs/LIMITATIONS.md) and the [revised manuscript](reports/publication_manuscript_revised.md).

Dataset citation: Kemp, Sleep-EDF Expanded, DOI [10.13026/C2X676](https://doi.org/10.13026/C2X676). Original collection: Kemp et al. (2000), DOI [10.1109/10.867928](https://doi.org/10.1109/10.867928). Original PhysioNet platform: Goldberger et al. (2000), DOI [10.1161/01.CIR.101.23.e215](https://doi.org/10.1161/01.CIR.101.23.e215). Verified related literature is in the [bibliography](docs/BIBLIOGRAPHY.md) and [revised audit](reports/revised_reference_audit.md).

[CITATION.cff](CITATION.cff) contains the confirmed author, Ravi Raja Akurathi. [LICENSE](LICENSE) grants MIT for authorized original code; [license scope](docs/LICENSE_SCOPE.md) separates data, third-party software and non-code content. No ORCID, affiliation, ethics approval, journal acceptance, public repository URL or archival DOI is invented. The [publication guide](docs/GITHUB_PUBLICATION.md) identifies decisions required before any staging or upload.
