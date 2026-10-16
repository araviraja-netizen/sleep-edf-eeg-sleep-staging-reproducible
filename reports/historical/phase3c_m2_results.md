> Historical measured report, curated export copy. Statements about unstarted later phases are time-specific history. Machine paths and links are adapted; excluded detailed assets are not bundled. Use the portable commands in docs/EXPERIMENTS.md rather than these historical local-archive commands.

# Phase 3C — full-budget M2 confirmation
Generated: 2026-10-15T09:42:06.586572+05:30.

**M2 phase3c-m2-s42-f5c42c57 completed on MPS float32**, status `early_stopped`, 17 completed epochs; best epoch 9 (1-based). M1 is preserved; M3/M4 and seeds43/44 were not launched.

M2 did not improve M1 in this registered seed-42 validation comparison. All five selected-checkpoint class F1 scores were lower. This is a validation finding; the frozen final test remains unevaluated.

## Registered comparison and full validation outcomes

Both models use the same 134,047 training epochs/54subjects and 30,266 validation epochs/12subjects/24recordings. Test 31,156/12subjects stays locked. Protocol A, Fpz-Cz, class mapping, normalization and all quality flags are unchanged; all 206 flagged validation epochs are included. M1 used its original Torch CE; M2 uses tested equivalent ordinary CE through the shared criterion interface.
Both use LR 0.001, batch 16, decay 0.01, max 25, patience 8, seed42, no scheduler, validation macro-F1 selection. Adam uses coupled L2; AdamW decouples decay. The contrast includes optimizer/regularization semantics. No preprocessing, weighting, architecture or subject allocation was changed.

| Model | Optimizer/loss | Accuracy | Macro-F1 | Balanced accuracy | Kappa | Best epoch | Completed epochs | Train+val attempt s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| M1 | AdamW / CE | 0.782462 | 0.693772 | 0.686035 | 0.692278 | 9 | 17 | 1527.55 |
| M2 | Adam / CE | 0.748662 | 0.626050 | 0.623867 | 0.643895 | 9 | 17 | 1386.17 |

M2−M1 absolute changes: accuracy -0.033800, macro-F1 -0.067722, balanced accuracy -0.062169, kappa -0.048382. These are selected validation differences from one optimization seed, not final test performance or statistical significance.

| Class | M1 precision | M1 recall | M1 F1 | M2 precision | M2 recall | M2 F1 | Support |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Wake | 0.924945 | 0.900242 | 0.912426 | 0.908227 | 0.907128 | 0.907677 | 10746 |
| N1 | 0.342360 | 0.391841 | 0.365433 | 0.286910 | 0.238457 | 0.260449 | 3162 |
| N2 | 0.815919 | 0.847076 | 0.831205 | 0.796589 | 0.810410 | 0.803440 | 11182 |
| N3 | 0.715385 | 0.662184 | 0.687757 | 0.616681 | 0.549842 | 0.581347 | 1264 |
| REM | 0.721619 | 0.628834 | 0.672039 | 0.545207 | 0.613497 | 0.577339 | 3912 |

N1 F1 change -0.104984; N3 F1 change -0.106410. N1/N3 precision and recall are retained in the table and comparison CSV. No class-specific threshold or model adjustment was made.

## Confusion matrices

Rows=true; columns=predicted. Order: Wake,N1,N2,N3,REM.

M1

```text
9674 842 133 1 96
503 1239 1081 1 338
109 762 9472 330 509
2 13 406 837 6
171 763 517 1 2460
```

M2

```text
9748 607 122 1 268
630 754 1052 0 726
168 523 9062 430 999
2 12 546 695 9
185 732 594 1 2400
```

## Convergence and computational requirements

Parameter count 40501; Conv/Linear forward MACs 4638464 per EEG epoch, approximate FLOPs 9276928 excluding norm/activation/pooling/backward/optimizer. Same architecture and 64-value penultimate representation as M1.
M2 peak process RSS 2.626GiB. End-of-run MPS allocator/driver values are stored separately and are not peak readings. Dataset access uses the existing bounded read-only recording memmaps; no normalized EEG copy is created.
Training accuracy/F1 are accumulated online with dropout and changing weights; validation is eval mode after each epoch. Timing is descriptive across separate executions: cache/thermal/system/software effects prevent a controlled optimizer speed claim. M1 historical per-epoch N1/N3 scores were not retained; only its selected-checkpoint class scores are compared.

| Model | Mean epoch s | Completed epoch total s | Time through best epoch s | Final validation F1 | Best validation F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| M1 | 89.76 | 1525.85 | 920.52 | 0.689557 | 0.693772 |
| M2 | 81.46 | 1384.80 | 754.95 | 0.618698 | 0.626050 |

M2 first-epoch validation macro-F1 0.600441, final 0.618698; realized best epoch 9. Best and latest weights have independent roles. Equal maximum budgets can stop at different realized epochs.

## Artifacts and verification

M2 run: `experiments/cnn/phase3c-m2-s42-f5c42c57`. Best: `experiments/cnn/phase3c-m2-s42-f5c42c57/reproducibility/checkpoints/epoch-000008-09227d58a753.pt`; latest: `experiments/cnn/phase3c-m2-s42-f5c42c57/reproducibility/checkpoints/epoch-000016-cd55973d4b6e.pt`. Immutable generations/checksums, optimizer, null scheduler, criterion, early-stop, CPU/MPS RNG, sampler/generators, source/software/config/fingerprints and authoritative epoch history are retained.
Validation export `experiments/cnn/phase3c-m2-s42-f5c42c57/performance/exports/validation-epoch000008-402b63c1` includes all 30,266 IDs, labels, five logits/probabilities, subject/recording/night/original index/onset/split and quality flags. Penultimate and pooled intermediate features are each [30266,64], with exact row mappings and checkpoint-bound hashes. Analysis: `experiments/cnn/phase3c-m2-s42-f5c42c57/analysis/validation-epoch000008-402b63c1`; PCA uses all rows and t-SNE a reproducibly sampled 500-row subset(seed42,perplexity 30,500 iterations). Saved class/subject metrics, calibration/ROC/PR and original-index-respecting hypnograms remain available.
Reviewable copies and M1/M2 class/confusion/learning plots: `results/phase3c_figures/`. `phase3c_validation_comparison.csv` and `phase3c_classwise_scores.csv` preserve machine-readable results. Analysis never imports/calls the trainer; original M1 plots/exports are read only.
Real-data MPS smoke checked exact normalization, finite forward/loss/gradients, CE equivalence and interrupted-epoch recovery. Automated tests include exact-CNN CPU optimizer/history equality and corrupted-latest recovery. Bitwise MPS reproducibility is not promised. Separate test and final integrity evidence are in `phase3c_test_results.*` and `phase3c_final_validation.json`.
The registry predates full training. All previous source/config/data/reports/pilot/M1 artifacts are checked against `phase3c_preservation_snapshot.json`; earlier test outputs are not overwritten. No packages/environments/downloads/raw EDF writes occurred.

## Commands and stopping boundary

```sh
PYTHONPATH=src ${PYTHON} scripts/run_phase3c_tests.py
PYTHONPATH=src ${PYTHON} -m sleep_research.phase3c register
PYTHONPATH=src ${PYTHON} -m sleep_research.phase3c smoke
PYTHONPATH=src ${PYTHON} -m sleep_research.phase3c train-m2 --resume auto
PYTHONPATH=src ${PYTHON} -m sleep_research.phase3c_analysis
```

M3/M4 and the predefined seeds42/43/44 plan are prepared in `configs/phase3c/`; resource discussion is in `phase3c_multiseed_plan.md`. M1 seed42 is reused, never retrained. One-seed validation selection cannot establish seed variability, population significance or test generalization. Historical R&K merging, offline Wake trimming/filtering, retained artifacts and filename-derived identities remain limitations.
**Stopped after M2 full training and validation analysis. No M3/M4, additional seed, advanced architecture or frozen final test evaluation.**


## Final validation evidence

**144 automated tests passed**, including all137 previous tests and7 Phase3C registration/
exact-CNN recovery checks, zero failures/errors/skips. All1791
prior files match their snapshot, including153 arrays, all124 M1
experiment files and all8 pilot runs. Earlier source, configs, normalization/splits, reports,
notebooks and test evidence remain unchanged; no shared infrastructure code was modified.

M2's17 completed epochs and best epoch9 were verified
against immutable checksums, optimizer/criterion/RNG/history states, full-epoch step counts,
exact validation IDs/labels/subjects/flags and saved metrics. Penultimate/intermediate
features, PCA and500-row seeded t-SNE mappings were checked. Retained M2 artifacts total
48.15MiB, excluding shared EEG. No M3/M4 directory, additional seed or
frozen-test evaluation was created. `phase3c_final_validation.json` records these checks.
