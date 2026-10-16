> Historical measured report, curated export copy. Statements about unstarted later phases are time-specific history. Machine paths and links are adapted; excluded detailed assets are not bundled. Use the portable commands in docs/EXPERIMENTS.md rather than these historical local-archive commands.

# Phase 3 — real Sleep-EDF 1D CNN baseline
Generated: 2026-10-14T18:27:29.662901+05:30.

Run **cnn1d-a-seed42-20261014-1800**, status **early_stopped**, actual **MPS float32** training.
Completed **17** epochs; latest epoch **17** and selected best epoch **9** (1-based display).
Checkpoint fields/history use zero-based completed_epoch. This report contains actual SC validation results; synthetic infrastructure tests and the bounded real smoke are separate evidence.

## Fixed data and training protocol

Protocol A only: 134047 training epochs/54 subjects/107 recordings; 30266 validation epochs/12 subjects/24 recordings. The 31156 frozen test epochs remain locked and have no predictions, embeddings or model metrics.
All 2235 Protocol A flagged epochs remain retained, including 1701 training and 206 validation epochs; 328 test flags are prior data-quality inventory only. No artifact exclusions, split changes or preprocessing regeneration.
Fpz-Cz native 100 Hz, 30 s / 3000 samples. The Phase 2 fixed train-only global normalization is unchanged: mean 2.4219063337607186e-05 µV, std 17.664351915765064 µV. Ordinary cross entropy; AdamW lr 0.001,weight decay 0.01; batch 16; seed 42; max 25; early patience 8; selection strictly highest validation five-class macro-F1; constant LR/no scheduler. Workers: 0; shuffled training, ordered validation.
Training counts [W,N1,N2,N3,R]: [43782, 15249, 47664, 9302, 18050]. Future weighted-loss metadata [0.6123384039102827, 1.7581087284412094, 0.5624664316884861, 2.8821113738980864, 1.4852853185595567]; not applied.

## Architecture and measured resources

CNN1D has 40501 parameters and a 64-dimensional penultimate embedding. Input [B,1,3000]; widths 16/32/64/64; Conv1d kernels 25/9/7/5, strides 2/2/2/1, symmetric padding 12/4/3/2, bias disabled; GroupNorm with 4 groups, ReLU and MaxPool1d(kernel=2,stride=2) in each block; adaptive global average pool; dropout 0.3; five logits.

| Layer | Output (without batch) | Parameters | Local support samples | Conv/Linear MACs |
| --- | --- | ---: | ---: | ---: |
| blocks.0.0 | [16, 1500] | 400 | 25 | 600000 |
| blocks.0.1/2 | [16, 1500] | 32 | — | — |
| blocks.0.3 | [16, 750] | 0 | 27 | — |
| blocks.1.0 | [32, 375] | 4608 | 59 | 1728000 |
| blocks.1.1/2 | [32, 375] | 64 | — | — |
| blocks.1.3 | [32, 187] | 0 | 67 | — |
| blocks.2.0 | [64, 94] | 14336 | 163 | 1347584 |
| blocks.2.1/2 | [64, 94] | 128 | — | — |
| blocks.2.3 | [64, 47] | 0 | 195 | — |
| blocks.3.0 | [64, 47] | 20480 | 451 | 962560 |
| blocks.3.1/2 | [64, 47] | 128 | — | — |
| blocks.3.3 | [64, 23] | 0 | 515 | — |
| embedding | [64] | 0 | — | — |
| classifier.0 | [64] | 0 | — | — |
| classifier.1 | [5] | 325 | — | 320 |

Forward Conv/Linear MACs per epoch: 4638464; approximate FLOPs 9276928. Normalization/activation/pooling/backward/optimizer are excluded. Final local convolutional support 515 samples / 5.15 s; GroupNorm time statistics and global pooling have whole 30 s support. Learned striding does not guarantee anti-aliasing.
Committed training+validation epoch time: 1525.85s; all attempt time: 1527.55s. Peak process RSS: 2.628 GiB. End-of-run MPS memory readings (not peak readings) are in `experiments/cnn/cnn1d-a-seed42-20261014-1800/reproducibility/resource_usage.json`. No full normalized array copy was created.

## Selected best checkpoint — validation only

Accuracy **0.782462**, five-class macro-F1 **0.693772**, balanced accuracy **0.686035**, Cohen kappa **0.692278**. These are validation/model-selection results, not an unbiased final test estimate.

| Stage | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| Wake | 0.924945 | 0.900242 | 0.912426 | 10746 |
| N1 | 0.342360 | 0.391841 | 0.365433 | 3162 |
| N2 | 0.815919 | 0.847076 | 0.831205 | 11182 |
| N3 | 0.715385 | 0.662184 | 0.687757 | 1264 |
| REM | 0.721619 | 0.628834 | 0.672039 | 3912 |

N1 has the lowest class F1 (0.365433). This remains a baseline limitation; no additional tuning or weighted-loss run was started.

Confusion matrix: rows=true, columns=predicted, order W,N1,N2,N3,R.

```text
9674 842 133 1 96
503 1239 1081 1 338
109 762 9472 330 509
2 13 406 837 6
171 763 517 1 2460
```

## Retained quality subsets — descriptive validation analysis

| Subset | Epochs | Accuracy | Five-class macro-F1 | Class counts W/N1/N2/N3/R |
| --- | ---: | ---: | ---: | --- |
| all | 30266 | 0.782462 | 0.693772 | [10746, 3162, 11182, 1264, 3912] |
| flagged | 206 | 0.839806 | 0.526858 | [173, 7, 15, 7, 4] |
| flag_free | 30060 | 0.782069 | 0.693995 | [10573, 3155, 11167, 1257, 3908] |

Subsets differ in size, subjects and class mix; differences do not establish artifact causality.
No sensitivity/exclusion training or test-driven adjustment was started.

## Saved evidence and reproducibility

Run directory: `experiments/cnn/cnn1d-a-seed42-20261014-1800`. Best/latest/previous immutable generations, byte checksums and atomic pointer indices are in `reproducibility/checkpoints/`; best checkpoint SHA-256 `36c092ac719ac2536687eac1858df597540cfeed9003e1d9b7c4df71c6c66e70`. Export pins the checkpoint for future inference.
Prediction export: `experiments/cnn/cnn1d-a-seed42-20261014-1800/performance/exports/validation-epoch000008-6ad9e935`. All 30,266 validation rows retain true/predicted labels, five logits/probabilities, subject/recording/night/original epoch/onset/split and quality flags. Embeddings [30266,64] and opt-in pooled intermediate [30266,64] have stable epoch mappings and hashed manifests.
Analysis: `experiments/cnn/cnn1d-a-seed42-20261014-1800/analysis/validation-epoch000008-6ad9e935`; confusion, per-class/subject metrics, ROC/PR/calibration, loss/accuracy/F1/LR curves, whole-validation PCA, seeded bounded 500-row t-SNE (seed 42/perplexity 30 / 500 iterations), recording-specific true/predicted hypnograms with original gaps.
Explanations: `experiments/cnn/cnn1d-a-seed42-20261014-1800/explanations/validation-epoch000008-6ad9e935`; first validation epoch of each true stage, selected independently of confidence/errors. Bounded saliency and compatible final Conv1d Grad-CAM bind exact weights; explanations are not evidence of causal physiology.
Config, software, source/config snapshot, seed, full dataset/split/normalization/protocol fingerprints and class weights are in reproducibility. Optimizer, null scheduler, early-stop, RNG/loader, global-step/history state are checkpointed. Incomplete epochs repeat; history is authoritative. No bitwise MPS resume guarantee.
Start/resume/load-best/regenerate/extend commands are in README and phase3_training_log.md. Analysis-only modules do not import or call Trainer.

## Validation and preservation

Previous suite: 104 passed. Final suite: 118 run, 0 failures, 0 errors, 0 skipped. Separate logs `phase3_previous_test_results.*` and `phase3_test_results.*`; old tests unchanged and legacy JSON writes redirected to SSD-local temporary paths.
Real MPS batch smoke verified exact frozen normalization and finite logits/loss/gradients, 128 training / 64 validation epochs and intentional interruption/resume. CNN CPU synthetic tests verify uninterrupted/resumed weights, metrics/history, corruption recovery, incompatible settings and test gates. These are checks, not real full-cohort metrics.
Notebook success: True; code cells: 7; PNG outputs: 11; errors: 0. Existing interpreter, temporary kernel specification; no kernel/environment installation.
Preserved prior files: 730 unchanged, including 153 arrays; Phase 1/2 baseline 552 unchanged. Only shared trainer/exporter gained backward-compatible Phase 3 extensions. Prior reports/notebooks/test logs/synthetic experiments remain byte-identical. No package install, raw write or dataset download.

## Limitations and stopping boundary

Validation was used for best-epoch selection and early stopping. Macro-F1 uses all five classes, zero F1 for absent classes; balanced accuracy averages recall of present classes. Epoch observations within subjects are correlated; subset/subject metrics and t-SNE are descriptive, with no population significance claim.
Historical R&K five-class merging, filename-derived identities, retained ADC rail contacts/edge effects, zero-phase filtering and annotation-based Wake trimming constrain interpretation. The single-epoch model has no surrounding-epoch sequence context. CPU equivalence does not guarantee identical MPS results across versions/devices/workers.
**Stopped after CNN baseline training and validation analysis. No frozen final test evaluation or CNN-LSTM/attention/Transformer training.**
