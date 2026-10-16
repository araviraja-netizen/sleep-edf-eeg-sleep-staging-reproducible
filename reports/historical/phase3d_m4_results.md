> Historical measured report, curated export copy. Statements about unstarted later phases are time-specific history. Machine paths and links are adapted; excluded detailed assets are not bundled. Use the portable commands in docs/EXPERIMENTS.md rather than these historical local-archive commands.

# Phase 3D — measured M4 validation results
Generated 2026-10-15T10:36:03.702267+05:30.
M4 `phase3d-m4-s42-7ffcd914`: early_stopped, 16 completed epochs, best epoch 8 (1-based).
Exact40501-parameter M1 CNN, AdamW lr0.001 decay0.01, batch16,float32,MPS,seed42,max25,patience8,no scheduler; only intended experimental change is weighted training CE. Shared Protocol A/train-only normalization/frozen54/12/12 subjects;134047 training,30266 complete validation epochs. All206 quality-flagged validation epochs retained. Test31156 remains locked.
Weights use N_train/(5*n_c) from training only, ordered Wake,N1,N2,N3,REM. Optimization is sum(w_y*CE)/sum(w_y). Validation loss is ordinary CE; validation_objective_loss is weighted CE. Training losses across objectives are not numerically comparable.

| Model | Accuracy | Macro-F1 | Balanced accuracy | Kappa | Best epoch | Duration s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| M1 | 0.782462 | 0.693772 | 0.686035 | 0.692278 | 9 | 1527.55 |
| M4 | 0.747142 | 0.680557 | 0.726422 | 0.656761 | 8 | 1357.03 |

Absolute differences M4−M1: {"accuracy": -0.03532016123703163, "balanced_accuracy": 0.0403867511394731, "cohen_kappa": -0.03551671555201885, "macro_f1": -0.013215520848727014, "n1_f1": 0.052351394319568756, "n1_precision": -0.004334352446912493, "n1_recall": 0.15496521189120804, "n2_f1": -0.05904701243153343, "n2_precision": 0.07040808331171511, "n2_recall": -0.16302986943301734, "n3_f1": -0.03478585608223994, "n3_precision": -0.17170040485829963, "n3_recall": 0.15506329113924056, "rem_f1": -0.018712320181049846, "rem_precision": -0.11310578927598047, "rem_recall": 0.07643149284253581, "wake_f1": -0.005883809868380663, "wake_precision": 0.01121042418102769, "wake_recall": -0.02149637074260191}

| Class | M1 precision | M1 recall | M1 F1 | M4 precision | M4 recall | M4 F1 | Support |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Wake | 0.924945 | 0.900242 | 0.912426 | 0.936155 | 0.878746 | 0.906543 | 10746 |
| N1 | 0.342360 | 0.391841 | 0.365433 | 0.338025 | 0.546806 | 0.417784 | 3162 |
| N2 | 0.815919 | 0.847076 | 0.831205 | 0.886327 | 0.684046 | 0.772158 | 11182 |
| N3 | 0.715385 | 0.662184 | 0.687757 | 0.543684 | 0.817247 | 0.652971 | 1264 |
| REM | 0.721619 | 0.628834 | 0.672039 | 0.608513 | 0.705266 | 0.653327 | 3912 |

Confusions: rows=true, columns=predicted, order Wake,N1,N2,N3,REM.

M1

```text
9674 842 133 1 96
503 1239 1081 1 338
109 762 9472 330 509
2 13 406 837 6
171 763 517 1 2460
```

M4

```text
9443 1041 39 2 221
339 1729 553 6 535
133 1543 7649 855 1002
1 19 194 1033 17
171 783 195 4 2759
```

| Model | Mean epoch s | Time through best s | Final macro-F1 |
| --- | ---: | ---: | ---: |
| M1 | 89.76 | 920.52 | 0.689557 |
| M4 | 84.75 | 712.35 | 0.664622 |

M4 process peak RSS 2.513GiB; end MPS readings are not peak memory. Architecture 40501 parameters, 4638464 Conv/Linear forward MACs per EEG epoch. No normalized dataset duplication.
Training accuracy/F1 are online with changing weights/dropout; validation is eval mode. Timing across separate runs includes thermal/cache/software effects. M1 historical per-epoch N1/N3 scores are unavailable; its selected checkpoint scores are compared. All differences are selected-validation, single-seed observations; no significance, seed stability or final test generalization claim.
Saved full validation export `experiments/cnn/phase3d-m4-s42-7ffcd914/performance/exports/validation-epoch000007-5f52a584`; analysis `experiments/cnn/phase3d-m4-s42-7ffcd914/analysis/validation-epoch000007-5f52a584`. These include five logits/probabilities, stable original epoch/subject/recording IDs and flags;64-value penultimate/pooled intermediate features; full PCA and seeded500-row t-SNE; class/subject metrics, confusion and original-index-respecting hypnograms. Checkpoint-bound saliency/Grad-CAM use the first epoch of each true validation class, selected without prediction confidence. Explanations describe predictions and do not establish causal physiology.
Comparison figures: `results/phase3d_figures/`; exact metrics and absolute differences: `phase3d_m1_m4_comparison.csv`; class rows: `phase3d_classwise_scores.csv`. Source/software/config/fingerprints/weights, optimizer/RNG/early-stop/authoritative histories and best/latest checkpoints remain in the permanent M4 experiment. Incomplete epochs repeat; bitwise MPS reproducibility is not promised.
**Stopped after M4 validation analysis. M1/M2/pilots preserved. No M3, additional seeds, advanced model or frozen test evaluation.**
