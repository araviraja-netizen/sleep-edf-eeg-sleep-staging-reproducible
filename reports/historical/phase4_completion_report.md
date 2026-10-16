> Historical measured report, curated export copy. Statements about unstarted later phases are time-specific history. Machine paths and links are adapted; excluded detailed assets are not bundled. Use the portable commands in docs/EXPERIMENTS.md rather than these historical local-archive commands.

# Phase 4 completion audit

All 15 registered experiments completed and their permanent artifacts passed the completion audit. This includes eight initial comparisons, four additional seed runs for the two shortlisted configurations, and three matched three-epoch ablation/control runs. No missing required analysis artifacts were found. No training, new inference or frozen-test evaluation was performed during this audit.

## Saved and verified

- Best/latest/previous checkpoints with checksum validation, optimizer and RNG states, experiment fingerprints and authoritative epoch histories.
- Predictions, five-class logits and probabilities, subject/recording/original epoch identifiers and finite 64-dimensional embeddings for the same 30,266 validation epochs from 12 subjects and 24 recordings in every run.
- Context masks and neighbor identifiers checked for every target; all 206 flagged validation epochs retained as registered.
- PCA, seeded t-SNE, class-wise confusion matrices, learning curves, transition analysis, per-recording hypnograms and bounded saliency/attention explanations. The global figure folder contains 93 PNG files.
- Consolidated comparison: 17 rows including preserved M1/M4 references; 85 class-wise rows and 34 transition-analysis rows.
- Executed notebook: five code cells, eight image outputs, zero error outputs. The notebook uses actual saved results and does not train or evaluate test subjects.

## Measured validation results

| Shortlisted configuration | Seeds | Macro-F1 mean ± sample SD | N1 F1 mean ± sample SD |
|---|---|---|---|
| M5 CNN-BiLSTM, cross-entropy | 42, 43, 44 | 0.742346 ± 0.001745 | 0.470314 ± 0.002837 |
| M7 CNN-Transformer, cross-entropy | 42, 43, 44 | 0.745397 ± 0.004025 | 0.467989 ± 0.024523 |

These configurations were shortlisted using validation results. The three-seed statistics describe observed variability; they are not final test performance or evidence of statistical significance. The bounded ablations use only three epochs and have parameter/cost confounds, so their results should be interpreted within that limited budget.

## Preservation and resources

The 15 experiment directories occupy approximately 0.801 GiB. Preservation checks covered 2,277 prior files, including 153 processed arrays: 2,276 remain byte-identical and the existing, preparation-approved inference export extension retains its approved checksum. Earlier checkpoints, reports, configurations, notebooks, preprocessing, split assignments and normalization are preserved. No source changes were needed for this completion audit.

The historical preparation suite recorded 183 passing tests. This completion audit separately verified completed run manifests, checkpoint state, prediction metrics, context alignment, analysis products and notebook outputs; it did not rerun or overwrite historical test evidence.

## Reviewable artifacts

- [Final comparison](phase4_final_comparison.md) and comparison CSV (additional archived evidence). [Availability](../../docs/ARTIFACT_AVAILABILITY.md)
- [Three-seed analysis](phase4_multiseed_analysis.md) and summary CSV (additional archived evidence). [Availability](../../docs/ARTIFACT_AVAILABILITY.md)
- [Ablation results](phase4_ablation_results.md)
- Research contribution analysis (additional archived evidence). [Availability](../../docs/ARTIFACT_AVAILABILITY.md)
- Executed notebook (additional archived evidence). [Availability](../../docs/ARTIFACT_AVAILABILITY.md)
- Detailed completion validation (additional archived evidence). [Availability](../../docs/ARTIFACT_AVAILABILITY.md), audit log (additional archived evidence). [Availability](../../docs/ARTIFACT_AVAILABILITY.md) and preservation validation (additional archived evidence). [Availability](../../docs/ARTIFACT_AVAILABILITY.md)
- Completion artifact checksums (additional archived evidence). [Availability](../../docs/ARTIFACT_AVAILABILITY.md)

No unresolved missing-artifact issue was found. Phase 5 and frozen final-test evaluation remain unstarted.
