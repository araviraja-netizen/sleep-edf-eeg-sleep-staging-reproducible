# Proposed supplementary material: structure, retained analyses and figure allocation

This is an editorial supplement proposal accompanying the [revised manuscript](publication_manuscript_revised.md). It includes retained explanatory text and original measurements, not a newly executed study. Main-text figures are reduced from sixteen to six; all sixteen saved PNG/PDF pairs remain unchanged. The allocation is proposed until a journal is selected. Internal filenames retain their historical numbering; the revised main and supplementary caption numbers determine presentation order.

## Main-manuscript essentials

| Revised figure | Saved artifact | Purpose |
| --- | --- | --- |
| 1 | [01_architectures](../results/publication_figures/01_architectures.pdf) | Make different context/capacity explicit |
| 2 | [04_primary_comparison](../results/publication_figures/04_primary_comparison.pdf) | Distinguish subject uncertainty and seed dispersion |
| 3 | [07_paired_intervals](../results/publication_figures/07_paired_intervals.pdf) | Show the three registered aggregate contrasts |
| 4 | [16_transition_errors](../results/publication_figures/16_transition_errors.pdf) | Keep the unfavorable transition outcome prominent |
| 5 | [06_classwise](../results/publication_figures/06_classwise.pdf) | Show minority-class precision, recall and F1 |
| 6 | [05_confusion_matrices](../results/publication_figures/05_confusion_matrices.pdf) | Show fixed seed-42 error structure; full-page placement |

Keep main Tables 1–4 for allocation, information/capacity/seeds, primary outcomes and transition/stable outcomes. If a journal allows fewer main figures, the illustrative confusion matrices are the first candidate for supplementary relocation; retain the transition figure in the main paper. Do not merge or redraw plots before choosing final journal dimensions. Figure 3's dark/pale interval meaning must remain in its caption because the saved image has no in-panel interval legend.

## Supplement S1. Dataset and preprocessing audit

Use [Phase 1 inspection](../results/phase1_dataset_inspection.md), [Phase 2 preprocessing](../results/phase2_preprocessing_report.md), [inventory](../results/dataset_inventory.csv), [stage counts](../results/publication_tables/02_stage_distribution.md), [saved response](../results/filter_response.json) and [normalization](../results/normalization.json) as authoritative technical attachments. If an archive is prepared, distribute a curated read-only copy under the applicable data terms, not raw EDFs by assumption.

Retain SC/ST separation; official filename pairing; common origin and exact half-open epochs; movement/unknown exclusions distinct from coverage and overhang; nonfinite/constant rejection; retained rail/edge flags; original epoch indexing; and the label-informed Wake eligibility definition. The archived report records SOS coefficients, padding, response, acquisition-filter limitations, split hashes and normalization precision. The main text displays a rounded mean/std; implementations must use saved full precision. No per-epoch filtering or per-subject/test-fitted normalization was substituted.

Figures S1–S2 give eligibility distributions and the historical training-only EEG/PSD/filter illustration. The top raster of S2 cannot be made vector or recover detail merely by a high-dpi container or PDF export. Preserve native resolution and avoid upscaling claims.

## Supplement S2. Complete outcomes, supports and uncertainty conventions

Proposed Tables S1–S6 map to the unchanged [nine-checkpoint identity table](../results/publication_tables/09_all_nine_checkpoints.md), [all-seed test outcomes](../results/publication_tables/10_all_seed_test_metrics.md), [primary class precision/recall/F1](../results/publication_tables/04a_classwise_display.md), [all class metrics](../results/publication_tables/13_all_classwise_metrics.md), [paired contrasts](../results/publication_tables/05_paired_comparisons.md) and [subject/recording metrics](../results/publication_tables/07_subjectwise_primary.md). Attach the [recording table](../results/publication_tables/14_recording_metrics.md) alongside the last table. Machine-readable CSV counterparts retain full precision.

The primary outcome pools epochs within each seed before computing five-class macro-F1 and then averages fixed-seed metrics. Subject plots average seed-specific subject scores and are not the primary estimator. Bootstrap intervals resample subjects, not individual epochs; replicate counts and undefined-value handling remain in the [registered analysis](../results/phase5_statistical_analysis.md). Seed sample SD is distinct from subject uncertainty. Do not replace M1's undefined one-seed SD with zero, pool seeds into a fictitious ensemble, or promote an individual seed from its test score.

For completeness, accuracy equals single-label multiclass micro-F1; weighted F1 uses true-class supports. Balanced accuracy averages recall of present true classes. Cohen's kappa uses empirical true/predicted marginals. Specificity is TN/(TN+FP). Undefined P/R/F1 is zero, including absent classes in the five-class macro denominator; undefined specificity, kappa, AUC or AP is null. Saved one-vs-rest ROC/PR metrics, multiclass Brier score (mean sum of five squared probability errors) and ECE (ten equal-width confidence bins, final bin including one) are retained. No calibration transform or threshold was fitted to test labels.

Figures S5–S7 retain true-N1 prediction distributions, all subjects in identifier order and the deterministic first-recording hypnogram. Quality results remain in [the unchanged quality table](../results/publication_tables/12_quality_subsets.md). All {{flagged_test}} flagged test epochs remain included; class-composition differences and the absence of expert artifact labels prevent an artifact-robustness conclusion. No new uncertainty calculation is introduced for descriptive subsets.

## Supplement S3. Secondary weighted-loss comparisons

[[SECONDARY]]

The saved descriptive N1 difference for M4 minus M1 is {{M4_n1_f1_delta}} (95% interval [{{M4_n1_f1_low}}, {{M4_n1_f1_high}}]); weighted minus ordinary M5 seed 42 is {{m5_wce_s42_n1_f1_delta}} (95% interval [{{m5_wce_s42_n1_f1_low}}, {{m5_wce_s42_n1_f1_high}}]). Neither is a new multiplicity-controlled primary claim. M4 accuracy {{M4_accuracy}} and weighted M5 accuracy {{m5_wce_s42_accuracy}} are retained on the same ratio scale as the source. The full [secondary comparison table](../results/publication_tables/05_paired_comparisons.md) preserves N3 and other outcomes.

## Supplement S4. Exploratory M8 and historical development

[[M8]]

**Proposed Table S7. Original bounded M8 validation ablations.**

[[ABLATION_TABLE]]

Source: [saved validation CSV](../results/publication_tables/11_validation_metrics.csv) and [original six-decimal report](../results/phase4_ablation_results.md). Three completed training epochs per run, all full training/validation targets, fresh initialization, ordinary CE, seed 42. The table reproduces the original reported rounding; full precision remains in the source. These are not final-test results. The full-budget M8 CE run above is not interchangeable with the short bounded control. Component removal changes parameter count and computation; there is no capacity-matched, gate-only, converged multi-seed factorial comparison.

Retain the entire [validation development table](../results/publication_tables/11_validation_metrics.md) as proposed Table S8, including M6, weighted and conditional-repeat configurations. The [optimizer pilot](../results/phase3b_optimizer_comparison.md), [Adam confirmation](../results/phase3c_m2_results.md) and [weighted-CNN validation report](../results/phase3d_m4_results.md) are historical development attachments, not extra final comparators. Earlier reports' pending-test language reflects their execution dates, not the current completion state. No claim that an optional M3 or a deferred M8 confirmation was executed is introduced.

## Supplement S5. Implementation, convergence and cost

[[COST]]

Keep [full computational measurements](../results/publication_tables/06_computational_cost.md) as proposed Table S9 and Figure S4. Float32 parameter storage is in KiB (1,024 bytes), serialized checkpoint sizes in bytes, training in min and inference/export in s. MACs count selected operations per central target, not complete FLOPs, energy or peak accelerator memory. Saved timing completeness flags and differing early-stop trajectories remain attached.

Figure S3 retains every primary validation curve and validation-selected epoch. The [original architecture protocol](../results/phase4_architecture_protocol.md), [baseline results](../results/phase3_cnn_results.md), [final configuration plan](../results/phase5_evaluation_plan.json) and [infrastructure report](../results/phase2_5_infrastructure_report.md) supply exact software/hardware, model and checkpoint semantics. Every completed epoch commits optimizer/RNG/sampler/early-stop/history state with recoverable prior generations; incomplete epochs repeat. Source hashes and software versions support reproducibility, but bitwise MPS repeatability is not promised.

## Supplement S6. Representations and descriptive explanations

[[REPRESENTATIONS]]

The references R7, R9 and R13 are defined in the [revised manuscript](publication_manuscript_revised.md). Figure S8 retains all saved PCA targets and the saved seeded t-SNE subset, without a new fit. The linked [representation lineage](../results/publication_tables/representation_lineage.json) retains source coordinates and IDs. Scatter layers may be rasterized inside PDF. Keep the model-independent projection axes warning; separation does not establish a better physiological representation.

Figures S9–S10 reuse the saved validation N1 example SC4061E:e000935. Heatmap slots 0–4 in S9 are relative offsets −2 through +2, with slot 2 central; S10 uses relative offsets directly. Absolute saliency scales differ by model. S10 lacks a quantitative attention colorbar in its saved export; the present caption limits interpretation to qualitative within-example patterns. If later journal typesetting needs a calibrated colorbar, this would require a separately versioned presentation export from the saved tensor, not new model inference. No such export is performed here.

## Supplement S7. Reproducibility attachments

Include the frozen plan/checkpoint table, [original numerical ledger](../results/publication_tables/manuscript_numbers.json), [original verification](../results/publication_verification_report.md), [revision verification](../results/manuscript_revision/verification.json), [reference audit](revised_reference_audit.md) and [author checklist](reviewer_readiness_report.md). Stable public archive URLs, licenses and any redaction requirements remain author actions. The source-code/processed-array/model artifacts are unchanged; no new experiment is proposed as a condition of this revision.

## Supplementary figure captions and existing exports

[[SUPPLEMENT_FIGURES]]
