> Historical measured report, curated export copy. Statements about unstarted later phases are time-specific history. Machine paths and links are adapted; excluded detailed assets are not bundled. Use the portable commands in docs/EXPERIMENTS.md rather than these historical local-archive commands.

# Phase4 consolidated measured validation campaign
Completed analysis 2026-10-16T08:47:22.870995+05:30.

Eight initial configurations, four conditional repeat runs, three equal-budget ablation/control runs; all artifacts verified. Same30266 validation target IDs; no test evaluation. M1/M4 original trained baselines reused.

| Key | Parameters | Accuracy | Macro-F1 | N1F1 | N3F1 | Training min |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| M1 | 40501 | 0.782462 | 0.693772 | 0.365433 | 0.687757 | 25.46 |
| M4 | 40501 | 0.747142 | 0.680557 | 0.417784 | 0.652971 | 22.62 |
| m5_ce_s42 | 65717 | 0.821417 | 0.744214 | 0.467091 | 0.696373 | 59.62 |
| m5_wce_s42 | 65717 | 0.802452 | 0.740190 | 0.512166 | 0.682712 | 73.80 |
| m6_ce_s42 | 57589 | 0.810084 | 0.731693 | 0.466500 | 0.659225 | 47.02 |
| m6_wce_s42 | 57589 | 0.797330 | 0.732925 | 0.494638 | 0.660036 | 51.79 |
| m7_ce_s42 | 107893 | 0.823234 | 0.741260 | 0.455379 | 0.689093 | 65.84 |
| m7_wce_s42 | 107893 | 0.782925 | 0.728220 | 0.482900 | 0.681803 | 66.59 |
| m8_ce_s42 | 44629 | 0.817221 | 0.740848 | 0.433903 | 0.723525 | 90.12 |
| m8_wce_s42 | 44629 | 0.787451 | 0.726145 | 0.481234 | 0.652135 | 57.68 |
| m5_ce_s43 | 65717 | 0.810613 | 0.742066 | 0.471419 | 0.703810 | 66.11 |
| m5_ce_s44 | 65717 | 0.820624 | 0.740758 | 0.472433 | 0.701288 | 66.07 |
| m7_ce_s43 | 107893 | 0.819401 | 0.749300 | 0.496251 | 0.703345 | 74.17 |
| m7_ce_s44 | 107893 | 0.820194 | 0.745630 | 0.452338 | 0.719871 | 65.59 |
| m8_ablation_control_s42 | 44629 | 0.802452 | 0.717430 | 0.406780 | 0.689629 | 10.79 |
| m8_ablation_no_multiscale_s42 | 31861 | 0.801262 | 0.704538 | 0.357660 | 0.696872 | 8.37 |
| m8_ablation_no_context_s42 | 19413 | 0.770270 | 0.651634 | 0.244262 | 0.663304 | 7.97 |

CE/weighted CE rows use matched architecture/batch/budget; examine N1 gains alongside primary macro-F1 and N3 errors. Class precision/recall/F1 and raw/normalized confusion figures expose tradeoffs. Per-subject, ROC/PR, learning curves and saved embeddings remain available per run. Transition errors exclude recording boundaries and invalid-index gaps.

See phase4_models_comparison.csv,phase4_classwise_results.csv,phase4_transition_results.csv,phase4_multiseed_analysis.md,phase4_ablation_results.md and phase4_research_contribution_analysis.md. Figures include cost versus validation performance, confusions and per-run PCA/t-SNE/subject/ROC/learning curves. Separate model PCA/t-SNE axes are not shared geometric spaces.
The best initial configuration is m5_ce_s42. This is validation selection, not statistically proven improvement or test generalization. No Phase5 or new training is invoked by report generation.
