# Saved results and independent checks

Historical primary final outcomes:

| Model | Seeds | Accuracy (%) | Macro-F1 | N1 F1 | N3 F1 | Balanced accuracy | Kappa |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CNN M1 | 1 | 80.18 | 0.7198 | 0.3823 | 0.7420 | 0.7088 | 0.7237 |
| CNN–BiLSTM M5 | 3 | 84.71 | 0.7771 | 0.4611 | 0.7892 | 0.7733 | 0.7872 |
| CNN–Transformer M7 | 3 | 84.48 | 0.7758 | 0.4752 | 0.7836 | 0.7664 | 0.7834 |

M5/M7 average all fixed seeds 42/43/44; M1 is seed 42 only. The [full-precision summary](../results/publication_tables/03_primary_test_summary.csv) separates seed sample SD from subject uncertainty. The [individual-seed table](../results/publication_tables/10_all_seed_test_metrics.csv), [class scores](../results/publication_tables/04_primary_classwise.csv), [paired comparisons](../results/publication_tables/05_paired_comparisons.csv), [validation development](../results/publication_tables/11_validation_metrics.csv) and [transition subsets](../results/publication_tables/08_transition_primary.csv) remain distinct. No new ranking is selected from them.

Transition/stable macro-F1: M1 0.5821/0.7289; M5 0.5636/0.8044; M7 0.5479/0.8042. Subsets contain 3,498 and 27,628 targets per checkpoint; 30 starts/gaps are neither. Label changes are defined retrospectively from consecutive original indices within a recording, not supplied as inputs or treated as localized physiological events.

The independent lightweight verifier matched **654 numerical checks** across nine checkpoints, including the original paired-bootstrap draws/intervals, without EEG, PyTorch or retraining. Its evidence is derived from checksum-verified frozen prediction counts and pre-existing transition matrices, not manually populated performance tables. [Source-to-table provenance](../provenance/source_to_table.json) binds each checkpoint, source CSV hash and aggregate derivation. Hashes establish file identity and internal traceability; they do not independently certify acquisition honesty or replace access to underlying evidence.

The final test set is consumed. New independent reproduction runs receive new fingerprints and may differ because of device/software/training nondeterminism. Their scores must be reported separately from these preserved historical measurements. No ensemble, test-winning seed, calibration fit or changed eligibility is introduced by this release.

Probability-based metrics present in original saved tables are retained as historical measurements, but cannot be independently recomputed from confusion counts alone. See [coverage and required inputs](REVIEWER_VERIFICATION.md). The original EEG/embedding/explanation/hypnogram figures are not in the proposed public source/evidence bundle.
