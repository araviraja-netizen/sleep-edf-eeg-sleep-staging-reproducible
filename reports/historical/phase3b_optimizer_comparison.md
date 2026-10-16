> Historical measured report, curated export copy. Statements about unstarted later phases are time-specific history. Machine paths and links are adapted; excluded detailed assets are not bundled. Use the portable commands in docs/EXPERIMENTS.md rather than these historical local-archive commands.

# Phase 3B — measured optimizer/loss pilot comparison

Generated: 2026-10-15T09:01:21.301786+05:30.



**Eight real MPS subset trials completed, seed 42 only; zero new full runs.**

Identical frozen 8,192 training / 4,096 validation subsets, three epochs per trial, same 40,501-parameter M1 architecture, initialization/shuffle seed, batch 16, float32, normalization and splits. All flags retained. Test remains locked.

These short subset trials cannot establish improvement over the fully trained M1 or estimate between-seed variability. RMSprop and class-balanced focal were tested/smoked but were not trained as pilot trials.


| Stage | Optimizer/LR | Loss | Macro-F1 | N1 F1 | N3 F1 | Accuracy | Bal. accuracy | Kappa | Best epoch | Seconds to best | Total train+val s |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| optimizer_pilot | AdamW/0.0003 | cross_entropy | 0.5112 | 0.0371 | 0.3902 | 0.7356 | 0.5112 | 0.6064 | 2 | 17.51 | 23.20 |
| optimizer_pilot | AdamW/0.001 | cross_entropy | 0.5297 | 0.0275 | 0.5581 | 0.7097 | 0.6005 | 0.5815 | 3 | 16.82 | 17.00 |
| optimizer_pilot | Adam/0.0003 | cross_entropy | 0.4924 | 0.0000 | 0.5173 | 0.7002 | 0.5726 | 0.5629 | 3 | 16.89 | 17.08 |
| optimizer_pilot | Adam/0.001 | cross_entropy | 0.5525 | 0.0000 | 0.6297 | 0.7354 | 0.6038 | 0.6146 | 3 | 17.07 | 17.24 |
| optimizer_pilot | SGD/0.01 | cross_entropy | 0.5182 | 0.0316 | 0.5972 | 0.7234 | 0.5563 | 0.5873 | 2 | 9.87 | 14.82 |
| optimizer_pilot | SGD/0.05 | cross_entropy | 0.1086 | 0.0000 | 0.0000 | 0.3728 | 0.2000 | 0.0000 | 1 | 5.12 | 14.74 |
| loss_pilot | Adam/0.001 | weighted_cross_entropy | 0.5848 | 0.2338 | 0.5856 | 0.6956 | 0.6026 | 0.5801 | 2 | 11.82 | 17.69 |
| loss_pilot | Adam/0.001 | focal | 0.5148 | 0.0317 | 0.5377 | 0.7063 | 0.5866 | 0.5740 | 3 | 17.54 | 17.73 |

Optimizer-stage selection: **adam_1e-3**, macro-F1 **0.552537**. This selection was recorded before the two loss trials. Same metric chooses each run’s retained best epoch; exact ties use registered order.
Full CE optimizer ranking is in the CSV. The selected optimizer’s CE trial is the loss-stage control; loss trials change only the objective. Training loss reductions differ; validation unweighted CE is stored for every trial and objective-specific validation loss is separate.

For the selected optimizer, weighted CE changed macro-F1 by +0.032297, N1 F1 by +0.233766, N3 F1 by -0.044076, and accuracy by -0.039795. This supports a matched full-budget confirmation, not an improvement claim over M1. Focal is deferred because this pilot did not improve the primary metric.
## Baseline context and limitations

M1, evaluated from its preserved saved predictions on these same 4,096 validation IDs: macro-F1 0.687128, N1 F1 0.338530, N3 F1 0.693182. M1 used all 134,047 training examples and17 epochs, so this is a descriptive reference, not a matched optimizer comparison.
Across-epoch F1 sample SD and seconds/epoch are saved in the CSV. They describe short-run fluctuations, not random-seed stability. All trials completed with finite losses/logits; the 16 real smoke updates also had finite gradients/parameters. Finite arithmetic does not guarantee useful classification: SGD at 0.05 collapsed to one predicted class at its selected epoch (kappa 0). Predicted class counts and a collapse indicator are retained. No failed trial was silently excluded. Learning curves, class-wise one-vs-rest/5x5 confusions, per-subject metrics, PCA and seeded 200-row t-SNE are saved under each run’s analysis folder.
All subsets span the frozen subject groups; sampling did not reallocate subjects. No flag removal, class resampling, test use or M1 fine-tuning. All initializations are fresh. The optimizer-specific LR probes are sparse; a slowly converging SGD configuration may be underestimated by three epochs. Shared numeric weight_decay 0.01 does not imply identical coupled/decoupled regularization.
Validation selection across candidates/epochs is optimistic; confirm a shortlist with matched full budgets and seeds 42/43/44 before drawing performance conclusions. No test generalization estimate, confidence interval or significance claim is made.

## Artifacts and commands

`phase3b_validation_comparison.csv` contains all measured trial metrics, class support/precision/recall/F1, resource/time information and exact checkpoint/export identities. `phase3b_pilot_runs.json` maps permanent run IDs; `phase3b_trial_registry.json` permits safe campaign continuation. Each run retains config/source/software/fingerprints, optimizer/null-scheduler/criterion/RNG/sampler/history state, best/latest/previous checkpoints, all five logits/probabilities/IDs/quality flags and 64-value penultimate/pooled intermediate features.
`phase3b_analysis_summary.json` maps analysis directories. `phase3b_figures/` contains comparison figures. `03b_optimizer_analysis.ipynb` reads saved artifacts only; it never launches training.
```sh
PYTHONPATH=src ${PYTHON} scripts/run_phase3b_tests.py
PYTHONPATH=src ${PYTHON} -m sleep_research.phase3b pilot
PYTHONPATH=src ${PYTHON} -m sleep_research.phase3b resume --run-id <EXISTING_PHASE3B_RUN_ID>
PYTHONPATH=src ${PYTHON} -m sleep_research.phase3b_analysis
```
Use `phase3b_experiment_plan.md` and `phase3b_full_run_proposals.json` for proposed full configurations and resource estimates. **Stopped after the bounded pilot; zero full runs launched.**



## Final validation evidence

All137 automated tests passed (previous118 plus19 new), with zero failures/errors/skips.
The executed notebook contains7 code cells,7 PNG outputs and zero errors. Legacy test code
and original result logs remain unchanged; two legacy JSON writes were redirected to SSD
temporary files. The model source,153 arrays, normalization/splits and all124 M1 run files
match the preservation snapshot. Two shared modules gained backward-compatible extensions.

Measured pilot training/validation attempt time: 139.50s total,
excluding setup/smoke/export/analysis/notebook. Retained pilot run artifacts:
66.84MiB. Reported pilot RSS is a cumulative process high-water value,
not an independent isolated-memory benchmark for each trial. All4096 validation rows per
trial retain metadata and the same18 flagged examples. Each best/latest checkpoint,
loss state, exported metric/embedding, class confusion and seeded projection was checked.
`phase3b_final_validation.json` records these checks. No full or multi-seed results exist.
