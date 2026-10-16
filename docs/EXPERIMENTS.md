# Reproducing the actual experimental pipeline

Use the [installation guide](INSTALLATION.md) and choose `PYTHON`, `WORKSPACE` and `DATASET_ROOT` as shown in the README. Workspaces must be new/empty and outside both the raw dataset and the public source tree. The `init` command renders path templates with the executing interpreter, copies the unchanged scientific source/tests, and installs a portable pilot-test driver in the generated workspace. It does not process EEG or train a model. All subsequent raw-data outputs stay in that workspace, including `data/processed`, `results` and `experiments/<family>/<run_id>/reproducibility/checkpoints`.

The scientific model/filter/loader/trainer/checkpoint/export implementations under `src` are byte-identical copies of the research project. The original historical CLI wrappers contain local-machine checks and signed-archive assumptions. Portable orchestration is added under `release_tools`; it calls the same scientific components and creates new reproducibility fingerprints. It does not edit old signed registries to make a different machine appear to be the historical environment. Existing historical checkpoints cannot be silently resumed under new identities.

## Dataset inspection, preprocessing, split and normalization

```sh
"$PYTHON" release_tools/pipeline.py init --workspace "$WORKSPACE" --dataset-root "$DATASET_ROOT"
"$PYTHON" release_tools/pipeline.py inspect --workspace "$WORKSPACE"
"$PYTHON" release_tools/pipeline.py preprocess-pilot --workspace "$WORKSPACE"
"$PYTHON" release_tools/pipeline.py preprocess-all --workspace "$WORKSPACE"
"$PYTHON" release_tools/pipeline.py verify-splits --workspace "$WORKSPACE"
"$PYTHON" release_tools/pipeline.py normalization --workspace "$WORKSPACE"
```

`inspect` invokes actual Phase 1 code. The two-night preprocessing pilot invokes the original inspection/filter/storage tests with runtime workspace/dataset bindings; no test source is rewritten. All-SC processing requires that pilot and its passing evidence. Per-recording outputs are atomic and resume only validated completions. Both the native filter and frozen subject assignment are unchanged. `normalization` verifies the actual train-only fit already performed in original Phase 2 consolidation; it does not refit on test/validation. Repeating preprocessing commands validates cached recordings; it does not alter raw EDFs.

The original split bytes are copied into the new workspace and checked against the original freeze hashes and generated seed-42 assignment. Do not relocate subjects, rebalance epoch populations, replace timing gaps or alter eligibility after seeing outcomes. Historical normalization hashes may differ in a new pipeline archive because source/path/software provenance contributes to its identity; scientific mean/std and retained target IDs should be compared separately and discrepancies documented. Different libraries/hardware do not imply bit-identical filtered arrays.

## CNN, optimizer and weighted-loss confirmations

```sh
"$PYTHON" release_tools/pipeline.py train --workspace "$WORKSPACE" --key M1 --resume auto
"$PYTHON" release_tools/pipeline.py train --workspace "$WORKSPACE" --key M2 --resume auto
"$PYTHON" release_tools/pipeline.py train --workspace "$WORKSPACE" --key M4 --resume auto
```

These commands are for manual local Terminal execution with MPS available. Preparation did not run them. M1 is the actual 40,501-parameter CNN with AdamW/ordinary CE; M2 changes optimizer to Adam; M4 changes training loss to class-weighted CE. LR 0.001, weight decay 0.01, batch 16, maximum 25 epochs, patience eight, no scheduler, float32 and validation macro-F1 selection remain as registered. Adam's coupled regularization and AdamW's decoupled decay differ despite matching numeric decay. Weighted CE uses only training counts `[43782,15249,47664,9302,18050]` and `N/(5*n_c)` with sum(weighted CE)/sum(weights). Ordinary validation loss remains separate. The short original Phase 3B search settings/objectives/optimizers are also exposed, but its subset pilot is not confused with these full-budget comparisons. M3 is unexecuted, not a reported result.

## Advanced models and repeated seeds

```sh
"$PYTHON" release_tools/pipeline.py campaign --workspace "$WORKSPACE" --resume auto
```

The original `CampaignEngine`, `Trainer`, objective factory, models, `ContextDataset` and streaming export are reused. The reproduction replays eight initial architecture/loss configurations, M5/M7 CE repeat seeds 43/44 and three M8 seed-42 three-epoch ablations. The original M5/M7 shortlist is fixed for this historical reproduction; fresh validation scores do not redefine the final-test roster. All configurations are saved before execution in a new immutable workspace registration. This is a methodological reproduction, not a new model-selection claim or permission to optimize on the consumed test cohort. Individual registered runs can also be invoked with `train --key m5_ce_s42`, etc.

M5/M6/M7 use the actual shared CNN encoder design. M5 uses a masked bidirectional LSTM; M6 uses central-query attention; M7 uses two compact Transformer blocks. M8 uses multiscale convolution and gated central/context fusion. `m8_ablation_control_s42`, `m8_ablation_no_multiscale_s42` and `m8_ablation_no_context_s42` retain fresh, equal three-epoch budgets. They are exploratory validation studies, not full-convergence module confirmation. Model sizes, masks and original boundary behavior are tested on synthetic inputs.

## Resume, checkpoint inspection and validation analysis

Repeat exactly the same training/campaign command with `--resume auto`. The actual checkpoint manager validates checksums/fingerprints and retains best/latest/previous roles and recoverable generations; every completed epoch commits model, optimizer, scheduler, RNG, sampler, early-stop and history state. Incomplete epochs repeat. A completed run is verified and skipped by the campaign rather than optimized again. Do not reduce budgets, change optimizer/weights or edit the source to force a resume. No bitwise MPS guarantee is made.

```sh
RUN="$WORKSPACE/experiments/cnn/reproduction-M1"
"$PYTHON" release_tools/pipeline.py checkpoints --workspace "$WORKSPACE" --run "$RUN"
"$PYTHON" release_tools/analyze.py --export /path/to/saved/validation/export --output verification_outputs/validation
```

The exact export path is recorded in each run's `performance/portable_result.json`. Analysis uses saved logits/probabilities/features and does not import the trainer. It writes to a separately chosen output directory. Per-run streaming exports preserve all five logits/probabilities, penultimate features and original central/context metadata locally, outside ordinary Git.

## Fixed final evaluation: review, freeze, then authorize

The historical nine checkpoint hashes are provided as evidence, not bundled weights. New trained reproduction checkpoints have different hashes. Do not call their outputs the original historical measurements. Before any final forward pass, every fixed final model must be complete and best-selected on validation: M1 seed 42; M5/M7 CE 42/43/44; secondary M4 and weighted M5 seed 42.

```sh
"$PYTHON" release_tools/pipeline.py plan-final --workspace "$WORKSPACE"
# Read results/portable_final_plan.json and the exact printed digest before continuing.
PLAN_SHA256=the_exact_digest_you_reviewed
"$PYTHON" release_tools/pipeline.py freeze-final --workspace "$WORKSPACE" --reviewed-plan-sha256 "$PLAN_SHA256"
"$PYTHON" release_tools/pipeline.py final-evaluate --workspace "$WORKSPACE" --authorize-final-evaluation --plan-sha256 "$PLAN_SHA256"
```

Freezing is distinct from authorization. The exact digest and explicit flag are required before the actual `FinalEpochDataset`/`FinalContextDataset` can load EEG; source/data/checkpoint drift is rejected. Full inference requires MPS. Checkpoint reproducibility directories are copied into a plan-owned final workspace; the original training directories stay unchanged. Original 2,048-target chunk export/commit/resume/merge functions are reused with exact canonical targets and masks. Repeating the authorized command resumes verified committed chunks. Development training is blocked after the final freeze.

```sh
"$PYTHON" release_tools/analyze.py --workspace "$WORKSPACE" --final-statistics --final-evaluation --output verification_outputs/final_statistics
"$PYTHON" release_tools/figures.py --output verification_outputs/publication_figures
```

The first command analyzes completed local reproduction exports using the original statistics routine and fixed seed groups. The second independently redraws selected published aggregate figures from the bundled historical counts; it never mixes a new reproduction's scores into the historical publication table. To analyze a single saved final export, provide `--export ... --final-evaluation --output ...`. Complete legacy publication plotting scripts remain exposed under `reports/publication_package`; their private-archive inputs are described in [availability](ARTIFACT_AVAILABILITY.md), not falsely advertised as fully bundled.

## Computation and storage

Observed source evidence is [resource_evidence.json](../provenance/resource_evidence.json). The fifteen advanced runs recorded **48,691.28 s / 13.53 h** of training-attempt time, excluding preprocessing/export/analysis and unrecorded interrupted-attempt time. The original pre-training full-budget projection was **18.6–49.4 h**, not a measured duration or guarantee. Baseline/optimizer/weighted confirmations and Phase 3B pilot add separate time. Primary training means are in the saved [cost table](../results/publication_tables/06_computational_cost.csv).

The local raw dataset measured about 8.12 GiB; processed EEG arrays about 4.64 GiB (4,979,551,584 bytes), shared across protocols and models. Keep at least 8 GiB free headroom; plan roughly 20 GiB for a new workspace beyond raw data if retaining extensive exports/recovery copies. This storage budget is an estimate, not a measured minimum. Memory-mapped float32 arrays, one-recording preprocessing, workers zero and bounded feature exports target the original 16-GB machine. No new real benchmark, full training or real final inference was run in release preparation.
