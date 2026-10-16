> Historical measured report, curated export copy. Statements about unstarted later phases are time-specific history. Machine paths and links are adapted; excluded detailed assets are not bundled. Use the portable commands in docs/EXPERIMENTS.md rather than these historical local-archive commands.

# Phase 2.5 — resumable training and research experiment infrastructure

Generated: 2026-10-14T17:38:43.890947+05:30 (Asia/Kolkata).

## Completion and boundaries

**Infrastructure built and validated. Real Sleep-EDF CNN training has not started.**
The preservation snapshot verifies **552 prior files**, including all 153 processed EEG arrays,
existing code, configurations, reports, manifests, normalization, frozen splits and both prior notebooks.
New reusable code is isolated in `src/sleep_research`; existing Phase 1/2 code was not changed.
No environment/package installation, upgrade, dataset download or preprocessing regeneration occurred.
All retained model runs use tiny synthetic signals. Their metrics are infrastructure evidence, not Sleep-EDF performance.
No real frozen-test logits, predictions, embeddings or performance metrics were computed.

## Data quality review

Protocol B: 12,020 rail epochs, 301 edge flags, 16 overlaps;
12,305 distinct flagged epochs. Protocol A: 2,226 rail epochs,
9 edge flags, 0 overlaps. Seven measured training-only waveform examples were saved.
Subject/recording/split/stage distributions and proposed pre-registered sensitivity analyses are in
`results/artifact_quality_review.md` and the quality CSVs. Existing primary data and labels remain unchanged.

## Checkpoint and resume implementation

Shared manager: `checkpoints.py`; reusable trainer: `trainer.py`; permanent runs/contracts: `experiments.py`.
Immutable checkpoint generations and byte-hash sidecars are validated with the restricted Torch loader before
an atomic pointer index publishes latest, best and previous together. A checksummed previous index permits
recovery if the current pointer file is damaged. Uncommitted orphan generations are never auto-selected.
A prior recoverable epoch is retained. Rolling roles and the prior index typically retain at most four
generations; checkpoints referenced by permanent exports are additionally pinned for later inference.
Best selection uses strictly highest validation macro-F1 over all five classes. Best/latest roles are independent.
Checkpoint checksums detect corruption; they are integrity evidence rather than authentication.

Payloads save model/optimizer/scheduler, early-stop counters/best score, completed/next epoch, global step,
Python/global-NumPy/CPU-Torch RNG, relevant CUDA/MPS RNG, loader generators, optional sampler/dataset state,
history, exact config, source revision and architecture/data/split/normalization/class/protocol fingerprints.
A bootstrap checkpoint stores completed=-1 and next=0. Every trained-and-validated completed epoch then
commits scheduler/early-stop/history state. An interruption before that commit repeats the entire epoch
from the prior checkpoint. Materialized history is rebuilt from the checkpoint without duplicates.
Manual rewind establishes a new committed timeline and discards future best state from the abandoned timeline.
Incompatible architecture/config/preprocessing/normalization/classes/subject splits/epoch manifests/protocols
or experiment identity are rejected, rather than silently restarting or adapting model weights.

CLI resume values: `--resume auto`, `--resume none`, `--resume /absolute/checkpoint/path.pt`.
Epoch budget extension is allowed; other training configuration changes are rejected. A fresh start preserves
existing checkpoints by requiring a new run. Both Terminal and Jupyter invoke the same `Trainer` class.
**Bitwise-identical Apple MPS resumption is not promised.** CPU equivalence is validated only for this
software/configuration fixture. Different backends, versions, workers or custom stochastic transforms may differ.
Custom RNGs need dataset/sampler state callbacks; worker-internal RNG state is not captured. Demo workers=0.

## Permanent research artifacts

Runs use `experiments/{cnn,cnn_lstm,attention,transformer}/<unique_run_id>/`; only the cnn family currently
has synthetic demonstration runs. The other family directories are ready for future authorized models.

- Reproducibility: config, software, seed/device, source hashes/Git revision if available, frozen contract,
  attempt durations/status, optimizer/scheduler/model checkpoints and integrity/fingerprint manifests.
- Performance: committed train/validation history and epoch-ID-aligned true/predicted labels, all five
  raw logits, five probabilities, subject/recording/night/original-index/onset metadata and split membership.
- Representations: penultimate features, row-to-epoch mapping and configurable pooled intermediate features.
  Intermediate tensors are off by default and have a width cap. No entire processed EEG dataset is copied.

Inference exports pin their source checkpoint so more features/explanations can be generated without retraining.
Optional bounded saliency and compatible Conv1d Grad-CAM are demonstrated on synthetic data. Attention capture
is tested with a compatible MultiheadAttention fixture returning weights; arbitrary Transformer layers may
require an adapter. Unsupported/non-returning attention layers fail explicitly. Explanations are not causal proof.

## Analysis-only interfaces and final-test gate

`analysis.py` imports no trainer and reads verified saved exports. It recalculates confusion matrices,
accuracy/precision/recall/F1, balanced accuracy, Cohen kappa, log loss/Brier/ECE, ROC/PR, subject metrics,
learning curves, PCA/t-SNE and predicted-versus-true hypnograms. Original recording boundaries and gaps are respected.
True labels/probabilities and all five logits remain available for recalibration and ROC/PR inputs.
PCA uses centered SVD with recorded components/variance. Exact NumPy/SciPy t-SNE matches the analytic gradient
in a finite-difference test and records seed, perplexity, iterations, subset IDs and KL divergence.
Its quadratic cost is bounded by a saved subsample. Scikit-learn and UMAP are absent; no packages were installed.
The optional UMAP function reports unavailable unless the package already exists.

Default inference/analysis gates reject test and mixed-test exports before model test performance is computed.
`--final-evaluation` is an explicit future-stage interface; no real-data final-evaluation invocation occurred.
Data-quality counts by test split are descriptive baseline statistics and do not bypass this gate.

## Validation evidence

Final suite: **104 tests; 0 failures, 0 errors, 0 skipped**.
The suite includes all 72 prior tests, plus interruption/continuation, CPU weights/Adam moments/StepLR/RNG
restoration, bootstrap recovery, best/latest independence, corrupted bytes/pointers, pinned retention,
history uniqueness, incompatibility rejection, exact exported IDs/probabilities/features, PCA/t-SNE,
confusion recalculation without training, gap-aware plotting, attention/Grad-CAM/saliency and CLI integration.
Old test code remains unchanged; the two legacy JSON-writing callbacks are redirected to temporary SSD-local
paths so original Phase 2 test evidence is never rewritten. Logs: `results/phase2_5_test_results.txt`.
Notebook: success=True; **5 code cells, 7 PNG outputs, 0 errors**.
The notebook uses the exact existing interpreter and a temporary local kernel with explicit cleanup.
Synthetic notebook interruption was inside epoch 1; it recovered next epoch 1
and ended at next epoch 3 with unique history [0, 1, 2].
Terminal evidence: `phase2_5_terminal_interrupt.txt` and `phase2_5_terminal_resume.txt`.
Notebook evidence: `phase2_5_notebook_execution.json`, `phase2_5_notebook_demo.json` and the executed notebook.

| Retained synthetic run | Completed epoch | Next epoch | Global step | Checkpoints | Exports | Attempt seconds |
| --- | --- | --- | --- | --- | --- | --- |
| cli-synthetic-test-116c21c2 | 0 | 1 | 5 | 2 | 0 | 0.116 |
| cli-synthetic-test-508f654c | 0 | 1 | 5 | 2 | 0 | 0.118 |
| cli-synthetic-test-75ae5fd6 | 0 | 1 | 5 | 2 | 0 | 0.117 |
| cli-synthetic-test-bac712b7 | 0 | 1 | 5 | 2 | 0 | 0.118 |
| cli-synthetic-test-d9119167 | 0 | 1 | 5 | 2 | 0 | 0.119 |
| notebook-synthetic-a80fee2b | 2 | 3 | 15 | 3 | 1 | 0.247 |
| terminal-synthetic-demo | 3 | 4 | 20 | 3 | 1 | 0.261 |

Permanent experiment storage: **2,362,877 bytes (2.253 MiB)**, excluding the existing EEG dataset.
All permanent model artifacts are marked synthetic. Trained synthetic checkpoints are retained for additional inference.

## Readiness and remaining scope

Shared epoch-boundary checkpointing, resume, permanent artifact export and analysis-only workflows are ready
for later explicitly authorized model experiments. The primary processed dataset and frozen split are unchanged.
Actual CNN/CNN-LSTM/attention/Transformer architecture selection, full training, scientific comparisons and the
frozen final-evaluation stage remain future work. Artifact sensitivity masks are proposals, not applied changes.
**Stopped after Phase 2.5 infrastructure. No full Sleep-EDF CNN training was started.**

## References and reproducibility

- [PyTorch reproducibility](https://docs.pytorch.org/docs/2.14/notes/randomness.html): backend/version reproducibility limits.
- [PyTorch serialization](https://docs.pytorch.org/docs/2.14/notes/serialization.html): state dictionaries and restricted loading.
- [van der Maaten & Hinton, Visualizing Data using t-SNE](https://www.jmlr.org/papers/v9/vandermaaten08a.html), JMLR 9 (2008).
- `configs/infrastructure_demo.yaml`, README commands and `src/sleep_research/` modules.
- `results/phase2_5_preservation_snapshot.json` and `phase2_5_artifact_manifest.json`.
- Prior dataset attribution and experimental protocol remain in the preserved Phase 1/2 reports.
