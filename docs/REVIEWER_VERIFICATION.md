# Independent verification without training

Run from the repository root with Python and NumPy installed:

```sh
PYTHON=python3
"$PYTHON" release_tools/verify.py --scope full --output verification_outputs/reviewer
```

The script imports neither PyTorch nor the research trainer/model/EEG loaders. It does not access a remote dataset. Formula implementations are independent of `src/sleep_research/phase5_statistics.py`; the actual research statistics implementation is also exposed for comparison.

## Evidence and privacy review

`evidence/aggregate_confusions.json` contains nine integer 5×5 overall matrices plus the pre-existing transition/stable matrices. `evidence/subject_confusions.json` contains the same outcomes partitioned into twelve clusters, with consistent aliases. There are no epoch rows, timestamps, probabilities, raw signals, intermediate representations, labels attached to individual epochs, demographics or individual names. Each checkpoint retains the same target population and true-class supports.

Cluster aliases preserve the original cluster order for exact registered bootstrap draws; a private alias map is retained locally and excluded. **Pseudonyms do not guarantee anonymity**, especially when the source dataset and split metadata are public. The author explicitly approved release of these subject counts and exact split identifiers; [the decisions](../provenance/release_decisions.json) record this separately from final inventory/publication authorization. This does not establish an ethics approval, legal opinion or guarantee of anonymity. If subject counts are declined, aggregate metrics remain verifiable, but subject-paired intervals cannot be independently reconstructed from pooled counts alone.

## What is actually recalculated

The verifier checks support totals and the sum of clusters against overall counts; calculates class precision, recall, F1, macro-F1, accuracy, balanced accuracy and kappa; checks all per-seed metrics and primary means/sample SD; verifies all primary class scores; recounts saved transition/stable matrices; checks subject/recording split hashes/grouping; and reconstructs the original 10,000 paired cluster-bootstrap draws and all stored fixed-seed/paired interval estimates. It matches the draw SHA-256 and uses the original quantile/confidence definitions.

It writes recalculated metrics/class scores/confusion counts/paired comparison CSV and a pass/fail JSON into the chosen output directory, leaving provided evidence unchanged. All score units and undefined-value conventions are explicit in source and [integrity documentation](REPRODUCIBILITY.md). Preparation obtained 654 matching numeric checks for all nine checkpoints; no changed statistic or selection rule was introduced.

Source-to-table lineage is in [source_to_table.json](../provenance/source_to_table.json): historical plan digest, checkpoint hashes, original prediction/export-manifest hashes, saved metrics/history hashes and aggregate transformations. Frozen tables were copied from saved evidence, then verified from counts. Hashes prove file consistency, not independently the truth of the original acquisition or training; that distinction is intentional.

## What additional inputs are needed

| Analysis | Available from lightweight bundle? | Additional approved inputs |
| --- | --- | --- |
| Accuracy/macro-F1/class P/R/F1/confusions | Yes | None |
| All fixed seeds and sample SD | Yes | None |
| Subject-paired intervals | Yes, author-approved counts included | Otherwise twelve paired subject matrices per checkpoint |
| Transition/stable aggregate scores | Yes, saved subset matrices | Original-grid metadata needed to independently reconstruct subset membership |
| Exact split identities/grouping | Yes, author-approved identifiers included | Raw files only for independent acquisition/filename checking |
| ROC/PR/AP/calibration/Brier/log loss | No | Approved per-epoch true labels and all probabilities, with checkpoint/target bindings |
| EEG examples/filter signal effects | No | Separately acquired raw EEG plus preprocessing |
| PCA/t-SNE/representations | No | Reviewed embeddings and exact saved coordinates/parameters, or independently reproduced models/data |
| Saliency/Grad-CAM/attention physiology | No | Approved saved explanation artifacts or a separately authorized analysis with models/data |
| Canonical target and transition eligibility from source | Partly: counts/hashes only | Original metadata/annotations and protocol manifests |
| Original training truth/checkpoint-resume archive | Source/protocol/hash evidence only | Separately approved original artifacts or a new reproduction |

For aggregate figures use `"$PYTHON" release_tools/figures.py --output verification_outputs/figures` with Matplotlib installed. It redraws selected panels from the verified counts and saved outcomes; it does not create new experimental observations or overwrite original saved figures. All detailed raw/model artifacts remain excluded unless separately approved under the release policy.

## Reduced public verification

```sh
"$PYTHON" release_tools/verify.py --scope aggregate --output verification_outputs/aggregate
```

This checks 162 saved metric equalities, using pooled overall and saved subset matrices without reading subject-count matrices or reconstructing confidence intervals. It still checks split hashes/night grouping if the complete split bundle is present. If all three split files are absent, the report explicitly marks split verification unavailable; a partially present bundle is rejected. `--scope auto` uses available evidence and reports its scope; `--scope full` requires both approved subject counts/bootstrap references and the complete split bundle. A reduced copy with both subject counts and split files removed passed verification during release finalization. Reduced mode removes a stale paired-comparison output rather than presenting an old interval file as recalculated.

If evidence is withheld in a future distribution, update its inventory, README, pipeline-input instructions and links before publication. Full EEG reproduction currently uses the included exact split manifests and cannot be claimed to run unchanged without them. No alternative split is silently generated. Pooled counts cannot reconstruct subject-paired uncertainty, probability metrics, original target eligibility or underlying physiology.
