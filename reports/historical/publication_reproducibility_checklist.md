> Historical measured report, curated export copy. Statements about unstarted later phases are time-specific history. Machine paths and links are adapted; excluded detailed assets are not bundled. Use the portable commands in docs/EXPERIMENTS.md rather than these historical local-archive commands.

# Publication reproducibility checklist

## Data and eligibility

- [x] Existing dataset `${DATASET_ROOT}` and interpreter `${PYTHON}` verified; no installation, environment creation, dataset download or raw-data modification.
- [x] SC only for model comparisons: 153 recordings / 78 subjects. ST remains separate.
- [x] Frozen 54/12/12 subject allocation; nights grouped; no subject leakage.
- [x] Five-class mapping, movement/unknown exclusions, exact 30-second alignment and original indices documented.
- [x] Continuous filter/SOS/padding, acquisition-filter limitations and fixed normalization documented.
- [x] Label-based 30-minute Wake margins and offline deployment limitation disclosed.
- [x] Same 31,156 targets, 22 test recordings and retained quality flags for all final checkpoints.

## Models and final evaluation

- [x] Exact source/configuration/checkpoint/data/software fingerprints retained in the immutable Phase5 plan.
- [x] All 9 checkpoints selected before test; no lucky-seed selection, ensembling or test-based threshold fitting.
- [x] CNN single-seed and contextual three-seed evidence explicitly distinguished.
- [x] Context masks stop at gaps and subject/recording/night/split boundaries; neighbor labels are not inputs.
- [x] Earlier histories, checkpoints, predictions and embeddings preserved byte-for-byte.
- [x] No new training, model forward, gradient extraction or final inference in Phase6.

## Statistics and saved representations

- [x] Recalculate individual-run and class-wise metrics from saved predictions.
- [x] Reproduce the registered 10000 subject-paired bootstrap draws and intervals exactly.
- [x] Primary metric is five-class macro-F1; missing-class and undefined-value conventions explicit.
- [x] Separate subject uncertainty from fixed training-seed sample SD; all individual seeds retained.
- [x] Report M5-versus-M7 interval crossing zero; avoid clinical-effectiveness and SOTA assertions.
- [x] Save CSV/Markdown tables, linked PCA/t-SNE coordinates, publication figures and captions.
- [x] All illustrative final plots use predefined seed42; recording selection is deterministic, not outcome-ranked.
- [x] Explainability figures reuse checkpoint-bound validation examples and make no causal claims.

## Independent reproduction from existing evidence

From the project root, using the existing interpreter:

```sh
PYTHONPATH=src ${PYTHON} reports/publication_package/build_publication.py
PYTHONPATH=src ${PYTHON} reports/publication_package/verify_publication.py
```

The builder reads saved artifacts, recalculates metrics/statistics and redraws figures; it does not import or call the trainer and does not repeat model inference. It writes only the new publication files. It is outside the locked earlier source directories so previous registries remain valid. The verifier checks source preservation, numbers, linked outputs and publication checksums. Do not rerun training or final inference to reproduce these tables.

## Author actions still required

- [ ] Supply author names/affiliations, contributor roles and corresponding-author details.
- [ ] Confirm funding, competing interests and applicable ethics/data-reuse statements.
- [ ] Provide a stable public code/archive link; local preservation alone is not public release.
- [ ] Choose a journal and adapt its formatting/declaration requirements.
- [ ] Complete final scientific and editorial author review and submission declarations.

No item above authorizes further model development on the consumed test cohort.
