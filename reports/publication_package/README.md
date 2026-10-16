# Publication build sources

This directory contains the Phase6 saved-artifact builder, verifier and manuscript/support-document templates. It is separate from the immutable earlier research source trees. The scripts write only new publication artifacts and never train, run model forward inference or alter Phase1–5 files.

Use the exact existing interpreter from the project root:

```sh
PYTHONPATH=src ${PYTHON} reports/publication_package/build_publication.py
PYTHONPATH=src ${PYTHON} reports/publication_package/verify_publication.py
```

The builder recalculates saved prediction metrics and the registered subject bootstrap, exports tables and renders figures. Templates use evidence-linked numeric tokens; the ledger, input hashes and output manifest live in `results/publication_tables`. References were verified online during preparation and are retained locally; no network, new data download or package installation is needed to rebuild. The preprocessing illustration is an explicitly labeled historical raster, while newly plotted quantitative figures have PDF counterparts and300-dpi PNGs.

The verifier checks the earlier preservation snapshot, metrics, exact bootstrap reproduction, model parameter/hyperparameter constants, partial MAC arithmetic, PCA/epoch lineage, figure resolution/checksums, document links, numeric tokens and reference metadata. Final-test predictions are already permanently saved. Do not invoke training or final inference to regenerate this package.
