# Release boundaries and checkpoint policy

> **Publication update (2026-10-16):** Published publicly at [https://github.com/araviraja-netizen/sleep-edf-eeg-sleep-staging-reproducible](https://github.com/araviraja-netizen/sleep-edf-eeg-sleep-staging-reproducible); initial reviewed commit `9ed9e3baf3b33054b74b5f068727fe503bcbe05b`. The preparation/review text below is retained as dated history. The 276-file initial inventory was explicitly approved and verified on GitHub; current hashes include the authorized publication-metadata update. No additional experimental results or physiological artifacts were released.

The author approved MIT for authorized original code, pseudonymized subject-level confusion counts, and exact frozen split identifiers. The final exact inventory still requires review before staging/commit; no public remote, upload or journal submission is authorized. See [recorded decisions](../provenance/release_decisions.json) and [license scope](LICENSE_SCOPE.md).

| Proposed content | Present in publication directory? | Decision before release |
| --- | --- | --- |
| Actual source, historical scripts/tests and registered scientific hyperparameters | Yes | Ownership/contributor authorization and scoped MIT confirmed by author |
| New portable wrappers and independent verifier | Yes | Scoped MIT approved; platform limits remain documented |
| Aggregate confusion matrices, metrics/tables, model/seed histories and selected non-physiological figures | Yes | Confirm publication/data-derived material terms and notices |
| Twelve pseudonymized subject confusion-count clusters | Yes | Explicit author release approval recorded; aliases are not guaranteed anonymous |
| Frozen split manifests with public dataset IDs | Yes | Explicit author release approval recorded; retain original dataset attribution |
| Curated manuscript/bibliography | Yes | Genuine software author metadata complete; no blanket manuscript/figure license implied; final inventory review pending |
| Curated historical reports with public recording identifiers and quality/count summaries | Yes | Review recording-level derived statistics/identifiers and applicable redistribution terms; linked physiological figures are withheld |
| Raw EDF/PSG/hypnogram files | No | Never uploaded by this task; obtain separately from PhysioNet |
| Bulk processed EEG / epoch manifests / original epoch-level physiological data | No | Not ordinary Git content; no release proposed here |
| Embeddings, detailed probabilities/labels/epoch IDs, attention/gradients, individual hypnograms | No | Separate license/privacy/content review; no automatic release |
| Full optimizer/RNG/checkpoint histories | No | Retain unchanged in the private research archive |
| Environments, caches, credentials, editor settings, Finder metadata | No | Never include |

## Trained weights

The [checkpoint inventory](../provenance/checkpoint_release_inventory.json) measures the nine selected historical resume checkpoint files and gives expected float32 parameter storage. Exact original checkpoint hashes, seeds and roles are in [the final-plan projection](../provenance/frozen_evaluation_projection.json). All original generations stay local. Small file size does not resolve redistribution rights or privacy.

Preferred future strategy, **only after separate approval**: selected inference-only weights in a versioned GitHub Release or a reviewed research archive. Git LFS is an option if a continuing large-artifact workflow warrants it, not an automatic choice; it is not required or installed here. Do not commit full checkpoint histories to normal Git. Pure inference weights differ from training-resume checkpoints, which include optimizer/RNG/early-stop/config/history states and may disclose additional provenance.

Any inference-only extraction must be a new file, never overwrite the original. Its manifest should include the source full-checkpoint hash, source model/key/seed, exact architecture/source/config/data contract, original selected epoch, extraction operation/tool version, tensor names/shapes/dtypes and the new artifact hash. A new weights-file hash will not equal the original full-checkpoint hash; preserve that parent/derived relationship rather than relabeling the old identity. Verify strict loading and prediction equivalence on separately authorized inputs before claiming it is interchangeable. This task does not extract, upload or approve weights.

## Optional epoch-level bundle

No complete epoch-level bundle has been prepared or copied into this proposal. If later approved, use a separate versioned, access-appropriate artifact release: five logits/probabilities, true/predicted labels, reviewed stable IDs, exact ordered target/checkpoint/manifest hashes and documented class order. Review traceable subject/time information, derived physiology, data terms, institutional policy and content rights first. A metadata/data-dictionary proposal alone does not establish that redistribution is permitted. Do not include embeddings or EEG automatically alongside predictions.

If subject counts are declined, retain pooled-metric verification but clearly mark paired subject intervals as not reconstructable from the reduced public bundle. Do not replace cluster bootstrap with epoch-IID intervals or fabricate data to fill the gap. All conditional policy decisions must be reflected in a regenerated exact publishing inventory and repeated audit before approval.
