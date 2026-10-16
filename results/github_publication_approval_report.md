# Publication approval report

Prepared 2026-10-16T15:28:29.337367+05:30. **Ready for final inventory review; publication is not authorized.**

Repository name: **sleep-edf-eeg-sleep-staging-reproducible**. The publication directory is the separate export with this basename, not the private research archive. Its exact machine-local location is recorded in the author's private finalization report, excluded from this release.

## Complete proposed staging inventory

[PUBLIC_FILE_INVENTORY.csv](../provenance/PUBLIC_FILE_INVENTORY.csv) is the complete path/byte/SHA-256 inventory; its identical public mirror is [github_publication_manifest.csv](github_publication_manifest.csv). [file_hashes.json](../provenance/file_hashes.json) seals payload hashes and the reviewed inventory digest; [reviewed_paths.nul](../provenance/reviewed_paths.nul) is the exact candidate staging allowlist. [size_summary.json](../provenance/size_summary.json) records the final count and total bytes. These are proposed staging controls, not evidence that files were staged or final inventory approval was granted. Four controls are explicitly excluded from recursive payload hashes; their sizes/path set and CSV mirror are verified separately. The external approval record must bind the exact inventory digest.

## Author and content decisions

| Decision | Recorded status |
| --- | --- |
| Software author | Ravi Raja Akurathi; given names Ravi Raja, family name Akurathi |
| Copyright | 2026 Ravi Raja Akurathi, explicitly confirmed |
| Original code ownership/contributor authorization | Confirmed by author; not independently certified |
| MIT for authorized original code only | Approved and written; excludes PhysioNet data/third-party/non-code science |
| Pseudonymized subject confusion counts | Explicitly approved |
| Exact frozen split identifiers | Explicitly approved |
| Citation | Schema-valid CFF 1.2.0; existing package version 0.2.0; no invented URL/date/DOI |
| Final exact inventory and initial commit | Awaiting review/authorization |
| GitHub owner/name/public visibility/remote/push | Awaiting separate authorization |
| Git commit email | Not supplied or guessed; use verified identity locally |

Subject-level matrices enable exact paired subject-bootstrap intervals without per-epoch predictions. Exact split identifiers enable fixed allocation, grouping and leakage checks. Aliases and public-dataset IDs can be linked to public physiology; there is no anonymity guarantee. Inclusion approval is not an ethics determination or a new data license. The [license-scope notice](../docs/LICENSE_SCOPE.md) retains dataset attribution and original terms. Manuscripts/figures are not blanket-relicensed by MIT.

## Security and reproduction review

The original research archive and the separate publication export serve different roles. The full private Phase 7 audit and inclusion/exclusion census remain private; only the public-safe `results/github_publication_audit.md` is a release audit. `reports/revised_reference_audit.md` is a scientific bibliography audit, not a duplicate publication audit. Private filenames, the cluster-to-original-subject mapping and local preparation logs are not copied into this export.

The lightweight bundle contains source, frozen settings, reviewed count evidence, selected aggregate reports/figures and provenance hashes. EEG, processed arrays, original predictions, checkpoints, model weights, embeddings and detailed physiological assets are excluded. Credential/path/large-file/type/metadata checks and research preservation findings are in the [validation record](../provenance/prepublication_validation.json). Original scientific sources, measured results and saved figure files remain unchanged; only export release tooling and release documentation were finalized.

176 CPU/synthetic tests passed with zero failures/errors, while 45 checks requiring excluded inputs remain unrun. Count verification matched 654 checks with the complete bundle; the reduced mode matched 162 and explicitly withholds subject-bootstrap claims. The reviewer notebook executed three cells/one aggregate image without errors. Synthetic checkpoint/resume and aggregate redraw checks passed. No installation, real training, new inference or test-driven model selection occurred. A literal Git clone check and long MPS end-to-end reproduction are not claimed completed.

## Safe next steps, after approval

1. Approve the exact current inventory digest and the intended GitHub owner/name/visibility. Changes require a fresh digest/review.
2. Use the external author approval record with [prepare_git.py](../release_tools/prepare_git.py), first `--check-only`, then `--stage` after inventory authorization. This initializes only the separate empty export and stages the explicit NUL allowlist. Do not use `git add .`.
3. Inspect `git status --short`, `git diff --cached --stat` and the complete cached file list/diff. Configure the genuine commit identity locally, then commit using [proposed_initial_commit.txt](../provenance/proposed_initial_commit.txt).
4. Clone that local commit into a new folder and run `release_tools/verify.py --scope full`. A directory-copy check already passed; this pending clone step is distinct.
5. Only after separate publication approval, use the existing VS Code GitHub sign-in and **Publish to GitHub** from this export, confirming the intended account/name/public visibility. No `gh` installation is needed. Exact steps are in [GITHUB_PUBLICATION.md](../docs/GITHUB_PUBLICATION.md).
6. After publication, capture the real URL, replace README's local/pre-publication status with an accurate published status and that URL, and add the real URL to CFF. Recheck metadata/hashes/verification and commit/push the reviewed documentation update. No URL is invented beforehand.

No Git repository, staged files, initial commit, remote or push was created by this review. Stop for the author's final approval.
