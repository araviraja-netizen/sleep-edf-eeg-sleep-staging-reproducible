# GitHub publication procedure — final review pending

Repository name: **sleep-edf-eeg-sleep-staging-reproducible**. The release export is a separate sibling directory of this name, never the private research archive. No remote URL is assigned yet.

Description: “Reproducible single-channel Sleep-EDF sleep staging: compact CNN/context-model comparisons, frozen subject splits, resumable experiments and independently verifiable results.”

Topics: `sleep-staging`, `sleep-edf`, `eeg`, `pytorch`, `reproducible-research`, `biomedical-signal-processing`, `time-series-classification`.

## Completed decisions and remaining approval

Ravi Raja Akurathi confirmed the author name, 2026 copyright, original code ownership/contributor authorization, scoped MIT, release of pseudonymized subject confusion counts and exact frozen split identifiers. [CITATION.cff](../CITATION.cff) validates against the official CFF 1.2.0 schema. Optional ORCID/affiliation/contact details were not supplied and are omitted. An existing package version is recorded; no public release date, DOI or repository URL is invented. See [license scope](LICENSE_SCOPE.md), [release decisions](../provenance/release_decisions.json) and [approval report](../results/github_publication_approval_report.md).

Review the complete [file inventory](../provenance/PUBLIC_FILE_INVENTORY.csv) and its exact SHA-256 in [file_hashes.json](../provenance/file_hashes.json). Category approvals do not authorize an unreviewed final inventory, commit, public remote or push. Any content change requires rehashing and renewed inventory review. Confirm the GitHub owner/name/visibility and the genuine Git commit email locally; never send account credentials or tokens in chat.

## Safe local Git preparation after inventory approval

No Git repository, index, initial commit or remote was created during preparation. The prepared [initial commit message](../provenance/proposed_initial_commit.txt) and NUL-delimited [explicit allowlist](../provenance/reviewed_paths.nul) are candidates awaiting final approval, not proof of staging. Do not use `git add .` or stage the private research directory.

Use an external approval JSON outside the export. Copy the recorded decisions, add the exact reviewed `inventory_sha256`, and set `publication_inventory_approved` to true only after the author approves that digest. Do not edit the hashed public decision record to record this later action; record it externally so approval cannot invalidate its own inventory. `prepare_git.py --check-only` validates the current candidate without invoking Git; `--stage` additionally requires final inventory approval and initializes only an empty new repository on `main`.

```sh
PYTHON=python3
APPROVAL_RECORD=/path/to/private/approved-release.json
REVIEWED_SHA256=the_exact_reviewed_inventory_digest
"$PYTHON" release_tools/prepare_git.py --check-only --reviewed-manifest-sha256 "$REVIEWED_SHA256" --approval-record "$APPROVAL_RECORD"
# Only after explicit approval of that exact inventory:
"$PYTHON" release_tools/prepare_git.py --stage --reviewed-manifest-sha256 "$REVIEWED_SHA256" --approval-record "$APPROVAL_RECORD"
git status --short
git diff --cached --stat
git diff --cached --name-only
```

The staging guard verifies inventory/payload/control sizes and hashes, path safety and the separate sensitive-content approvals before any Git call. It rejects an ancestor repository, existing history or pre-existing staged changes requiring separate review. It never commits, creates a remote or pushes. Configure the genuine author identity locally; the name is Ravi Raja Akurathi, but the Git commit email must be supplied by the author (a verified GitHub noreply address is an option). No email or global identity configuration is invented.

After reviewing the cached diff and confirming identity, create the initial commit with `git commit -F provenance/proposed_initial_commit.txt`. Then perform a literal local clone into a new directory and run `release_tools/verify.py --scope full` there. The isolated directory-copy checks already passed; a Git clone remains pending because no first commit is authorized yet. If a Git history is found at publication time, audit it rather than reinitializing or rewriting it.

## Publish using the existing VS Code GitHub sign-in

`gh` is not installed and is unnecessary. Open **only the separate export** in VS Code. Use Accounts to check the existing GitHub sign-in, then Source Control to inspect the approved commit. Do not use “Commit All” to bypass the allowlist. After the author separately approves the actual GitHub owner/name and **public** visibility, choose **Publish to GitHub**, select the intended account/name and public visibility, and verify the displayed remote. This action creates a remote and uploads files; do not perform it before that approval. See [VS Code's official Source Control guide](https://code.visualstudio.com/docs/sourcecontrol/quickstart).

After upload, verify the visible file list, license/citation rendering and absence of excluded data. Record the actual resulting URL. Immediately update README's release-status paragraph to say published and link the real repository, and add that same real URL to `CITATION.cff` as `repository-code`. Update any current release-status documentation that still says unpublished; historical audit statements remain dated history. Revalidate CFF, rebuild hashes/inventory, rerun verification and commit/push that reviewed metadata update. Do not leave README describing only a local proposal after publication. Never manufacture a URL, author email, release date, tag or DOI beforehand.

A version tag or Zenodo archival integration is a later reviewed action. Weight files, full optimizer/RNG recovery archives and detailed physiological/epoch-level artifacts are not part of this source release. Never force-push or alter visibility without explicit authorization.
