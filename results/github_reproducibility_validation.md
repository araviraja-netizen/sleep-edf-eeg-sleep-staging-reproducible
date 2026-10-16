# Finalization reproducibility verification

Refreshed 2026-10-16T15:28:29.337367+05:30. The final staging inventory and publication actions remain pending author approval; metadata and the two evidence-category approvals are complete.

| Verification | Actual result |
| --- | --- |
| CPU/synthetic tests | 176 passed; zero failures/errors; 45 additional-input checks not run |
| Full saved-count verification | 654 matching checks; nine fixed checkpoints; original 10,000 paired subject draws |
| Reduced verification | 162 matching checks; no subject bootstrap claim; absent split bundle explicitly reported |
| Synthetic smoke | Actual CNN/advanced models and checkpoint/resume passed; no real EEG |
| Reviewer notebook | Three code cells, one aggregate image, zero errors; no inference/training |
| Aggregate figure redraw | Passed from saved confusion counts; saved publication figures unchanged |
| Citation | Official CFF 1.2.0 schema passed; confirmed author and scoped MIT |
| Scientific source/script/test fidelity | Byte identity checked against original archive; see current machine-readable record |
| File/security/link/preservation checks | Current findings in [prepublication_validation.json](../provenance/prepublication_validation.json) |
| Isolated arbitrary-path copy | Count verification/imports checked without Git; actual clone remains pending approved commit |
| Git or network publication | No initialization, stage, commit, public remote or upload |

The full historical test suite cannot run from a lightweight source/evidence bundle without excluded raw/processed/experiment artifacts. Full EEG/MPS reproduction commands were not executed end to end. Version pins describe the observed environment, not a tested cross-platform installation lock. No packages were installed. Pseudonymization does not guarantee anonymity. Hashes establish consistency, not independent acquisition/training truth or legal ownership. Original statistical definitions and research artifacts were not changed. See [reviewer scope](../docs/REVIEWER_VERIFICATION.md).
