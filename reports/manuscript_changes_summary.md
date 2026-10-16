> Curated public-export copy. Historical originals remain unchanged. Links to excluded evidence point to the artifact-availability guide; this is not the complete private publication archive.

# Manuscript changes summary

The revision is saved separately as [publication_manuscript_revised.md](publication_manuscript_revised.md). The original manuscript (additional evidence). [Availability and reproduction inputs](../docs/ARTIFACT_AVAILABILITY.md), earlier abstracts/highlights/cover letter, reports, saved tables, figures and experimental artifacts remain the historical record. This revision does not silently update those companion documents to the new title or abstract; authors must reconcile them when preparing a journal-specific package.

## Scientific and organizational changes

| Original location | Revised location | Change and scientific purpose |
| --- | --- | --- |
| Title; abstract; Sections 1–3 | New title; abstract; Introduction | Position the study as a configuration-level comparison with class/transition tradeoffs. State context, capacity and replication differences directly in the abstract; avoid an isolated architectural-effect interpretation |
| Architecture and limitations in Sections 6, 17–18 | Methods 2.3, Table 2; Discussion 4.1/4.4 | Put single-epoch M1 versus five-slot M5/M7, parameter counts and seed identities side by side; distinguish shared data controls from unmatched factors |
| Seed and uncertainty details across Sections 8–10 | Methods 2.4–2.5; Table 3 and captions | Separate one baseline seed from three contextual seeds; distinguish seed sample SD, conditional subject intervals and ensembles; retain all seeds without test-based selection |
| Transition results in Section 13 | Abstract; Results 3.2/Table 4/Figure 4; Discussion 4.2; Conclusion | Elevate the unfavorable contextual transition result to a principal finding. Define the preceding-label subset precisely. Remove the speculative smoothing interpretation; no causal explanation is established |
| Brief literature listing in Section 3 | Introduction 1.1 | Synthesize representation/context, minority-class objectives, subject independence versus cross-cohort transfer, and efficiency; distinguish cited designs from these compact implementations |
| Thirteen verified references | Fifteen traceable references | Retain R1–R13; add original PhysioNet platform citation R14 and verified IITNet R15. Separate dataset, original EEG study and platform citations |
| Twenty-one numbered sections | Five main sections with Methods/Results/Discussion subsections | Use conventional journal organization while keeping architecture, preprocessing, context boundaries, training selection and statistical definitions accessible |
| Exploratory M8 in Section 15 | Supplement S4 / proposed Tables S7–S8 | Retain full-budget validation and bounded-ablation values; preserve short-budget, one-seed, capacity/confounding and no-test limitations |
| Sixteen embedded main figures | Six proposed main figures, ten supplementary | Prioritize inputs, aggregate results/uncertainty, transitions and class errors; keep every saved asset with an explicit caption/filename crosswalk |
| Sections 12, 14–16 and less-central diagnostics | Supplements S1–S7 | Retain weighted-loss details, cost accounting, convergence, representations, explanations, quality/subgroup analyses and reproducibility evidence |
| Machine-style table headings and short captions | Tables 1–4 and complete captions | Use explicit model IDs, human-readable labels and units; define ratio versus percentage scores, support population, averaging, subset membership and illustrative seeds |
| General reproducibility/declarations | Data/code section, author declarations and readiness report | Distinguish local preservation from public code availability; identify missing author and journal information without inventing statements |

## Numerical and artifact integrity

No measured value, frozen model ranking, split, model selection, checkpoint, saved prediction or original statistic was changed. The revised text uses the original formatted numerical ledger; tables are rendered from unchanged saved CSVs/facts. Full-precision values remain in the original tables. Architecture text and constants are carried from the verified original Methods with new subheadings. Supplemental secondary/cost/M8/representation narrative is retained from the original rather than silently discarded.

The bounded M8 table additionally exposes original parameter counts, N1 scores and six-decimal ablation differences from existing evidence. These are presentation of previously measured results, not a new ablation or significance test. No bootstrap, t-SNE, PCA, calibration fit, model forward pass or explanation extraction was executed.

The revision-specific verification (additional evidence). [Availability and reproduction inputs](../docs/ARTIFACT_AVAILABILITY.md) checks numeric source bindings, main tables, saved aggregate/individual-seed/class/subset relationships, reference identifiers, figure/caption order, local links and before/after preservation hashes. The verification log (additional evidence). [Availability and reproduction inputs](../docs/ARTIFACT_AVAILABILITY.md) records the actual run. Historical model tests are preserved; this documentation-only change does not claim a fresh model-training/resume test run.

## Presentation decisions and limits

All original figure files remain unchanged. Their review found no obvious clipped content at inspection resolution, but native readability does not guarantee readability at final column width. Dense main figures and supplementary multi-panel figures need journal-size proofing. The attention illustration lacks a quantitative colorbar, and the preprocessing illustration embeds a historical raster; these are explicitly recorded rather than described as flawless print-ready graphics. Figure S9's context-slot indexing is clarified in its new caption.

The revision helper and templates live in manuscript_revision (additional evidence). [Availability and reproduction inputs](../docs/ARTIFACT_AVAILABILITY.md). They add an editorial workflow only and do not modify the frozen research source. Scientific limitations cannot be repaired through wording: balanced training-seed replication, matched context/capacity and independent-cohort evidence remain absent. No new experiment or journal submission is started.
