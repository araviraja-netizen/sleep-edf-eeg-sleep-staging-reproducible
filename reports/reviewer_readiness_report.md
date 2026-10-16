> Curated public-export copy. Historical originals remain unchanged. Links to excluded evidence point to the artifact-availability guide; this is not the complete private publication archive.

# Reviewer readiness report

The separate [revised manuscript](publication_manuscript_revised.md) is ready for scientific author review. It is **not yet a journal-specific submission package**: author declarations, public code availability, figure size proofing and target-journal formatting remain unresolved. The contribution is an auditable comparison of existing compact model families, with explicit aggregate/class/transition tradeoffs. This revision does not provide new experimental evidence.

## Scientific review assessment

| Reviewer issue | Revision response | Remaining limitation |
| --- | --- | --- |
| Is this a fair architecture comparison? | Abstract, Methods Table 2 and Discussion distinguish shared targets/preprocessing/splits from one-versus-five-epoch inputs and different capacities | Configuration contrasts do not isolate recurrence or attention; capacity/context controls are absent |
| Are seed counts comparable? | M1 seed 42 only; M5/M7 seeds 42, 43, 44 are explicit throughout | Baseline training variance cannot be estimated from one seed; subject bootstrap does not repair it |
| Does context help every target? | Transition results are prominent in abstract, Results Table 4/Figure 4, discussion and conclusion | Context models have lower descriptive transition macro-F1; no new causal or subgroup-significance claim |
| How strong is the inference? | Paired subject bootstrap, fixed-seed conditioning and multiplicity interval scope remain explicit | Small single-cohort holdout and fixed split limit coverage/generalization; class/subset comparisons remain descriptive |
| Was the test set used for selection? | Validation shortlist/checkpoint selection and frozen nine-checkpoint plan remain distinct from final outcomes | Test is consumed; no best-test-seed selection, new model ranking or development on this holdout is justified |
| What is novel? | Expanded synthesis acknowledges prior CNN/recurrent/attention/context/efficiency/imbalance work | No new architecture, exhaustive novelty search, SOTA or clinical-effectiveness claim |
| Is N1 handled adequately? | Per-class precision/recall/F1, secondary loss references and N1 limitations retained | N1 remains weak; weighting evidence has one seed per secondary configuration |
| Do transition errors imply smoothing? | Speculative mechanism removed; precise preceding-label subset definition added | No mechanism, event localization or physiological transition analysis was tested |
| Are M8 ablations confirmatory? | Moved to supplementary proposal with exact values, budgets and capacity counts | Short, single-seed, non-capacity-matched ablations cannot establish component efficacy; M8 has no test result |
| Is this deployable online? | Label-informed Wake eligibility, noncausal filtering and future context are explicit | No prospective causal/streaming or autonomous sleep-boundary evaluation |
| Are small models necessarily efficient? | Units, operation omissions, realized stopping times and inference/export timing limits retained | No controlled kernel latency, energy or deployment-memory study |
| Are embeddings or explanations causal? | Supplemental captions identify saved validation examples, independent projection coordinates and descriptive scope | No causal physiology or cross-model saliency-magnitude conclusion |

## Actual saved figure inspection

All sixteen PNGs were visually reviewed in an overview; architecture, confusion matrices, class-wise bars, paired intervals, preprocessing, embeddings, saliency/Grad-CAM and attention were also inspected individually. Dimensions, DPI, one-page PDF structure, embedded-font markers and original manifest checksums are checked programmatically in figure_review.json (additional evidence). [Availability and reproduction inputs](../docs/ARTIFACT_AVAILABILITY.md) and figure_review.csv (additional evidence). [Availability and reproduction inputs](../docs/ARTIFACT_AVAILABILITY.md). Existing PNG/PDF bytes are preserved. PDF checks inspect headers/structure, not a separate rendered proof of each PDF.

All PNGs report approximately 300 dpi. Their pixel widths range from 2,063 to 3,634 and heights from 901 to 3,634. All PDF counterparts contain a MediaBox, one page and embedded-font stream markers. These facts do not mean every element is vector: preprocessing includes a historical raster and embedding scatter layers can be rasterized. Native visual inspection found no obvious clipping, missing model panels or contradictory data labels. Quantitative source values are verified separately from visual appearance.

| Saved figure | Placement | Specific finding and required editorial treatment |
| --- | --- | --- |
| 01_architectures | Main 1 | Input pathways, counts and heads are readable when enlarged; dense 8-point source text should not be reduced to a narrow column. Caption clarifies shared design versus independent fitted weights |
| 04_primary_comparison | Main 2 | Distinct subject-interval and seed-point panels; M1 one point and context three. Retain panel descriptions and ratio scale |
| 07_paired_intervals | Main 3 | Zero line and interval endpoints visible; no in-panel interval-type legend. Caption must retain dark 95% versus pale 98.333% explanation |
| 16_transition_errors | Main 4 | Clear labels and bars; no uncertainty bars. Caption identifies descriptive fixed-seed means and precise subset definition |
| 06_classwise | Main 5 | All five classes and P/R/F1 panels present; small legend. Full-width PDF still needs font-size proof at final journal width |
| 05_confusion_matrices | Main 6 | Count and row-normalized panels present for seed 42; raw counts readable enlarged. Six-panel portrait layout needs a full page; rounded zero fractions need not mean zero errors |
| 02_stage_distribution | S1 | Count and percentage axes are distinct; descriptive all-subject eligibility, not train-fitted weights |
| 03_preprocessing | S2 | EEG µV, PSD µV²/Hz and analytic response dB units are present. Historical training raster has a lower native resolution and will not gain detail by PDF/upscaling; full-width placement required |
| 08_training_convergence | S3 | Validation curves and selected-epoch stars present for every primary seed; small legends. Not a plot of equal realized epochs or test selection |
| 09_parameter_cost | S4 | Model labels visible; time includes inference/export and is not controlled latency. Preserve measurement caveat and use vector PDF for labels |
| 10_N1_errors | S5 | True-N1-conditioned fractions, with N1 bar equal to recall. Do not label the diagonal as an error fraction |
| 15_subjectwise | S6 | All subject IDs visible with rotated labels; no outcome-based ordering. Equal-subject display differs from pooled primary estimator |
| 11_hypnograms | S7 | Original-time axis and stage ordering visible; SC4081E example retained deterministically. Three-panel dense time trace belongs full-width in supplement |
| 12_embeddings | S8 | PCA/t-SNE labels and common color legend present; scatter overplotting and small legend limit small-format reading. Separate model coordinates are not comparable axes |
| 13_explanations | S9 | Gradient and Grad-CAM scales are present; model-specific scales and context heatmap slots 0–4 need caption interpretation. Full-width placement; do not compare saliency magnitudes as physiology |
| 14_attention | S10 | Four heads and relative key/query offsets visible; **no quantitative colorbar**. Existing figure is suitable only for qualitative pattern illustration; a later separately versioned presentation export from saved weights would be needed for numerical color interpretation |

Font-size risk is material. The saved builder uses a 9-point base font, some 7-point legends and 8-point architecture/count text at native widths around 8–12 inches. Scaling a 12-inch-wide panel to an illustrative 180-mm journal width reduces 9-point text to about 5.3 points; 7-point legends become about 4.1 points. These are editorial scaling calculations, not experimental measurements or a claimed journal policy. High pixel DPI and a PDF container do not fix undersized printed text. After journal selection, proof the dense panels at actual size; prepare separately versioned layouts with larger fonts or split panels using only saved artifacts if needed. This revision supplies the review and allocation, not new figure exports or a claim of print acceptance.

## Numerical, link and preservation verification

The independent [revision verifier](manuscript_revision/verify_revision.py) checks the rendered manuscript against its evidence-linked template; original numeric display formatting; shared split/class/target counts; saved single-seed and model-level summaries; class-wise means; transition/stable support accounting; registered paired intervals; caption order; reference coverage; and local document/image links across all five deliverables. It also rehashes all files in the before-revision preservation snapshot, including the original manuscript and all processed arrays/experimental artifacts. The machine-readable result (additional evidence). [Availability and reproduction inputs](../docs/ARTIFACT_AVAILABILITY.md) and log (additional evidence). [Availability and reproduction inputs](../docs/ARTIFACT_AVAILABILITY.md) are the authority for actual pass/fail counts.

The completed verification passed 524 checks, including 103 local links and all sixteen PNG/PDF pairs. All 8,599 captured pre-existing files were checked; every research artifact, including 153 processed EEG arrays, remained byte-identical. A subsequent recheck found a change to macOS Finder metadata `results/.DS_Store`; that folder-presentation file is recorded separately from research preservation and was not restored or used as scientific evidence. Exact byte-identical counts and any Finder metadata changes are recorded in the machine-readable verification. The original manuscript SHA-256 is `34724b02054dd63869142ae63779f8b81364b47aaeb8febe2ea0b1c3fef3cdc7`. The rendered narrative has 120 source-bound numeric occurrences covering 100 unique original facts; generated tables and verbatim method/supplement details are checked separately against preserved inputs.

No old test output is overwritten. No model code was changed, and no training/resume test, forward pass, new inference, gradient, t-SNE/PCA fit or bootstrap run was invoked for this revision. Historical model-test evidence remains historical; new documentation checks are reported separately. External reference verification uses original saved metadata plus fresh primary-source checks; publisher access restrictions are disclosed in the [reference audit](revised_reference_audit.md), rather than treated as broken local links or successful live access.

Reproduce only the editorial checks with the existing interpreter:

```sh
cd ${WORKSPACE}
${PYTHON} reports/manuscript_revision/verify_revision.py
```

## Information the authors must supply

- Author names, order, affiliations, ORCIDs where required, corresponding-author address/email and contributor roles.
- Funding sources/grant identifiers, competing interests, acknowledgments and confirmation of all author contributions and approval.
- Applicable institutional ethics/data-reuse determination and consent wording for secondary public-data research; do not assume a waiver or invent an approval number.
- Public code/artifact archive URL, release/version or DOI, license, software/data accessibility and any restrictions on redistribution. The local source-hash record is not an existing public repository or Git commit.
- Target journal and article type, plus final author review of the scientific positioning and all references.
- Any required statement describing assistance in manuscript preparation, according to actual use and the selected journal's policy.

## Journal-dependent work still pending

Check the chosen journal's actual instructions before asserting compliance: manuscript/abstract length and headings; citation order/style; article type and contribution expectations; reporting/data/code checklists; anonymization; table/figure limits and supplementary-file conventions; PDF/TIFF/EPS acceptance, column width, minimum text size, resolution and color accessibility; mathematical notation and abbreviation style; graphical abstract/highlights if requested; ethics/consent/conflict/funding/AI-use declarations; and cover-letter/originality/submission attestations. These are pending checks, not invented universal requirements.

The original abstract, highlights and cover letter remain preserved and must be reconciled to the revised title/abstract at journal adaptation. No submission or new experiment is initiated. Additional matched-capacity/context, balanced-seed or independent-cohort evidence is outside this editorial revision and cannot be obtained by tuning against the consumed test set.
