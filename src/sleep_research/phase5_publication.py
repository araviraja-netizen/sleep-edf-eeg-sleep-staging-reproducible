"""Preparation documents and manuscript rendering; no training or test inference."""
from pathlib import Path
import json
import pandas as pd
from .common import fingerprint,digest,json_write,now


def markdown_table(frame):
    cols=list(frame.columns)
    return '| '+' | '.join(cols)+' |\n| '+' | '.join(['---']*len(cols))+' |\n'+'\n'.join('| '+' | '.join(str(v) for v in r)+' |' for r in frame.itertuples(index=False,name=None))


def preparation_reports(project,body):
    project=Path(project);out=project/'results';sha=fingerprint(body)
    modeltable=markdown_table(pd.DataFrame([dict(key=r['key'],seed=r['seed'],role=r['role'],parameters=r['parameters'],best_epoch=r['best_epoch_1_based'],checkpoint=r['checkpoint'],sha256=r['checkpoint_sha256']) for r in body['models']]))
    (out/'phase5_preregistered_evaluation_plan.md').write_text(f'''# Preregistered final evaluation — review before freezing

Plan `{body['plan_id']}`; SHA-256 `{sha}`. Immutable machine plan: `results/phase5_evaluation_plan.json`; readable exact configuration: `configs/phase5_final_evaluation.yaml`.

This is preparation. No frozen-test prediction has been computed. Freezing requires a separate reviewed digest; running requires a separate explicit authorization flag and the same digest. Integrity hashes bind artifacts; they are not cryptographic authentication of user identity.

## Fixed model comparisons

{modeltable}

Primary model-level comparisons: M5 minus M1, M7 minus M1, M7 minus M5 on five-class macro-F1. M5 and M7 summarize all three predefined seeds42/43/44 by mean metrics, never by test-selected seed or probability ensemble. M1 has only seed42; unequal seed evidence is explicitly disclosed. Secondary descriptive comparisons: M4 minus M1, weighted M5 seed42 minus ordinary M5 seed42. No M8 or other model is added based on test results.

## Frozen data and inference

SC only; Protocol A, Fpz-Cz100Hz, 3000samples/30seconds. Preserve continuous 0.3–35Hz forward/backward filter, label-based30-minute Wake margins, all internal Wake, fixed training population normalization and original split. Classes in order: Wake0,N1=1,N2=2,N3=3(original3+4),REM4; movement/unknown excluded. No refitting, preprocessing regeneration, calibration fitting, threshold tuning or ensemble selection.

Test subjects: {', '.join(body['test_subjects'])}. Exactly31,156 central targets,12 subjects,22 recordings. Subject assignments are disjoint from54 train and12 validation subjects, with both nights grouped. Epoch-ID sequence fingerprint: `{body['test_epoch_ids_sha256']}`. Frozen data contract:

```json
{json.dumps(body['dataset_contract'],indent=2)}
```

One argmax prediction per canonical central epoch. M1/M4 use one epoch; M5/M7 use centered five-epoch EEG context. Neighbors remain within subject/recording/night/split, stop at invalid-index gaps, and use zero padding plus explicit masks. Neighboring labels are never inputs. All central targets and quality flags remain included. This is offline future-context inference, not real-time deployment. Normalization is unchanged and trained on train subjects only.

Inference: float32, MPS required, batch16, no workers, at most2 recordings cached. Shared exporter processes resumable2048-target chunks in copied evaluation experiments, preserving original checkpoints/artifacts and checkpoint optimizer/RNG metadata. Only committed, hash-verified chunks count; unreferenced committed chunks are safely adopted. Final merged exports must match all canonical IDs/labels before analysis. Interrupted chunks repeat; no duplicate final rows or silent overwrite of verified output. No gradient optimization, training or checkpoint reselection.

## Metrics and uncertainty

Primary metric five-class macro-F1: mean of five per-class `2TP/(2TP+FP+FN)`. Undefined class F1/precision/recall is zero, and all five classes remain in macro denominators even for a subject missing classes. Report true/predicted support. Accuracy/micro-F1 denominator is all valid targets; weighted F1 weights true support; balanced accuracy averages recall over true-present classes. Kappa is `(accuracy-chance)/(1-chance)` using empirical true/predicted marginals; null if chance=1. Specificity is `TN/(TN+FP)`, null without negatives. One-vs-rest ROC AUC uses trapezoids and AP the step recall integral; null if a class has no positives or negatives. No interpolation to invent missing-class scores.

Calibration: top-label ECE with10 equal-width confidence bins, left-inclusive/right-exclusive except final bin including1; weight by bin count/N. Brier score averages the sum of five squared errors, range0–2. Save raw/row-normalized confusion matrices, ROC/PR, reliability, subject/recording metrics, flagged/flag-free descriptions, true/predicted hypnograms and N1/N3 precision/recall/F1.

Paired cluster bootstrap: 10,000 draws with RNG seed20261016; draw12 subjects uniformly with replacement, retaining all of each selected subject's epochs and nights with multiplicity. Reuse identical draws across models/seeds. Compute pooled confusion-based metrics and differences. For multi-seed groups average fixed-seed metrics within each draw; do not combine probabilities. Report95% percentile intervals, plus98.333% marginal intervals for macro-F1 in the three primary comparisons (Bonferroni approximate simultaneous family coverage; other metrics descriptive). Secondary analyses are descriptive; no significance declaration from higher point estimates. Undefined kappa replicates are excluded with an explicit valid-replicate count. Rare classes and only12 subjects limit coverage and external generalization.

Training-seed sample SD(ddof1), range and individual metrics are separate from subject-bootstrap intervals; M1/M4/weighted-M5 single-seed SD is null. These intervals condition on fitted seeds and do not quantify uncertainty from retraining, split selection, scorer or dataset shift.

Transitions compare a target with its immediately preceding consecutive original epoch in the same recording/night/subject; gaps/boundaries excluded, never labels as inputs. Quality subsets are descriptive after final evaluation, never a basis for new exclusions or tuning.

## Representations and research constraints

PCA uses saved64-dimensional embeddings. t-SNE uses at most500 uniformly sampled targets, seed42, perplexity30 (bounded by sample size),500 iterations. IDs, labels and checkpoint linkage are permanent; axes are independent between models. Centroid between/within scatter is descriptive. No UMAP dependency installation. Predefine first five original-order examples for each mistaken pair among Wake,N1,N2; save identity and probabilities without outcome-driven model changes. Reuse checkpoint-linked validation saliency/Grad-CAM/compatible attention figures, clearly labeled validation; do not interpret them as causal physiology.

M8 remains exploratory, with optional full-budget validation definitions deferred. No additional training is needed for the registered M1/M5/M7 comparison. If optional M8 training is explicitly started before freezing, all its registered runs must finish first; M8 remains outside this test plan. Nothing after test access may alter checkpoints, preprocessing, model set, primary comparisons, seeds, ensemble choices or thresholds. A new development study would need a fresh holdout.

## Source and permanent evidence

Source files and package/platform versions are locked in the JSON/YAML plan. Existing experiment source revisions and configs remain preserved. Test output directory: `experiments/final_evaluation/{body['plan_id']}`. Final result-bearing reports/CSVs are generated only when all nine registered exports genuinely complete. See `phase5_execution_instructions.md` for review/freeze/authorize/resume commands. No final freeze or authorization has been executed by Codex.
''')
    (out/'phase5_model_selection.md').write_text('''# Validation-only final model selection

M1 is the original40,501-parameter CNN with AdamW and ordinary CE, saved seed42. M5 is65,717-parameter CNN-BiLSTM; M7 is107,893-parameter compact CNN-Transformer. M5/M7 ordinary CE were shortlisted by the sealed Phase4 validation ranking and completed seeds42/43/44. All checkpoint selection uses best complete-validation five-class macro-F1. No additional hyperparameter or seed choice uses the frozen test.

M5 three-seed validation macro-F1 is0.742346±0.001745; M7 is0.745397±0.004025 (sample SD). M1 seed42 is0.693772. M5/M7 mean N1F1 is0.470314/0.467989; M1 is0.365433. M1 has fewer seeds; comparisons confound architecture and input context and are not significance claims. Seed42's initial winner was M5, whereas M7 has the higher three-seed mean; retain both as predefined comparisons rather than cherry-picking.

Secondary class-imbalance references are M4(weighted AdamW CNN seed42) and M5 weighted CE seed42. M4 validation macro-F1/N1F1 is0.680557/0.417784; weighted M5 is0.740190/0.512166. Weighting improved N1 on these validation runs but did not maximize macro-F1. No new weighted seeds or additional models are selected after test outcomes.

M8 is exploratory. It is excluded from the final test roster; its full-budget CE seed42 macro-F1 is0.740848, N1F1=0.433903. Short ablations do not establish architectural novelty. Full-budget optional confirmation remains a deferred validation proposal. All reported values here are measured validation metrics, never final test results.
''')
    command=f'''cd /Volumes/Mac/projects/sleep-stage-classification
export PYTHONPATH="$PWD/src"
/Volumes/Mac/envs/dl-m4/bin/python -m sleep_research.phase5 validate
/Volumes/Mac/envs/dl-m4/bin/python -m sleep_research.phase5 review'''
    (out/'phase5_execution_instructions.md').write_text(f'''# Phase 5 — regular Terminal execution

Prepared only. Codex has not frozen the plan, authorized or computed any real test prediction. Use the existing interpreter, no install or preprocessing. Stop to review the immutable plan and manuscript draft first.

## Validate and review

```sh
{command}
```

The validator checks all15 completed Phase4 runs, prior files including153 arrays, original source bytes, selected checkpoint sidecars/states and validation exports. New Phase5 files are additive: the Phase4 historical source identity is verified over its original98-file set, preserving its registry and source unchanged. The old Phase4 CLI's whole-tree source guard intentionally detects additions; use the Phase5 validator for this completed campaign rather than editing or re-registering Phase4.

Run all existing/new automated tests with separate preserved evidence:

```sh
/Volumes/Mac/envs/dl-m4/bin/python scripts/run_phase5_tests.py
/Volumes/Mac/envs/dl-m4/bin/python scripts/validate_phase5_notebook.py
```

## Freeze after reviewing the exact report

Read `results/phase5_preregistered_evaluation_plan.md`, model selection and novelty review. The following digest pins this exact plan. Freezing records review but does not run inference:

```sh
/Volumes/Mac/envs/dl-m4/bin/python -m sleep_research.phase5 freeze --reviewed-plan-sha256 {sha}
```

## Explicitly authorize and run final evaluation

Only execute this after reviewing/freezing. It authorizes all nine fixed checkpoint test exports and analysis; no training occurs. Same command resumes an interrupted evaluation, reusing verified chunks without duplicate final rows:

```sh
/usr/bin/caffeinate -i /Volumes/Mac/envs/dl-m4/bin/python -u -m sleep_research.phase5 run --authorize-final-evaluation --plan-sha256 {sha}
```

Do not use the earlier Phase4 campaign command for Phase5. Persistent log and status survive Codex closure; keep the Terminal open. Caffeinate inhibits idle sleep, not power loss or reboots. Incomplete chunks repeat; completed validated exports are reused. File corruption is a hard stop, never silent retraining.

## Monitor and regenerate without inference

```sh
/Volumes/Mac/envs/dl-m4/bin/python -m sleep_research.phase5 status
tail -n 60 -f experiments/final_evaluation/{body['plan_id']}/evaluation.log
# After all nine inference exports complete; reads saved test evidence only:
/Volumes/Mac/envs/dl-m4/bin/python -m sleep_research.phase5 analyze --authorize-final-evaluation --plan-sha256 {sha}
/Volumes/Mac/envs/dl-m4/bin/python scripts/validate_phase5_notebook.py --show-authorized-final-results --plan-sha256 {sha}
```

Preparation notebook execution requires no authorization and displays validation evidence plus the pending final plan. After final evaluation, the explicit notebook option permits reading committed test results only; it never calls inference or training.

## Optional M8 validation proposal — deferred, not required

No additional validation training is approved or launched by preparation. Inspect the nine immutable prospective configs with:

```sh
/Volumes/Mac/envs/dl-m4/bin/python -m sleep_research.phase5 optional-validation-definitions
```

If you explicitly decide to undertake these validation-only ablations BEFORE freezing the final plan, the following distinct authorization starts/resumes the nine-run25-epoch/patience8 campaign through the existing trainer/checkpoint manager. It is blocked once the final plan is frozen. M8 stays exploratory and outside this final test roster. If the campaign is started, finish it before freezing.

```sh
mkdir -p results/phase5_optional_validation
# Same command resumes each interrupted epoch and skips complete exported runs:
/usr/bin/caffeinate -i /Volumes/Mac/envs/dl-m4/bin/python -u -m sleep_research.phase5 optional-validation-run --authorize-additional-validation-training >> results/phase5_optional_validation/terminal.log 2>&1
# Monitor:
tail -n 60 -f results/phase5_optional_validation/terminal.log
```

Prospective cost from measured3-epoch M8 control/no-multiscale/no-context runs: about3.60/2.79/2.66 minutes per epoch, respectively; nine25-epoch runs about11.3 training hours before early stopping, plus validation export/analysis and thermal/I/O variability. Allow12–24 hours and at least4GiB SSD reserve. Actual extra runs and novelty confirmation are unmeasured. Primary test inference plus analysis is estimated0.5–2 hours on this Mac (not benchmarked on test); nine models contain280,404 central predictions and approximately69MiB penultimate float32 embeddings, with chunk+merged duplication, CSV probabilities/logits and figures. Reserve2GiB SSD and approximately2–5GiB process memory. These are planning estimates, not measured test throughput or peaks.
''')
    (out/'phase5_reproducibility_report.md').write_text(f'''# Phase 5 reproducibility — prepared, not evaluated

Created {now()}. Plan `{body['plan_id']}`; SHA-256 `{sha}`. Frozen test remains unaccessed by preparation. Existing interpreter/dataset verified; no environments/packages/downloads/preprocessing changes. Full original preservation snapshot: `results/phase5_preservation_snapshot.json`. Phase4 verification uses its exact historical source file set and all immutable experiment markers, retaining153 processed arrays and original experiment bytes. No Git repository was present; source-file SHA-256 maps substitute for a commit and are embedded in the immutable plan alongside software versions and original experiment revisions.

Reused infrastructure: Experiment, CheckpointManager, frozen_dataset_contract, original CNN/advanced constructors, memory-map loader, Phase4 context boundary implementation, export_inference/load_export/analyze_export, ROC/calibration/representations. Test adapters require final authorization; old train/validation loaders remain unchanged. Separate evaluation copies hold checkpoint optimizer/RNG/source metadata and pins, so original runs are never modified by final export.2048-target chunk commit references and final merged manifests bind exact IDs, quality flags, model and source fingerprints. A single-process advisory lock prevents simultaneous writers. Incomplete chunk outputs repeat/adopt safely; corrupt completed chunks stop. No MPS bitwise reproducibility guarantee.

Separate evidence will be written to phase5_test_results.json/txt, phase5_project_validation.json, phase5_notebook_execution.json and phase5_preparation_validation.json. Preparation must pass all before the freeze command is usable. Final result-bearing outputs remain absent until nine verified test exports complete; final analysis is regenerable from those exports without training/new inference. Frozen source/report edits are rejected.
''')
    research_report(project,body,False)


def research_report(project,body,final=False):
    project=Path(project);target=project/'reports/final_sleep_stage_classification_research_report.md';target.parent.mkdir(exist_ok=True)
    measured=pd.read_csv(project/'results/phase4_models_comparison.csv')
    selected=measured[measured.key.isin([r['key'] for r in body['models']])]
    table=markdown_table(selected[['key','parameters','accuracy','macro_f1','n1_f1','n3_f1']].round(6))
    finaltext=(project/'results/phase5_final_test_results.md').read_text() if final else 'Pending explicit local Terminal authorization. No real test predictions, scores or uncertainty estimates have been computed. This section intentionally contains no placeholders presented as measurements.'
    statistical=(project/'results/phase5_statistical_analysis.md').read_text() if final else 'Predefined analysis:10,000 paired subject-cluster bootstrap replicates, seed20261016,95% intervals, approximate multiplicity-adjusted primary comparison intervals. Seed variability remains distinct. See the preregistered evaluation plan; no test confidence intervals exist yet.'
    target.write_text(f'''# Automatic Sleep Stage Classification on Sleep-EDF Expanded

Status: {'final frozen-test exports analyzed' if final else 'publication preparation; validation findings measured, final test pending'}. Plan `{body['plan_id']}`; digest `{fingerprint(body)}`. Generated {now()}.

## 1. Abstract

We implemented a reproducible single-channel EEG research pipeline with frozen subject-independent splits, persistent resumable checkpoints and explicit final-evaluation control.153 Sleep Cassette recordings from78 subjects yielded195,469 Protocol A epochs. Primary training/validation used54/12 subjects and134,047/30,266 targets. The baseline CNN validation macro-F1 was0.693772(seed42). CNN-BiLSTM and compact CNN-Transformer ordinary-CE variants achieved validation three-seed means0.742346±0.001745 and0.745397±0.004025(sample SD). These are validation-selected findings. The proposed M8 combination remains exploratory. {'Final test metrics are reported below without post-test tuning.' if final else 'Final performance is intentionally unreported until authorized inference on the12 frozen test subjects.'}

## 2. Introduction

Automatic sleep staging assigns30-second PSG intervals to Wake,N1,N2,N3 or REM. The project examines compact EEG models under an audited fixed protocol. Related methods already combine convolution, temporal recurrence and attention; our implementations are controlled compact variants rather than reproductions or claims of novel standard building blocks. [DeepSleepNet](https://arxiv.org/abs/1703.04046), [SeqSleepNet](https://arxiv.org/abs/1809.10932) and [SleepTransformer](https://arxiv.org/abs/2105.11043) provide architectural context.

## 3. Research motivation and objectives

Evaluate whether controlled architecture and class-weighted-loss changes improve five-class macro-F1 and N1 performance without subject leakage. Separate validation selection from final generalization; document computational cost, uncertainty and offline protocol limitations. Avoid optimizing a lucky training seed using test outcomes.

## 4. Dataset and data-quality audit

Local Phase1 found153 paired SC recordings from78 subjects and44 ST recordings from22 subjects, zero discovery/pairing errors, all398 supplied file checksums matching. Native Fpz-Cz and Pz-Oz were available throughout; only Fpz-Cz is used. ST remains inventoried separately. Sleep labels map W0,1→1,2→2,3/4→3,R→4; movement/unknown have no supervised label. Complete half-open30-second sample-grid epochs require verified common origins and unique annotation coverage. Original recording indices retain gaps; no timing shift or invented samples. See Phase1 inventory/quality report for annotation exclusions versus missing coverage. Dataset source: [PhysioNet Sleep-EDF Expanded](https://physionet.org/content/sleep-edfx/1.0.0/).

All153 SC recordings processed without failure. Protocol B retains414,961 epochs; Protocol A retains195,469. Quality flags are retained, including2,226 rail-contact and9 recording-edge flags in A;30266 validation targets contain206 flagged epochs. Flags are descriptive and do not prove clinical artifact severity or annotation correctness. No additional artifact rejection or interpolation is applied.

## 5. EEG preprocessing and Wake trimming

Continuous native100Hz Fpz-Cz EEG is calibrated to microvolts, filtered using `butter(4,[0.3,35],fs=100,btype='bandpass',output='sos')` then float64 `sosfiltfilt` with odd padding27 before epoch extraction. Forward/backward response is squared; acquisition filters cannot be undone. Store unnormalized float32 epochs once and normalize on access using the fixed training-only global population mean/std. Protocol A uses up to60 original epochs before first and after last valid sleep, retaining internal Wake. Label-based trimming, zero-phase filtering and future context make this offline evaluation, not a deployable sleep-boundary detector. No preprocessing was regenerated in Phase5.

## 6. Deep-learning model architectures

M1/M4 share40501 parameters: original1D CNN with64-value penultimate features. M5 CNN-BiLSTM65717 and M7 compact CNN-Transformer107893 use centered five-epoch EEG windows and predict only the central target. Original IDs, split/subject/recording/night boundaries and annotation gaps constrain context; explicit masks/zero padding represent missing neighbors. Neighbor labels are never inputs. M6 attention and M8 multiscale gated-context models remain validation comparisons. Architecture sources/config fingerprints and math are preserved in Phase4 documentation; these variants are not literal published-model reproductions.

## 7. Optimizers and loss functions

The original AdamW/ordinary-CE baseline is preserved. Phase3B implemented AdamW,Adam,momentum SGD and optional RMSprop, weighted CE/focal variants with bounded validation pilots. Full comparison controls use AdamW lr0.001,decay0.01,batch16,max25,patience8,no scheduler. Class weights use training counts[43782,15249,47664,9302,18050] and N/(5*n_c); validation/test never set weights. Adam's coupled L2 and AdamW's decoupled decay differ despite the same numerical0.01. The Phase3B Adam pilot favored ordinary CE at0.552537 macro-F1 and weighted CE at0.584834 on its bounded subset; this did not predict full-budget superiority. Full M2 Adam/CE confirmation yielded0.626050 validation macro-F1 versus M1 AdamW/CE0.693772, both seed42. M4 weighted AdamW improved N1F1 while reducing macro-F1. These are separate validation studies, with pilot/full budgets explicitly distinguished. Phase5 does not train or adapt any loss.

## 8. Experimental configuration

Frozen seed42 subject allocation54/12/12 with107/24/22 recordings and134047/30266/31156 Protocol A epochs. Preserve train-only normalization and split hashes. MPS float32 on AppleM4 MacBook Air16GB; external SSD memory maps. Existing trainer/checkpoint manager commits every completed epoch, including optimizer/RNG/sampler/history, with prior recovery and checksums. Training seeds42/43/44 where completed. All15 Phase4 runs and their permanent artifacts verified. The final roster is9 fixed checkpoints; M1/M4/weighted M5 each have one seed. No extra training is required for primary evaluation.

## 9. Validation-based model selection

{table}

The sealed seed42 validation ranking selected ordinary-CE M5/M7 for seeds43/44. Mean validation macro-F1 is0.742346/0.745397, and mean N1F1 is0.470314/0.467989. Weighted M5 seed42 raises N1F1 to0.512166 while macro-F1=0.740190. Improvements and class tradeoffs are validation observations; architecture comparisons also differ in EEG context. No final seed is chosen using test results, and no ensemble is proposed.

## 10. Frozen final-test results

{finaltext}

## 11. Ablation studies

M8 CE seed42 full-budget validation macro-F1=0.740848,N1F1=0.433903,N3F1=0.723525. Matched fresh three-epoch control/no-multiscale/no-context scores are0.717430/0.704538/0.651634; deltas−0.012892/−0.065796 relative to bounded control. These short runs suggest hypotheses but confound removed components with capacity/computation and convergence. Deferred full-budget25-epoch,patience8 ablations across seeds42/43/44 are defined separately and cannot run after final freeze. No novelty/superiority claim is supported by current evidence alone.

## 12. Statistical and uncertainty analysis

{statistical}

## 13. Results and discussion

The completed validation campaign favors the shortlisted context-aware CE variants on macro-F1 relative to M1, while weighted CE can increase N1 at an overall tradeoff. Attention or multiscale design does not uniformly dominate the controlled alternatives. Convergence plots, quality subsets, transitions and per-subject analyses should accompany pooled metrics. Validation-selected runs do not provide an unbiased final estimate. Seed sample SD and subject uncertainty reflect different variation sources. No clinical effectiveness or significance is inferred from these point estimates.

## 14. Comparison with related literature

Use [the source-linked novelty review](../results/phase5_research_novelty_review.md). Published architecture choices contextualize recurrence, multiscale extraction and attention. External paper scores are not ranked against these results: different Sleep-EDF subsets, channels, scoring conventions, Wake treatment, cross-validation and subject allocation prevent a controlled numeric comparison. A compact masked gated combination may offer a useful reproducible design study, but a broader systematic search and adequate matched ablations are needed before a novelty claim.

## 15. Practical limitations

Only one cohort and one frozen split;12 final-test subjects; single-channel manual R&K-derived labels; unequal seed counts; validation-selected shortlist; class imbalance/N1 difficulty; quality flags without expert artifact relabeling; unknown scorer uncertainty; capacity/context/training-time confounds; no external ST/generalization evaluation. Label-based Wake trimming, noncausal filtering, bidirectional/future EEG context limit online inference. Attention, saliency and Grad-CAM are descriptive sensitivity maps, not causal physiology. Parameter efficiency does not prove battery or deployment efficiency; MPS memory/runtime vary with thermal load and software. No SOTA or novelty claim is established.

## 16. Conclusion and future work

A complete reproducible pipeline and measured validation campaign support a fixed primary comparison of CNN, CNN-BiLSTM and compact CNN-Transformer. {'Authorized final results are reported without tuning; future development requires a separate untouched evaluation cohort.' if final else 'Final test conclusions remain pending the explicit reviewed Terminal command.'} Future work may examine adequately controlled M8 ablations, external validation, causal preprocessing/context and scoring uncertainty under a separately approved protocol.

## Reproducibility and references

Exact checkpoints, source-file digests, package versions, model/data fingerprints, uncertainty parameters and CLI commands are in the Phase5 plan and reproducibility report. Figures can be regenerated from saved outputs without retraining. Earlier artifacts remain immutable. See the linked literature review for verified primary references and the preparation/final evidence files for actual execution status.
''')
