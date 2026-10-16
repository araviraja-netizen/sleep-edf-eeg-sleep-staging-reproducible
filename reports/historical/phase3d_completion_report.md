> Historical measured report, curated export copy. Statements about unstarted later phases are time-specific history. Machine paths and links are adapted; excluded detailed assets are not bundled. Use the portable commands in docs/EXPERIMENTS.md rather than these historical local-archive commands.

# Phase3D completion and Phase3C/3D notebooks
Verified 2026-10-15T10:42:40.417513+05:30.

**M4 manual Terminal training and automatic validation postprocessing completed successfully.** The persistent log ends with `M4 validation export, explanations and comparison complete.` No training process remained when checked. No new training or inference was launched in this follow-up; existing artifacts were sufficient.

M4 `phase3d-m4-s42-7ffcd914` early-stopped after 16 completed epochs. Best epoch:8 (1-based). Training+validation duration:22.62minutes. Mean epoch:84.75seconds. Peak process RSS:2.513GiB (macOS high-water, not isolated GPU peak). Retained M4 folder:48.35MiB. Same40501-parameter CNN, AdamW lr0.001/decay0.01, seed42,batch16,25-epoch maximum,patience8,no scheduler,float32,MPS.

## Actual complete-validation comparison

All rows use the same30266 validation epochs/12subjects/24recordings,134047 training epochs/54subjects, ProtocolA, Fpz-Cz and frozen training-only normalization. All206 quality-flagged validation epochs are retained. The31156 test epochs remain locked. M3 is unlaunched.

| Model | Optimizer/loss | Accuracy | Macro-F1 | N1 F1 | N3 F1 | Balanced accuracy | Kappa |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| M1 | AdamW / ordinary_cross_entropy | 0.782462 | 0.693772 | 0.365433 | 0.687757 | 0.686035 | 0.692278 |
| M2 | Adam / ordinary_cross_entropy | 0.748662 | 0.626050 | 0.260449 | 0.581347 | 0.623867 | 0.643895 |
| M4 | AdamW / weighted_cross_entropy | 0.747142 | 0.680557 | 0.417784 | 0.652971 | 0.726422 | 0.656761 |

M4−M1: N1 F1+0.052351, macro-F1-0.013216, accuracy-0.035320, N3 F1-0.034786, balanced accuracy+0.040387.
N1 recall rose from 0.391841 to 0.546806, while precision changed from 0.342360 to 0.338025. N3 recall also rose, with a larger precision reduction and lower F1. N2 F1 fell from 0.831205 to 0.772158. Thus weighting improved minority recall but did not improve the primary five-class macro-F1 metric. M1 retains the highest observed macro-F1 among these three completed configurations.
These are single-seed validation findings after model selection, not statistical significance, multi-seed stability or final-test generalization. Adam vs AdamW comparisons include coupled versus decoupled regularization semantics.

## Missing notebooks resolved

- `notebooks/03c_adam_confirmation_analysis.ipynb`: M1 vs M2 full-budget Adam confirmation. Executed7code cells,8PNG outputs,zero errors.
- `notebooks/03d_weighted_adamw_analysis.ipynb`: M1 vs M4 weighted-loss confirmation, with M1/M2/M4 context. Executed8code cells,9PNG outputs,zero errors.
- `notebooks/03b_optimizer_analysis.ipynb` remains unchanged as the historical8192/4096-epoch,three-epoch pilot analysis. It is not being repurposed for full-budget results.
Both new notebooks use the exact existing interpreter through temporary SSD-local kernel specifications. They read saved artifacts only, verify IDs/checksums/metrics and display class confusions, convergence, subject scores, PCA/t-SNE and hypnograms;03d also displays checkpoint-bound saliency/Grad-CAM. No environment/kernel installation.

## Saving and integrity audit

Already saved by the manual run: immutable best/latest/previous checkpoints and hash sidecars; optimizer/null scheduler/criterion/RNG/sampler/early-stop states; authoritative histories; exact hyperparameters, class weights, source/software and frozen data fingerprints; all30266 predictions/labels/five logits/probabilities/epoch-subject-recording IDs/flags; penultimate and pooled intermediate[30266,64] features; whole-validation PCA and reproducibly sampled500-row t-SNE; all five class and subject metrics;24hypnograms; class/confusion/convergence comparison figures; five representative saliency/Grad-CAM examples bound to the selected checkpoint.
The follow-up audit recomputed classification metrics from saved probabilities, checked canonical ordered IDs/labels and flags, verified selected best epoch and full-dataset step counts, loaded checksummed best/latest/previous states, checked finite representations and explanation arrays, validated PCA/t-SNE row mappings and every hashed analysis file. No missing required saved artifact was found.
New evidence: `phase3d_completion_validation.json`, `phase3d_final_validation.json`, `phase3c_completion_notebook_execution.json`, `phase3d_completion_notebook_execution.json`, `phase3d_completed_models_validation_comparison.csv`, and `phase3d_completion_artifact_manifest.json`. Original preparation/test reports remain historical and are preserved;162tests passed at preparation. This follow-up ran artifact validation and both notebooks, not a new training test campaign.
Preservation is checked against `phase3d_completion_preservation_snapshot.json`, including all pre-existing reports/notebooks/experiments and153processed arrays. See `phase3d_completion_preservation_validation.json`. M1,M2,pilots and the manually completed M4 run are unchanged.

## Read-only follow-up commands

```sh
cd ${WORKSPACE}
export PYTHONPATH="$PWD/src"
${PYTHON} -m sleep_research.phase3d_analysis status
${PYTHON} scripts/validate_phase3d_completion.py
${PYTHON} scripts/validate_phase3cd_notebooks.py 3c
${PYTHON} scripts/validate_phase3cd_notebooks.py 3d
```

Notebook execution updates only the selected new notebook and its completion execution evidence. The audit refreshes its new completion JSON/CSV. It does not retrain. Existing published Phase3C/3D result reports and run artifacts are preserved. No resume or additional training is needed to view these results.
**Stopped after completion verification and notebook execution. No M3, additional seed, advanced architecture or frozen final test evaluation.**
