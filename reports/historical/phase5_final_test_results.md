> Historical measured report, curated export copy. Statements about unstarted later phases are time-specific history. Machine paths and links are adapted; excluded detailed assets are not bundled. Use the portable commands in docs/EXPERIMENTS.md rather than these historical local-archive commands.

# Frozen final-test results

Plan `phase5-39f5f570696c`; SHA-256 `0c959604d09802c645f320a258f036000e3d1ec1dfac7092dba0ed87a233410f`. All nine preselected checkpoint exports verified on the identical 31,156 epochs, 12 subjects, 22 recordings. No tuning/training or lucky-seed selection.

       key  accuracy  macro_f1    n1_f1    n3_f1
        M1  0.801836  0.719774 0.382271 0.742009
 m5_ce_s42  0.845712  0.777613 0.472644 0.795033
 m5_ce_s43  0.847798  0.783506 0.485037 0.785148
 m5_ce_s44  0.847766  0.770253 0.425558 0.787495
 m7_ce_s42  0.843882  0.771246 0.443104 0.789847
 m7_ce_s43  0.850366  0.789187 0.528961 0.778212
 m7_ce_s44  0.840127  0.766957 0.453608 0.782824
        M4  0.796476  0.737975 0.467555 0.746510
m5_wce_s42  0.840159  0.785514 0.520438 0.789121

M1/M4/weighted M5 have one seed; primary M5/M7 have three. Primary model-level estimates average metrics across all predefined seeds without ensembling. See class-wise, subject/recording, quality, transition and uncertainty artifacts. No new model selection is permitted from these outcomes.
