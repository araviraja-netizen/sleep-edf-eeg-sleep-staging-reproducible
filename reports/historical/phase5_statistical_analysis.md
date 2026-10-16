> Historical measured report, curated export copy. Statements about unstarted later phases are time-specific history. Machine paths and links are adapted; excluded detailed assets are not bundled. Use the portable commands in docs/EXPERIMENTS.md rather than these historical local-archive commands.

# Subject-aware uncertainty

Paired cluster bootstrap: sample 12 subjects with replacement; keep all epochs and nights; same draws for all models and seeds; pool weighted subject confusion counts; average metrics across fixed seeds (no ensemble)

10,000 paired resamples, seed20261016; 95% percentile intervals. The three predefined primary comparisons additionally show Bonferroni98.333% marginal percentile intervals (approximate simultaneous family coverage); secondary comparisons remain descriptive. No null-hypothesis p-values or superiority declarations. Only12 independent subjects; percentile coverage and rare-class intervals can be unstable. Subject resampling conditions on the trained seeds; seed sample SD is reported separately and is null for M1. Five-class F1 uses zero for undefined classes; balanced accuracy averages classes present; undefined kappa is null and valid-replicate denominators are reported. See phase5_subject_bootstrap.json and phase5_paired_comparisons.csv.

 primary_macro_f1_comparison              a              b            metric  estimate   lower95   upper95  lower_simultaneous  upper_simultaneous
                       False M5_fixed_seeds M1_fixed_seeds          accuracy  0.045256  0.035935  0.055052                 NaN                 NaN
                        True M5_fixed_seeds M1_fixed_seeds          macro_f1  0.057350  0.047530  0.069295            0.045508            0.071819
                       False M5_fixed_seeds M1_fixed_seeds             n1_f1  0.078808  0.048768  0.119130                 NaN                 NaN
                       False M5_fixed_seeds M1_fixed_seeds             n3_f1  0.047217  0.018184  0.079523                 NaN                 NaN
                       False M5_fixed_seeds M1_fixed_seeds balanced_accuracy  0.064517  0.056045  0.077885                 NaN                 NaN
                       False M5_fixed_seeds M1_fixed_seeds       cohen_kappa  0.063505  0.051268  0.077043                 NaN                 NaN
                       False M7_fixed_seeds M1_fixed_seeds          accuracy  0.042956  0.034082  0.052074                 NaN                 NaN
                        True M7_fixed_seeds M1_fixed_seeds          macro_f1  0.056023  0.045091  0.068489            0.042657            0.071234
                       False M7_fixed_seeds M1_fixed_seeds             n1_f1  0.092953  0.063431  0.128447                 NaN                 NaN
                       False M7_fixed_seeds M1_fixed_seeds             n3_f1  0.041619  0.004374  0.079773                 NaN                 NaN
                       False M7_fixed_seeds M1_fixed_seeds balanced_accuracy  0.057555  0.045829  0.072360                 NaN                 NaN
                       False M7_fixed_seeds M1_fixed_seeds       cohen_kappa  0.059727  0.047710  0.072471                 NaN                 NaN
                       False M7_fixed_seeds M5_fixed_seeds          accuracy -0.002300 -0.007865  0.001993                 NaN                 NaN
                        True M7_fixed_seeds M5_fixed_seeds          macro_f1 -0.001327 -0.007773  0.003887           -0.009130            0.004885
                       False M7_fixed_seeds M5_fixed_seeds             n1_f1  0.014145 -0.005273  0.030434                 NaN                 NaN
                       False M7_fixed_seeds M5_fixed_seeds             n3_f1 -0.005598 -0.017386  0.004103                 NaN                 NaN
                       False M7_fixed_seeds M5_fixed_seeds balanced_accuracy -0.006962 -0.013994 -0.000870                 NaN                 NaN
                       False M7_fixed_seeds M5_fixed_seeds       cohen_kappa -0.003777 -0.011544  0.002516                 NaN                 NaN
                       False             M4             M1          accuracy -0.005360 -0.025081  0.008871                 NaN                 NaN
                       False             M4             M1          macro_f1  0.018202 -0.001637  0.036606                 NaN                 NaN
                       False             M4             M1             n1_f1  0.085283  0.038749  0.117787                 NaN                 NaN
                       False             M4             M1             n3_f1  0.004501 -0.051735  0.067974                 NaN                 NaN
                       False             M4             M1 balanced_accuracy  0.062381  0.045581  0.079911                 NaN                 NaN
                       False             M4             M1       cohen_kappa  0.002440 -0.021046  0.020336                 NaN                 NaN
                       False     m5_wce_s42      m5_ce_s42          accuracy -0.005553 -0.014877  0.003524                 NaN                 NaN
                       False     m5_wce_s42      m5_ce_s42          macro_f1  0.007901 -0.003623  0.020041                 NaN                 NaN
                       False     m5_wce_s42      m5_ce_s42             n1_f1  0.047794  0.002234  0.093399                 NaN                 NaN
                       False     m5_wce_s42      m5_ce_s42             n3_f1 -0.005913 -0.027396  0.012541                 NaN                 NaN
                       False     m5_wce_s42      m5_ce_s42 balanced_accuracy  0.022217  0.012558  0.032481                 NaN                 NaN
                       False     m5_wce_s42      m5_ce_s42       cohen_kappa -0.003875 -0.015865  0.008883                 NaN                 NaN
