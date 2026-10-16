| model | seeds | parameters | accuracy | macro_F1 | N1_F1 | subject_95_CI |
| --- | --- | --- | --- | --- | --- | --- |
| CNN | 1 | 40501 | 0.8018 | 0.7198 (single seed) | 0.3823 | [0.6853, 0.7477] |
| CNN-BiLSTM | 3 | 65717 | 0.8471 | 0.7771 ± 0.0066 | 0.4611 | [0.7472, 0.8005] |
| CNN-Transformer | 3 | 107893 | 0.8448 | 0.7758 ± 0.0118 | 0.4752 | [0.7460, 0.7994] |

Macro-F1 ± sample SD across fixed training seeds; the interval column is separately computed paired-subject uncertainty conditional on those seeds. Scores are ratios0–1.
