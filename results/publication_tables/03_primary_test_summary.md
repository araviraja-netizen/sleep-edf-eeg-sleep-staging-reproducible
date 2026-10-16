| model | seeds | parameters | accuracy_mean | accuracy_seed_sd | macro_f1_mean | macro_f1_seed_sd | macro_f1_subject_CI_low | macro_f1_subject_CI_high | n1_f1_mean | n1_f1_seed_sd | n3_f1_mean | n3_f1_seed_sd | balanced_accuracy_mean | balanced_accuracy_seed_sd | cohen_kappa_mean | cohen_kappa_seed_sd |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CNN | 1 | 40501 | 0.8018 | — | 0.7198 | — | 0.6853 | 0.7477 | 0.3823 | — | 0.7420 | — | 0.7088 | — | 0.7237 | — |
| CNN-BiLSTM | 3 | 65717 | 0.8471 | 0.0012 | 0.7771 | 0.0066 | 0.7472 | 0.8005 | 0.4611 | 0.0314 | 0.7892 | 0.0052 | 0.7733 | 0.0076 | 0.7872 | 0.0018 |
| CNN-Transformer | 3 | 107893 | 0.8448 | 0.0052 | 0.7758 | 0.0118 | 0.7460 | 0.7994 | 0.4752 | 0.0468 | 0.7836 | 0.0059 | 0.7664 | 0.0128 | 0.7834 | 0.0079 |

Scores are 0–1. Means/seed SD are computed across predefined fixed training seeds; M1 SD is undefined. Subject bootstrap intervals condition on those fitted seeds, and are not seed-variability intervals.
