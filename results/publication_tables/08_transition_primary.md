| model | subset | epochs_per_seed | macro_f1 | accuracy |
| --- | --- | --- | --- | --- |
| CNN | transition | 3498 | 0.5821 | 0.6032 |
| CNN | stable | 27628 | 0.7289 | 0.8269 |
| CNN-BiLSTM | transition | 3498 | 0.5636 | 0.5708 |
| CNN-BiLSTM | stable | 27628 | 0.8044 | 0.8820 |
| CNN-Transformer | transition | 3498 | 0.5479 | 0.5652 |
| CNN-Transformer | stable | 27628 | 0.8042 | 0.8801 |

Target is a transition if the previous consecutive original epoch within subject/recording/night differs in true label. Boundaries/gaps excluded. Fixed-seed means, descriptive only.
