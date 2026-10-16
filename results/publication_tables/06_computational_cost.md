| model | seeds | parameters | float32_weight_KiB | approximate_forward_MACs_per_target | training_minutes_mean | inference_export_seconds_mean | best_checkpoint_bytes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| CNN | 1 | 40501 | 158.2070 | 4638464 | 25.4591 | 10.1497 | 540332 |
| CNN-BiLSTM | 3 | 65717 | 256.7070 | 23313920 | 63.9316 | 31.0382 | 857139 |
| CNN-Transformer | 3 | 107893 | 421.4570 | 23525120 | 68.5332 | 30.3188 | 1382699 |

Measured training-attempt and chunk inference/export wall times are from separate executions on M4/MPS, not controlled kernel benchmarks. Analytic MACs exclude normalization, activation, pooling, biases, softmax, and elementwise recurrent operations; two FLOPs per MAC is only a partial operation convention. Checkpoint bytes include metadata/state, whereas weight KiB counts trainable float32 parameters only.
