"""Small epoch-level CNN; no sequence context or test-set decisions."""

import torch
from torch import nn


class CNN1D(nn.Module):
    def __init__(self, channels=(16, 32, 64, 64), kernels=(25, 9, 7, 5),
                 strides=(2, 2, 2, 1), groups=4, dropout=0.3, classes=5):
        super().__init__()
        if classes != 5 or not (len(channels) == len(kernels) == len(strides) == 4):
            raise ValueError("Four convolution blocks and five sleep classes are required")
        if any(k % 2 != 1 or k < 1 for k in kernels) or any(c % groups for c in channels):
            raise ValueError("Use positive odd kernels and channels divisible by normalization groups")
        self.blocks = nn.ModuleList()
        previous = 1
        for width, kernel, stride in zip(channels, kernels, strides):
            self.blocks.append(nn.Sequential(
                nn.Conv1d(previous, width, kernel, stride=stride, padding=kernel // 2, bias=False),
                nn.GroupNorm(groups, width), nn.ReLU(), nn.MaxPool1d(2)))
            previous = width
        self.embedding = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(1))
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(previous, classes))

    def extract_features(self, values):
        if values.ndim != 3 or values.shape[1:] != (1, 3000):
            raise ValueError("CNN1D expects [batch,1,3000] normalized float32 EEG")
        for block in self.blocks:
            values = block(values)
        return self.embedding(values)

    def forward(self, values):
        return self.classifier(self.extract_features(values))


def architecture_details(model):
    """Analytic temporal support and Conv/Linear MACs per single epoch.

    MACs omit normalization, activations, pooling and dropout. Global pooling
    spans the whole recording epoch; convolutional receptive fields describe
    local pre-pooling features, before GroupNorm's time-wide statistics.
    """
    length, width, receptive, jump, macs = 3000, 1, 1, 1, 0
    layers = []
    for i, block in enumerate(model.blocks):
        conv = block[0]
        length = (length + 2 * conv.padding[0] - conv.kernel_size[0]) // conv.stride[0] + 1
        receptive += (conv.kernel_size[0] - 1) * jump
        jump *= conv.stride[0]
        count = length * conv.out_channels * width * conv.kernel_size[0]
        macs += count
        layers.append(dict(name=f"blocks.{i}.0", output=[conv.out_channels, length],
                           parameters=sum(p.numel() for p in conv.parameters()),
                           local_receptive_field_samples=receptive, input_jump=jump, macs=count))
        layers.append(dict(name=f"blocks.{i}.1/2", operation="GroupNorm + ReLU",
                           output=[conv.out_channels, length], parameters=sum(p.numel() for p in block[1].parameters())))
        receptive += jump
        jump *= 2
        length //= 2
        width = conv.out_channels
        layers.append(dict(name=f"blocks.{i}.3", operation="MaxPool1d(2)", output=[width, length],
                           parameters=0, local_receptive_field_samples=receptive, input_jump=jump))
    classifier = model.classifier[1]
    macs += width * 5
    layers += [dict(name="embedding", operation="AdaptiveAvgPool1d(1) + Flatten", output=[width], parameters=0),
               dict(name="classifier.0", operation="Dropout", output=[width], parameters=0),
               dict(name="classifier.1", operation="Linear", output=[5], parameters=sum(p.numel() for p in classifier.parameters()), macs=width * 5)]
    return dict(input=[1, 3000], layers=layers, parameters=sum(p.numel() for p in model.parameters()),
                embedding_dimension=width, conv_linear_macs_per_epoch=macs,
                approximate_conv_linear_flops_per_epoch=2 * macs,
                final_local_receptive_field_samples=receptive,
                final_local_receptive_field_seconds=receptive / 100,
                global_support_seconds=30,
                note="GroupNorm aggregates each group's channels/time; strided learned convolutions are not an explicit anti-alias filter. MACs exclude norm/pool/activation/backward/optimizer.")
