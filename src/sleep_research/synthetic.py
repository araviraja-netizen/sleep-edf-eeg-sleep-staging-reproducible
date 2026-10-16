"""Small explicit synthetic fixtures; never read Sleep-EDF EEG for model training."""

import copy
import numpy as np
import torch

from .checkpoints import seed_everything
from .common import fingerprint
from .experiments import CLASSES, Experiment, frozen_dataset_contract
from .trainer import Trainer


class SyntheticEpochs:
    def __init__(self, split, count, signal_samples=128, seed=42):
        if split not in ("train", "validation", "test"):
            raise ValueError("Unknown synthetic split")
        self.split = split
        rng = np.random.default_rng(seed)
        labels = np.arange(count, dtype=np.int64) % 5
        time = np.arange(signal_samples) / 100
        phase = rng.uniform(-np.pi, np.pi, count)
        signals = np.sin(2 * np.pi * (2 + 2 * labels[:, None]) * time + phase[:, None])
        signals += 0.2 * labels[:, None] + rng.normal(0, 0.12, size=signals.shape)
        self.signals = signals[:, None, :].astype(np.float32)
        self.labels = labels
        self.metadata = []
        counters = {}
        for i in range(count):
            subject = f"SYN_{split.upper()}_S{i % 4}"
            recording = subject + "_N1"
            counter = counters.get(recording, 0)
            original = counter + counter // 7  # Deliberate original-index gaps for plotting tests.
            counters[recording] = counter + 1
            self.metadata.append(dict(epoch_id=f"{recording}:e{original:06d}", subject_id=subject, recording_id=recording,
                                      night=1, original_epoch_index=original, onset_seconds=30 * original, split=split))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return {"eeg": torch.from_numpy(self.signals[index].copy()), "label": int(self.labels[index]), **self.metadata[index]}


class SyntheticTinyCNN(torch.nn.Module):
    """Infrastructure fixture, not an evaluated Sleep-EDF model."""
    def __init__(self, channels=4, embedding_dimension=16, dropout=0.2, classes=5):
        super().__init__()
        self.features = torch.nn.Sequential(torch.nn.Conv1d(1, channels, kernel_size=7, padding=3),
                                            torch.nn.ReLU(), torch.nn.AdaptiveAvgPool1d(16), torch.nn.Flatten())
        self.embedding = torch.nn.Sequential(torch.nn.Linear(channels * 16, embedding_dimension),
                                             torch.nn.ReLU(), torch.nn.Dropout(dropout))
        self.classifier = torch.nn.Linear(embedding_dimension, classes)

    def forward(self, values):
        return self.classifier(self.embedding(self.features(values)))


def synthetic_contract(train, validation, real_reference=None):
    def identity(dataset):
        from hashlib import sha256
        return {"inputs_sha256": sha256(dataset.signals.tobytes()).hexdigest(), "labels": dataset.labels.tolist(),
                "metadata": dataset.metadata}
    identity_value = {"train": identity(train), "validation": identity(validation)}
    contract = {"dataset": "Synthetic infrastructure fixture", "synthetic": True, "preprocessing": fingerprint({"synthetic_samples": train.signals.shape[-1]}),
                "normalization": fingerprint({"synthetic_units": "arbitrary", "normalization_applied": False}),
                "subject_split": fingerprint({"train": sorted({r['subject_id'] for r in train.metadata}),
                                              "validation": sorted({r['subject_id'] for r in validation.metadata})}),
                "epoch_manifest": fingerprint(identity_value), "protocol": fingerprint("synthetic fixture, no Wake trimming"),
                "class_mapping": fingerprint(CLASSES)}
    if real_reference is not None:
        contract["reference_frozen_sleep_edf"] = real_reference
    return contract


def build_demo_trainer(project, config, run_id=None, bind_real_reference=True):
    config = copy.deepcopy(config)
    if not config.get("synthetic"):
        raise PermissionError("Phase 2.5 demo entrypoint permits synthetic training only")
    if config["optimizer"]["name"] != "Adam" or config["scheduler"]["name"] != "StepLR":
        raise ValueError("Synthetic fixture supports Adam/StepLR; generic Trainer accepts other supplied optimizers")
    seed_everything(config["seed"])
    torch.set_num_threads(1)
    data = config["synthetic_dataset"]
    train = SyntheticEpochs("train", data["train_samples"], data["signal_samples"], config["seed"])
    validation = SyntheticEpochs("validation", data["validation_samples"], data["signal_samples"], config["seed"] + 1)
    arch = {key: value for key, value in config["architecture"].items() if key != "name"}
    model = SyntheticTinyCNN(**arch)
    opt = config["optimizer"]
    optimizer = torch.optim.Adam(model.parameters(), lr=opt["learning_rate"], weight_decay=opt["weight_decay"])
    sched = config["scheduler"]
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=sched["step_size"], gamma=sched["gamma"])
    reference = frozen_dataset_contract(project, "A") if bind_real_reference else None
    contract = synthetic_contract(train, validation, reference)
    experiment = Experiment(project, config["family"], config, contract, run_id)
    return Trainer(model, optimizer, scheduler, train, validation, experiment, config)
