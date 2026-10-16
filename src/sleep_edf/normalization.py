"""Stable population moments fitted to training epochs only, separately per protocol."""

from dataclasses import dataclass
import csv
from pathlib import Path

import numpy as np

from .phase2_utils import atomic_json


@dataclass
class RunningMoments:
    count: int = 0
    mean: float = 0.0
    m2: float = 0.0

    def update(self, values):
        values = np.asarray(values, dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError("Nonfinite values cannot contribute to normalization")
        count = values.size
        if not count:
            return
        mean = float(values.mean())
        m2 = float(np.sum((values - mean) ** 2))
        combined = self.count + count
        delta = mean - self.mean
        self.m2 += m2 + delta * delta * self.count * count / combined
        self.mean += delta * count / combined
        self.count = combined

    def parameters(self):
        if not self.count:
            raise ValueError("Cannot fit normalization without training samples")
        std = float(np.sqrt(max(0, self.m2 / self.count)))
        if not np.isfinite(std) or std <= 0:
            raise ValueError("Training normalization standard deviation is not positive/finite")
        return {"sample_count": self.count, "mean_microvolts": self.mean,
                "std_microvolts": std, "ddof": 0}


def normalize_epoch(epoch, parameters):
    std, mean = parameters["std_microvolts"], parameters["mean_microvolts"]
    if not np.isfinite(std) or std <= 0 or not np.isfinite(mean):
        raise ValueError("Invalid normalization parameters")
    epoch = np.asarray(epoch)
    if not np.isfinite(epoch).all():
        raise ValueError("Cannot normalize a nonfinite epoch")
    return ((epoch.astype(np.float64) - mean) / std).astype(np.float32)


def fit_normalization(completions, assignments, processed_root, split_hash, fingerprint, output_path, batch_epochs=128):
    moments = {"A": RunningMoments(), "B": RunningMoments()}
    epochs = {"A": 0, "B": 0}
    contributors = {"A": set(), "B": set()}
    records = {"A": [], "B": []}
    for completion in completions:
        subject = completion["subject_id"]
        if assignments[subject] != "train":
            continue  # Neither validation nor test values are read for fitted moments.
        directory = Path(processed_root) / "recordings" / completion["recording_id"]
        values = np.load(directory / "epochs.npy", mmap_mode="r", allow_pickle=False)
        with (directory / "epoch_metadata.csv").open(newline="") as handle:
            flags = np.array([row["wake_trimmed_inclusion"] == "1" for row in csv.DictReader(handle)])
        if len(flags) != len(values):
            raise ValueError("Normalization metadata/array count mismatch")
        for start in range(0, len(values), batch_epochs):
            batch = values[start:start + batch_epochs]
            moments["B"].update(batch)
            moments["A"].update(batch[flags[start:start + batch_epochs]])
        for protocol, count in (("A", int(flags.sum())), ("B", len(values))):
            epochs[protocol] += count
            if count:
                contributors[protocol].add(subject)
                records[protocol].append(completion["recording_id"])
        del values
    normalization = {"method": "global_zscore", "fit_split": "train", "applied_on_load": True,
                     "stored_array_units": "microvolts", "normalized_units": "dimensionless",
                     "split_manifest_sha256": split_hash, "pipeline_fingerprint": fingerprint, "protocols": {}}
    for protocol in ("A", "B"):
        normalization["protocols"][protocol] = {**moments[protocol].parameters(), "epoch_count": epochs[protocol],
            "training_subjects": sorted(contributors[protocol]), "training_recordings": records[protocol]}
        if moments[protocol].count != epochs[protocol] * 3000:
            raise ValueError("Normalization sample/epoch count mismatch")
    atomic_json(output_path, normalization)
    return normalization
