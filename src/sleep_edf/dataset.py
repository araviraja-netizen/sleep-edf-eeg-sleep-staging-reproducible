"""Memory-mapped epochs with frozen metadata and train-only normalization on access.

This sequence can be passed to a later PyTorch DataLoader. It does not import torch,
construct a model, train, or generate predictions. Each sample carries a stable epoch ID.
"""

from collections import OrderedDict
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .inspection import sha256_file
from .normalization import normalize_epoch
from .preprocessing import verify_completion


class EpochDataset:
    def __init__(self, project, protocol="A", split="train", normalize=True, max_open_recordings=2):
        self.project = Path(project).resolve()
        if protocol not in ("A", "B") or split not in ("train", "validation", "test", None):
            raise ValueError("Select protocol A/B and train/validation/test (or all for analysis)")
        if max_open_recordings < 1:
            raise ValueError("At least one recording cache slot is required")
        self.protocol, self.normalize = protocol, normalize
        self.max_open_recordings = max_open_recordings
        self._cache = OrderedDict()
        results = self.project / "results"
        self.state = json.loads((results / "phase2_state.json").read_text())
        if self.state["status"] != "complete":
            raise ValueError("Dataset consolidation is incomplete")
        self.processed_root = (self.project / self.state["processed_root"]).resolve()
        if not self.processed_root.is_relative_to(self.project / "data/processed"):
            raise ValueError("Invalid processed data root")
        for relative, expected in self.state["manifest_sha256"].items():
            if sha256_file(self.project / relative) != expected:
                raise ValueError(f"Frozen dataset artifact changed: {relative}")
        if sha256_file(results / "subject_splits.csv") != self.state["split_manifest_sha256"]:
            raise ValueError("Subject split changed")
        normalization = json.loads((results / "normalization.json").read_text())
        if (normalization["pipeline_fingerprint"] != self.state["pipeline_fingerprint"]
                or normalization["split_manifest_sha256"] != self.state["split_manifest_sha256"]):
            raise ValueError("Normalization provenance differs from dataset/splits")
        self.parameters = normalization["protocols"][protocol]
        with (results / "subject_splits.csv").open(newline="") as handle:
            assignments = {r["subject_id"]: r["split"] for r in csv.DictReader(handle)}
        if any(assignments[s] != "train" for s in self.parameters["training_subjects"]):
            raise ValueError("Normalization contains a non-training subject")
        index = results / f"protocol_{protocol.lower()}_index.csv"
        dtypes = {"recording_id": "category", "subject_id": "category", "split": "category",
                  "night": "uint8", "original_epoch_index": "int32", "array_row": "int32",
                  "label": "uint8", "onset_seconds": "int32"}
        frame = pd.read_csv(index, dtype=dtypes)
        if split is not None:
            frame = frame[frame["split"] == split]
        self.index = frame.reset_index(drop=True)
        if self.index.epoch_id.duplicated().any():
            raise ValueError("Duplicate epoch IDs")
        self.assignments = assignments

    def __len__(self):
        return len(self.index)

    def _recording(self, recording_id):
        if recording_id in self._cache:
            self._cache.move_to_end(recording_id)
            return self._cache[recording_id]
        directory = self.processed_root / "recordings" / recording_id
        completion = verify_completion(directory, self.state["pipeline_fingerprint"], hashes=False)
        values = np.load(directory / "epochs.npy", mmap_mode="r", allow_pickle=False)
        with (directory / "epoch_metadata.csv").open(newline="") as handle:
            metadata = list(csv.DictReader(handle))
        self._cache[recording_id] = (values, metadata, completion)
        while len(self._cache) > self.max_open_recordings:
            _, (old_values, _, _) = self._cache.popitem(last=False)
            old_values._mmap.close()
        return self._cache[recording_id]

    def __getitem__(self, index):
        row = self.index.iloc[index]
        values, metadata, completion = self._recording(str(row.recording_id))
        array_row = int(row.array_row)
        saved = metadata[array_row]
        if (int(saved["original_epoch_index"]) != int(row.original_epoch_index)
                or int(saved["label"]) != int(row.label)
                or int(saved["onset_seconds"]) != int(row.onset_seconds)
                or completion["subject_id"] != str(row.subject_id)
                or completion["night"] != int(row.night)
                or self.assignments[str(row.subject_id)] != str(row.split)
                or (self.protocol == "A" and saved["wake_trimmed_inclusion"] != "1")):
            raise ValueError("Epoch/sample/label/subject/split alignment mismatch")
        eeg = normalize_epoch(values[array_row], self.parameters) if self.normalize else values[array_row].copy()
        return {"eeg": eeg.reshape(1, 3000), "label": int(row.label), "epoch_id": str(row.epoch_id),
                "subject_id": str(row.subject_id), "recording_id": str(row.recording_id),
                "night": int(row.night), "original_epoch_index": int(row.original_epoch_index),
                "onset_seconds": int(row.onset_seconds), "split": str(row.split)}

    def close(self):
        for values, _, _ in self._cache.values():
            values._mmap.close()
        self._cache.clear()

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_cache"] = OrderedDict()  # Worker processes reopen their own read-only memmaps.
        return state
