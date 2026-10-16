"""Immutable checkpoint generations and atomically committed latest/best/previous roles."""

import json
from pathlib import Path
import random
import shutil
import uuid

import numpy as np
import torch

from .common import atomic_file, digest, fingerprint, json_write

REQUIRED_FINGERPRINTS = {"architecture", "training_configuration", "preprocessing", "normalization",
                         "class_mapping", "subject_split", "epoch_manifest", "protocol"}
REQUIRED_STATE = {"model", "optimizer", "scheduler", "early_stopping", "completed_epoch", "next_epoch",
                  "global_step", "rng", "sampler", "history", "configuration", "fingerprints", "experiment_id"}


class CorruptCheckpoint(ValueError):
    pass


class IncompatibleCheckpoint(ValueError):
    pass


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def capture_rng(include_mps=False):
    legacy = np.random.get_state()
    state = {"python": random.getstate(), "numpy": {"kind": legacy[0], "keys": legacy[1].tolist(),
             "position": legacy[2], "has_gauss": legacy[3], "cached_gaussian": legacy[4]},
             "torch_cpu": torch.get_rng_state(), "torch_cuda": None, "torch_mps": None}
    if torch.cuda.is_initialized():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    if include_mps and torch.backends.mps.is_available():
        state["torch_mps"] = torch.mps.get_rng_state()
    return state


def restore_rng(state):
    random.setstate(state["python"])
    legacy = state["numpy"]
    np.random.set_state((legacy["kind"], np.array(legacy["keys"], dtype=np.uint32), legacy["position"],
                         legacy["has_gauss"], legacy["cached_gaussian"]))
    torch.set_rng_state(state["torch_cpu"].cpu())
    if state.get("torch_cuda") is not None:
        if not torch.cuda.is_available():
            raise IncompatibleCheckpoint("Saved CUDA RNG requires an available CUDA backend")
        torch.cuda.set_rng_state_all(state["torch_cuda"])
    if state.get("torch_mps") is not None:
        if not torch.backends.mps.is_available():
            raise IncompatibleCheckpoint("Saved MPS RNG requires an available MPS backend")
        torch.mps.set_rng_state(state["torch_mps"].cpu())


class CheckpointManager:
    """One writer per experiment; Trainer holds the experiment lock while saving.

    Pointer commit happens after immutable checkpoint + sidecar checksum validation.
    The previous pointer index is independently checksummed. Uncommitted orphan files
    are never selected by auto resume. Loads use PyTorch's restricted weights-only loader.
    """
    def __init__(self, directory, fingerprints, experiment_id):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        if not REQUIRED_FINGERPRINTS.issubset(fingerprints):
            raise ValueError("Missing dataset/model resume fingerprints")
        self.fingerprints = dict(fingerprints)
        self.experiment_id = experiment_id
        self.recovered_path = None
        self.recovery_notes = []

    def _validate_payload(self, state):
        if not isinstance(state, dict) or not REQUIRED_STATE.issubset(state):
            raise CorruptCheckpoint("Missing required checkpoint state")
        if state["fingerprints"] != self.fingerprints or state["experiment_id"] != self.experiment_id:
            raise IncompatibleCheckpoint("Architecture/config/dataset/split/normalization/class mapping mismatch")
        completed = state["completed_epoch"]
        if not isinstance(completed, int) or completed < -1 or state["next_epoch"] != completed + 1:
            raise CorruptCheckpoint("Invalid completed/next epoch")
        if [entry["epoch"] for entry in state["history"]] != list(range(completed + 1)):
            raise CorruptCheckpoint("History has missing/duplicate/uncommitted epoch entries")
        if state["global_step"] < 0:
            raise CorruptCheckpoint("Negative global step")

    def read(self, path):
        path = Path(path)
        try:
            metadata = json.loads(path.with_suffix(path.suffix + ".json").read_text())
            if (metadata["format"] != "sleep-research-checkpoint-v1"
                    or path.stat().st_size != metadata["bytes"] or digest(path) != metadata["sha256"]):
                raise CorruptCheckpoint("Checkpoint byte checksum/size/format mismatch")
            if metadata["fingerprints"] != self.fingerprints or metadata["experiment_id"] != self.experiment_id:
                raise IncompatibleCheckpoint("Checkpoint sidecar fingerprints or experiment ID differ")
            state = torch.load(path, map_location="cpu", weights_only=True)
            self._validate_payload(state)
            if state["completed_epoch"] != metadata["completed_epoch"]:
                raise CorruptCheckpoint("Checkpoint/sidecar epoch mismatch")
            return state
        except IncompatibleCheckpoint:
            raise
        except Exception as exc:
            if isinstance(exc, CorruptCheckpoint):
                raise
            raise CorruptCheckpoint(f"Cannot verify/load {path.name}: {exc}") from exc

    def _index(self, path):
        envelope = json.loads(path.read_text())
        body = envelope["body"]
        if envelope["sha256"] != fingerprint(body) or body["schema"] != 1:
            raise CorruptCheckpoint("Checkpoint pointer index checksum/schema mismatch")
        for role in ("latest", "previous", "best"):
            name = body.get(role)
            if name is not None and Path(name).name != name:
                raise CorruptCheckpoint("Checkpoint pointer must name a local generation")
        return body

    def _indices(self):
        indices = []
        for name in ("index.json", "index.previous.json"):
            path = self.directory / name
            if path.exists():
                try:
                    indices.append(self._index(path))
                except (OSError, ValueError, KeyError) as exc:
                    self.recovery_notes.append(f"Rejected {name}: {exc}")
        return indices

    def discover(self):
        indices = self._indices()
        if not indices:
            return {}
        return {role: str(self.directory / name) if name else None
                for role, name in ((role, indices[0].get(role)) for role in ("latest", "best", "previous"))}

    def pin(self, path, reference):
        """Keep checkpoints used by permanent inference exports beyond rolling retention."""
        path = Path(path).resolve()
        if path.parent != self.directory.resolve():
            raise ValueError("Pin a validated local checkpoint generation")
        self.read(path)
        target = self.directory / "pins.json"
        pins = json.loads(target.read_text()) if target.exists() else {"schema": 1, "checkpoints": {}}
        entry = pins["checkpoints"].setdefault(path.name, {"sha256": digest(path), "references": []})
        if entry["sha256"] != digest(path):
            raise CorruptCheckpoint("Pinned checkpoint bytes changed")
        if reference not in entry["references"]:
            entry["references"].append(reference)
        json_write(target, pins)

    def recover(self, resume="auto"):
        self.recovery_notes = []
        if resume == "none":
            if any(self.directory.glob("*.pt")) or any(self.directory.glob("index*.json")):
                raise FileExistsError("--resume none requires a new experiment; existing checkpoints are preserved")
            return None
        if resume != "auto":
            path = Path(resume).expanduser().resolve()
            state = self.read(path)
            if path.parent != self.directory.resolve():
                imported = self.directory / f"imported-epoch-{state['completed_epoch']:06d}-{uuid.uuid4().hex[:12]}.pt"
                with path.open("rb") as source, atomic_file(imported, binary=True) as target:
                    shutil.copyfileobj(source, target)
                metadata = json.loads(path.with_suffix(path.suffix + ".json").read_text())
                json_write(imported.with_suffix(".pt.json"), metadata)
                self.read(imported)
                path = imported
            self.recovered_path = path
            return state
        indices = self._indices()
        if not indices:
            if any(self.directory.glob("*.pt")) or any(self.directory.glob("index*.json")):
                raise CorruptCheckpoint("No valid committed pointer index; refusing a silent fresh start")
            return None
        # The current committed timeline takes precedence over the old pointer index,
        # including an explicit manual rewind to an earlier compatible checkpoint.
        for index in indices:
            candidates = []
            for role in ("latest", "previous", "best"):
                name = index.get(role)
                if name and name not in candidates:
                    candidates.append(name)
            valid = []
            for name in candidates:
                try:
                    state = self.read(self.directory / name)
                    if state["completed_epoch"] <= index["completed_epoch"]:
                        valid.append((state["completed_epoch"], name, state))
                except IncompatibleCheckpoint:
                    raise  # Valid but incompatible state is never treated as mere corruption.
                except CorruptCheckpoint as exc:
                    self.recovery_notes.append(f"Rejected {name}: {exc}")
            if valid:
                _, name, state = max(valid, key=lambda item: item[0])
                self.recovered_path = self.directory / name
                return state
        raise CorruptCheckpoint("All committed checkpoint candidates are invalid")

    def save(self, state, validation_macro_f1=None):
        self._validate_payload(state)
        if validation_macro_f1 is not None and not 0 <= validation_macro_f1 <= 1:
            raise ValueError("Best checkpoint requires a finite validation macro-F1 in [0,1]")
        name = f"epoch-{state['completed_epoch']:06d}-{uuid.uuid4().hex[:12]}.pt"
        target = self.directory / name
        with atomic_file(target, binary=True) as handle:
            torch.save(state, handle)
        metadata = {"format": "sleep-research-checkpoint-v1", "sha256": digest(target), "bytes": target.stat().st_size,
                    "completed_epoch": state["completed_epoch"], "next_epoch": state["next_epoch"],
                    "experiment_id": self.experiment_id, "fingerprints": self.fingerprints}
        json_write(target.with_suffix(".pt.json"), metadata)
        self.read(target)  # Verify the actual serialized bytes before publishing a pointer.
        indices = self._indices()
        old = indices[0] if indices else None
        previous = self.recovered_path.name if self.recovered_path is not None and self.recovered_path.parent == self.directory else (old.get("latest") if old else None)
        best, best_score = (old.get("best"), old.get("best_score")) if old else (None, None)
        if best is not None:
            try:
                best_state = self.read(self.directory / best)
            except CorruptCheckpoint:
                self.recovery_notes.append("Prior best checkpoint is corrupt; selecting among recoverable epochs")
                best, best_score = None, None
        if previous is not None:
            try:
                prior = self.read(self.directory / previous)
            except CorruptCheckpoint:
                previous = None
            else:
                if prior["completed_epoch"] >= state["completed_epoch"]:
                    raise ValueError("Checkpoint epochs must advance; duplicate saves are forbidden")
                if best is not None and best_state["completed_epoch"] > prior["completed_epoch"]:
                    best, best_score = None, None  # Manual rewind discards future model-selection state.
                if best is None and prior["history"]:
                    best, best_score = previous, prior["history"][-1]["validation_macro_f1"]
        if validation_macro_f1 is not None and (best_score is None or validation_macro_f1 > best_score):
            best, best_score = name, float(validation_macro_f1)
        body = {"schema": 1, "latest": name, "previous": previous, "best": best, "best_score": best_score,
                "completed_epoch": state["completed_epoch"], "next_epoch": state["next_epoch"]}
        if old:
            json_write(self.directory / "index.previous.json", {"body": old, "sha256": fingerprint(old)})
        json_write(self.directory / "index.json", {"body": body, "sha256": fingerprint(body)})
        self.recovered_path = target
        # Bound storage while keeping current roles and the prior committed index recoverable.
        protected = {body.get(role) for role in ("latest", "previous", "best")}
        if old:
            protected |= {old.get(role) for role in ("latest", "previous", "best")}
        pins_path = self.directory / "pins.json"
        if pins_path.exists():
            pins = json.loads(pins_path.read_text())
            protected |= set(pins["checkpoints"])
        for path in self.directory.glob("*.pt"):
            if path.name not in protected:
                path.unlink()
                path.with_suffix(".pt.json").unlink(missing_ok=True)
        return target
