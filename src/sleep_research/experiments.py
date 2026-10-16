"""Permanent experiment directories, exact configs and reusable dataset contracts."""

from contextlib import contextmanager
import fcntl
import importlib.metadata
import inspect
import json
from pathlib import Path
import platform
import re
import subprocess
import sys
import uuid

import torch

from .common import digest, fingerprint, json_write, now

FAMILIES = ("cnn", "cnn_lstm", "attention", "transformer")
CLASSES = {"Wake": 0, "N1": 1, "N2": 2, "N3": 3, "REM": 4}


def frozen_dataset_contract(project, protocol="A"):
    project = Path(project)
    if protocol not in ("A", "B"):
        raise ValueError("Protocol must be A or B")
    state = json.loads((project / "results/phase2_state.json").read_text())
    if state["status"] != "complete":
        raise ValueError("Frozen processed dataset is incomplete")
    for relative, expected in state["manifest_sha256"].items():
        if digest(project / relative) != expected:
            raise ValueError(f"Frozen dataset manifest changed: {relative}")
    return {"dataset": "Sleep-EDF Expanded", "study": "SC", "preprocessing": state["pipeline_fingerprint"],
            "normalization": digest(project / "results/normalization.json"),
            "subject_split": state["split_manifest_sha256"],
            "epoch_manifest": state["manifest_sha256"]["results/epoch_manifest.csv"],
            "protocol": fingerprint({"protocol": protocol, "wake_config": state["pipeline_identity"]["wake_trimming"]}),
            "class_mapping": fingerprint({"classes": CLASSES, "N3_original_stages": [3, 4], "excluded": ["movement", "unknown"]}),
            "protocol_name": protocol, "synthetic": False}


def model_fingerprint(model, architecture_config):
    try:
        source = inspect.getsource(type(model))
    except (OSError, TypeError):
        source = repr(type(model))
    value = {"class": f"{type(model).__module__}.{type(model).__qualname__}", "source": source,
             "architecture_config": architecture_config, "modules": repr(model),
             "parameters": {name: {"shape": list(value.shape), "dtype": str(value.dtype)}
                            for name, value in model.state_dict().items()}}
    return fingerprint(value)


def resume_fingerprints(model, config, dataset_contract):
    values = {key: dataset_contract[key] for key in ("preprocessing", "normalization", "subject_split", "epoch_manifest", "protocol", "class_mapping")}
    values["architecture"] = model_fingerprint(model, config["architecture"])
    # Epoch budget may be extended; optimization/data order settings must not change.
    values["training_configuration"] = fingerprint({key: value for key, value in config.items() if key not in ("max_epochs", "run_id")})
    return values


def software_versions():
    result = {"python": sys.version, "python_executable": sys.executable, "platform": platform.platform()}
    for name in ("torch", "numpy", "scipy", "pandas", "matplotlib", "scikit-learn", "umap-learn"):
        try:
            result[name] = str(importlib.metadata.version(name))
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def source_revision(project):
    project = Path(project)
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=project, capture_output=True, text=True)
    files = sorted((project / "src").rglob("*.py"))
    hashes = {str(path.relative_to(project)): digest(path) for path in files}
    return {"git_commit": result.stdout.strip() if result.returncode == 0 else None,
            "source_sha256": hashes, "source_snapshot_sha256": fingerprint(hashes)}


class Experiment:
    def __init__(self, project, family, config=None, contract=None, run_id=None):
        self.project = Path(project).resolve()
        if family not in FAMILIES:
            raise ValueError("Use cnn/cnn_lstm/attention/transformer experiment family")
        if run_id is None:
            run_id = now().replace(":", "").replace("+", "_").replace(".", "-") + "_" + uuid.uuid4().hex[:8]
        if not re.fullmatch(r"[A-Za-z0-9_-]+", run_id):
            raise ValueError("Run ID must be a local alphanumeric identifier")
        self.run_id = run_id
        self.path = self.project / "experiments" / family / run_id
        metadata_path = self.path / "reproducibility/experiment.json"
        if metadata_path.exists():
            self.metadata = json.loads(metadata_path.read_text())
            if contract is not None and contract != self.metadata["dataset_contract"]:
                raise ValueError("Experiment dataset contract cannot change")
            if config is not None:
                before = {k: v for k, v in self.metadata["configuration"].items() if k not in ("max_epochs", "run_id")}
                after = {k: v for k, v in config.items() if k not in ("max_epochs", "run_id")}
                if before != after:
                    raise ValueError("Experiment hyperparameters/architecture cannot change during resume")
        else:
            if self.path.exists():
                raise ValueError("Experiment directory exists without a valid creation manifest")
            if config is None or contract is None:
                raise ValueError("New experiments require exact config and dataset contract")
            self.path.mkdir(parents=True)
            for folder in ("reproducibility/checkpoints", "performance", "representations", "analysis", "explanations"):
                (self.path / folder).mkdir(parents=True)
            self.metadata = {"experiment_id": run_id, "family": family, "created_at": now(), "status": "created",
                             "configuration": config, "dataset_contract": contract, "software": software_versions(),
                             "source_revision": source_revision(self.project), "device": config["device"],
                             "random_seed": config["seed"], "synthetic": contract["synthetic"],
                             "training_attempts": [], "mps_bitwise_resume_guarantee": False}
            json_write(metadata_path, self.metadata)
            json_write(self.path / "reproducibility/configuration.json", config)

    @contextmanager
    def lock(self):
        with (self.path / ".experiment.lock").open("a+") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            yield

    def update(self, **values):
        self.metadata.update(values)
        json_write(self.path / "reproducibility/experiment.json", self.metadata)

    def record_attempt(self, status, seconds, start_epoch, next_epoch, resume, recovery_notes):
        attempts = list(self.metadata["training_attempts"])
        attempts.append({"finished_at": now(), "status": status, "seconds": seconds,
                         "start_epoch": start_epoch, "next_epoch": next_epoch, "resume": str(resume),
                         "recovery_notes": list(recovery_notes)})
        self.update(status=status, training_attempts=attempts,
                    total_training_attempt_seconds=sum(value["seconds"] for value in attempts))
