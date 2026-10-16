"""Atomic, SSD-local writes and reproducible Phase 2 configuration."""

from contextlib import contextmanager
from datetime import datetime
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import tempfile
from zoneinfo import ZoneInfo

import yaml

from .inspection import sha256_file


def timestamp():
    return datetime.now(ZoneInfo("Asia/Kolkata")).isoformat()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


@contextmanager
def atomic_writer(path, newline=None):
    """Fsync then rename a same-directory temporary file; old output survives failure."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "w", newline=newline) as handle:
            yield handle
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_json(path, value):
    with atomic_writer(path) as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def load_config(project):
    project = Path(project).resolve()
    config = yaml.safe_load((project / "configs/preprocessing.yaml").read_text())
    wake = yaml.safe_load((project / "configs/wake_trimming.yaml").read_text())
    dataset, python = Path(config["dataset_root"]), Path(config["python_executable"])
    if not dataset.is_dir() or not python.is_file() or not os.access(python, os.X_OK):
        raise RuntimeError("STOP: specified dataset or executable interpreter is missing")
    import sys
    if Path(sys.executable).resolve() != python.resolve():
        raise RuntimeError(f"STOP: use the existing interpreter {python}")
    if Path(config["project_root"]).resolve() != project:
        raise ValueError("Configured project root differs from the working project")
    if config["study"] != "SC" or config["channel"] != "EEG Fpz-Cz":
        raise ValueError("Phase 2 uses SC Fpz-Cz only")
    if (config["sampling_frequency_hz"], config["epoch_seconds"], config["samples_per_epoch"]) != (100, 30, 3000):
        raise ValueError("Native 100 Hz, 30-second, 3000-sample grid required; no resampling")
    if config["storage_dtype"] != "float32" or config["storage_units"] != "microvolts":
        raise ValueError("Phase 2 storage must be float32 microvolts")
    filt = config["filter"]
    if (filt["design"] != "butterworth" or filt["representation"] != "sos" or filt["method"] != "sosfiltfilt"
            or filt["prototype_order"] != 4 or filt["padtype"] not in ("odd", "even", "constant")
            or not 0 < filt["low_hz"] < filt["high_hz"] < config["sampling_frequency_hz"] / 2
            or filt["padlen_samples"] < 1 or filt["boundary_flag_seconds"] < 0):
        raise ValueError("Invalid stable fourth-order SOS filter configuration")
    if config["quality"] != {"nonfinite_raw": "reject_recording", "nonfinite_filtered": "reject_recording",
                             "constant_raw_epoch": "reject_epoch", "adc_rail_contacts": "flag_only",
                             "amplitude_threshold_microvolts": None}:
        raise ValueError("Unsupported quality policy")
    norm = config["normalization"]
    if ({k: v for k, v in norm.items() if k != "batch_epochs"} !=
            {"method": "global_zscore", "fit_split": "train", "per_protocol": True, "ddof": 0, "apply": "dataset_loader"}
            or norm["batch_epochs"] < 1):
        raise ValueError("Unsupported normalization policy")
    if config["split"] != {"seed": 42, "train_subjects": 54, "validation_subjects": 12, "test_subjects": 12,
                            "algorithm": "numpy_default_rng_permutation_of_sorted_subject_ids"}:
        raise ValueError("Frozen subject split policy must remain unchanged")
    if (wake["protocols"]["A"]["context_epochs_before_sleep"],
            wake["protocols"]["A"]["context_epochs_after_sleep"],
            wake["protocols"]["A"]["sleep_labels"]) != (60, 60, [1, 2, 3, 4]):
        raise ValueError("Unsupported Wake trimming policy")
    output = (project / config["processed_root"]).resolve()
    if not output.is_relative_to(project / "data/processed") or output.is_relative_to(dataset.resolve()):
        raise ValueError("Generated arrays must stay inside the SSD project processed-data directory")
    code_files = ["annotations.py", "discovery.py", "edf.py", "inspection.py", "phase2_utils.py",
                  "splits.py", "trimming.py", "preprocessing.py", "normalization.py", "dataset.py", "phase2.py"]
    code = {name: sha256_file(project / "src/sleep_edf" / name) for name in code_files}
    versions = {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "mne", "PyYAML")}
    identity = {"preprocessing": config, "wake_trimming": wake, "source_sha256": code,
                "package_versions": versions}
    fingerprint = canonical_hash(identity)
    return config, wake, fingerprint, identity
