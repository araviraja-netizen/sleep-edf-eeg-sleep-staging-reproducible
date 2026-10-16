"""Atomic experiment writes, snapshots, fingerprints and frozen-test access checks."""

from contextlib import contextmanager
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import tempfile
from zoneinfo import ZoneInfo


def now():
    return datetime.now(ZoneInfo("Asia/Kolkata")).isoformat()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


@contextmanager
def atomic_file(path, binary=False, newline=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "wb" if binary else "w", **({} if binary else {"newline": newline})) as handle:
            yield handle
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def json_write(path, value):
    with atomic_file(path) as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def guard_evaluation(splits, final_evaluation=False):
    splits = set(str(value) for value in splits)
    if not splits or splits - {"train", "validation", "test"}:
        raise ValueError("Every evaluation artifact must declare train/validation/test membership")
    if "test" in splits and not final_evaluation:
        raise PermissionError("Frozen-test performance is gated; explicitly start final evaluation first")


def preserve_previous(project, create=False):
    """Hash every prior file, including processed arrays, while excluding editable docs."""
    project = Path(project).resolve()
    path = project / "results/phase2_5_preservation_snapshot.json"
    if path.exists():
        snapshot = json.loads(path.read_text())
    elif create:
        roots = ["src/sleep_edf", "data/processed", "configs", "notebooks", "scripts", "tests", "results"]
        files = [file for root in roots for file in (project / root).rglob("*") if file.is_file()
                 and "__pycache__" not in file.parts]
        # Called once before Phase 2.5 files are created. New results never enter the frozen snapshot.
        snapshot = {str(file.relative_to(project)): digest(file) for file in sorted(files)}
        json_write(path, snapshot)
    else:
        raise ValueError("Missing Phase 2.5 preservation snapshot")
    for relative, expected in snapshot.items():
        if digest(project / relative) != expected:
            raise ValueError(f"Preserved Phase 1/2 artifact changed: {relative}")
    return snapshot
