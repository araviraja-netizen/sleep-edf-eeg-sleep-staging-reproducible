"""Seeded subject-independent splits with immutable, hash-checked test membership."""

import csv
import json
from pathlib import Path

import numpy as np

from .inspection import sha256_file
from .phase2_utils import atomic_writer, atomic_json, timestamp

SPLIT_NAMES = ("train", "validation", "test")


def assign_subjects(subjects, seed=42, sizes=(54, 12, 12)):
    subjects = sorted(subjects)
    if len(subjects) != len(set(subjects)) or len(subjects) != sum(sizes):
        raise ValueError("Expected unique subjects matching split sizes")
    shuffled = np.random.default_rng(seed).permutation(subjects)
    assignments = {}
    offset = 0
    for split, size in zip(SPLIT_NAMES, sizes):
        for subject in shuffled[offset:offset + size]:
            assignments[str(subject)] = split
        offset += size
    return {subject: assignments[subject] for subject in subjects}


def check_assignments(assignments, recording_rows):
    if set(assignments.values()) - set(SPLIT_NAMES):
        raise ValueError("Unknown split name")
    if any(row["study"] != "SC" for row in recording_rows):
        raise ValueError("ST cannot enter SC splits")
    if set(assignments) != {r["subject_id"] for r in recording_rows}:
        raise ValueError("Split subjects differ from SC inventory")
    sets = {name: {s for s, split in assignments.items() if split == name} for name in SPLIT_NAMES}
    if any(sets[a] & sets[b] for i, a in enumerate(SPLIT_NAMES) for b in SPLIT_NAMES[i + 1:]):
        raise ValueError("Subject overlap")
    return {name: len(values) for name, values in sets.items()}


def freeze_splits(results, recording_rows):
    results = Path(results)
    results.mkdir(parents=True, exist_ok=True)
    subjects = sorted({r["subject_id"] for r in recording_rows})
    if len(subjects) != 78 or len(recording_rows) != 153:
        raise ValueError("Expected the verified 78-subject, 153-recording SC cohort")
    subjects_path = results / "subject_splits.csv"
    recordings_path = results / "recording_splits.csv"
    frozen_path = results / "subject_split_freeze.json"
    expected = assign_subjects(subjects)
    if frozen_path.exists():
        frozen = json.loads(frozen_path.read_text())
        if (not subjects_path.exists() or not recordings_path.exists()
                or sha256_file(subjects_path) != frozen["subject_manifest_sha256"]
                or sha256_file(recordings_path) != frozen["recording_manifest_sha256"]):
            raise ValueError("Frozen split manifest changed or disappeared; refusing to redefine test set")
        with subjects_path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        assignments = {r["subject_id"]: r["split"] for r in rows}
        if len(rows) != 78 or assignments != expected:
            raise ValueError("Frozen assignment differs from seed-42 policy/cohort")
        with recordings_path.open(newline="") as handle:
            saved = list(csv.DictReader(handle))
        actual = {(r["recording_id"], r["subject_id"], str(r["night"]), assignments[r["subject_id"]])
                  for r in recording_rows}
        if len(saved) != 153 or {(r["recording_id"], r["subject_id"], r["night"], r["split"]) for r in saved} != actual:
            raise ValueError("Frozen recording assignments differ from SC inventory")
        check_assignments(assignments, recording_rows)
        return assignments, frozen
    if subjects_path.exists() or recordings_path.exists():
        raise ValueError("Unfrozen partial split artifacts exist; refusing to overwrite assignments")
    counts = check_assignments(expected, recording_rows)
    with atomic_writer(subjects_path, newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["dataset", "study", "subject_id", "split", "seed"])
        writer.writeheader()
        for subject, split in expected.items():
            writer.writerow(dict(dataset="Sleep-EDF Expanded", study="SC", subject_id=subject, split=split, seed=42))
    with atomic_writer(recordings_path, newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["dataset", "study", "subject_id", "recording_id", "night", "split"])
        writer.writeheader()
        for row in sorted(recording_rows, key=lambda r: r["recording_id"]):
            writer.writerow(dict(dataset="Sleep-EDF Expanded", study="SC", subject_id=row["subject_id"],
                                 recording_id=row["recording_id"], night=row["night"], split=expected[row["subject_id"]]))
    frozen = {"version": "sc-subject-split-v1", "seed": 42, "subject_counts": counts,
              "subject_manifest_sha256": sha256_file(subjects_path),
              "recording_manifest_sha256": sha256_file(recordings_path),
              "test_subjects": sorted(s for s, split in expected.items() if split == "test"),
              "created_at": timestamp(), "policy": "Frozen for all future models; no test-performance optimization"}
    atomic_json(frozen_path, frozen)
    return expected, frozen
