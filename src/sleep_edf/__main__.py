"""Run the reproducible Phase 1 inventory; writes only inside the project."""

import argparse
import csv
from datetime import datetime
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import time
import tomllib
from zoneinfo import ZoneInfo

from .discovery import discover, subject_groups
from .inspection import inspect_pair, inspect_signal, sha256_file, study_summary


def write_json(path: Path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def run(config_path: Path) -> int:
    config = tomllib.loads(config_path.read_text())
    dataset = Path(config["dataset_root"])
    python = Path(config["python_executable"])
    # Environment checks precede any output creation.
    if not dataset.is_dir() or not python.is_file() or not os.access(python, os.X_OK):
        raise SystemExit("STOP: configured dataset or executable Python is missing")
    if Path(sys.executable).resolve() != python.resolve():
        raise SystemExit(f"STOP: use {python}, not {sys.executable}")
    if (config["epoch_seconds"] != 30 or config["initial_channel"] != "EEG Fpz-Cz"
            or config["future_channel"] != "EEG Pz-Oz" or config["training_study"] != "SC"
            or config["retain_all_wake"] is not True):
        raise SystemExit("STOP: config violates Phase 1 protocol")
    project = Path(config["project_root"]).resolve()
    if project == dataset.resolve() or project.is_relative_to(dataset.resolve()):
        raise SystemExit("STOP: project output must be outside the raw dataset")
    results = project / "results"
    results.mkdir(parents=True, exist_ok=True)
    started = datetime.now(ZoneInfo("Asia/Kolkata")).isoformat()
    tick = time.monotonic()
    pairs, issues = discover(dataset)
    print(f"Discovered {len(pairs)} recording identities, {len(issues)} discovery issues", flush=True)
    checksum_reference = {}
    reference_path = dataset / "SHA256SUMS.txt"
    if reference_path.exists():
        for line in reference_path.read_text().splitlines():
            digest, relative = line.split(maxsplit=1)
            checksum_reference[relative.strip().lstrip("*")] = digest
    rows, file_manifest = [], []
    for index, pair in enumerate(pairs, 1):
        row = inspect_pair(pair, dataset)
        for kind, paths in (("psg", pair.psg), ("hypnogram", pair.hypnogram)):
            for path in paths:
                relative = str(path.relative_to(dataset))
                stat = path.stat()
                digest = sha256_file(path) if config["verify_checksums"] else None
                expected = checksum_reference.get(relative)
                match = digest == expected if digest and expected else None
                file_manifest.append({"path": relative, "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns,
                                      "sha256": digest, "reference_sha256": expected, "checksum_matches": match})
                if len(paths) == 1:
                    row[f"{kind}_sha256"] = digest
                    row[f"{kind}_checksum_matches"] = match
        rows.append(row)
        if index % 20 == 0 or index == len(pairs):
            print(f"Inspected and hashed {index}/{len(pairs)} recordings", flush=True)
    indexed = {entry["path"] for entry in file_manifest}
    for path in sorted(dataset.rglob("*")):
        if not path.is_file() or str(path.relative_to(dataset)) in indexed:
            continue
        relative = str(path.relative_to(dataset))
        stat = path.stat()
        digest = sha256_file(path) if config["verify_checksums"] else None
        expected = checksum_reference.get(relative)
        file_manifest.append({"path": relative, "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns,
                              "sha256": digest, "reference_sha256": expected,
                              "checksum_matches": digest == expected if digest and expected else None})
    manifest_keys = {entry["path"] for entry in file_manifest}
    checksum_missing = sorted(set(checksum_reference) - manifest_keys)
    checksum_mismatch = [entry["path"] for entry in file_manifest if entry["checksum_matches"] is False]
    checksum_unverified = [entry["path"] for entry in file_manifest
                           if entry["checksum_matches"] is None and entry["path"] != "SHA256SUMS.txt"]
    records_path = dataset / "RECORDS"
    listed_records = set(records_path.read_text().splitlines()) if records_path.exists() else set()
    actual_records = {str(path.relative_to(dataset)) for pair in pairs for path in pair.psg}
    detailed = {}
    by_id = {pair.identity.recording_id: pair for pair in pairs}
    for recording_id in config["inspection_recordings"]:
        pair = by_id.get(recording_id)
        if pair is None or not pair.paired or pair.identity.study != "SC":
            raise ValueError(f"Detailed SC inspection recording missing/ambiguous: {recording_id}")
        print(f"Scanning native Fpz-Cz EEG incrementally: {recording_id}", flush=True)
        detailed[recording_id] = inspect_signal(pair.psg[0], config["signal_chunk_seconds"])
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (results / "dataset_inventory.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value
                             for key, value in row.items()})
    summary = study_summary(rows)
    write_json(results / "dataset_summary.json", summary)
    write_json(results / "detailed_inspection.json", detailed)
    write_json(results / "subject_groups.json", subject_groups(pairs))
    write_json(results / "discovery_issues.json", issues)
    write_json(results / "raw_file_manifest.json", sorted(file_manifest, key=lambda item: item["path"]))
    packages = {dist.metadata["Name"]: dist.version for dist in importlib.metadata.distributions()}
    (results / "environment_observed.txt").write_text(
        "# Observed existing environment; reference only, not an installation command.\n"
        + "\n".join(f"{name}=={version}" for name, version in sorted(packages.items(), key=lambda item: item[0].lower())) + "\n")
    mps = {}
    try:
        import torch
        mps = {"torch_version": torch.__version__, "built": torch.backends.mps.is_built(),
               "available": torch.backends.mps.is_available()}
    except ImportError:
        mps = {"available": False, "reason": "torch not installed"}
    source_paths = [*project.glob("src/**/*.py"), *project.glob("tests/**/*.py"),
                    *project.glob("scripts/**/*.py"), config_path.resolve(),
                    project / "README.md", project / "AGENTS.md", project / "pyproject.toml", project / ".gitignore"]
    source_hashes = {str(path.relative_to(project)): sha256_file(path)
                     for path in sorted(set(source_paths)) if path.exists() and path.is_relative_to(project)}
    provenance = {"started_at_Asia_Kolkata": started,
                  "completed_at_Asia_Kolkata": datetime.now(ZoneInfo("Asia/Kolkata")).isoformat(),
                  "elapsed_seconds": time.monotonic() - tick, "python_executable": sys.executable,
                  "python_version": sys.version, "platform": platform.platform(), "config": config,
                  "source_sha256": source_hashes, "mps": mps,
                  "dataset_source": "https://physionet.org/content/sleep-edfx/1.0.0/",
                  "edf_source": "https://www.edfplus.info/specs/edf.html",
                  "edfplus_source": "https://www.edfplus.info/specs/edfplus.html",
                  "checksum_reference_sha256": sha256_file(reference_path) if reference_path.exists() else None,
                  "checksum_verified_files": sum(entry["checksum_matches"] is True for entry in file_manifest),
                  "checksum_mismatch": checksum_mismatch, "checksum_missing_files": checksum_missing,
                  "checksum_unverified_files": checksum_unverified,
                  "RECORDS_missing_psgs": sorted(listed_records - actual_records),
                  "RECORDS_unlisted_psgs": sorted(actual_records - listed_records),
                  "raw_write_operations": 0, "wake_trimming": False, "training": False}
    write_json(results / "provenance.json", provenance)
    print(json.dumps(summary, indent=2), flush=True)
    errors = (issues or checksum_missing or checksum_mismatch or checksum_unverified
              or provenance["RECORDS_missing_psgs"] or provenance["RECORDS_unlisted_psgs"]
              or any(row["inspection_status"] != "ok" for row in rows))
    return 1 if errors else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/phase1.toml"))
    raise SystemExit(run(parser.parse_args().config))


if __name__ == "__main__":
    main()
