"""Phase 2 pilot/full orchestration with Phase 1 preservation and resumable outputs."""

from collections import Counter, defaultdict
import csv
import fcntl
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

from .annotations import STAGE_NAMES
from .discovery import discover
from .inspection import sha256_file
from .normalization import fit_normalization
from .phase2_utils import load_config, atomic_json, atomic_writer, timestamp
from .preprocessing import process_recording, verify_completion, frequency_response, peak_rss_bytes
from .splits import freeze_splits, check_assignments

INDEX_FIELDS = ["epoch_id", "subject_id", "recording_id", "night", "original_epoch_index",
                "array_row", "onset_seconds", "label", "split"]
EPOCH_FIELDS = ["epoch_id", "dataset", "study", "subject_id", "recording_id", "night",
                "original_epoch_index", "onset_seconds", "label", "split", "wake_trimmed_inclusion",
                "full_recording_inclusion", "processed_data_location", "array_row", "storage_units",
                "preprocessing_version", "pipeline_fingerprint", "raw_rail_contact_samples", "filter_boundary_flag"]


def preserve_phase1(project):
    results = project / "results"
    snapshot_path = results / "phase2_phase1_snapshot.json"
    if snapshot_path.exists():
        snapshot = json.loads(snapshot_path.read_text())
    else:
        original = json.loads((results / "artifact_manifest.json").read_text())
        snapshot = {name: digest for name, digest in original.items()
                    if name.startswith("results/") or name == "notebooks/01_dataset_inspection.ipynb"}
        snapshot["results/artifact_manifest.json"] = sha256_file(results / "artifact_manifest.json")
    for relative, digest in snapshot.items():
        if sha256_file(project / relative) != digest:
            raise ValueError(f"Phase 1 output changed: {relative}")
    if not snapshot_path.exists():
        atomic_json(snapshot_path, snapshot)
    return snapshot


def load_inventory(project):
    with (project / "results/dataset_inventory.csv").open(newline="") as handle:
        all_rows = list(csv.DictReader(handle))
    sc_rows = [row for row in all_rows if row["study"] == "SC"]
    if len(sc_rows) != 153 or any(row["inspection_status"] != "ok" for row in sc_rows):
        raise ValueError("Phase 1 SC inventory is not complete/valid")
    return sc_rows


def consolidate(project, config, fingerprint, identity, processed_root, completions, assignments, frozen, exceptions, run_info):
    results = project / "results"
    fields_per_recording = ["recording_id", "subject_id", "night", "split", "original_annotation_valid_epochs",
                           "usable_full_epochs", "usable_trimmed_epochs", "wake_epochs_removed",
                           "constant_raw_supervised_epochs_rejected", "raw_rail_contact_samples",
                           "usable_epochs_with_rail_contacts", "first_valid_sleep_epoch", "last_valid_sleep_epoch",
                           "trim_start_epoch", "trim_stop_epoch_exclusive", "no_valid_sleep",
                           "raw_min_uv", "raw_max_uv", "raw_mean_uv", "raw_std_uv", "filtered_min_uv",
                           "filtered_max_uv", "filtered_mean_uv", "filtered_std_uv", "process_peak_rss_bytes"]
    counts = {protocol: {split: Counter() for split in ("all", "train", "validation", "test")}
              for protocol in ("original_annotation_valid", "A", "B")}
    subject_counts = defaultdict(Counter)
    quality = Counter()
    # Multi-artifact publication ends with phase2_state.json as the authoritative commit.
    with atomic_writer(results / "epoch_manifest.csv", newline="") as epoch_handle, \
            atomic_writer(results / "protocol_a_index.csv", newline="") as a_handle, \
            atomic_writer(results / "protocol_b_index.csv", newline="") as b_handle, \
            atomic_writer(results / "recording_preprocessing_summary.csv", newline="") as record_handle:
        epoch_writer = csv.DictWriter(epoch_handle, fieldnames=EPOCH_FIELDS)
        indexes = {"A": csv.DictWriter(a_handle, fieldnames=INDEX_FIELDS), "B": csv.DictWriter(b_handle, fieldnames=INDEX_FIELDS)}
        record_writer = csv.DictWriter(record_handle, fieldnames=fields_per_recording)
        epoch_writer.writeheader()
        record_writer.writeheader()
        for writer in indexes.values():
            writer.writeheader()
        for completion in completions:
            subject, recording = completion["subject_id"], completion["recording_id"]
            split = assignments[subject]
            for label, stage in enumerate(STAGE_NAMES):
                count = completion["annotation_audit"][f"{stage}_epochs"]
                for group in ("all", split):
                    counts["original_annotation_valid"][group][label] += count
                subject_counts[(subject, "original_annotation_valid")][label] += count
            directory = Path(processed_root) / "recordings" / recording
            location = str((directory / "epochs.npy").relative_to(project))
            with (directory / "epoch_metadata.csv").open(newline="") as handle:
                for meta in csv.DictReader(handle):
                    label = int(meta["label"])
                    original = int(meta["original_epoch_index"])
                    row = dict(epoch_id=f"{recording}:e{original:06d}", dataset=config["dataset"], study="SC",
                               subject_id=subject, recording_id=recording, night=completion["night"],
                               original_epoch_index=original, onset_seconds=original * 30, label=label, split=split,
                               wake_trimmed_inclusion=meta["wake_trimmed_inclusion"], full_recording_inclusion=1,
                               processed_data_location=location, array_row=meta["array_row"], storage_units="microvolts",
                               preprocessing_version=config["version"], pipeline_fingerprint=fingerprint,
                               raw_rail_contact_samples=meta["raw_rail_contact_samples"], filter_boundary_flag=meta["filter_boundary_flag"])
                    epoch_writer.writerow(row)
                    for protocol, included in (("A", meta["wake_trimmed_inclusion"] == "1"), ("B", True)):
                        if included:
                            indexes[protocol].writerow({field: row[field] for field in INDEX_FIELDS})
                            for group in ("all", split):
                                counts[protocol][group][label] += 1
                            subject_counts[(subject, protocol)][label] += 1
            rec = {key: completion.get(key) for key in fields_per_recording}
            rec.update(split=split, original_annotation_valid_epochs=completion["annotation_audit"]["valid_epochs"])
            for kind in ("raw", "filtered"):
                for metric in ("min_uv", "max_uv", "mean_uv", "std_uv"):
                    rec[f"{kind}_{metric}"] = completion[f"{kind}_signal"][metric]
            record_writer.writerow(rec)
            for key in ("constant_raw_supervised_epochs_rejected", "constant_raw_epochs_all", "raw_rail_contact_samples",
                        "usable_epochs_with_rail_contacts"):
                quality[key] += completion[key]
            for key in ("movement_epochs", "unknown_epochs", "missing_epochs", "overlap_epochs", "invalid_timing_epochs",
                        "unrecognized_epochs", "out_of_signal_annotation_segments", "out_of_signal_supervised_annotation_segments"):
                quality[key] += completion["annotation_audit"][key]
    with atomic_writer(results / "stage_distribution_comparison.csv", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["protocol", "split", "label", "stage", "epoch_count", "percent_of_protocol_split"])
        writer.writeheader()
        for protocol, splits in counts.items():
            for split, counter in splits.items():
                total = sum(counter.values())
                for label, stage in enumerate(STAGE_NAMES):
                    writer.writerow(dict(protocol=protocol, split=split, label=label, stage=stage, epoch_count=counter[label],
                                         percent_of_protocol_split=100 * counter[label] / total if total else 0))
    with atomic_writer(results / "per_subject_class_counts.csv", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["subject_id", "split", "protocol", "label", "stage", "epoch_count"])
        writer.writeheader()
        for subject in sorted(assignments):
            for protocol in counts:
                for label, stage in enumerate(STAGE_NAMES):
                    writer.writerow(dict(subject_id=subject, split=assignments[subject], protocol=protocol, label=label,
                                         stage=stage, epoch_count=subject_counts[(subject, protocol)][label]))
    with atomic_writer(results / "per_subject_epoch_counts.csv", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["subject_id", "split", "original_annotation_valid_epochs", "full_epochs", "trimmed_epochs"])
        writer.writeheader()
        for subject in sorted(assignments):
            writer.writerow(dict(subject_id=subject, split=assignments[subject],
                                 original_annotation_valid_epochs=sum(subject_counts[(subject, "original_annotation_valid")].values()),
                                 full_epochs=sum(subject_counts[(subject, "B")].values()),
                                 trimmed_epochs=sum(subject_counts[(subject, "A")].values())))
    normalization = fit_normalization(completions, assignments, processed_root, frozen["subject_manifest_sha256"], fingerprint,
                                      results / "normalization.json", config["normalization"]["batch_epochs"])
    leakage = {"subject_counts": check_assignments(assignments, load_inventory(project)), "subject_overlap": 0,
               "recording_split_conflicts": 0, "protocols_share_assignments": True,
               "split_manifest_sha256": frozen["subject_manifest_sha256"], "frozen_test_subjects": frozen["test_subjects"],
               "normalization_fit_split": "train", "normalization_validation_subjects": 0, "normalization_test_subjects": 0,
               "no_test_performance_used": True, "no_class_balancing": True, "training_performed": False}
    for protocol in ("A", "B"):
        params = normalization["protocols"][protocol]
        if any(assignments[subject] != "train" for subject in params["training_subjects"]):
            raise ValueError("Normalization subject leakage")
        if params["epoch_count"] != sum(counts[protocol]["train"].values()):
            raise ValueError("Normalization epoch eligibility differs from training manifest")
    atomic_json(results / "phase2_leakage_checks.json", leakage)
    atomic_json(results / "filter_response.json", frequency_response(config))
    atomic_json(results / "phase2_exceptions.json", exceptions)
    state = {"status": "complete" if not exceptions and len(completions) == 153 else "partial",
             "pipeline_fingerprint": fingerprint, "processed_root": str(Path(processed_root).relative_to(project)),
             "split_manifest_sha256": frozen["subject_manifest_sha256"], "processed_recordings": len(completions),
             "recording_ids": [c["recording_id"] for c in completions],
             "counts": {protocol: {split: {stage: counter[label] for label, stage in enumerate(STAGE_NAMES)}
                                    for split, counter in splits.items()} for protocol, splits in counts.items()},
             "quality": dict(quality), "normalization": normalization, "leakage": leakage,
             "no_valid_sleep_recordings": [c["recording_id"] for c in completions if c["no_valid_sleep"]],
             "array_storage_bytes": sum(c["output_bytes"]["epochs.npy"] for c in completions),
             "processed_directory_bytes": sum(p.stat().st_size for p in Path(processed_root).rglob("*") if p.is_file()),
             "process_peak_rss_bytes": peak_rss_bytes(), "run": run_info, "completed_at": timestamp(),
             "pipeline_identity": identity}
    manifest_names = ["epoch_manifest.csv", "protocol_a_index.csv", "protocol_b_index.csv", "normalization.json",
                      "subject_splits.csv", "recording_splits.csv", "subject_split_freeze.json"]
    state["manifest_sha256"] = {f"results/{name}": sha256_file(results / name) for name in manifest_names}
    atomic_json(results / "phase2_state.json", state)
    return state


def run_phase2(project, mode):
    project = Path(project).resolve()
    config, wake, fingerprint, identity = load_config(project)
    results = project / "results"
    snapshot = preserve_phase1(project)
    rows = load_inventory(project)
    assignments, frozen = freeze_splits(results, rows)
    pairs, issues = discover(Path(config["dataset_root"]))
    if issues:
        raise ValueError(f"Dataset discovery changed: {issues}")
    by_id = {p.identity.recording_id: p for p in pairs if p.identity.study == "SC"}
    if set(by_id) != {r["recording_id"] for r in rows}:
        raise ValueError("SC discovery differs from Phase 1 inventory")
    processed_root = project / config["processed_root"] / fingerprint[:16]
    processed_root.mkdir(parents=True, exist_ok=True)
    atomic_json(processed_root / "pipeline_identity.json", identity)
    pilot_path = results / "phase2_pilot.json"
    if mode == "all":
        pilot = json.loads(pilot_path.read_text())
        if (pilot["successful"] is not True or pilot["pipeline_fingerprint"] != fingerprint
                or pilot["split_manifest_sha256"] != frozen["subject_manifest_sha256"]
                or pilot["test_results_sha256"] != sha256_file(results / "phase2_pilot_test_results.json")):
            raise ValueError("A successful current two-recording pilot with passing tests is required")
        selected = rows
    else:
        selected = [r for r in rows if r["recording_id"] in config["pilot_recordings"]]
        if len(selected) != 2 or len({r["subject_id"] for r in selected}) != 1:
            raise ValueError("Expected two pilot nights from one SC subject")
    estimate_bytes = sum(int(r["valid_epochs"]) * 3000 * 4 for r in selected)
    if shutil.disk_usage(project).free < estimate_bytes + 2 * 1024 ** 3:
        raise ValueError("Insufficient SSD space for arrays plus metadata/headroom")
    completions, exceptions = [], []
    resumed = 0
    started, tick = timestamp(), time.monotonic()
    with (processed_root / ".run.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for index, row in enumerate(selected, 1):
            recording = row["recording_id"]
            try:
                completion, cached = process_recording(by_id[recording], row, config, wake, fingerprint, processed_root)
                completions.append(completion)
                resumed += int(cached)
                print(f"[{index}/{len(selected)}] {recording}: {'validated cache' if cached else 'processed'}; "
                      f"B={completion['usable_full_epochs']}, A={completion['usable_trimmed_epochs']}", flush=True)
            except Exception as exc:
                failure = {"recording_id": recording, "at": timestamp(), "error": f"{type(exc).__name__}: {exc}",
                           "traceback": traceback.format_exc()}
                exceptions.append(failure)
                atomic_json(results / f"phase2_{mode}_exceptions.json", exceptions)
                print(f"FAILED {recording}: {failure['error']}", flush=True)
        if exceptions:
            atomic_json(results / f"phase2_{mode}_exceptions.json", exceptions)
            raise RuntimeError(f"{len(exceptions)} failed recordings; validated completed outputs are resumable")
        run_info = {"started_at": started, "elapsed_seconds": time.monotonic() - tick,
                    "resumed_recordings": resumed, "new_recordings": len(completions) - resumed,
                    "mode": mode, "command": f"{sys.executable} scripts/run_phase2.py --{mode}"}
        if mode == "pilot":
            pilot = {"successful": False, "pipeline_fingerprint": fingerprint,
                     "split_manifest_sha256": frozen["subject_manifest_sha256"],
                     "processed_root": str(processed_root.relative_to(project)),
                     "recordings": [c["recording_id"] for c in completions], "run": run_info,
                     "full_epochs": sum(c["usable_full_epochs"] for c in completions),
                     "trimmed_epochs": sum(c["usable_trimmed_epochs"] for c in completions)}
            atomic_json(pilot_path, pilot)
            subprocess.run([sys.executable, str(project / "scripts/run_phase2_tests.py"), "--pilot"], cwd=project, check=True)
            pilot.update(successful=True, test_results_sha256=sha256_file(results / "phase2_pilot_test_results.json"))
            atomic_json(pilot_path, pilot)
            atomic_json(results / "phase2_pilot_exceptions.json", [])
            print("Two-recording pilot passed. Full SC processing is now eligible.", flush=True)
        else:
            state = consolidate(project, config, fingerprint, identity, processed_root, completions, assignments, frozen, exceptions, run_info)
            atomic_json(results / "phase2_all_exceptions.json", [])
            print(json.dumps({"recordings": state["processed_recordings"], "counts": state["counts"],
                              "array_storage_bytes": state["array_storage_bytes"]}, indent=2), flush=True)
    preserve_phase1(project)
    return completions
