"""Describe frozen numeric flags without changing the dataset or model eligibility."""

from collections import Counter, defaultdict
import csv
import json
from pathlib import Path

import mne
import numpy as np

from sleep_edf.edf import read_header
from .common import atomic_file, digest, json_write

STAGES = ("Wake", "N1", "N2", "N3", "REM")


def review_quality(project):
    project = Path(project)
    results = project / "results"
    state = json.loads((results / "phase2_state.json").read_text())
    if digest(results / "epoch_manifest.csv") != state["manifest_sha256"]["results/epoch_manifest.csv"]:
        raise ValueError("Frozen epoch manifest changed")
    groups = {name: defaultdict(Counter) for name in ("subject", "recording", "split", "stage", "split_stage")}
    totals = {protocol: Counter() for protocol in ("A", "B")}
    examples = {}
    contacts = Counter()
    flagged_path = results / "quality_flagged_epochs.csv"
    with (results / "epoch_manifest.csv").open(newline="") as source, atomic_file(flagged_path, newline="") as target:
        fields = ["epoch_id", "subject_id", "recording_id", "night", "original_epoch_index", "onset_seconds", "label", "split",
                  "wake_trimmed_inclusion", "raw_rail_contact_samples", "filter_boundary_flag", "flag_overlap", "processed_data_location", "array_row"]
        writer = csv.DictWriter(target, fieldnames=fields)
        writer.writeheader()
        for row in csv.DictReader(source):
            rail_count = int(row["raw_rail_contact_samples"])
            rail, edge, primary = rail_count > 0, row["filter_boundary_flag"] == "1", row["wake_trimmed_inclusion"] == "1"
            stage = STAGES[int(row["label"])]
            keys = {"subject": row["subject_id"], "recording": row["recording_id"], "split": row["split"],
                    "stage": stage, "split_stage": row["split"] + ":" + stage}
            values = {"usable_epochs": 1, "rail_epochs": int(rail), "edge_epochs": int(edge),
                      "both_flags": int(rail and edge), "any_flag": int(rail or edge),
                      "rail_only": int(rail and not edge), "edge_only": int(edge and not rail),
                      "flag_free": int(not (rail or edge)), "rail_contact_samples": rail_count}
            for protocol, included in (("A", primary), ("B", True)):
                if included:
                    totals[protocol].update(values)
                    for name, key in keys.items():
                        groups[name][(protocol, key)].update(values)
            if rail or edge:
                writer.writerow({**{key: row[key] for key in fields if key != "flag_overlap"}, "flag_overlap": int(rail and edge)})
            if rail:
                contacts["1" if rail_count == 1 else "2-9" if rail_count < 10 else "10-99" if rail_count < 100 else "100+"] += 1
            if row["split"] == "train":
                if rail:
                    key = "rail_" + stage
                    if key not in examples or rail_count > int(examples[key]["raw_rail_contact_samples"]):
                        examples[key] = dict(row)
                if edge and "edge" not in examples:
                    examples["edge"] = dict(row)
                if rail and edge and "overlap" not in examples:
                    examples["overlap"] = dict(row)
    if totals["B"]["rail_epochs"] != 12020 or totals["B"]["edge_epochs"] != 301:
        raise ValueError("Flag totals differ from the Phase 2 baseline")
    metric_fields = list(totals["B"])
    for dimension, values in groups.items():
        with atomic_file(results / f"quality_by_{dimension}.csv", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["protocol", dimension, *metric_fields, "rail_percent", "edge_percent"])
            writer.writeheader()
            for (protocol, key), counts in sorted(values.items()):
                writer.writerow({"protocol": protocol, dimension: key, **counts,
                                 "rail_percent": 100 * counts["rail_epochs"] / counts["usable_epochs"],
                                 "edge_percent": 100 * counts["edge_epochs"] / counts["usable_epochs"]})
    summary = {"protocols": {p: dict(c) for p, c in totals.items()}, "rail_contact_density_epoch_counts_B": dict(contacts),
               "training_examples": examples, "dataset_unchanged": True, "exclusions_applied": False,
               "frozen_test_performance_computed": False, "phase2_pipeline_fingerprint": state["pipeline_fingerprint"]}
    json_write(results / "artifact_quality_summary.json", summary)
    return summary


def quality_waveforms(project, summary):
    import matplotlib.pyplot as plt

    project = Path(project)
    results = project / "results"
    state = json.loads((results / "phase2_state.json").read_text())
    dataset = Path(state["pipeline_identity"]["preprocessing"]["dataset_root"])
    with (results / "dataset_inventory.csv").open(newline="") as handle:
        inventory = {r["recording_id"]: r for r in csv.DictReader(handle)}
    root = results / "quality_review_figures"
    root.mkdir(exist_ok=True)
    paths = []
    for name, row in summary["training_examples"].items():
        if row["split"] != "train":
            raise PermissionError("Quality waveform examples must use training subjects")
        source = dataset / inventory[row["recording_id"]]["psg_path"]
        raw = mne.io.read_raw_edf(source, include=["EEG Fpz-Cz"], preload=False, verbose="ERROR")
        original = int(row["original_epoch_index"])
        try:
            raw_uv = raw.get_data(start=original * 3000, stop=(original + 1) * 3000)[0] * 1e6
        finally:
            raw.close()
        array = np.load(project / row["processed_data_location"], mmap_mode="r", allow_pickle=False)
        filtered_uv = array[int(row["array_row"])].copy()
        del array
        header = read_header(source)
        channel = header.labels.index("EEG Fpz-Cz")
        fig, ax = plt.subplots(figsize=(12, 4), constrained_layout=True)
        seconds = np.arange(3000) / 100
        ax.plot(seconds, raw_uv, color="grey", alpha=0.7, linewidth=0.7, label="Raw calibrated EEG")
        ax.plot(seconds, filtered_uv, linewidth=0.7, label="Frozen continuous-filtered EEG")
        for value in (header.physical_min[channel], header.physical_max[channel]):
            ax.axhline(value, color="red", linestyle="--", linewidth=0.7, alpha=0.6)
        ax.set(title=f"TRAIN {row['epoch_id']} ({STAGES[int(row['label'])]}); rail samples={row['raw_rail_contact_samples']}, edge={row['filter_boundary_flag']}",
               xlabel="Seconds within original epoch", ylabel="EEG (µV)", xlim=(0, 30))
        ax.legend(loc="upper right")
        path = root / f"{name}_{row['epoch_id'].replace(':', '_')}.png"
        fig.savefig(path, dpi=140)
        plt.close(fig)
        paths.append(str(path.relative_to(project)))
    json_write(results / "quality_review_figures.json", paths)
    return paths
