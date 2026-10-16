"""Full SC integrity, normalization and dataset-loading checks after the pilot gate."""

from collections import Counter
import csv
import json
from pathlib import Path
import unittest

import numpy as np

from sleep_edf.annotations import audit_epochs, STAGE_NAMES
from sleep_edf.dataset import EpochDataset
from sleep_edf.discovery import discover
from sleep_edf.edf import read_header
from sleep_edf.inspection import load_segments, sha256_file
from sleep_edf.normalization import RunningMoments, normalize_epoch
from sleep_edf.phase2 import preserve_phase1
from sleep_edf.phase2_utils import atomic_json
from sleep_edf.preprocessing import verify_completion
from sleep_edf.trimming import select_wake_protocols

PROJECT = Path(__file__).resolve().parents[1]


class FullDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.state = json.loads((PROJECT / "results/phase2_state.json").read_text())
        cls.root = PROJECT / cls.state["processed_root"]
        with (PROJECT / "results/subject_splits.csv").open(newline="") as handle:
            cls.assignments = {r["subject_id"]: r["split"] for r in csv.DictReader(handle)}

    def test_phase1_preserved_and_no_models_or_st_arrays(self):
        preserve_phase1(PROJECT)
        self.assertEqual(self.state["status"], "complete")
        directories = sorted(p.name for p in (self.root / "recordings").iterdir() if p.is_dir() and not p.name.startswith("."))
        self.assertEqual(len(directories), 153)
        self.assertTrue(all(name.startswith("SC") for name in directories))
        self.assertEqual([p.name for p in (PROJECT / "checkpoints").iterdir()], [".gitkeep"])

    def test_frozen_subject_splits_and_both_nights(self):
        groups = {name: {s for s, split in self.assignments.items() if split == name} for name in ("train", "validation", "test")}
        self.assertEqual([len(groups[n]) for n in ("train", "validation", "test")], [54, 12, 12])
        self.assertFalse(groups["train"] & groups["validation"] or groups["train"] & groups["test"] or groups["validation"] & groups["test"])
        frozen = json.loads((PROJECT / "results/subject_split_freeze.json").read_text())
        self.assertEqual(sorted(groups["test"]), frozen["test_subjects"])
        for relative, digest in self.state["manifest_sha256"].items():
            self.assertEqual(sha256_file(PROJECT / relative), digest)
        with (PROJECT / "results/recording_splits.csv").open(newline="") as handle:
            recordings = list(csv.DictReader(handle))
        self.assertEqual(len(recordings), 153)
        self.assertTrue(all(self.assignments[r["subject_id"]] == r["split"] for r in recordings))

    def test_all_completion_checksums_metadata_labels_and_trim_boundaries(self):
        dataset = Path(self.state["pipeline_identity"]["preprocessing"]["dataset_root"])
        pairs = {p.identity.recording_id: p for p in discover(dataset)[0] if p.identity.study == "SC"}
        actual = {protocol: Counter() for protocol in ("A", "B")}
        for recording in self.state["recording_ids"]:
            directory = self.root / "recordings" / recording
            completion = verify_completion(directory, self.state["pipeline_fingerprint"])
            pair = pairs[recording]
            header = read_header(pair.psg[0])
            audit = audit_epochs(load_segments(pair.psg[0], pair.hypnogram[0]), completion["native_n_samples"], header.rate("EEG Fpz-Cz"))
            with (directory / "epoch_metadata.csv").open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            usable_labels = [None] * len(audit.labels)
            for row in rows:
                original, label = int(row["original_epoch_index"]), int(row["label"])
                self.assertEqual(audit.labels[original], label, (recording, original))
                usable_labels[original] = label
                actual["B"][label] += 1
                if row["wake_trimmed_inclusion"] == "1":
                    actual["A"][label] += 1
            selection = select_wake_protocols(usable_labels)
            self.assertEqual(completion["trim_start_epoch"], selection.start_epoch)
            self.assertEqual(completion["trim_stop_epoch_exclusive"], selection.stop_epoch_exclusive)
            for row in rows:
                original = int(row["original_epoch_index"])
                self.assertEqual(row["wake_trimmed_inclusion"] == "1", bool(selection.trimmed[original]))
                if row["wake_trimmed_inclusion"] == "0":
                    self.assertEqual(int(row["label"]), 0)
        for protocol in ("A", "B"):
            self.assertEqual({name: actual[protocol][label] for label, name in enumerate(STAGE_NAMES)}, self.state["counts"][protocol]["all"])

    def test_consolidated_epoch_manifest_alignment_and_quality_flags(self):
        expected_total = sum(self.state["counts"]["B"]["all"].values())
        count = trimmed = rail_epochs = boundary_epochs = 0
        previous_recording, previous_index = "", -1
        with (PROJECT / "results/epoch_manifest.csv").open(newline="") as handle:
            for row in csv.DictReader(handle):
                recording, original = row["recording_id"], int(row["original_epoch_index"])
                if recording != previous_recording:
                    self.assertGreater(recording, previous_recording)
                    previous_recording, previous_index = recording, -1
                self.assertGreater(original, previous_index)
                previous_index = original
                self.assertEqual(row["epoch_id"], f"{recording}:e{original:06d}")
                self.assertEqual(int(row["onset_seconds"]), original * 30)
                self.assertEqual(row["split"], self.assignments[row["subject_id"]])
                self.assertEqual(row["study"], "SC")
                self.assertEqual(row["pipeline_fingerprint"], self.state["pipeline_fingerprint"])
                self.assertIn(int(row["label"]), range(5))
                self.assertEqual(row["full_recording_inclusion"], "1")
                self.assertTrue((PROJECT / row["processed_data_location"]).is_file())
                count += 1
                trimmed += int(row["wake_trimmed_inclusion"])
                rail_epochs += int(int(row["raw_rail_contact_samples"]) > 0)
                boundary_epochs += int(row["filter_boundary_flag"])
        self.assertEqual(count, expected_total)
        self.assertEqual(trimmed, sum(self.state["counts"]["A"]["all"].values()))
        self.assertEqual(rail_epochs, self.state["quality"]["usable_epochs_with_rail_contacts"])
        atomic_json(PROJECT / "results/phase2_manifest_validation.json",
                    {"epoch_rows": count, "trimmed_rows": trimmed, "unique_epoch_ids": True,
                     "sample_label_alignment": True, "subject_overlap": 0,
                     "epochs_with_raw_rail_contacts": rail_epochs, "recording_edge_flagged_epochs": boundary_epochs})

    def test_normalized_training_population_and_no_fit_leakage(self):
        normalization = json.loads((PROJECT / "results/normalization.json").read_text())
        moments = {"A": RunningMoments(), "B": RunningMoments()}
        for recording in self.state["recording_ids"]:
            directory = self.root / "recordings" / recording
            completion = json.loads((directory / "completion.json").read_text())
            if self.assignments[completion["subject_id"]] != "train":
                continue
            values = np.load(directory / "epochs.npy", mmap_mode="r")
            with (directory / "epoch_metadata.csv").open(newline="") as handle:
                flags = np.array([r["wake_trimmed_inclusion"] == "1" for r in csv.DictReader(handle)])
            for start in range(0, len(values), 128):
                batch = values[start:start + 128]
                moments["B"].update(normalize_epoch(batch, normalization["protocols"]["B"]))
                moments["A"].update(normalize_epoch(batch[flags[start:start + 128]], normalization["protocols"]["A"]))
            del values
        checks = {}
        for protocol in ("A", "B"):
            params = normalization["protocols"][protocol]
            self.assertTrue(all(self.assignments[s] == "train" for s in params["training_subjects"]))
            count = sum(self.state["counts"][protocol]["train"].values())
            self.assertEqual(params["sample_count"], count * 3000)
            post = moments[protocol].parameters()
            self.assertAlmostEqual(post["mean_microvolts"], 0, places=6)
            self.assertAlmostEqual(post["std_microvolts"], 1, places=6)
            checks[protocol] = {"normalized_training_mean": post["mean_microvolts"],
                                "normalized_training_population_std": post["std_microvolts"],
                                "training_sample_count": post["sample_count"], "validation_fit_samples": 0, "test_fit_samples": 0}
        atomic_json(PROJECT / "results/phase2_normalization_validation.json", checks)

    def test_loader_all_protocols_and_splits_preserve_metadata_and_normalization(self):
        for protocol in ("A", "B"):
            for split in ("train", "validation", "test"):
                dataset = EpochDataset(PROJECT, protocol, split, max_open_recordings=1)
                self.assertEqual(len(dataset), sum(self.state["counts"][protocol][split].values()))
                for index in (0, len(dataset) // 2, len(dataset) - 1):
                    row = dataset[index]
                    self.assertEqual(row["split"], split)
                    self.assertEqual(self.assignments[row["subject_id"]], split)
                    self.assertEqual(row["eeg"].shape, (1, 3000))
                    self.assertEqual(row["eeg"].dtype, np.float32)
                    self.assertTrue(np.isfinite(row["eeg"]).all())
                    self.assertEqual(row["epoch_id"], f"{row['recording_id']}:e{row['original_epoch_index']:06d}")
                    self.assertEqual(row["onset_seconds"], row["original_epoch_index"] * 30)
                self.assertLessEqual(len(dataset._cache), 1)
                dataset.close()
