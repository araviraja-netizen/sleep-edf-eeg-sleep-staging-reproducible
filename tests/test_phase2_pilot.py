"""Integration checks required by the two-recording preprocessing gate."""

import csv
import json
from pathlib import Path
import unittest

import mne
import numpy as np

from sleep_edf.annotations import audit_epochs
from sleep_edf.discovery import discover
from sleep_edf.edf import read_header
from sleep_edf.inspection import load_segments
from sleep_edf.phase2_utils import load_config
from sleep_edf.preprocessing import filter_continuous, verify_completion, valid_signal_labels
from sleep_edf.trimming import select_wake_protocols

PROJECT = Path(__file__).resolve().parents[1]


class PilotIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config, cls.wake, cls.fingerprint, _ = load_config(PROJECT)
        cls.pilot = json.loads((PROJECT / "results/phase2_pilot.json").read_text())
        cls.root = PROJECT / cls.pilot["processed_root"]
        cls.pairs = {p.identity.recording_id: p for p in discover(Path(cls.config["dataset_root"]))[0]}

    def test_complete_validated_two_sc_nights_and_resume(self):
        self.assertEqual(self.pilot["recordings"], ["SC4001E", "SC4002E"])
        for recording in self.pilot["recordings"]:
            path = self.root / "recordings" / recording
            completion_hash = (path / "completion.json").stat().st_mtime_ns
            completion = verify_completion(path, self.fingerprint)
            self.assertEqual(completion["subject_id"], "SC400")
            self.assertEqual((path / "completion.json").stat().st_mtime_ns, completion_hash)
            self.assertEqual(completion["raw_signal"]["nonfinite_samples"], 0)
            self.assertEqual(completion["filtered_signal"]["nonfinite_samples"], 0)

    def test_samples_labels_trimming_match_raw_continuous_reference(self):
        for recording in self.pilot["recordings"]:
            pair = self.pairs[recording]
            header = read_header(pair.psg[0])
            rate = header.rate("EEG Fpz-Cz")
            raw = mne.io.read_raw_edf(pair.psg[0], include=["EEG Fpz-Cz"], preload=False, verbose="ERROR")
            try:
                values = raw.get_data()[0] * 1e6
            finally:
                raw.close()
            audit = audit_epochs(load_segments(pair.psg[0], pair.hypnogram[0]), len(values), rate)
            labels, _ = valid_signal_labels(audit.labels, values.reshape(-1, 3000))
            selection = select_wake_protocols(labels)
            reference = filter_continuous(values, self.config)
            directory = self.root / "recordings" / recording
            epochs = np.load(directory / "epochs.npy", mmap_mode="r")
            with (directory / "epoch_metadata.csv").open(newline="") as handle:
                metadata = list(csv.DictReader(handle))
            originals = [int(r["original_epoch_index"]) for r in metadata]
            self.assertEqual(originals, np.flatnonzero(selection.full).tolist())
            self.assertEqual([int(r["label"]) for r in metadata], [labels[k] for k in originals])
            self.assertEqual([r["wake_trimmed_inclusion"] == "1" for r in metadata], selection.trimmed[originals].tolist())
            for array_row in (0, len(epochs) // 2, len(epochs) - 1):
                original = originals[array_row]
                np.testing.assert_array_equal(epochs[array_row], reference[original * 3000:(original + 1) * 3000].astype(np.float32))
            if recording == "SC4002E":
                self.assertEqual(len(metadata), 2829)  # Exactly one Movement epoch excluded.
                self.assertTrue(any(b - a > 1 for a, b in zip(originals, originals[1:])))
            del epochs, reference, values
