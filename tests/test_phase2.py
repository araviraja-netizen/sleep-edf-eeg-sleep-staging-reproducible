"""Phase 2 scientific invariants and adversarial storage/leakage checks."""

import csv
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from sleep_edf.inspection import sha256_file
from sleep_edf.normalization import RunningMoments, fit_normalization, normalize_epoch
from sleep_edf.phase2_utils import atomic_writer, load_config
from sleep_edf.preprocessing import filter_continuous, frequency_response, verify_completion, valid_signal_labels, META_FIELDS
from sleep_edf.splits import assign_subjects, check_assignments, freeze_splits
from sleep_edf.trimming import select_wake_protocols

PROJECT = Path(__file__).resolve().parents[1]


class WakeTests(unittest.TestCase):
    def test_sixty_epochs_of_context_and_internal_wake(self):
        labels = [0] * 300
        labels[100], labels[150], labels[200] = 1, 0, 4
        result = select_wake_protocols(labels)
        self.assertEqual((result.start_epoch, result.stop_epoch_exclusive), (40, 261))
        self.assertEqual(int(result.trimmed.sum()), 221)
        self.assertTrue(result.trimmed[150])
        self.assertFalse(result.trimmed[39])
        self.assertFalse(result.trimmed[261])

    def test_invalid_between_sleep_boundaries_does_not_shift_grid(self):
        labels = [0, 1, None, 0, 2, -1, 0]
        result = select_wake_protocols(labels, before=1, after=1)
        self.assertEqual(np.flatnonzero(result.trimmed).tolist(), [0, 1, 3, 4])
        self.assertEqual(np.flatnonzero(result.full).tolist(), [0, 1, 3, 4, 6])

    def test_wake_only_is_flagged_secondary_still_keeps_wake(self):
        result = select_wake_protocols([0] * 10)
        self.assertTrue(result.no_valid_sleep)
        self.assertEqual(result.full.sum(), 10)
        self.assertEqual(result.trimmed.sum(), 0)
        self.assertIsNone(result.first_sleep)

    def test_recording_edges_cap_context(self):
        result = select_wake_protocols([1, 0, 0, 4])
        self.assertEqual((result.start_epoch, result.stop_epoch_exclusive), (0, 4))
        self.assertTrue(result.trimmed.all())

    def test_one_sleep_epoch_closed_last_boundary(self):
        labels = [0] * 140
        labels[70] = 2
        result = select_wake_protocols(labels)
        self.assertEqual((result.start_epoch, result.stop_epoch_exclusive), (10, 131))
        self.assertEqual(result.trimmed.sum(), 121)

    def test_all_invalid_and_empty(self):
        for labels in ([], [None, -1, 5, 9]):
            result = select_wake_protocols(labels)
            self.assertTrue(result.no_valid_sleep)
            self.assertEqual(result.full.sum(), 0)

    def test_nonwake_epochs_are_never_trimmed_out(self):
        labels = [0] * 100 + [1, 2, 3, 4] + [0] * 100
        result = select_wake_protocols(labels)
        self.assertTrue(result.trimmed[100:104].all())
        removed = np.flatnonzero(result.full & ~result.trimmed)
        self.assertTrue(all(labels[k] == 0 for k in removed))

    def test_negative_context_rejected(self):
        with self.assertRaises(ValueError):
            select_wake_protocols([1], before=-1)


class SplitTests(unittest.TestCase):
    def test_seeded_reproducible_54_12_12(self):
        subjects = [f"SC4{k:02d}" for k in range(78)]
        a = assign_subjects(subjects)
        b = assign_subjects(list(reversed(subjects)))
        self.assertEqual(a, b)
        self.assertEqual([list(a.values()).count(s) for s in ("train", "validation", "test")], [54, 12, 12])
        self.assertNotEqual(a, assign_subjects(subjects, seed=43))

    def test_duplicate_subjects_rejected(self):
        with self.assertRaises(ValueError):
            assign_subjects(["SC400", "SC400"], sizes=(1, 1, 0))

    def test_st_rejected(self):
        with self.assertRaises(ValueError):
            check_assignments({"ST701": "train"}, [{"study": "ST", "subject_id": "ST701"}])

    def test_missing_subject_rejected(self):
        with self.assertRaises(ValueError):
            check_assignments({"SC400": "train"}, [{"study": "SC", "subject_id": "SC401"}])

    def test_frozen_manifests_not_rewritten_and_tampering_rejected(self):
        with (PROJECT / "results/dataset_inventory.csv").open(newline="") as handle:
            rows = [r for r in csv.DictReader(handle) if r["study"] == "SC"]
        with tempfile.TemporaryDirectory(dir=PROJECT / "results") as tmp:
            root = Path(tmp)
            a, frozen = freeze_splits(root, rows)
            stat = (root / "subject_splits.csv").stat().st_mtime_ns
            b, other = freeze_splits(root, rows)
            self.assertEqual(a, b)
            self.assertEqual(frozen, other)
            self.assertEqual((root / "subject_splits.csv").stat().st_mtime_ns, stat)
            with (root / "subject_splits.csv").open("a") as handle:
                handle.write("tamper\n")
            with self.assertRaisesRegex(ValueError, "Frozen split"):
                freeze_splits(root, rows)


class FilterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_config(PROJECT)[0]

    def test_sos_order_and_squared_cutoff_gain(self):
        response = frequency_response(self.config)
        self.assertEqual(np.asarray(response["sos"]).shape, (4, 6))
        self.assertEqual(response["one_pass_bandpass_order"], 8)
        self.assertEqual(response["forward_backward_effective_order"], 16)
        for hz in (0.3, 35):
            index = response["frequencies_hz"].index(hz)
            self.assertAlmostEqual(response["zero_phase_gain_db"][index], -6.020599913, places=6)

    def test_passband_gain_and_zero_phase(self):
        time = np.arange(30000) / 100
        raw = np.sin(2 * np.pi * 10 * time)
        filtered = filter_continuous(raw, self.config)
        section = slice(2000, -2000)
        gain = np.dot(filtered[section], raw[section]) / np.dot(raw[section], raw[section])
        quadrature = 2 * np.mean(filtered[section] * np.cos(2 * np.pi * 10 * time[section]))
        self.assertAlmostEqual(gain, 1, places=3)
        self.assertLess(abs(quadrature), 1e-5)

    def test_out_of_band_attenuation(self):
        time = np.arange(60000) / 100
        for hz in (0.05, 45):
            raw = np.sin(2 * np.pi * hz * time)
            filtered = filter_continuous(raw, self.config)
            self.assertLess(np.std(filtered[10000:-10000]) / np.std(raw[10000:-10000]), 0.005)

    def test_dc_rejected(self):
        self.assertLess(np.max(np.abs(filter_continuous(np.full(12000, 27.0), self.config))), 1e-7)

    def test_nonfinite_and_too_short_rejected(self):
        for values in (np.full(100, np.nan), np.full(100, np.inf), np.ones(20)):
            with self.assertRaises(ValueError):
                filter_continuous(values, self.config)

    def test_continuous_filter_preserves_length_and_differs_from_per_epoch_filtering(self):
        rng = np.random.default_rng(42)
        raw = rng.normal(size=9000) + np.linspace(0, 20, 9000)
        continuous = filter_continuous(raw, self.config)
        segmented = np.concatenate([filter_continuous(epoch, self.config) for epoch in raw.reshape(3, 3000)])
        self.assertEqual(continuous.shape, raw.shape)
        self.assertGreater(np.max(np.abs(continuous[2990:3010] - segmented[2990:3010])), 0.05)

    def test_exact_constant_raw_epoch_excluded_without_relabeling(self):
        epochs = np.stack([np.arange(3000), np.zeros(3000), np.arange(3000)])
        labels, constants = valid_signal_labels([1, 0, None], epochs)
        self.assertEqual(labels, [1, None, None])
        self.assertEqual(constants.tolist(), [False, True, False])

    def test_invalid_raw_signal_grid_rejected(self):
        with self.assertRaises(ValueError):
            valid_signal_labels([1], np.ones((2, 3000)))
        with self.assertRaises(ValueError):
            valid_signal_labels([1], np.full((1, 3000), np.nan))


def create_storage_fixture(root, fingerprint="fixture"):
    root = Path(root)
    values = np.arange(9000, dtype=np.float32).reshape(3, 3000)
    np.save(root / "epochs.npy", values)
    with (root / "epoch_metadata.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=META_FIELDS)
        writer.writeheader()
        for i, original in enumerate([0, 2, 3]):
            writer.writerow(dict(array_row=i, original_epoch_index=original, onset_seconds=original * 30, label=i,
                                 wake_trimmed_inclusion=int(i > 0), full_recording_inclusion=1,
                                 raw_rail_contact_samples=0, filter_boundary_flag=0))
    completion = {"status": "complete", "pipeline_fingerprint": fingerprint, "source_sha256": {"psg": "a", "hypnogram": "b"},
                  "usable_full_epochs": 3, "usable_trimmed_epochs": 2, "signal_duration_seconds": 120,
                  "output_sha256": {n: sha256_file(root / n) for n in ("epochs.npy", "epoch_metadata.csv")},
                  "output_bytes": {n: (root / n).stat().st_size for n in ("epochs.npy", "epoch_metadata.csv")}}
    (root / "completion.json").write_text(json.dumps(completion))
    return completion


class StorageTests(unittest.TestCase):
    def test_memmap_reload_alignment_original_indices_not_contiguous(self):
        with tempfile.TemporaryDirectory(dir=PROJECT / "results") as tmp:
            create_storage_fixture(tmp)
            self.assertEqual(verify_completion(tmp, "fixture")["usable_full_epochs"], 3)
            values = np.load(Path(tmp) / "epochs.npy", mmap_mode="r")
            self.assertIsInstance(values, np.memmap)
            np.testing.assert_array_equal(values[1], np.arange(3000, 6000))

    def test_wrong_source_or_config_cannot_be_resumed(self):
        with tempfile.TemporaryDirectory(dir=PROJECT / "results") as tmp:
            create_storage_fixture(tmp)
            with self.assertRaisesRegex(ValueError, "fingerprint"):
                verify_completion(tmp, "changed")
            with self.assertRaisesRegex(ValueError, "source"):
                verify_completion(tmp, "fixture", {"psg": "changed"})

    def test_corrupt_processed_bytes_cannot_be_skipped(self):
        with tempfile.TemporaryDirectory(dir=PROJECT / "results") as tmp:
            create_storage_fixture(tmp)
            path = Path(tmp) / "epochs.npy"
            with path.open("r+b") as handle:
                handle.seek(-1, 2)
                handle.write(b"X")
            with self.assertRaisesRegex(ValueError, "checksum"):
                verify_completion(tmp, "fixture")

    def test_missing_completion_cannot_be_skipped(self):
        with tempfile.TemporaryDirectory(dir=PROJECT / "results") as tmp:
            np.save(Path(tmp) / "epochs.npy", np.zeros((1, 3000), dtype=np.float32))
            with self.assertRaisesRegex(ValueError, "Incomplete"):
                verify_completion(tmp, "fixture")

    def test_invalid_label_in_metadata_rejected(self):
        with tempfile.TemporaryDirectory(dir=PROJECT / "results") as tmp:
            completion = create_storage_fixture(tmp)
            path = Path(tmp) / "epoch_metadata.csv"
            with path.open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            rows[0]["label"] = "9"
            with path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=META_FIELDS)
                writer.writeheader(); writer.writerows(rows)
            completion["output_bytes"][path.name] = path.stat().st_size
            completion["output_sha256"][path.name] = sha256_file(path)
            (Path(tmp) / "completion.json").write_text(json.dumps(completion))
            with self.assertRaisesRegex(ValueError, "metadata"):
                verify_completion(tmp, "fixture")

    def test_atomic_writer_failure_preserves_original(self):
        with tempfile.TemporaryDirectory(dir=PROJECT / "results") as tmp:
            path = Path(tmp) / "test.txt"
            path.write_text("old")
            with self.assertRaises(RuntimeError):
                with atomic_writer(path) as handle:
                    handle.write("partial")
                    raise RuntimeError("simulated interruption")
            self.assertEqual(path.read_text(), "old")
            self.assertEqual(len(list(Path(tmp).iterdir())), 1)


class NormalizationTests(unittest.TestCase):
    def test_streaming_population_moments(self):
        values = np.random.default_rng(42).normal(50, 10, 10001)
        moments = RunningMoments()
        for batch in np.array_split(values, 19):
            moments.update(batch)
        params = moments.parameters()
        self.assertEqual(params["sample_count"], len(values))
        self.assertAlmostEqual(params["mean_microvolts"], values.mean(), places=12)
        self.assertAlmostEqual(params["std_microvolts"], values.std(), places=12)

    def test_float32_dimensionless_normalization(self):
        values = np.arange(3000, dtype=np.float32)
        moments = RunningMoments(); moments.update(values)
        result = normalize_epoch(values, moments.parameters())
        self.assertEqual(result.dtype, np.float32)
        self.assertAlmostEqual(float(result.mean()), 0, places=6)
        self.assertAlmostEqual(float(result.std()), 1, places=6)

    def test_validation_and_test_extremes_do_not_affect_fit(self):
        with tempfile.TemporaryDirectory(dir=PROJECT / "results") as tmp:
            root = Path(tmp)
            train = root / "recordings/SC4001E"; train.mkdir(parents=True)
            np.save(train / "epochs.npy", np.arange(6000, dtype=np.float32).reshape(2, 3000))
            with (train / "epoch_metadata.csv").open("w") as handle:
                handle.write("wake_trimmed_inclusion\n0\n1\n")
            completions = [{"subject_id": s, "recording_id": r} for s, r in
                           [("SC400", "SC4001E"), ("SC401", "SC4011E"), ("SC402", "SC4021E")]]
            assignments = {"SC400": "train", "SC401": "validation", "SC402": "test"}
            # Validation/test files deliberately do not exist: fitting must not read them.
            params = fit_normalization(completions, assignments, root, "split", "config", root / "normalization.json")
            self.assertAlmostEqual(params["protocols"]["B"]["mean_microvolts"], 2999.5)
            self.assertAlmostEqual(params["protocols"]["A"]["mean_microvolts"], 4499.5)
            self.assertEqual(params["protocols"]["A"]["training_subjects"], ["SC400"])
            self.assertEqual(params["protocols"]["B"]["sample_count"], 6000)

    def test_nonfinite_zero_variance_or_empty_fit_rejected(self):
        with self.assertRaises(ValueError):
            RunningMoments().parameters()
        moments = RunningMoments(); moments.update(np.ones(10))
        with self.assertRaises(ValueError):
            moments.parameters()
        with self.assertRaises(ValueError):
            RunningMoments().update([np.nan])
        with self.assertRaises(ValueError):
            normalize_epoch([1, 2], {"mean_microvolts": 0, "std_microvolts": 0})


if __name__ == "__main__":
    unittest.main()
