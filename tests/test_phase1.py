"""Scientific invariants, ambiguous inputs, and real-dataset smoke tests."""

from dataclasses import replace
from fractions import Fraction
from pathlib import Path
import tempfile
import unittest

from sleep_edf.annotations import ANNOTATION_TO_LABEL, Segment, annotation_label, audit_epochs
from sleep_edf.discovery import discover, parse_filename, subject_groups
from sleep_edf.edf import read_header, verify_common_origin
from sleep_edf.inspection import load_segments

DATASET = Path("/Volumes/Mac/datasets/sleep-edfx")


class FilenameTests(unittest.TestCase):
    def test_sc_identity_and_scorer(self):
        psg = parse_filename("SC4001E0-PSG.edf")
        hyp = parse_filename("SC4001EC-Hypnogram.edf")
        self.assertEqual(psg.recording_id, hyp.recording_id)
        self.assertEqual(psg.subject_id, "SC400")
        self.assertEqual(psg.night, 1)
        self.assertEqual(hyp.marker, "C")

    def test_st_identity(self):
        self.assertEqual(parse_filename("ST7012JP-Hypnogram.edf").subject_id, "ST701")

    def test_official_expanded_sc_acquisition_codes_f_and_g(self):
        self.assertEqual(parse_filename("SC4261F0-PSG.edf").recording_id,
                         parse_filename("SC4261FM-Hypnogram.edf").recording_id)
        self.assertEqual(parse_filename("SC4281G0-PSG.edf").subject_id, "SC428")

    def test_reject_malformed_or_wrong_acquisition(self):
        for name in ["SC4001J0-PSG.edf", "ST7011E0-PSG.edf", "SC4003E0-PSG.edf",
                     "SC4001EO-PSG.edf", "SC4001E0-Hypnogram.edf", "random.edf"]:
            with self.subTest(name=name), self.assertRaises(ValueError):
                parse_filename(name)

    def _files(self, root, names):
        for name in names:
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()

    def test_pairing_ignores_scorer_character(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._files(root, ["sleep-cassette/SC4011E0-PSG.edf", "sleep-cassette/SC4011EH-Hypnogram.edf"])
            pairs, issues = discover(root)
            self.assertEqual(issues, [])
            self.assertTrue(pairs[0].paired)

    def test_missing_pair_is_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._files(root, ["sleep-cassette/SC4011E0-PSG.edf"])
            pairs, issues = discover(root)
            self.assertFalse(pairs[0].paired)
            self.assertEqual(issues[0]["hypnogram_count"], 0)

    def test_orphan_hypnogram_is_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._files(root, ["sleep-cassette/SC4011EH-Hypnogram.edf"])
            pairs, issues = discover(root)
            self.assertFalse(pairs[0].paired)
            self.assertEqual(issues[0]["psg_count"], 0)

    def test_multiple_scorers_are_ambiguous(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._files(root, ["sleep-cassette/SC4011E0-PSG.edf",
                               "sleep-cassette/SC4011EH-Hypnogram.edf",
                               "sleep-cassette/SC4011EC-Hypnogram.edf"])
            pairs, issues = discover(root)
            self.assertFalse(pairs[0].paired)
            self.assertEqual(issues[0]["hypnogram_count"], 2)

    def test_duplicate_psg_is_ambiguous(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._files(root, ["sleep-cassette/SC4011E0-PSG.edf",
                               "sleep-cassette/copy/SC4011E0-PSG.edf",
                               "sleep-cassette/SC4011EC-Hypnogram.edf"])
            pairs, issues = discover(root)
            self.assertFalse(pairs[0].paired)
            self.assertEqual(issues[0]["psg_count"], 2)

    def test_unrecognized_edf_is_not_silently_ignored(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._files(root, ["bad.edf"])
            pairs, issues = discover(root)
            self.assertEqual(pairs, [])
            self.assertEqual(len(issues), 1)

    def test_nights_group_together_and_study_ids_do_not_collide(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._files(root, ["sleep-cassette/SC4011E0-PSG.edf", "sleep-cassette/SC4012E0-PSG.edf",
                               "sleep-telemetry/ST7011J0-PSG.edf", "sleep-telemetry/ST7012J0-PSG.edf"])
            groups = subject_groups(discover(root)[0])
            self.assertEqual(groups, {"SC401": ["SC4011E", "SC4012E"], "ST701": ["ST7011J", "ST7012J"]})


class AnnotationTests(unittest.TestCase):
    def test_mapping(self):
        self.assertEqual(ANNOTATION_TO_LABEL, {"Sleep stage W": 0, "Sleep stage 1": 1,
            "Sleep stage 2": 2, "Sleep stage 3": 3, "Sleep stage 4": 3, "Sleep stage R": 4})

    def test_exclusions_and_unrecognized_are_never_wake(self):
        for description in ["Movement time", "Sleep stage M", "Sleep stage ?", "", "UNKNOWN"]:
            self.assertIsNone(annotation_label(description))

    def audit(self, segments, seconds=90):
        return audit_epochs(segments, int(seconds * 100), Fraction(100))

    def test_preserve_long_wake_and_expand_multiple_epochs(self):
        result = self.audit([Segment(0, 90, "Sleep stage W")])
        self.assertEqual(result.labels, (0, 0, 0))

    def test_merge_original_n3_n4(self):
        result = self.audit([Segment(0, 30, "Sleep stage 3"), Segment(30, 30, "Sleep stage 4")], 60)
        self.assertEqual(result.labels, (3, 3))
        self.assertEqual(result.metrics["N3_epochs"], 2)

    def test_half_open_boundaries(self):
        result = self.audit([Segment(0, 30, "Sleep stage W"), Segment(30, 60, "Sleep stage R")])
        self.assertEqual(result.labels, (0, 4, 4))
        self.assertEqual(result.metrics["overlap_epochs"], 0)

    def test_never_create_out_of_signal_epochs(self):
        result = self.audit([Segment(0, 60, "Sleep stage W"), Segment(60, 86400, "Sleep stage ?")], 60)
        self.assertEqual(len(result.labels), 2)
        self.assertEqual(result.metrics["unknown_epochs"], 0)
        self.assertEqual(result.metrics["out_of_signal_annotation_segments"], 1)
        self.assertEqual(result.metrics["out_of_signal_annotation_seconds"], 86400)

    def test_supervised_overhang_keeps_only_supported_complete_epochs(self):
        result = self.audit([Segment(0, 90, "Sleep stage 2")], 65)
        self.assertEqual(result.labels, (2, 2))
        self.assertEqual(result.metrics["trailing_signal_seconds"], 5)
        self.assertEqual(result.metrics["out_of_signal_supervised_annotation_segments"], 1)

    def test_unknown_movement_and_unrecognized_have_separate_counts(self):
        result = self.audit([Segment(0, 30, "Sleep stage ?"), Segment(30, 30, "Movement time"),
                             Segment(60, 30, "Other")])
        self.assertEqual(result.labels, (None, None, None))
        self.assertEqual(result.reasons, ("unknown", "movement", "unrecognized"))
        self.assertEqual(result.metrics["annotation_missing_seconds"], 0)

    def test_missing_coverage(self):
        result = self.audit([Segment(30, 30, "Sleep stage 1")])
        self.assertEqual(result.labels, (None, 1, None))
        self.assertEqual(result.metrics["missing_epochs"], 2)
        self.assertEqual(result.metrics["annotation_missing_seconds"], 60)

    def test_overlap_invalidates_even_same_label(self):
        result = self.audit([Segment(0, 60, "Sleep stage 2"), Segment(30, 60, "Sleep stage 2")])
        self.assertEqual(result.labels, (2, None, 2))
        self.assertEqual(result.metrics["annotation_overlap_seconds"], 30)

    def test_off_grid_onset_is_rejected_without_shifting(self):
        result = self.audit([Segment(0.01, 60, "Sleep stage 2")])
        self.assertEqual(result.metrics["valid_epochs"], 0)
        self.assertEqual(result.metrics["invalid_timing_annotation_segments"], 1)
        self.assertEqual(result.metrics["invalid_timing_epochs"], 3)

    def test_off_grid_duration_rejects_entire_segment(self):
        result = self.audit([Segment(0, 45, "Sleep stage 2")])
        self.assertEqual(result.metrics["valid_epochs"], 0)
        self.assertEqual(result.metrics["invalid_timing_epochs"], 2)
        self.assertEqual(result.metrics["missing_epochs"], 1)

    def test_negative_onset_is_rejected(self):
        result = self.audit([Segment(-30, 90, "Sleep stage 2")])
        self.assertEqual(result.metrics["valid_epochs"], 0)
        self.assertEqual(result.metrics["out_of_signal_annotation_seconds"], 30)

    def test_nonfinite_nonpositive_annotations_count_as_malformed(self):
        result = self.audit([Segment(float("nan"), 30, "Sleep stage W"),
                             Segment(0, float("inf"), "Sleep stage W"),
                             Segment(0, 0, "Sleep stage W"), Segment(0, -30, "Sleep stage W")])
        self.assertEqual(result.metrics["invalid_timing_annotation_segments"], 4)
        self.assertEqual(result.metrics["valid_epochs"], 0)
        self.assertEqual(result.metrics["missing_epochs"], 3)

    def test_sub_epoch_recording(self):
        result = self.audit([Segment(0, 30, "Sleep stage W")], 20)
        self.assertEqual(result.labels, ())
        self.assertEqual(result.metrics["trailing_signal_seconds"], 20)

    def test_sample_rate_must_support_exact_epoch_samples(self):
        with self.assertRaises(ValueError):
            audit_epochs([], 1000, Fraction(100, 7))
        self.assertEqual(audit_epochs([], 1000, Fraction(100, 3)).metrics["samples_per_epoch"], 1000)

    def test_label_and_rejection_counts_partition_complete_epochs(self):
        result = self.audit([Segment(0, 30, "Sleep stage W"), Segment(30, 30, "Sleep stage ?")])
        reasons = ("valid", "missing", "overlap", "invalid_timing", "partial_annotation",
                   "movement", "unknown", "unrecognized")
        self.assertEqual(sum(result.metrics[f"{r}_epochs"] for r in reasons), len(result.labels))


@unittest.skipUnless(DATASET.is_dir(), "Configured dataset missing; real-data checks unavailable")
class DatasetIntegrationTests(unittest.TestCase):
    def test_all_actual_files_pair_and_subject_nights_are_unique(self):
        pairs, issues = discover(DATASET)
        self.assertEqual(issues, [])
        self.assertTrue(pairs)
        self.assertTrue(all(p.paired for p in pairs))
        keys = [(p.identity.subject_id, p.identity.night) for p in pairs]
        self.assertEqual(len(keys), len(set(keys)))

    def test_all_headers_native_rates_origin_size_and_channels(self):
        for pair in discover(DATASET)[0]:
            with self.subTest(recording=pair.identity.recording_id):
                psg, hyp = read_header(pair.psg[0]), read_header(pair.hypnogram[0])
                verify_common_origin(psg, hyp)
                self.assertTrue(psg.size_valid)
                self.assertTrue(hyp.size_valid)
                self.assertEqual(psg.rate("EEG Fpz-Cz"), 100)
                self.assertEqual(psg.rate("EEG Pz-Oz"), 100)
                self.assertGreater(psg.duration, 0)

    def test_origin_mismatch_fails_closed(self):
        pair = discover(DATASET)[0][0]
        psg, hyp = read_header(pair.psg[0]), read_header(pair.hypnogram[0])
        with self.assertRaisesRegex(ValueError, "start mismatch"):
            verify_common_origin(psg, replace(hyp, start=hyp.start.replace(second=1)))

    def test_discontinuous_psg_fails_closed(self):
        pair = discover(DATASET)[0][0]
        psg, hyp = read_header(pair.psg[0]), read_header(pair.hypnogram[0])
        with self.assertRaisesRegex(ValueError, "timekeeping"):
            verify_common_origin(replace(psg, reserved="EDF+D"), hyp)

    def test_two_sc_nights_timing_and_no_overhang_epochs(self):
        selected = [pair for pair in discover(DATASET)[0] if pair.identity.recording_id in ("SC4001E", "SC4002E")]
        self.assertEqual(len(selected), 2)
        self.assertEqual({p.identity.subject_id for p in selected}, {"SC400"})
        for pair in selected:
            header = read_header(pair.psg[0])
            rate = header.rate("EEG Fpz-Cz")
            samples = header.records * header.samples_per_record[header.labels.index("EEG Fpz-Cz")]
            audit = audit_epochs(load_segments(pair.psg[0], pair.hypnogram[0]), samples, rate)
            self.assertEqual(audit.metrics["invalid_timing_annotation_segments"], 0)
            self.assertEqual(audit.metrics["missing_epochs"], 0)
            self.assertEqual(audit.metrics["overlap_epochs"], 0)
            self.assertEqual(len(audit.labels), int(header.duration // 30))
            self.assertGreater(audit.metrics["Wake_epochs"], 0)
            self.assertGreater(audit.metrics["out_of_signal_annotation_segments"], 0)


if __name__ == "__main__":
    unittest.main()
