"""Incremental recording metadata, annotation audit and bounded EEG inspection."""

from collections import Counter
import hashlib
from pathlib import Path

from .annotations import Segment, audit_epochs
from .discovery import RecordingPair
from .edf import read_header, verify_common_origin

INITIAL_CHANNEL = "EEG Fpz-Cz"
FUTURE_CHANNEL = "EEG Pz-Oz"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_segments(psg_path: Path, hypnogram_path: Path) -> list[Segment]:
    """Compare headers and EDF+ first TAL before reading relative annotations."""
    import mne

    psg, hyp = read_header(psg_path), read_header(hypnogram_path)
    verify_common_origin(psg, hyp)
    if not psg.size_valid or not hyp.size_valid:
        raise ValueError("EDF file size does not match header")
    # The single annotations channel begins at the end of the EDF header.
    with hypnogram_path.open("rb") as handle:
        handle.seek(hyp.header_bytes)
        first_tal = handle.read(min(256, hyp.samples_per_record[0] * 2)).split(b"\x00", 1)[0]
    fields = first_tal.split(b"\x14")
    if len(fields) < 3 or fields[1] != b"" or fields[0] not in (b"+0", b"+0.0"):
        raise ValueError("Nonzero/unsupported EDF+ timekeeping origin; no timestamp shifts applied")
    annotations = mne.read_annotations(hypnogram_path)
    if annotations.orig_time is not None:
        raise ValueError("Unexpected absolute MNE annotation origin; explicit review required")
    return [Segment(float(onset), float(duration), str(description))
            for onset, duration, description in zip(annotations.onset, annotations.duration, annotations.description)]


def inspect_pair(pair: RecordingPair, root: Path) -> dict:
    identity = pair.identity
    row = {"recording_id": identity.recording_id, "study": identity.study,
           "subject_id": identity.subject_id, "night": identity.night,
           "psg_count": len(pair.psg), "hypnogram_count": len(pair.hypnogram),
           "psg_path": "|".join(str(p.relative_to(root)) for p in pair.psg),
           "hypnogram_path": "|".join(str(p.relative_to(root)) for p in pair.hypnogram),
           "paired": pair.paired, "inspection_status": "unpaired", "inspection_error": ""}
    if not pair.paired:
        return row
    try:
        psg, hyp = read_header(pair.psg[0]), read_header(pair.hypnogram[0])
        rates = {label: float(psg.rate(label)) for label in psg.labels}
        row.update({"psg_start_local": psg.start.isoformat(), "hypnogram_start_local": hyp.start.isoformat(),
                    "start_times_match": psg.start == hyp.start,
                    "psg_size_valid": psg.size_valid, "hypnogram_size_valid": hyp.size_valid,
                    "psg_bytes": psg.file_bytes, "hypnogram_bytes": hyp.file_bytes,
                    "channel_names": list(psg.labels), "native_sample_rates_hz": rates,
                    "channel_units": dict(zip(psg.labels, psg.units)),
                    "acquisition_prefilters": dict(zip(psg.labels, psg.prefilters)),
                    "fpz_cz_available": INITIAL_CHANNEL in psg.labels,
                    "pz_oz_available": FUTURE_CHANNEL in psg.labels,
                    "signal_duration_seconds": float(psg.duration),
                    "hypnogram_scorer": pair.hypnogram[0].name[7]})
        if INITIAL_CHANNEL not in psg.labels:
            raise ValueError("Initial EEG Fpz-Cz channel is missing")
        rate = psg.rate(INITIAL_CHANNEL)
        samples = psg.records * psg.samples_per_record[psg.labels.index(INITIAL_CHANNEL)]
        row.update({"fpz_cz_sample_rate_hz": float(rate), "fpz_cz_n_samples": samples,
                    "pz_oz_sample_rate_hz": rates.get(FUTURE_CHANNEL)})
        segments = load_segments(pair.psg[0], pair.hypnogram[0])
        row.update(audit_epochs(segments, samples, rate).metrics)
        row["inspection_status"] = "ok"
    except (OSError, ValueError, RuntimeError) as exc:
        row["inspection_status"] = "error"
        row["inspection_error"] = f"{type(exc).__name__}: {exc}"
    return row


def inspect_signal(path: Path, chunk_seconds: int = 600) -> dict:
    """Scan only Fpz-Cz, without filtering/resampling; aggregate numeric quality.

    MNE's EDF calibration returns volts. Moment sums use float64. This is a
    numeric sanity check, not an expert artifact or physiological quality assessment.
    """
    import mne
    import numpy as np

    header = read_header(path)
    rate = header.rate(INITIAL_CHANNEL)
    raw = mne.io.read_raw_edf(path, include=[INITIAL_CHANNEL], preload=False, verbose="ERROR")
    try:
        native_samples = header.records * header.samples_per_record[header.labels.index(INITIAL_CHANNEL)]
        if raw.n_times != native_samples or raw.info["sfreq"] != float(rate):
            raise ValueError("MNE/native header EEG sample grid mismatch")
        total = finite_count = equal_adjacent = 0
        sum_uv = sum_squares_uv = 0.0
        minimum, maximum = float("inf"), float("-inf")
        previous = None
        chunk_samples = int(chunk_seconds * rate)
        if chunk_samples <= 0:
            raise ValueError("chunk_seconds must be positive")
        for start in range(0, raw.n_times, chunk_samples):
            values = raw.get_data(start=start, stop=min(raw.n_times, start + chunk_samples))[0] * 1e6
            mask = np.isfinite(values)
            finite = values[mask]
            total += len(values)
            finite_count += len(finite)
            if len(finite):
                minimum = min(minimum, float(finite.min()))
                maximum = max(maximum, float(finite.max()))
                sum_uv += float(finite.sum())
                sum_squares_uv += float(np.dot(finite, finite))
            equal_adjacent += int(np.count_nonzero(values[1:] == values[:-1]))
            if previous is not None:
                equal_adjacent += int(previous == values[0])
            previous = values[-1]
        mean = sum_uv / finite_count if finite_count else None
        std = float(np.sqrt(max(0, sum_squares_uv / finite_count - mean ** 2))) if finite_count else None
        return {"samples_scanned": total, "nonfinite_samples": total - finite_count,
                "min_uv": minimum if finite_count else None, "max_uv": maximum if finite_count else None,
                "mean_uv": mean, "std_uv": std, "equal_adjacent_sample_pairs": equal_adjacent,
                "chunk_seconds": chunk_seconds, "channel": INITIAL_CHANNEL,
                "quality_scope": "Numeric checks only; no physiological/artifact screening"}
    finally:
        raw.close()


def study_summary(rows: list[dict]) -> dict:
    from .annotations import STAGE_NAMES

    result = {}
    for study in ("SC", "ST"):
        selected = [row for row in rows if row["study"] == study]
        ok = [row for row in selected if row["inspection_status"] == "ok"]
        counts = {name: sum(row[f"{name}_epochs"] for row in ok) for name in STAGE_NAMES}
        valid = sum(counts.values())
        metrics = {key: sum(row[key] for row in ok) for key in (
            "complete_signal_epochs", "valid_epochs", "missing_epochs", "overlap_epochs",
            "invalid_timing_epochs", "partial_annotation_epochs", "movement_epochs", "unknown_epochs",
            "unrecognized_epochs", "invalid_timing_annotation_segments", "unknown_annotation_segments",
            "movement_annotation_segments", "unrecognized_annotation_segments",
            "out_of_signal_annotation_segments", "out_of_signal_supervised_annotation_segments",
            "out_of_signal_annotation_seconds", "annotation_missing_seconds", "annotation_overlap_seconds",
            "trailing_signal_seconds")}
        nights = Counter(row["subject_id"] for row in selected)
        result[study] = {"recordings": len(selected), "psg_files": sum(r["psg_count"] for r in selected),
                         "hypnogram_files": sum(r["hypnogram_count"] for r in selected),
                         "subjects": len(nights), "paired_recordings": sum(r["paired"] for r in selected),
                         "inspected_ok": len(ok), "inspection_errors": len(selected) - len(ok),
                         "subjects_with_two_nights": sum(n == 2 for n in nights.values()),
                         "subjects_with_one_night": sum(n == 1 for n in nights.values()),
                         "class_epoch_counts": counts,
                         "class_percent_of_valid": {name: (100 * count / valid if valid else 0)
                                                    for name, count in counts.items()},
                         "fpz_cz_available_recordings": sum(r.get("fpz_cz_available", False) for r in selected),
                         "pz_oz_available_recordings": sum(r.get("pz_oz_available", False) for r in selected),
                         **metrics}
    return result
