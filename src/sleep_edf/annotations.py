"""Five-class mapping and conservative, signal-bounded 30-second epoch audit."""

from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
import math

STAGE_NAMES = ("Wake", "N1", "N2", "N3", "REM")
ANNOTATION_TO_LABEL = {
    "Sleep stage W": 0, "Sleep stage 1": 1, "Sleep stage 2": 2,
    "Sleep stage 3": 3, "Sleep stage 4": 3, "Sleep stage R": 4,
}
MOVEMENT = {"Movement time", "Sleep stage M"}
UNKNOWN = {"Sleep stage ?"}
EPOCH_SECONDS = 30
TOLERANCE_SECONDS = 1e-7  # Floating serialization tolerance, never a sample/timing shift.


def annotation_label(description: str) -> int | None:
    return ANNOTATION_TO_LABEL.get(description)


@dataclass(frozen=True)
class Segment:
    onset: float
    duration: float
    description: str


@dataclass(frozen=True)
class EpochAudit:
    labels: tuple[int | None, ...]
    reasons: tuple[str, ...]
    metrics: dict


def _on_grid(value: float) -> bool:
    return abs(value - round(value / EPOCH_SECONDS) * EPOCH_SECONDS) <= TOLERANCE_SECONDS


def audit_epochs(segments: list[Segment], n_samples: int, sfreq: Fraction | float) -> EpochAudit:
    """Never attach annotations to MNE Raw: that can crop evidence of overhang.

    All complete PSG-grid epochs are audited. A supervised label requires a single
    nonnegative, grid-aligned segment covering the entire epoch. Any intersecting
    competing segment invalidates it, even if both have the same stage.
    Malformed segments are counted and their finite positive intervals are rejected.
    """
    rate = Fraction(str(sfreq)) if not isinstance(sfreq, Fraction) else sfreq
    if n_samples < 0 or rate <= 0 or (rate * EPOCH_SECONDS).denominator != 1:
        raise ValueError("30 seconds must contain an exact integer number of native EEG samples")
    samples_per_epoch = int(rate * EPOCH_SECONDS)
    count = n_samples // samples_per_epoch
    duration = float(Fraction(n_samples) / rate)
    claims = [[] for _ in range(count)]
    intervals = []
    metrics = Counter()
    metrics["annotation_segments"] = len(segments)
    descriptions = Counter(s.description for s in segments)
    for segment in segments:
        start, length = segment.onset, segment.duration
        finite = math.isfinite(start) and math.isfinite(length) and math.isfinite(start + length)
        positive = finite and length > 0
        end = start + length if finite else float("nan")
        aligned = positive and start >= 0 and _on_grid(start) and _on_grid(end)
        if not aligned:
            metrics["invalid_timing_annotation_segments"] += 1
        if segment.description in MOVEMENT:
            metrics["movement_annotation_segments"] += 1
        elif segment.description in UNKNOWN:
            metrics["unknown_annotation_segments"] += 1
        elif annotation_label(segment.description) is None:
            metrics["unrecognized_annotation_segments"] += 1
        if not positive:
            continue
        if start < 0 or end > duration + TOLERANCE_SECONDS:
            metrics["out_of_signal_annotation_segments"] += 1
            metrics["out_of_signal_annotation_seconds"] += max(0, min(end, 0) - start) + max(0, end - max(start, duration))
            if annotation_label(segment.description) is not None:
                metrics["out_of_signal_supervised_annotation_segments"] += 1
        left, right = max(0.0, start), min(duration, end)
        if right <= left:
            continue
        intervals.append((left, right))
        first = max(0, math.floor(start / EPOCH_SECONDS))
        last = min(count, math.ceil(end / EPOCH_SECONDS))
        for k in range(first, last):
            ep_start, ep_end = k * EPOCH_SECONDS, (k + 1) * EPOCH_SECONDS
            if end <= ep_start + TOLERANCE_SECONDS or start >= ep_end - TOLERANCE_SECONDS:
                continue
            full = start <= ep_start + TOLERANCE_SECONDS and end >= ep_end - TOLERANCE_SECONDS
            claims[k].append((segment.description, aligned, full))
    labels, reasons = [], []
    for contenders in claims:
        label = None
        if not contenders:
            reason = "missing"
        elif len(contenders) > 1:
            reason = "overlap"
        else:
            description, aligned, full = contenders[0]
            if not aligned:
                reason = "invalid_timing"
            elif not full:
                reason = "partial_annotation"
            elif description in MOVEMENT:
                reason = "movement"
            elif description in UNKNOWN:
                reason = "unknown"
            elif description not in ANNOTATION_TO_LABEL:
                reason = "unrecognized"
            else:
                reason = "valid"
                label = annotation_label(description)
        labels.append(label)
        reasons.append(reason)
    covered = overlap = 0.0
    previous_end = 0.0
    for start, end in sorted(intervals):
        covered += max(0, end - max(start, previous_end))
        overlap += max(0, min(end, previous_end) - start)
        previous_end = max(previous_end, end)
    reason_counts = Counter(reasons)
    for reason in ("valid", "missing", "overlap", "invalid_timing", "partial_annotation",
                   "movement", "unknown", "unrecognized"):
        metrics[f"{reason}_epochs"] = reason_counts[reason]
    for key in ("invalid_timing_annotation_segments", "movement_annotation_segments",
                "unknown_annotation_segments", "unrecognized_annotation_segments",
                "out_of_signal_annotation_segments", "out_of_signal_annotation_seconds",
                "out_of_signal_supervised_annotation_segments"):
        metrics.setdefault(key, 0)
    metrics = dict(metrics)  # Remaining entries include structured/non-numeric metadata.
    metrics.update({"complete_signal_epochs": count, "samples_per_epoch": samples_per_epoch,
                    "signal_duration_seconds": duration,
                    "trailing_signal_seconds": float(Fraction(n_samples % samples_per_epoch) / rate),
                    "annotation_covered_seconds": covered,
                    "annotation_missing_seconds": max(0, duration - covered),
                    "annotation_overlap_seconds": overlap,
                    "annotation_first_onset": min((s.onset for s in segments if math.isfinite(s.onset)), default=None),
                    "annotation_last_end": max((s.onset + s.duration for s in segments
                                                if math.isfinite(s.onset + s.duration)), default=None),
                    "description_counts": dict(sorted(descriptions.items()))})
    label_counts = Counter(label for label in labels if label is not None)
    for label, name in enumerate(STAGE_NAMES):
        metrics[f"{name}_epochs"] = label_counts[label]
    return EpochAudit(tuple(labels), tuple(reasons), dict(metrics))
