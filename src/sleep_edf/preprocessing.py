"""Continuous SOS preprocessing and atomic, resumable, per-recording epoch storage."""

import csv
import json
import os
from pathlib import Path
import resource
import shutil
import sys
import tempfile
import time

import mne
import numpy as np
from scipy import signal

from .annotations import audit_epochs, STAGE_NAMES
from .edf import read_header
from .inspection import load_segments, sha256_file
from .phase2_utils import atomic_json, timestamp
from .trimming import select_wake_protocols

META_FIELDS = ["array_row", "original_epoch_index", "onset_seconds", "label",
               "wake_trimmed_inclusion", "full_recording_inclusion", "raw_rail_contact_samples",
               "filter_boundary_flag"]


def peak_rss_bytes():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def design_filter(config):
    filt = config["filter"]
    return signal.butter(filt["prototype_order"], [filt["low_hz"], filt["high_hz"]],
                         btype="bandpass", fs=config["sampling_frequency_hz"], output="sos")


def filter_continuous(values, config):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("Continuous EEG must be one finite 1D signal; no nonfinite imputation")
    padlen = config["filter"]["padlen_samples"]
    if len(values) <= padlen + 1:
        raise ValueError("Continuous recording is too short for specified filtering padding")
    output = signal.sosfiltfilt(design_filter(config), values, padtype=config["filter"]["padtype"], padlen=padlen)
    if not np.isfinite(output).all():
        raise ValueError("Filtered EEG contains nonfinite values")
    return output


def frequency_response(config):
    sos = design_filter(config)
    hz = np.array([0.05, 0.1, 0.3, 0.5, 1, 5, 10, 20, 30, 35, 40, 45, 49])
    frequencies, response = signal.freqz_sos(sos, worN=hz, fs=config["sampling_frequency_hz"])
    return {"sos": sos.tolist(), "prototype_order": config["filter"]["prototype_order"],
            "one_pass_bandpass_order": 2 * config["filter"]["prototype_order"],
            "forward_backward_effective_order": 4 * config["filter"]["prototype_order"],
            "padtype": config["filter"]["padtype"], "padlen_samples": config["filter"]["padlen_samples"],
            "frequencies_hz": frequencies.tolist(),
            "one_pass_gain_db": (20 * np.log10(np.maximum(np.abs(response), 1e-20))).tolist(),
            "zero_phase_gain_db": (40 * np.log10(np.maximum(np.abs(response), 1e-20))).tolist()}


def signal_stats(values):
    return {"samples": len(values), "nonfinite_samples": int(np.count_nonzero(~np.isfinite(values))),
            "min_uv": float(np.min(values)), "max_uv": float(np.max(values)),
            "mean_uv": float(np.mean(values, dtype=np.float64)),
            "std_uv": float(np.std(values, dtype=np.float64))}


def valid_signal_labels(labels, raw_epochs):
    """Reject exact constant raw epochs while preserving the original label grid."""
    raw_epochs = np.asarray(raw_epochs)
    if raw_epochs.ndim != 2 or len(labels) != len(raw_epochs) or not np.isfinite(raw_epochs).all():
        raise ValueError("Signal/annotation grid mismatch or nonfinite raw samples")
    constants = np.ptp(raw_epochs, axis=1) == 0
    return [None if constants[k] else label for k, label in enumerate(labels)], constants


def verify_completion(directory, fingerprint, source_hashes=None, hashes=True):
    """Validate bytes, row/label alignment and sample finiteness before reuse."""
    directory = Path(directory)
    completion_path = directory / "completion.json"
    if not completion_path.exists():
        raise ValueError(f"Incomplete recording output: {directory}")
    completion = json.loads(completion_path.read_text())
    if completion["pipeline_fingerprint"] != fingerprint or completion["status"] != "complete":
        raise ValueError("Completion fingerprint/status mismatch")
    if source_hashes is not None and completion["source_sha256"] != source_hashes:
        raise ValueError("Completion source hashes differ from current raw recordings")
    for name in ("epochs.npy", "epoch_metadata.csv"):
        path = directory / name
        if not path.is_file() or path.stat().st_size != completion["output_bytes"][name]:
            raise ValueError(f"Missing/truncated processed artifact: {path}")
        if hashes and sha256_file(path) != completion["output_sha256"][name]:
            raise ValueError(f"Processed artifact checksum mismatch: {path}")
    array = np.load(directory / "epochs.npy", mmap_mode="r", allow_pickle=False)
    if array.dtype != np.dtype("float32") or array.shape != (completion["usable_full_epochs"], 3000):
        raise ValueError("Processed array dtype/shape mismatch")
    with (directory / "epoch_metadata.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != len(array):
        raise ValueError("Metadata/sample count mismatch")
    previous = -1
    trimmed = 0
    for i, row in enumerate(rows):
        original = int(row["original_epoch_index"])
        if (int(row["array_row"]) != i or original <= previous or float(row["onset_seconds"]) != original * 30
                or int(row["label"]) not in range(5) or row["full_recording_inclusion"] != "1"
                or row["wake_trimmed_inclusion"] not in ("0", "1")
                or (original + 1) * 30 > completion["signal_duration_seconds"]):
            raise ValueError("Invalid stored epoch metadata/alignment")
        previous = original
        trimmed += int(row["wake_trimmed_inclusion"])
    if trimmed != completion["usable_trimmed_epochs"]:
        raise ValueError("Stored trim inclusion count differs from completion")
    if hashes:
        for start in range(0, len(array), 128):
            if not np.isfinite(array[start:start + 128]).all():
                raise ValueError("Stored processed signal contains nonfinite values")
    del array
    return completion


def process_recording(pair, inventory_row, config, wake_config, fingerprint, processed_root):
    if pair.identity.study != "SC" or not pair.paired or inventory_row["inspection_status"] != "ok":
        raise ValueError("Only verified, paired SC recordings can be processed")
    tick = time.monotonic()
    psg, hyp = pair.psg[0], pair.hypnogram[0]
    source_hashes = {"psg": sha256_file(psg), "hypnogram": sha256_file(hyp)}
    if (source_hashes["psg"] != inventory_row["psg_sha256"]
            or source_hashes["hypnogram"] != inventory_row["hypnogram_sha256"]):
        raise ValueError("Raw source no longer matches verified Phase 1 SHA-256")
    final = Path(processed_root) / "recordings" / pair.identity.recording_id
    if final.exists():
        return verify_completion(final, fingerprint, source_hashes), True
    header = read_header(psg)
    rate = header.rate(config["channel"])
    if rate != config["sampling_frequency_hz"] or header.units[header.labels.index(config["channel"])] != "uV":
        raise ValueError("Expected native 100 Hz EEG with microvolt calibration; no resampling/unit guesses")
    n_samples = header.records * header.samples_per_record[header.labels.index(config["channel"])]
    audit = audit_epochs(load_segments(psg, hyp), n_samples, rate)
    if audit.metrics["valid_epochs"] != int(inventory_row["valid_epochs"]):
        raise ValueError("Current annotation audit differs from saved Phase 1 count")
    raw = mne.io.read_raw_edf(psg, include=[config["channel"]], preload=False, verbose="ERROR")
    values = np.empty(n_samples, dtype=np.float64)
    try:
        if raw.n_times != n_samples or raw.info["sfreq"] != float(rate):
            raise ValueError("MNE/native EEG grid mismatch")
        chunk_samples = int(config["read_chunk_seconds"] * rate)
        if chunk_samples <= 0:
            raise ValueError("Read chunk must be positive")
        for start in range(0, n_samples, chunk_samples):
            stop = min(n_samples, start + chunk_samples)
            values[start:stop] = raw.get_data(start=start, stop=stop)[0] * 1e6
    finally:
        raw.close()
    nonfinite = int(np.count_nonzero(~np.isfinite(values)))
    if nonfinite:
        raise ValueError(f"{nonfinite} nonfinite raw samples; whole recording rejected, no interpolation")
    n_complete = len(audit.labels)
    samples_per_epoch = config["samples_per_epoch"]
    raw_epochs = values[:n_complete * samples_per_epoch].reshape(n_complete, samples_per_epoch)
    labels, constants = valid_signal_labels(audit.labels, raw_epochs)
    rail_contacts = np.zeros(n_complete, dtype=np.int32)
    channel_index = header.labels.index(config["channel"])
    lower, upper = header.physical_min[channel_index], header.physical_max[channel_index]
    for start in range(0, n_complete, 128):
        batch = raw_epochs[start:start + 128]
        rail_contacts[start:start + 128] = np.count_nonzero(
            np.isclose(batch, lower, atol=1e-8, rtol=0) | np.isclose(batch, upper, atol=1e-8, rtol=0), axis=1)
    wake = wake_config["protocols"]["A"]
    selection = select_wake_protocols(labels, wake["context_epochs_before_sleep"], wake["context_epochs_after_sleep"])
    indices = np.flatnonzero(selection.full)
    if not len(indices):
        raise ValueError("No supervised epochs remain after signal-quality checks")
    raw_summary = signal_stats(values)
    # Apply one forward/backward pass over the full continuous finite recording,
    # including unknown/movement intervals. Annotation exclusions do not break EEG continuity.
    filtered = filter_continuous(values, config)
    filtered_summary = signal_stats(filtered)
    filtered_epochs = filtered[:n_complete * samples_per_epoch].reshape(n_complete, samples_per_epoch)
    parent = final.parent
    parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{pair.identity.recording_id}.tmp-", dir=parent))
    try:
        target = temporary / "epochs.npy"
        array = np.lib.format.open_memmap(target, mode="w+", dtype=np.float32, shape=(len(indices), samples_per_epoch))
        for start in range(0, len(indices), 128):
            array[start:start + 128] = filtered_epochs[indices[start:start + 128]].astype(np.float32)
        array.flush()
        del array
        with target.open("r+b") as handle:
            os.fsync(handle.fileno())
        with (temporary / "epoch_metadata.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=META_FIELDS)
            writer.writeheader()
            for row, original in enumerate(indices):
                onset, end = int(original) * 30, (int(original) + 1) * 30
                boundary = onset < config["filter"]["boundary_flag_seconds"] or end > float(header.duration) - config["filter"]["boundary_flag_seconds"]
                writer.writerow(dict(array_row=row, original_epoch_index=int(original), onset_seconds=onset,
                                     label=labels[original], wake_trimmed_inclusion=int(selection.trimmed[original]),
                                     full_recording_inclusion=1, raw_rail_contact_samples=int(rail_contacts[original]),
                                     filter_boundary_flag=int(boundary)))
            handle.flush()
            os.fsync(handle.fileno())
        full_counts = {name: sum(labels[k] == label for k in indices) for label, name in enumerate(STAGE_NAMES)}
        trimmed_indices = np.flatnonzero(selection.trimmed)
        trimmed_counts = {name: sum(labels[k] == label for k in trimmed_indices) for label, name in enumerate(STAGE_NAMES)}
        completion = {"status": "complete", "recording_id": pair.identity.recording_id,
                      "subject_id": pair.identity.subject_id, "night": pair.identity.night,
                      "study": "SC", "dataset": config["dataset"], "preprocessing_version": config["version"],
                      "pipeline_fingerprint": fingerprint, "source_sha256": source_hashes,
                      "storage_units": "microvolts", "storage_dtype": "float32", "stored_normalized": False,
                      "signal_duration_seconds": float(header.duration), "native_n_samples": n_samples,
                      "annotation_audit": audit.metrics,
                      "constant_raw_epochs_all": int(constants.sum()),
                      "constant_raw_supervised_epochs_rejected": sum(label is not None and constants[k]
                                                                     for k, label in enumerate(audit.labels)),
                      "raw_rail_contact_samples": int(rail_contacts.sum()),
                      "usable_epochs_with_rail_contacts": int(np.count_nonzero(rail_contacts[indices])),
                      "raw_signal": raw_summary, "filtered_signal": filtered_summary,
                      "usable_full_epochs": len(indices), "usable_trimmed_epochs": len(trimmed_indices),
                      "full_class_counts": full_counts, "trimmed_class_counts": trimmed_counts,
                      "wake_epochs_removed": full_counts["Wake"] - trimmed_counts["Wake"],
                      "first_valid_sleep_epoch": selection.first_sleep, "last_valid_sleep_epoch": selection.last_sleep,
                      "trim_start_epoch": selection.start_epoch, "trim_stop_epoch_exclusive": selection.stop_epoch_exclusive,
                      "no_valid_sleep": selection.no_valid_sleep,
                      "acquisition_prefilter": header.prefilters[channel_index],
                      "output_sha256": {name: sha256_file(temporary / name) for name in ("epochs.npy", "epoch_metadata.csv")},
                      "output_bytes": {name: (temporary / name).stat().st_size for name in ("epochs.npy", "epoch_metadata.csv")},
                      "completed_at": timestamp(), "elapsed_seconds": time.monotonic() - tick,
                      "process_peak_rss_bytes": peak_rss_bytes()}
        # Convert NumPy counters to native ints before JSON serialization.
        completion["constant_raw_supervised_epochs_rejected"] = int(completion["constant_raw_supervised_epochs_rejected"])
        atomic_json(temporary / "completion.json", completion)
        verify_completion(temporary, fingerprint, source_hashes)
        os.replace(temporary, final)  # Publish the complete directory only after validation.
        return completion, False
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
