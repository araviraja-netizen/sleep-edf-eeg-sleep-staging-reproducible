"""Raw/continuous-filtered EEG, PSD, response and original-grid Wake protocol plots."""

import csv
import json
from pathlib import Path

import mne
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

from .annotations import audit_epochs, STAGE_NAMES
from .edf import read_header
from .inspection import load_segments


def plot_filter_response(config):
    from .preprocessing import design_filter
    frequencies, h = signal.freqz_sos(design_filter(config), worN=10000, fs=100)
    fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
    ax.plot(frequencies, 20 * np.log10(np.maximum(np.abs(h), 1e-15)), label="One pass")
    ax.plot(frequencies, 40 * np.log10(np.maximum(np.abs(h), 1e-15)), label="Forward + backward (used)")
    for hz in (config["filter"]["low_hz"], config["filter"]["high_hz"]):
        ax.axvline(hz, color="grey", linestyle="--", linewidth=0.8)
    ax.set(xlabel="Frequency (Hz)", ylabel="Amplitude gain (dB)", ylim=(-100, 2), xlim=(0, 50),
           title="Butterworth N=4 band-pass in SOS form; zero-phase response is squared")
    ax.legend()
    return fig


def plot_preprocessing_recording(project, recording_id):
    project = Path(project)
    state = json.loads((project / "results/phase2_state.json").read_text())
    identity = state["pipeline_identity"]["preprocessing"]
    directory = project / state["processed_root"] / "recordings" / recording_id
    completion = json.loads((directory / "completion.json").read_text())
    with (project / "results/dataset_inventory.csv").open(newline="") as handle:
        inventory = next(row for row in csv.DictReader(handle) if row["recording_id"] == recording_id)
    with (directory / "epoch_metadata.csv").open(newline="") as handle:
        metadata = list(csv.DictReader(handle))
    selected = next((row for row in metadata if int(row["label"]) > 0), metadata[0])
    original = int(selected["original_epoch_index"])
    psg = Path(identity["dataset_root"]) / inventory["psg_path"]
    hyp = Path(identity["dataset_root"]) / inventory["hypnogram_path"]
    raw = mne.io.read_raw_edf(psg, include=["EEG Fpz-Cz"], preload=False, verbose="ERROR")
    try:
        raw_uv = raw.get_data(start=original * 3000, stop=(original + 1) * 3000)[0] * 1e6
    finally:
        raw.close()
    stored = np.load(directory / "epochs.npy", mmap_mode="r")
    filtered_uv = stored[int(selected["array_row"])].copy()
    del stored
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), constrained_layout=True)
    time = np.arange(3000) / 100
    axes[0].plot(time, raw_uv, color="grey", alpha=0.6, linewidth=0.6, label="Raw calibrated EEG")
    axes[0].plot(time, filtered_uv, linewidth=0.7, label="Continuous recording filtered before extraction")
    axes[0].set(xlabel="Seconds within epoch", ylabel="EEG (µV)", xlim=(0, 30),
                title=f"{recording_id}, original epoch {original}, {STAGE_NAMES[int(selected['label'])]}")
    axes[0].legend()
    for values, name in ((raw_uv, "Raw"), (filtered_uv, "Filtered")):
        frequencies, psd = signal.welch(values, fs=100, nperseg=1000, noverlap=500, detrend="constant", scaling="density")
        axes[1].semilogy(frequencies, np.maximum(psd, 1e-15), label=name)
    axes[1].set(xlabel="Frequency (Hz)", ylabel="PSD (µV²/Hz)", xlim=(0, 50),
                title="Welch PSD of the same EEG epoch (10 s Hann windows, 50% overlap)")
    axes[1].legend()
    header = read_header(psg)
    audit = audit_epochs(load_segments(psg, hyp), completion["native_n_samples"], header.rate("EEG Fpz-Cz"))
    original_labels = np.array([np.nan if label is None else label for label in audit.labels])
    trimmed_labels = np.full_like(original_labels, np.nan)
    for row in metadata:
        if row["wake_trimmed_inclusion"] == "1":
            trimmed_labels[int(row["original_epoch_index"])] = int(row["label"])
    positions = np.array([0, 2, 3, 4, 1])
    hyp_fig, hyp_axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True, constrained_layout=True)
    for ax, labels, name in zip(hyp_axes, (original_labels, trimmed_labels),
                                ("Original annotations, all Wake", "Protocol A: 30-minute Wake context, original timeline")):
        display_labels = np.full_like(labels, np.nan)
        good = np.isfinite(labels)
        display_labels[good] = positions[labels[good].astype(int)]
        ax.stairs(display_labels, np.arange(len(labels) + 1) * 30 / 3600, linewidth=0.8)
        if completion["trim_start_epoch"] is not None:
            for boundary in (completion["trim_start_epoch"], completion["trim_stop_epoch_exclusive"]):
                ax.axvline(boundary * 30 / 3600, color="orange", linestyle="--", linewidth=0.8)
        ax.set(title=name, ylabel="Stage", yticks=range(5), yticklabels=["Wake", "REM", "N1", "N2", "N3"],
               ylim=(4.5, -0.5), xlim=(0, completion["signal_duration_seconds"] / 3600))
    hyp_axes[-1].set_xlabel("Hours from original PSG start (indices never shifted)")
    return fig, hyp_fig
