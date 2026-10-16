"""Notebook plots from small native EEG windows and audited epoch metadata."""

from pathlib import Path

from .annotations import STAGE_NAMES, audit_epochs
from .edf import read_header
from .inspection import INITIAL_CHANNEL, load_segments


def plot_recording(psg: Path, hypnogram: Path):
    import mne
    import numpy as np
    import matplotlib.pyplot as plt

    header = read_header(psg)
    rate = header.rate(INITIAL_CHANNEL)
    samples = header.records * header.samples_per_record[header.labels.index(INITIAL_CHANNEL)]
    audit = audit_epochs(load_segments(psg, hypnogram), samples, rate)
    labels = np.array([np.nan if label is None else label for label in audit.labels])
    nonwake = np.flatnonzero(np.isfinite(labels) & (labels != 0))
    epoch = int(nonwake[0]) if len(nonwake) else 0
    start = epoch * audit.metrics["samples_per_epoch"]
    stop = start + audit.metrics["samples_per_epoch"]
    raw = mne.io.read_raw_edf(psg, include=[INITIAL_CHANNEL], preload=False, verbose="ERROR")
    try:
        if raw.n_times != samples or raw.info["sfreq"] != float(rate):
            raise ValueError("MNE/native EEG sample grid mismatch")
        volts = raw.get_data(start=start, stop=stop)[0]
    finally:
        raw.close()
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), constrained_layout=True)
    axes[0].plot(np.arange(len(volts)) / float(rate), volts * 1e6, linewidth=0.6)
    stage = STAGE_NAMES[int(labels[epoch])] if np.isfinite(labels[epoch]) else "Excluded/uncovered"
    axes[0].set(title=f"{psg.name}: EEG Fpz-Cz, epoch {epoch}, {stage}, onset {epoch * 30}s",
                xlabel="Seconds within 30-second epoch", ylabel="EEG (µV)", xlim=(0, 30))
    # Plot conventional sleep-depth order; map explicit labels without changing labels.
    positions = np.array([0, 2, 3, 4, 1])  # Wake, REM, N1, N2, N3 on display axis.
    plotted = np.full_like(labels, np.nan)
    good = np.isfinite(labels)
    plotted[good] = positions[labels[good].astype(int)]
    for k in range(len(labels)):
        if not good[k]:
            axes[1].axvspan(k * 30 / 3600, (k + 1) * 30 / 3600, color="grey", alpha=0.3)
    axes[1].stairs(plotted, np.arange(len(labels) + 1) * 30 / 3600, linewidth=0.7)
    axes[1].set(title="Full PSG: all valid Wake retained; grey marks excluded/uncovered epochs",
                xlabel="Hours from PSG start", ylabel="Stage", yticks=range(5),
                yticklabels=["Wake", "REM", "N1", "N2", "N3"], ylim=(4.5, -0.5))
    counts = [audit.metrics[f"{name}_epochs"] for name in STAGE_NAMES]
    bars = axes[2].bar(STAGE_NAMES, counts, color=["#e8b44d", "#63b9ca", "#3d91b3", "#4d5ca8", "#c6759c"])
    axes[2].bar_label(bars, padding=3)
    axes[2].set(title="Supervised epoch counts (30 seconds each)", ylabel="Epoch count")
    return fig, audit
