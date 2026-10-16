"""Saved-artifact analysis only. This module never imports or calls the trainer."""

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .common import atomic_file, digest, guard_evaluation, json_write, now
from .metrics import CLASS_NAMES, classification_metrics, binary_curves, validate_predictions
from .representations import pca, exact_tsne


def load_export(export, final_evaluation=False):
    export = Path(export).resolve()
    manifest = json.loads((export / "manifest.json").read_text())
    guard_evaluation(manifest["splits"], final_evaluation)  # Gate before reading logits or embeddings.
    root = export.parents[2]
    for relative, expected in manifest["files"].items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or digest(path) != expected:
            raise ValueError(f"Export integrity/path violation: {relative}")
    frame = pd.read_csv(root / manifest["predictions"])
    guard_evaluation(frame["split"], final_evaluation)
    if sorted(set(frame["split"])) != sorted(manifest["splits"]):
        raise ValueError("Actual prediction split differs from export declaration")
    if len(frame) != manifest["sample_count"] or frame.epoch_id.duplicated().any():
        raise ValueError("Prediction count/unique epoch-ID mismatch")
    labels = frame.true_label.to_numpy()
    probabilities = frame[[f"probability_{i}" for i in range(5)]].to_numpy(dtype=float)
    validate_predictions(labels, probabilities)
    logits = frame[[f"logit_{i}" for i in range(5)]].to_numpy(dtype=float)
    if not np.isfinite(logits).all():
        raise ValueError("Nonfinite saved logits")
    exp = np.exp(logits - logits.max(axis=1, keepdims=True))
    if (not np.allclose(probabilities, exp / exp.sum(axis=1, keepdims=True), atol=1e-6)
            or not np.array_equal(frame.predicted_label.to_numpy(), probabilities.argmax(axis=1))):
        raise ValueError("Saved logits/probabilities/predicted labels disagree")
    mapping = pd.read_csv(root / manifest["feature_mapping"])
    if mapping.epoch_id.tolist() != frame.epoch_id.tolist() or mapping.feature_row.tolist() != list(range(len(frame))):
        raise ValueError("Feature-to-epoch mapping mismatch")
    features = np.load(root / manifest["feature_files"]["penultimate"], mmap_mode="r", allow_pickle=False)
    if features.ndim != 2 or len(features) != len(frame) or not np.isfinite(features).all():
        raise ValueError("Saved embedding shape/finiteness mismatch")
    return manifest, frame, features, root


def plot_history(history_path):
    history = json.loads(Path(history_path).read_text())
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    epochs = [row["epoch"] + 1 for row in history]
    axes[0].plot(epochs, [r["train_loss"] for r in history], label="Training")
    axes[0].plot(epochs, [r["validation_loss"] for r in history], label="Validation")
    axes[0].set(xlabel="Completed epoch (1-based display)", ylabel="Loss", title="Committed history only")
    axes[0].legend()
    axes[1].plot(epochs, [r["validation_macro_f1"] for r in history], marker="o")
    axes[1].set(xlabel="Completed epoch", ylabel="Validation macro-F1 (five classes)", ylim=(0, 1))
    return fig


def contiguous_blocks(original_indices):
    original_indices = np.asarray(original_indices, dtype=int)
    if len(original_indices) and np.any(np.diff(original_indices) <= 0):
        raise ValueError("Hypnogram indices must be unique and ascending within each recording")
    return np.split(np.arange(len(original_indices)), np.flatnonzero(np.diff(original_indices) != 1) + 1)


def analyze_export(export, output=None, final_evaluation=False, tsne_max_samples=500, tsne_perplexity=30, tsne_iterations=500, seed=42, hypnograms=True):
    manifest, frame, features, root = load_export(export, final_evaluation)
    output = Path(output) if output else root / "analysis" / manifest["export_id"]
    output.mkdir(parents=True, exist_ok=True)
    labels = frame.true_label.to_numpy()
    probabilities = frame[[f"probability_{i}" for i in range(5)]].to_numpy(dtype=float)
    metrics = classification_metrics(labels, probabilities)
    metrics.update(synthetic=manifest["synthetic"], evaluation_stage=manifest["evaluation_stage"],
                   export_id=manifest["export_id"], generated_at=now(), training_called=False)
    json_write(output / "metrics.json", metrics)
    matrix = np.array(metrics["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
    ax.imshow(matrix, cmap="Blues")
    for i in range(5):
        for j in range(5):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center")
    ax.set(xticks=range(5), yticks=range(5), xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
           xlabel="Predicted", ylabel="True", title="Synthetic confusion matrix" if manifest["synthetic"] else "Confusion matrix")
    fig.savefig(output / "confusion_matrix.png", dpi=140); plt.close(fig)
    curves = {name: binary_curves(labels == i, probabilities[:, i]) for i, name in enumerate(CLASS_NAMES)}
    json_write(output / "roc_pr_curves.json", curves)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for name, curve in curves.items():
        if curve["defined"]:
            axes[0].plot(curve["false_positive_rate"], curve["true_positive_rate"], label=f"{name} AUC={curve['roc_auc']:.3f}")
            axes[1].plot(curve["recall"], curve["precision"], label=f"{name} AP={curve['average_precision']:.3f}")
    axes[0].plot([0, 1], [0, 1], color="grey", linestyle="--")
    axes[0].set(xlabel="False-positive rate", ylabel="True-positive rate", title="One-vs-rest ROC", xlim=(0, 1), ylim=(0, 1))
    axes[1].set(xlabel="Recall", ylabel="Precision", title="One-vs-rest PR", xlim=(0, 1), ylim=(0, 1))
    for ax in axes:
        if ax.lines:
            ax.legend(fontsize=8)
    fig.savefig(output / "roc_pr.png", dpi=140); plt.close(fig)
    calibration = [row for row in metrics["calibration_bins"] if row["count"]]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    axes[0].plot([0, 1], [0, 1], color="grey", linestyle="--")
    axes[0].plot([r["confidence"] for r in calibration], [r["accuracy"] for r in calibration], marker="o")
    axes[0].set(xlabel="Mean confidence", ylabel="Empirical accuracy", title=f"Reliability diagram; ECE={metrics['expected_calibration_error']:.4f}", xlim=(0, 1), ylim=(0, 1))
    axes[1].hist(probabilities.max(axis=1), bins=np.linspace(0, 1, 11))
    axes[1].set(xlabel="Maximum probability", ylabel="Epochs", title="Confidence distribution")
    fig.savefig(output / "calibration.png", dpi=140); plt.close(fig)
    subject_metrics = {}
    for subject, group in frame.groupby("subject_id", observed=True):
        subject_metrics[str(subject)] = classification_metrics(group.true_label.to_numpy(), group[[f"probability_{i}" for i in range(5)]].to_numpy())
    json_write(output / "subject_metrics.json", subject_metrics)
    fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
    ax.bar(list(subject_metrics), [value["macro_f1"] for value in subject_metrics.values()])
    ax.set(xlabel="Subject", ylabel="Five-class macro-F1", ylim=(0, 1), title="Per-subject analysis")
    ax.tick_params(axis="x", rotation=60)
    fig.savefig(output / "subject_metrics.png", dpi=140); plt.close(fig)
    history = root / manifest.get("history_snapshot", "performance/history.json")
    if history.exists():
        fig = plot_history(history); fig.savefig(output / "learning_curves.png", dpi=140); plt.close(fig)
    projections, pca_params = pca(features, 2)
    json_write(output / "pca_parameters.json", pca_params)
    with atomic_file(output / "pca_coordinates.csv", newline="") as handle:
        writer = csv.writer(handle); writer.writerow(["epoch_id", "true_label", "pc1", "pc2"])
        writer.writerows((row.epoch_id, int(row.true_label), float(coords[0]), float(coords[1])) for row, coords in zip(frame.itertuples(), projections))
    methods = {"pca": projections}
    selections = {"pca": np.arange(len(frame))}
    if len(features) >= 4:
        selection = np.sort(np.random.default_rng(seed).choice(len(features), min(len(features), tsne_max_samples), replace=False))
        perplexity = min(tsne_perplexity, max(2, (len(selection) - 1) / 3))
        if len(selection) < 4:
            raise ValueError("t-SNE sample cap must allow at least four rows")
        positions, parameters = exact_tsne(features[selection], perplexity, tsne_iterations, seed, max_samples=tsne_max_samples)
        parameters["selection"] = "seeded uniform, without class balancing"
        parameters["epoch_ids"] = frame.iloc[selection].epoch_id.tolist()
        json_write(output / "tsne_parameters.json", parameters)
        with atomic_file(output / "tsne_coordinates.csv", newline="") as handle:
            writer = csv.writer(handle); writer.writerow(["epoch_id", "true_label", "tsne1", "tsne2"])
            writer.writerows((frame.iloc[i].epoch_id, int(labels[i]), float(coords[0]), float(coords[1])) for i, coords in zip(selection, positions))
        methods["tsne"], selections["tsne"] = positions, selection
    for name, coordinates in methods.items():
        fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
        selected_labels = labels[selections[name]]
        for label, stage in enumerate(CLASS_NAMES):
            included = selected_labels == label
            if included.any():
                ax.scatter(coordinates[included, 0], coordinates[included, 1], s=12, alpha=0.7, label=stage)
        ax.set(title=f"{'Synthetic ' if manifest['synthetic'] else ''}{name.upper()} stored embeddings", xlabel=f"{name} 1", ylabel=f"{name} 2")
        ax.legend(); fig.savefig(output / f"{name}.png", dpi=140); plt.close(fig)
    positions = np.array([0, 2, 3, 4, 1])
    for recording, group in (frame.groupby("recording_id", observed=True) if hypnograms else []):
        group = group.sort_values("original_epoch_index")
        original = group.original_epoch_index.to_numpy(dtype=int)
        fig, ax = plt.subplots(figsize=(11, 4), constrained_layout=True)
        for block_number, block in enumerate(contiguous_blocks(original)):
            if not len(block):
                continue
            edges = np.r_[original[block], original[block[-1]] + 1] * 30 / 3600
            for key, label, color in (("true_label", "True", "black"), ("predicted_label", "Predicted", "#397abd")):
                ax.stairs(positions[group.iloc[block][key].to_numpy(dtype=int)], edges,
                          label=label if block_number == 0 else None, color=color, linewidth=0.8)
        ax.set(title=str(recording) + " — gaps and original recording boundaries preserved", xlabel="Hours from original onset",
               ylabel="Stage", yticks=range(5), yticklabels=["Wake", "REM", "N1", "N2", "N3"], ylim=(4.5, -0.5))
        ax.legend(); fig.savefig(output / f"hypnogram_{str(recording)}.png", dpi=140); plt.close(fig)
    json_write(output / "analysis_manifest.json", {"source_export": str(Path(export).resolve()), "source_manifest_sha256": digest(Path(export) / "manifest.json"),
               "synthetic": manifest["synthetic"], "training_called": False, "seed": seed, "final_evaluation": final_evaluation,
               "files": {p.name: digest(p) for p in output.iterdir() if p.is_file() and p.name != "analysis_manifest.json"}})
    return output
