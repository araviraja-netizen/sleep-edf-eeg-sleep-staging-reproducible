"""Checkpoint-based inference/artifact export without a training loop."""

import csv
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import uuid

import numpy as np
import torch

from .checkpoints import CheckpointManager
from .common import digest, guard_evaluation, json_write, now
from .experiments import model_fingerprint

PREDICTION_FIELDS = ["epoch_id", "subject_id", "recording_id", "night", "original_epoch_index", "onset_seconds", "split",
                     "true_label", "predicted_label", *[f"logit_{i}" for i in range(5)], *[f"probability_{i}" for i in range(5)]]


def load_model(model, experiment, checkpoint="best", device="cpu"):
    expected = experiment.metadata["resume_fingerprints"]
    if model_fingerprint(model, experiment.metadata["configuration"]["architecture"]) != expected["architecture"]:
        raise ValueError("Inference architecture differs from the retained experiment")
    manager = CheckpointManager(experiment.path / "reproducibility/checkpoints", expected, experiment.run_id)
    if checkpoint == "auto":
        state = manager.recover("auto")
        path = manager.recovered_path
    else:
        path = manager.discover().get(checkpoint) if checkpoint in ("best", "latest", "previous") else checkpoint
        if path is None:
            raise ValueError(f"No {checkpoint} checkpoint exists")
        state = manager.read(path)
    model.load_state_dict(state["model"], strict=True)
    model.to(device).eval()
    return model, state, Path(path)


def _pooled(output, pooling, max_dimension):
    if isinstance(output, tuple):
        output = output[0]
    if not isinstance(output, torch.Tensor):
        raise ValueError("Selected layer must return a tensor or a tensor-first tuple")
    if pooling == "mean_last" and output.ndim > 2:
        output = output.mean(dim=-1)
    elif pooling == "mean_time" and output.ndim > 2:
        output = output.mean(dim=1)
    elif pooling != "flatten":
        if pooling not in ("mean_last", "mean_time"):
            raise ValueError("Unknown feature pooling")
    output = output.flatten(start_dim=1)
    if output.shape[1] > max_dimension or not torch.isfinite(output).all():
        raise ValueError("Feature tensor is nonfinite or exceeds configured storage width; choose pooled features")
    return output.detach().cpu().numpy().astype(np.float32)


class _DiskFeatures:
    """Bounded batch writes; embeddings stay on SSD instead of a tensor list."""
    def __init__(self,path,count):self.path,self.count,self.offset,self.array=path,count,0,None
    def append(self,batch):
        if self.array is None:self.array=np.lib.format.open_memmap(self.path,mode='w+',dtype=np.float32,shape=(self.count,batch.shape[1]))
        stop=self.offset+len(batch)
        if stop>self.count:raise ValueError('Export exceeds declared dataset size')
        self.array[self.offset:stop]=batch;self.offset=stop
    def publish(self,target):
        if self.offset!=self.count:raise ValueError('Streaming export has missing rows')
        self.array.flush();shutil.copyfile(self.path,target)
    def close(self):
        if self.array is not None:self.array.flush();self.array._mmap.close();self.array=None


def export_inference(model, loader, experiment, checkpoint="best", device="cpu", penultimate_layer="embedding",
                     intermediate_layers=None, max_feature_dimension=512, final_evaluation=False, metadata_fields=None, streaming=False):
    extra_fields = list(metadata_fields or [])
    if len(set(extra_fields)) != len(extra_fields) or set(extra_fields).intersection(PREDICTION_FIELDS):
        raise ValueError("Additional metadata fields must be unique and cannot replace prediction columns")
    model, state, checkpoint_path = load_model(model, experiment, checkpoint, device)
    modules = dict(model.named_modules())
    specs = {"penultimate": {"name": penultimate_layer, "pooling": "flatten"}}
    for specification in intermediate_layers or []:
        key = specification["name"]
        if key in specs:
            raise ValueError("Duplicate feature layer name")
        specs[key] = specification
    for specification in specs.values():
        if specification["name"] not in modules:
            raise ValueError(f"Unknown feature layer: {specification['name']}")
    latest, buffers, rows, outputs = {}, {key: [] for key in specs}, [], []
    spool=tempfile.TemporaryDirectory(prefix=".feature-spool-",dir=experiment.path) if streaming else None
    if streaming:buffers={key:_DiskFeatures(Path(spool.name)/(str(i)+".npy"),len(loader.dataset)) for i,key in enumerate(specs)}
    hooks = []
    for key, specification in specs.items():
        def capture(module, inputs, output, key=key, specification=specification):
            latest[key] = _pooled(output, specification.get("pooling", "flatten"), max_feature_dimension)
        hooks.append(modules[specification["name"]].register_forward_hook(capture))
    try:
        with torch.no_grad():
            for batch in loader:
                count = len(batch["label"])
                guard_evaluation(batch["split"], final_evaluation)
                latest.clear()
                logits = model(batch["eeg"].to(device))
                if logits.shape != (count, 5) or not torch.isfinite(logits).all():
                    raise ValueError("Inference requires finite five-class logits")
                if set(latest) != set(specs):
                    raise ValueError("A configured feature layer did not execute")
                outputs.append(logits.cpu().numpy().astype(np.float32))
                for key in specs:
                    if len(latest[key]) != count:
                        raise ValueError("Feature row count differs from epoch metadata")
                    buffers[key].append(latest[key])
                for i in range(count):
                    row = {}
                    for key in PREDICTION_FIELDS[:7] + extra_fields:
                        value = batch[key][i]
                        row[key] = value.item() if isinstance(value, torch.Tensor) else value
                    row["true_label"] = int(batch["label"][i])
                    rows.append(row)
    finally:
        for hook in hooks:
            hook.remove()
        if spool is not None and sys.exc_info()[0] is not None:
            for buffer in buffers.values():buffer.close()
            spool.cleanup()
    if not rows:
        if spool is not None:spool.cleanup()
        raise ValueError("Cannot export empty inference artifacts")
    splits = sorted({row["split"] for row in rows})
    guard_evaluation(splits, final_evaluation)
    if len(splits) != 1 or len({row["epoch_id"] for row in rows}) != len(rows):
        raise ValueError("An export must contain one split with unique epoch IDs")
    logits = np.concatenate(outputs)
    exp = np.exp(logits.astype(np.float64) - logits.max(axis=1, keepdims=True))
    probabilities = (exp / exp.sum(axis=1, keepdims=True)).astype(np.float32)
    export_id = f"{splits[0]}-epoch{state['completed_epoch']:06d}-{uuid.uuid4().hex[:8]}"
    performance = experiment.path / "performance/exports" / export_id
    representation = experiment.path / "representations" / export_id
    performance.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix=".export-", dir=experiment.path))
    try:
        perf, rep = temp / "performance", temp / "representations"
        perf.mkdir(); rep.mkdir()
        with (perf / "predictions.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=PREDICTION_FIELDS + extra_fields)
            writer.writeheader()
            for i, row in enumerate(rows):
                row.update(predicted_label=int(probabilities[i].argmax()))
                row.update({f"logit_{label}": float(logits[i, label]) for label in range(5)})
                row.update({f"probability_{label}": float(probabilities[i, label]) for label in range(5)})
                writer.writerow(row)
            handle.flush(); os.fsync(handle.fileno())
        manager = CheckpointManager(experiment.path / "reproducibility/checkpoints",
                                    experiment.metadata["resume_fingerprints"], experiment.run_id)
        committed_history = manager.recover("auto")["history"]
        json_write(perf / "history.json", committed_history)
        with (rep / "feature_epoch_mapping.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["feature_row", "epoch_id", "subject_id", "recording_id", "original_epoch_index", "split"])
            writer.writeheader()
            for i, row in enumerate(rows):
                writer.writerow({"feature_row": i, **{key: row[key] for key in ("epoch_id", "subject_id", "recording_id", "original_epoch_index", "split")}})
            handle.flush(); os.fsync(handle.fileno())
        feature_files = {}
        for key, values in buffers.items():
            name = "penultimate.npy" if key == "penultimate" else "intermediate-" + key.replace(".", "_") + ".npy"
            if name in feature_files.values():
                raise ValueError("Feature layer file-name collision")
            if streaming:values.publish(rep/name)
            else:np.save(rep / name, np.concatenate(values), allow_pickle=False)
            with (rep / name).open("r+b") as handle:
                os.fsync(handle.fileno())
            feature_files[key] = name
        files = {str((performance / "predictions.csv").relative_to(experiment.path)): digest(perf / "predictions.csv")}
        files[str((performance / "history.json").relative_to(experiment.path))] = digest(perf / "history.json")
        for path in rep.iterdir():
            files[str((representation / path.name).relative_to(experiment.path))] = digest(path)
        manifest = {"schema": 1, "export_id": export_id, "experiment_id": experiment.run_id, "created_at": now(),
                    "splits": splits, "evaluation_stage": "frozen_final" if final_evaluation else "development",
                    "synthetic": experiment.metadata["synthetic"], "dataset_contract": experiment.metadata["dataset_contract"],
                    "completed_epoch": state["completed_epoch"], "checkpoint": str(checkpoint_path.relative_to(experiment.path)),
                    "checkpoint_sha256": digest(checkpoint_path), "sample_count": len(rows),
                    "feature_specs": specs, "feature_files": {key: str((representation / value).relative_to(experiment.path)) for key, value in feature_files.items()},
                    "feature_mapping": str((representation / "feature_epoch_mapping.csv").relative_to(experiment.path)),
                    "predictions": str((performance / "predictions.csv").relative_to(experiment.path)), "files": files}
        manifest["history_snapshot"] = str((performance / "history.json").relative_to(experiment.path))
        if extra_fields:
            manifest["additional_metadata_fields"] = extra_fields
        json_write(perf / "manifest.json", manifest)
        manager.pin(checkpoint_path, f"inference-export:{export_id}")
        os.replace(rep, representation)
        os.replace(perf, performance)  # Only this publication exposes an analyzable export.
        return performance
    finally:
        if spool is not None:
            for buffer in buffers.values():buffer.close()
            spool.cleanup()
        if temp.exists():
            shutil.rmtree(temp)
