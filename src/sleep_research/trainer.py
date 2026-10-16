"""Shared epoch-boundary trainer for synthetic checks and authorized real training."""

import csv
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from .checkpoints import CheckpointManager, capture_rng, restore_rng
from .common import atomic_file, json_write, guard_evaluation
from .experiments import resume_fingerprints, source_revision
from .metrics import classification_metrics


class SimulatedInterruption(RuntimeError):
    """Synthetic smoke-test interruption; real interrupts follow the same rollback rule."""


def metadata_rows(batch):
    keys = ("epoch_id", "subject_id", "recording_id", "night", "original_epoch_index", "onset_seconds", "split")
    if any(key not in batch for key in keys):
        raise ValueError("Every sample needs stable epoch/subject/recording/night/index/onset/split metadata")
    count = len(batch["label"])
    rows = []
    for i in range(count):
        row = {}
        for key in keys:
            value = batch[key][i]
            row[key] = value.item() if isinstance(value, torch.Tensor) else value
        rows.append(row)
    return rows


def evaluate_model(model, loader, device="cpu", final_evaluation=False, criterion=None):
    model.eval()
    labels, logits, metadata = [], [], []
    loss_sum = 0.0
    objective_sum, objective_weight = 0.0, 0.0
    with torch.no_grad():
        for batch in loader:
            rows = metadata_rows(batch)
            guard_evaluation([row["split"] for row in rows], final_evaluation)
            inputs, true = batch["eeg"].to(device), batch["label"].long().to(device)
            output = model(inputs)
            if output.shape != (len(true), 5) or not torch.isfinite(output).all():
                raise ValueError("Model must return five finite logits per epoch")
            loss_sum += float(torch.nn.functional.cross_entropy(output, true, reduction="sum"))
            if criterion is not None:
                weight = float(criterion.normalizer(true))
                objective_sum += float(criterion(output, true)) * weight
                objective_weight += weight
            labels.append(true.cpu().numpy())
            logits.append(output.cpu().numpy())
            metadata.extend(rows)
    if not labels:
        raise ValueError("Evaluation loader is empty")
    labels, logits = np.concatenate(labels), np.concatenate(logits)
    exp = np.exp(logits.astype(np.float64) - logits.max(axis=1, keepdims=True))
    probabilities = exp / exp.sum(axis=1, keepdims=True)
    result = {"labels": labels, "logits": logits, "probabilities": probabilities, "metadata": metadata,
            "metrics": classification_metrics(labels, probabilities), "loss": loss_sum / len(labels)}
    if criterion is not None:
        result["objective_loss"] = objective_sum / objective_weight
    return result


class Trainer:
    def __init__(self, model, optimizer, scheduler, train_dataset, validation_dataset, experiment, config,
                 train_sampler=None, validation_sampler=None, criterion=None):
        self.model, self.optimizer, self.scheduler = model, optimizer, scheduler
        self.experiment, self.config = experiment, dict(config)
        self.device = torch.device(config["device"])
        if self.device.type == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but unavailable in this process")
        self.model.to(self.device)
        self.criterion = criterion.to(self.device) if criterion is not None else None
        self.train_generator = torch.Generator().manual_seed(config["seed"])
        self.validation_generator = torch.Generator().manual_seed(config["seed"] + 1)
        workers = config.get("num_workers", 0)
        self.train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=train_sampler is None,
                                       sampler=train_sampler, generator=self.train_generator, num_workers=workers)
        self.validation_loader = DataLoader(validation_dataset, batch_size=config["batch_size"], shuffle=False,
                                            sampler=validation_sampler, generator=self.validation_generator, num_workers=workers)
        self.fingerprints = resume_fingerprints(model, config, experiment.metadata["dataset_contract"])
        if self.criterion is not None:
            from .objectives import objective_fingerprint
            self.fingerprints["objective"] = objective_fingerprint(self.criterion)
        self.manager = CheckpointManager(experiment.path / "reproducibility/checkpoints", self.fingerprints, experiment.run_id)
        self.history, self.global_step, self.next_epoch = [], 0, 0
        self.early_stopping = {"best_score": None, "bad_epochs": 0, "patience": config["early_stopping"]["patience"],
                               "min_delta": config["early_stopping"].get("min_delta", 0.0), "stopped": False}
        self.completed_training_seconds = 0.0

    def _sampler_state(self):
        result = {"train_generator": self.train_generator.get_state(), "validation_generator": self.validation_generator.get_state()}
        for name, loader in (("train", self.train_loader), ("validation", self.validation_loader)):
            sampler = loader.sampler
            result[f"{name}_sampler"] = sampler.state_dict() if hasattr(sampler, "state_dict") else None
            result[f"{name}_sampler_generator"] = sampler.generator.get_state() if isinstance(getattr(sampler, "generator", None), torch.Generator) else None
            result[f"{name}_dataset"] = loader.dataset.state_dict() if hasattr(loader.dataset, "state_dict") else None
        return result

    def _restore_sampler(self, state):
        self.train_generator.set_state(state["train_generator"])
        self.validation_generator.set_state(state["validation_generator"])
        for name, loader in (("train", self.train_loader), ("validation", self.validation_loader)):
            if state[f"{name}_sampler"] is not None:
                if not hasattr(loader.sampler, "load_state_dict"):
                    raise ValueError("Sampler no longer supports state restoration")
                loader.sampler.load_state_dict(state[f"{name}_sampler"])
            if state[f"{name}_sampler_generator"] is not None:
                loader.sampler.generator.set_state(state[f"{name}_sampler_generator"])
            if state[f"{name}_dataset"] is not None:
                loader.dataset.load_state_dict(state[f"{name}_dataset"])

    def _state(self, completed_epoch):
        result = {"schema_version": 1, "experiment_id": self.experiment.run_id, "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(), "scheduler": self.scheduler.state_dict() if self.scheduler else None,
                "early_stopping": self.early_stopping, "completed_epoch": completed_epoch, "next_epoch": completed_epoch + 1,
                "global_step": self.global_step, "rng": capture_rng(self.device.type == "mps"),
                "sampler": self._sampler_state(), "history": self.history,
                "dataset_contract": self.experiment.metadata["dataset_contract"],
                "source_revision": source_revision(self.experiment.project),
                "configuration": self.config, "fingerprints": self.fingerprints,
                "completed_training_seconds": self.completed_training_seconds, "device": str(self.device),
                "mps_bitwise_resume_guarantee": False}
        if self.criterion is not None:
            result["criterion"] = self.criterion.state_dict()
        return result

    def restore(self, resume="auto"):
        state = self.manager.recover(resume)
        if state is None:
            self.manager.save(self._state(-1))  # Bootstrap permits rollback even inside the first epoch.
        else:
            self.model.load_state_dict(state["model"], strict=True)
            self.optimizer.load_state_dict(state["optimizer"])
            if self.criterion is not None:
                self.criterion.load_state_dict(state["criterion"], strict=True)
            if (state["scheduler"] is None) != (self.scheduler is None):
                raise ValueError("Scheduler compatibility mismatch")
            if self.scheduler is not None:
                self.scheduler.load_state_dict(state["scheduler"])
            self.early_stopping = state["early_stopping"]
            self.history = state["history"]
            self.global_step, self.next_epoch = state["global_step"], state["next_epoch"]
            self.completed_training_seconds = state.get("completed_training_seconds", 0.0)
            self._restore_sampler(state["sampler"])
            restore_rng(state["rng"])
        self._history_artifacts()  # Repair any interrupted materialization from committed history.

    def _history_artifacts(self):
        json_write(self.experiment.path / "performance/history.json", self.history)
        fields = ["epoch", "train_loss", "validation_loss", "validation_accuracy", "validation_macro_f1", "global_step", "learning_rate", "seconds"]
        if self.config.get("track_training_metrics", False):
            fields += ["training_accuracy", "training_macro_f1"]
        if self.criterion is not None:
            fields += ["validation_objective_loss", "validation_n1_f1", "validation_n3_f1"]
        with atomic_file(self.experiment.path / "performance/history.csv", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(self.history)

    def fit(self, max_epochs=None, resume="auto", interrupt_callback=None):
        max_epochs = self.config["max_epochs"] if max_epochs is None else max_epochs
        if max_epochs < 0:
            raise ValueError("Epoch budget must be nonnegative")
        tick, status, start_epoch = time.monotonic(), "failed", 0
        with self.experiment.lock():
            try:
                self.restore(resume)
                start_epoch = self.next_epoch
                self.experiment.update(status="running", resume_fingerprints=self.fingerprints, requested_epoch_budget=max_epochs)
                for epoch in range(self.next_epoch, max_epochs):
                    if self.early_stopping["stopped"]:
                        break
                    epoch_tick = time.monotonic()
                    self.model.train()
                    loss_sum, samples = 0.0, 0
                    loss_weight = 0.0
                    training_matrix = np.zeros((5, 5), dtype=np.int64)
                    epoch_learning_rate = self.optimizer.param_groups[0]["lr"]
                    for batch_index, batch in enumerate(self.train_loader):
                        if any(row["split"] != "train" for row in metadata_rows(batch)):
                            raise PermissionError("Training loader contains non-training epochs")
                        inputs, labels = batch["eeg"].to(self.device), batch["label"].long().to(self.device)
                        self.optimizer.zero_grad(set_to_none=True)
                        logits = self.model(inputs)
                        if logits.shape != (len(labels), 5) or not torch.isfinite(logits).all():
                            raise ValueError("Training requires finite five-class logits")
                        loss = self.criterion(logits, labels) if self.criterion is not None else torch.nn.functional.cross_entropy(logits, labels)
                        if not torch.isfinite(loss):
                            raise ValueError("Nonfinite training loss")
                        loss.backward()
                        if self.config.get("gradient_clip_norm") is not None:
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config["gradient_clip_norm"], error_if_nonfinite=True)
                        self.optimizer.step()
                        self.global_step += 1
                        weight = float(self.criterion.normalizer(labels)) if self.criterion is not None else len(labels)
                        loss_sum += float(loss.detach()) * weight
                        loss_weight += weight
                        if self.config.get("track_training_metrics", False):
                            true = labels.detach().cpu().numpy()
                            predicted = logits.detach().argmax(dim=1).cpu().numpy()
                            training_matrix += np.bincount(5 * true + predicted, minlength=25).reshape(5, 5)
                        samples += len(labels)
                        interval = self.config.get("progress_batches", 0)
                        if interval and (batch_index + 1) % interval == 0:
                            print(f"epoch={epoch + 1} train_batch={batch_index + 1}/{len(self.train_loader)} loss={loss_sum / loss_weight:.5f}", flush=True)
                        if interrupt_callback:
                            interrupt_callback(epoch, batch_index, self.global_step)
                    if not samples:
                        raise ValueError("Empty training loader")
                    # Validation is the only model-selection population. Test metadata is rejected before forward.
                    for name in ("split",):
                        dataset = self.validation_loader.dataset
                        if hasattr(dataset, name) and getattr(dataset, name) != "validation":
                            raise PermissionError("Validation dataset must declare validation membership")
                    validation = evaluate_model(self.model, self.validation_loader, str(self.device), criterion=self.criterion)
                    if any(row["split"] != "validation" for row in validation["metadata"]):
                        raise PermissionError("Model selection accepts validation membership only")
                    score = validation["metrics"]["macro_f1"]
                    early = self.early_stopping
                    if early["best_score"] is None or score > early["best_score"] + early["min_delta"]:
                        early["best_score"], early["bad_epochs"] = score, 0
                    else:
                        early["bad_epochs"] += 1
                    early["stopped"] = early["bad_epochs"] >= early["patience"]
                    if self.scheduler is not None:
                        if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                            self.scheduler.step(score)
                        else:
                            self.scheduler.step()
                    seconds = time.monotonic() - epoch_tick
                    entry = {"epoch": epoch, "train_loss": loss_sum / loss_weight, "validation_loss": validation["loss"],
                             "validation_accuracy": validation["metrics"]["accuracy"], "validation_macro_f1": score,
                             "global_step": self.global_step, "learning_rate": self.optimizer.param_groups[0]["lr"], "seconds": seconds}
                    if self.criterion is not None:
                        entry.update(validation_objective_loss=validation["objective_loss"],
                                     validation_n1_f1=validation["metrics"]["classes"]["N1"]["f1"],
                                     validation_n3_f1=validation["metrics"]["classes"]["N3"]["f1"])
                    if self.config.get("track_training_metrics", False):
                        diagonal = training_matrix.diagonal().astype(float)
                        denominator = training_matrix.sum(axis=0) + training_matrix.sum(axis=1)
                        f1 = np.divide(2 * diagonal, denominator, out=np.zeros(5), where=denominator != 0)
                        entry.update(training_accuracy=float(diagonal.sum() / samples), training_macro_f1=float(f1.mean()),
                                     learning_rate=epoch_learning_rate)
                    if any(old["epoch"] == epoch for old in self.history):
                        raise ValueError("Duplicate epoch history")
                    self.history.append(entry)
                    self.completed_training_seconds += seconds
                    self.manager.save(self._state(epoch), score)
                    self.next_epoch = epoch + 1
                    self._history_artifacts()
                    if self.config.get("progress_batches", 0):
                        print(f"committed_epoch={epoch + 1} validation_macro_f1={score:.6f} validation_accuracy={entry['validation_accuracy']:.6f} seconds={seconds:.1f}", flush=True)
                status = "early_stopped" if self.early_stopping["stopped"] else "completed"
                return self.history
            except (KeyboardInterrupt, SimulatedInterruption):
                status = "interrupted"
                raise
            finally:
                self.experiment.record_attempt(status, time.monotonic() - tick, start_epoch, self.next_epoch, resume, self.manager.recovery_notes)
