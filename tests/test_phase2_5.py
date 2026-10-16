"""Synthetic-only checkpoint/research infrastructure tests; frozen EEG is not trained on."""

from copy import deepcopy
import csv
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import uuid

import numpy as np
import torch
import yaml

from sleep_research.analysis import analyze_export, contiguous_blocks, load_export
from sleep_research.checkpoints import CheckpointManager, CorruptCheckpoint, IncompatibleCheckpoint, REQUIRED_FINGERPRINTS, capture_rng, restore_rng
from sleep_research.common import atomic_file, fingerprint, guard_evaluation
from sleep_research.explanations import saliency, grad_cam_1d, attention_weights
from sleep_research.experiments import Experiment, model_fingerprint, resume_fingerprints
from sleep_research.inference import export_inference, load_model
from sleep_research.metrics import classification_metrics, confusion_matrix, binary_curves
from sleep_research.representations import pca, exact_tsne, joint_probabilities, tsne_objective_gradient, optional_umap
from sleep_research.synthetic import build_demo_trainer, SyntheticEpochs, SyntheticTinyCNN
from sleep_research.trainer import SimulatedInterruption

PROJECT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT / "results/.phase2_5_matplotlib"))
CONFIG = yaml.safe_load((PROJECT / "configs/infrastructure_demo.yaml").read_text())


def config_small():
    config = deepcopy(CONFIG)
    config["synthetic_dataset"].update(train_samples=30, validation_samples=20)
    config["batch_size"] = 10
    config["max_epochs"] = 3
    config["analysis"].update(tsne_perplexity=5, tsne_iterations=300)
    return config


def fixture_payload(epoch, scores, config=None):
    fingerprints = {name: "fixture-" + name for name in REQUIRED_FINGERPRINTS}
    model = torch.nn.Linear(2, 5)
    optimizer = torch.optim.Adam(model.parameters())
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1)
    return dict(model=model.state_dict(), optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                early_stopping={"best_score": max(scores) if scores else None}, completed_epoch=epoch, next_epoch=epoch + 1,
                global_step=max(0, epoch + 1), rng=capture_rng(), sampler={}, configuration=config or {}, fingerprints=fingerprints,
                history=[{"epoch": i, "validation_macro_f1": score} for i, score in enumerate(scores)], experiment_id="fixture")


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=PROJECT / "results")
        self.payload = fixture_payload(-1, [])
        self.manager = CheckpointManager(self.tmp.name, self.payload["fingerprints"], "fixture")

    def tearDown(self):
        self.tmp.cleanup()

    def test_latest_best_and_previous_are_independent(self):
        self.manager.save(self.payload)
        self.manager.save(fixture_payload(0, [0.8]), 0.8)
        best_path = Path(self.manager.discover()["best"])
        best_bytes = best_path.read_bytes()
        self.manager.save(fixture_payload(1, [0.8, 0.6]), 0.6)
        self.assertEqual(self.manager.read(self.manager.discover()["latest"])["completed_epoch"], 1)
        self.assertEqual(self.manager.read(self.manager.discover()["best"])["completed_epoch"], 0)
        self.assertEqual(self.manager.read(self.manager.discover()["previous"])["completed_epoch"], 0)
        self.assertEqual(best_path.read_bytes(), best_bytes)

    def test_corrupted_latest_recovers_previous_completed_epoch(self):
        self.manager.save(self.payload)
        self.manager.save(fixture_payload(0, [0.4]), 0.4)
        self.manager.save(fixture_payload(1, [0.4, 0.5]), 0.5)
        Path(self.manager.discover()["latest"]).write_bytes(b"corrupted")
        state = self.manager.recover("auto")
        self.assertEqual(state["next_epoch"], 1)
        self.assertTrue(self.manager.recovery_notes)
        self.manager.save(fixture_payload(1, [0.4, 0.45]), 0.45)
        self.assertEqual(self.manager.recover()["completed_epoch"], 1)

    def test_corrupt_pointer_index_uses_independently_validated_previous_index(self):
        self.manager.save(self.payload)
        self.manager.save(fixture_payload(0, [0.3]), 0.3)
        (Path(self.tmp.name) / "index.json").write_text("{incomplete")
        self.assertEqual(self.manager.recover()["next_epoch"], 0)

    def test_all_corrupt_does_not_silently_restart(self):
        self.manager.save(self.payload)
        Path(self.manager.discover()["latest"]).write_bytes(b"broken")
        with self.assertRaises(CorruptCheckpoint):
            self.manager.recover()

    def test_each_incompatible_fingerprint_is_rejected(self):
        path = self.manager.save(self.payload)
        for name in REQUIRED_FINGERPRINTS:
            fingerprints = dict(self.payload["fingerprints"]); fingerprints[name] = "changed"
            other = CheckpointManager(self.tmp.name, fingerprints, "fixture")
            with self.subTest(name=name), self.assertRaises(IncompatibleCheckpoint):
                other.recover(str(path))

    def test_none_refuses_overwriting_and_duplicate_history_rejected(self):
        self.manager.save(self.payload)
        with self.assertRaises(FileExistsError):
            self.manager.recover("none")
        payload = fixture_payload(1, [0.2, 0.3]); payload["history"][1]["epoch"] = 0
        with self.assertRaises(CorruptCheckpoint):
            self.manager.save(payload, 0.3)

    def test_explicit_older_checkpoint_rewinds_history_and_auto_continues_new_timeline(self):
        self.manager.save(self.payload)
        older = self.manager.save(fixture_payload(0, [0.2]), 0.2)
        self.manager.save(fixture_payload(1, [0.2, 0.9]), 0.9)
        self.manager.save(fixture_payload(2, [0.2, 0.9, 0.7]), 0.7)
        self.assertEqual(self.manager.recover(str(older))["next_epoch"], 1)
        self.manager.save(fixture_payload(1, [0.2, 0.4]), 0.4)
        self.assertEqual(self.manager.recover()["completed_epoch"], 1)
        self.assertLessEqual(self.manager.read(self.manager.discover()["best"])["completed_epoch"], 1)

    def test_atomic_failed_write_preserves_previous_bytes(self):
        path = Path(self.tmp.name) / "atomic.json"; path.write_text("old")
        with self.assertRaises(RuntimeError):
            with atomic_file(path) as handle:
                handle.write("incomplete"); raise RuntimeError("power loss fixture")
        self.assertEqual(path.read_text(), "old")

    def test_rng_round_trip_python_numpy_and_torch(self):
        state = capture_rng()
        expected = (random.random(), np.random.random(), torch.rand(3))
        restore_rng(state)
        self.assertEqual(random.random(), expected[0])
        self.assertEqual(np.random.random(), expected[1])
        torch.testing.assert_close(torch.rand(3), expected[2], rtol=0, atol=0)

    def test_checkpoint_generations_have_bounded_retention(self):
        self.manager.save(self.payload)
        scores = []
        for epoch in range(8):
            scores.append(0.8 if epoch == 0 else 0.2)
            self.manager.save(fixture_payload(epoch, scores), scores[-1])
        self.assertLessEqual(len(list(Path(self.tmp.name).glob("*.pt"))), 4)
        self.assertEqual(self.manager.read(self.manager.discover()["best"])["completed_epoch"], 0)

    def test_export_pinned_checkpoint_survives_rolling_retention(self):
        self.manager.save(self.payload)
        pinned = self.manager.save(fixture_payload(0, [0.1]), 0.1)
        self.manager.pin(pinned, "synthetic-export-fixture")
        scores = [0.1]
        for epoch in range(1, 7):
            scores.append(0.2 + epoch / 20)
            self.manager.save(fixture_payload(epoch, scores), scores[-1])
        self.assertEqual(self.manager.read(pinned)["completed_epoch"], 0)


class TrainerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=PROJECT / "results")
        self.root = Path(self.tmp.name)
        self.config = config_small()

    def tearDown(self):
        self.tmp.cleanup()

    def test_mid_epoch_resume_matches_uninterrupted_cpu_weights_optimizer_scheduler_and_history(self):
        baseline = build_demo_trainer(self.root, self.config, "baseline", bind_real_reference=False)
        baseline.fit(resume="none")
        interrupted = build_demo_trainer(self.root, self.config, "resume", bind_real_reference=False)
        def stop(epoch, batch, step):
            if epoch == 1 and batch == 1:
                raise SimulatedInterruption("fixture")
        with self.assertRaises(SimulatedInterruption):
            interrupted.fit(resume="none", interrupt_callback=stop)
        state = interrupted.manager.recover()
        self.assertEqual(state["completed_epoch"], 0)
        self.assertEqual(state["next_epoch"], 1)
        self.assertEqual(state["global_step"], 3)
        resumed = build_demo_trainer(self.root, self.config, "resume", bind_real_reference=False)
        resumed.fit(resume="auto")
        self.assertEqual([r["epoch"] for r in resumed.history], [0, 1, 2])
        self.assertEqual(resumed.global_step, baseline.global_step)
        for name, value in baseline.model.state_dict().items():
            torch.testing.assert_close(value, resumed.model.state_dict()[name], rtol=0, atol=0)
        for key, values in baseline.optimizer.state_dict()["state"].items():
            for field, value in values.items():
                torch.testing.assert_close(value, resumed.optimizer.state_dict()["state"][key][field], rtol=0, atol=0)
        self.assertEqual(baseline.scheduler.state_dict(), resumed.scheduler.state_dict())
        for a, b in zip(baseline.history, resumed.history):
            self.assertEqual({k:v for k,v in a.items() if k != "seconds"}, {k:v for k,v in b.items() if k != "seconds"})

    def test_interruption_inside_first_epoch_recovers_bootstrap(self):
        trainer = build_demo_trainer(self.root, self.config, "first", bind_real_reference=False)
        with self.assertRaises(SimulatedInterruption):
            trainer.fit(resume="none", interrupt_callback=lambda *args: (_ for _ in ()).throw(SimulatedInterruption()))
        self.assertEqual(trainer.manager.recover()["next_epoch"], 0)
        fresh = build_demo_trainer(self.root, self.config, "first", bind_real_reference=False)
        fresh.fit(max_epochs=1)
        self.assertEqual([r["epoch"] for r in fresh.history], [0])

    def test_extended_epoch_budget_allowed_but_changed_optimizer_rejected(self):
        trainer = build_demo_trainer(self.root, self.config, "extend", bind_real_reference=False)
        trainer.fit(max_epochs=1)
        config = deepcopy(self.config); config["max_epochs"] = 4
        fresh = build_demo_trainer(self.root, config, "extend", bind_real_reference=False)
        fresh.fit()
        self.assertEqual(fresh.next_epoch, 4)
        altered = deepcopy(config); altered["optimizer"]["learning_rate"] *= 2
        with self.assertRaises(ValueError):
            build_demo_trainer(self.root, altered, "extend", bind_real_reference=False)

    def test_architecture_dropout_change_changes_fingerprint(self):
        model1, model2 = SyntheticTinyCNN(dropout=0.2), SyntheticTinyCNN(dropout=0.4)
        self.assertNotEqual(model_fingerprint(model1, {}), model_fingerprint(model2, {}))

    def test_repeat_completed_resume_does_not_duplicate_history(self):
        trainer = build_demo_trainer(self.root, self.config, "done", bind_real_reference=False)
        trainer.fit()
        fresh = build_demo_trainer(self.root, self.config, "done", bind_real_reference=False)
        fresh.fit()
        self.assertEqual([r["epoch"] for r in fresh.history], [0, 1, 2])

    def test_test_dataset_rejected_before_model_evaluation(self):
        trainer = build_demo_trainer(self.root, self.config, "gate", bind_real_reference=False)
        trainer.validation_loader = torch.utils.data.DataLoader(SyntheticEpochs("test", 10), batch_size=5)
        with self.assertRaises(PermissionError):
            trainer.fit(max_epochs=1)
        self.assertEqual(trainer.manager.recover()["next_epoch"], 0)


class ArtifactAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=PROJECT / "results")
        self.root = Path(self.tmp.name)
        self.config = config_small()
        self.trainer = build_demo_trainer(self.root, self.config, "artifacts", bind_real_reference=False)
        self.trainer.fit(max_epochs=2)
        self.export = export_inference(self.trainer.model, self.trainer.validation_loader, self.trainer.experiment,
                                       intermediate_layers=[{"name":"features.0","pooling":"mean_last"}])

    def tearDown(self):
        self.tmp.cleanup()

    def test_predictions_probabilities_embeddings_and_ids_align(self):
        manifest, frame, features, root = load_export(self.export)
        self.assertEqual(frame.epoch_id.tolist(), [r["epoch_id"] for r in self.trainer.validation_loader.dataset.metadata])
        self.assertEqual(features.shape, (20, 16))
        self.assertEqual(np.load(root / manifest["feature_files"]["features.0"]).shape, (20, 4))
        np.testing.assert_allclose(frame[[f"probability_{i}" for i in range(5)]].sum(axis=1), 1, atol=1e-6)
        self.assertTrue(manifest["synthetic"])
        self.assertEqual(json.loads((root / manifest['history_snapshot']).read_text()), self.trainer.history)

    def test_metrics_and_pca_tsne_are_recalculated_without_trainer(self):
        with patch("sleep_research.trainer.Trainer.fit", side_effect=AssertionError("Analysis called training")):
            output = analyze_export(self.export, tsne_perplexity=5, tsne_iterations=300, tsne_max_samples=20)
        saved = json.loads((output / "metrics.json").read_text())
        _, frame, _, _ = load_export(self.export)
        expected = confusion_matrix(frame.true_label.to_numpy(), frame.predicted_label.to_numpy())
        np.testing.assert_array_equal(saved["confusion_matrix"], expected)
        self.assertTrue((output / "pca_coordinates.csv").exists())
        self.assertTrue((output / "tsne_coordinates.csv").exists())
        self.assertFalse(saved["training_called"])

    def test_test_and_mixed_test_exports_are_gated(self):
        manifest_path = self.export / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["splits"] = ["validation", "test"]
        manifest_path.write_text(json.dumps(manifest))
        with self.assertRaises(PermissionError):
            load_export(self.export)

    def test_corrupt_feature_artifacts_are_rejected(self):
        manifest = json.loads((self.export / "manifest.json").read_text())
        path = self.trainer.experiment.path / manifest["feature_files"]["penultimate"]
        with path.open("r+b") as handle:
            handle.seek(-1, 2); handle.write(b"X")
        with self.assertRaises(ValueError):
            load_export(self.export)

    def test_original_hypnogram_gaps_are_broken_into_blocks(self):
        blocks = contiguous_blocks([0, 1, 4, 5, 9])
        self.assertEqual([b.tolist() for b in blocks], [[0, 1], [2, 3], [4]])

    def test_model_loading_saliency_grad_cam_without_training(self):
        model, _, _ = load_model(SyntheticTinyCNN(), self.trainer.experiment)
        batch = next(iter(self.trainer.validation_loader))["eeg"][:2]
        gradients, targets = saliency(model, batch)
        cams, _ = grad_cam_1d(model, batch, "features.0")
        self.assertEqual(gradients.shape, batch.shape)
        self.assertEqual(cams.shape, batch.shape)
        self.assertTrue(np.isfinite(gradients).all() and np.isfinite(cams).all())
        self.assertTrue((cams >= 0).all() and (cams <= 1).all())
        with self.assertRaises(ValueError):
            grad_cam_1d(model, batch, "classifier")


class MetricsRepresentationTests(unittest.TestCase):
    def test_perfect_five_class_metrics_and_calibration(self):
        metrics = classification_metrics(np.arange(5), np.eye(5))
        for key in ("accuracy", "macro_f1", "balanced_accuracy", "cohen_kappa"):
            self.assertEqual(metrics[key], 1)
        self.assertEqual(metrics["multiclass_brier_score"], 0)
        self.assertEqual(metrics["expected_calibration_error"], 0)

    def test_absent_classes_and_undefined_kappa_conventions(self):
        metrics = classification_metrics(np.array([0,0]), np.array([[1,0,0,0,0],[1,0,0,0,0]]))
        self.assertEqual(metrics["macro_f1"], 0.2)
        self.assertEqual(metrics["balanced_accuracy"], 1)
        self.assertIsNone(metrics["cohen_kappa"])

    def test_tied_binary_scores_and_undefined_curves(self):
        curve = binary_curves([False,True,False,True], [0.5]*4)
        self.assertEqual(curve["roc_auc"], 0.5)
        self.assertEqual(curve["average_precision"], 0.5)
        self.assertFalse(binary_curves([True,True],[0.1,0.9])["defined"])

    def test_pca_known_low_rank_and_coordinate_mapping(self):
        x = np.arange(12, dtype=float)
        positions, parameters = pca(np.column_stack([x,2*x,3*x]))
        self.assertAlmostEqual(parameters["explained_variance_ratio"][0], 1)
        self.assertEqual(positions.shape, (12,2))

    def test_tsne_gradient_matches_finite_difference(self):
        rng = np.random.default_rng(42)
        probabilities = joint_probabilities(rng.normal(size=(8,4)), perplexity=3)
        positions = rng.normal(size=(8,2))
        value, gradient = tsne_objective_gradient(positions, probabilities)
        epsilon = 1e-5
        a, b = positions.copy(), positions.copy()
        a[2,1] += epsilon; b[2,1] -= epsilon
        estimate = (tsne_objective_gradient(a, probabilities)[0] - tsne_objective_gradient(b, probabilities)[0]) / (2*epsilon)
        self.assertAlmostEqual(gradient[2,1], estimate, places=7)

    def test_exact_tsne_seeded_shape_and_bounds(self):
        features = np.random.default_rng(42).normal(size=(16,5))
        first, _ = exact_tsne(features, perplexity=4, iterations=300, seed=42)
        second, _ = exact_tsne(features, perplexity=4, iterations=300, seed=42)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(first.shape, (16,2))
        with self.assertRaises(ValueError):
            exact_tsne(features, max_samples=8)

    def test_frozen_test_guard_is_explicit(self):
        with self.assertRaises(PermissionError):
            guard_evaluation(["test"])
        guard_evaluation(["test"], final_evaluation=True)  # Synthetic access-control fixture, no real test metrics.

    def test_attention_capture_compatible_architecture_and_token_bound(self):
        class AttentionFixture(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.attention = torch.nn.MultiheadAttention(4,2,batch_first=True); self.head=torch.nn.Linear(4,5)
            def forward(self,x):
                values, weights = self.attention(x,x,x,need_weights=True,average_attn_weights=False)
                return self.head(values.mean(dim=1))
        model=AttentionFixture(); inputs=torch.rand(2,6,4)
        weights=attention_weights(model,inputs,"attention",max_tokens=8)
        self.assertEqual(weights.shape,(2,2,6,6))
        np.testing.assert_allclose(weights.sum(axis=-1),1,atol=1e-6)
        with self.assertRaises(ValueError):
            attention_weights(model,inputs,"attention",max_tokens=4)


class TerminalTests(unittest.TestCase):
    def test_terminal_checkpoint_discovery_and_resume(self):
        run_id = "cli-synthetic-test-"+uuid.uuid4().hex[:8]
        env=dict(os.environ); env["PYTHONPATH"]=str(PROJECT/'src')
        command=[sys.executable,"-m","sleep_research.cli","demo","--project",str(PROJECT),"--run-id",run_id,"--skip-analysis","--max-epochs","1"]
        a=subprocess.run(command+["--resume","none"],cwd=PROJECT,env=env,capture_output=True,text=True)
        self.assertEqual(a.returncode,0,a.stdout+a.stderr)
        b=subprocess.run(command+["--resume","auto"],cwd=PROJECT,env=env,capture_output=True,text=True)
        self.assertEqual(b.returncode,0,b.stdout+b.stderr)
        history=json.loads((PROJECT/'experiments/cnn'/run_id/'performance/history.json').read_text())
        self.assertEqual([r['epoch'] for r in history],[0])


if __name__ == "__main__":
    unittest.main()
