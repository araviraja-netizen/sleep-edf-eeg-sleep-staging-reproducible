"""CNN/resume failure cases. Training fixtures here are explicitly synthetic."""

import copy
import ast
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import DataLoader

from models.cnn1d import CNN1D, architecture_details
from sleep_research.cnn_baseline import read_config, make_trainer
from sleep_research.synthetic import SyntheticEpochs, synthetic_contract
from sleep_research.trainer import SimulatedInterruption
from sleep_research.real_data import ResearchEpochDataset
from sleep_research.inference import export_inference
from sleep_research.analysis import load_export
from sleep_research.checkpoints import IncompatibleCheckpoint

PROJECT = Path(__file__).resolve().parents[1]


class FlaggedSynthetic(SyntheticEpochs):
    def __getitem__(self, index):
        sample = super().__getitem__(index)
        return dict(sample, raw_rail_contact_samples=int(index % 3 == 0),
                    filter_boundary_flag=int(index == 0), quality_flagged=int(index % 3 == 0))


class CNNTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_shape_embedding_and_parameter_report(self):
        model = CNN1D().eval()
        values = torch.rand(3, 1, 3000)
        self.assertEqual(model(values).shape, (3, 5))
        self.assertEqual(model.extract_features(values).shape, (3, 64))
        details = architecture_details(model)
        self.assertEqual(details['parameters'], sum(p.numel() for p in model.parameters()))
        self.assertEqual(details['final_local_receptive_field_samples'], 515)
        actual = []
        handles = [block.register_forward_hook(lambda m, x, y: actual.append(list(y.shape[1:]))) for block in model.blocks]
        model(values)
        for h in handles:
            h.remove()
        reported = [r['output'] for r in details['layers'] if r['name'].endswith('.3')]
        self.assertEqual(actual, reported)

    def test_finite_backward_and_invalid_dimensions(self):
        model = CNN1D()
        loss = torch.nn.functional.cross_entropy(model(torch.rand(5, 1, 3000)), torch.arange(5))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters()))
        for shape in ((2, 3000), (2, 2, 3000), (2, 1, 2999)):
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                model(torch.zeros(shape))

    def test_group_normalization_has_no_cross_sample_statistics(self):
        model = CNN1D().eval()
        one = torch.rand(1, 1, 3000)
        other = torch.rand(1, 1, 3000) * 100
        torch.testing.assert_close(model(one), model(torch.cat([one, other]))[:1], atol=2e-6, rtol=2e-6)


class RealIntegrationTests(unittest.TestCase):
    def test_final_test_and_mixed_loading_disallowed(self):
        for split in ('test', None):
            with self.assertRaises(PermissionError):
                ResearchEpochDataset(PROJECT, split)

    def test_real_metadata_quality_and_normalization_match_existing_loader(self):
        dataset = ResearchEpochDataset(PROJECT, 'train', max_open_recordings=2)
        try:
            self.assertEqual(len(dataset), 134047)
            row = dataset.index.iloc[0]
            sample = dataset[0]
            array, metadata, _ = dataset._recording(sample['recording_id'])
            saved = metadata[int(row.array_row)]
            expected = ((array[int(row.array_row)].astype(np.float64)-dataset.parameters['mean_microvolts'])/dataset.parameters['std_microvolts']).astype(np.float32)
            np.testing.assert_array_equal(sample['eeg'][0], expected)
            self.assertEqual(sample['raw_rail_contact_samples'], int(saved['raw_rail_contact_samples']))
            self.assertEqual(sample['filter_boundary_flag'], int(saved['filter_boundary_flag']))
            self.assertEqual(sample['original_epoch_index'] * 30, sample['onset_seconds'])
            self.assertEqual(dataset.assignments[sample['subject_id']], 'train')
        finally:
            dataset.close()


class CNNResumeTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.tmp = tempfile.TemporaryDirectory(dir=PROJECT/'results')
        self.root = Path(self.tmp.name)
        self.config = read_config(PROJECT)
        self.config.update(device='cpu', synthetic=True, max_epochs=2, batch_size=5, progress_batches=0)
        self.config['early_stopping']['patience'] = 10
        self.train = FlaggedSynthetic('train', 10, 3000)
        self.validation = FlaggedSynthetic('validation', 10, 3000)
        self.contract = synthetic_contract(self.train, self.validation)

    def tearDown(self):
        self.tmp.cleanup()

    def build(self, run_id, config=None):
        return make_trainer(self.root, config or self.config, self.train, self.validation, run_id, self.contract)

    def test_cnn_cpu_interruption_repeats_incomplete_epoch_exactly(self):
        baseline = self.build('uninterrupted'); baseline.fit(resume='none')
        stopped = self.build('interrupted')
        def stop(epoch, batch, step):
            if epoch == 1 and batch == 0:
                raise SimulatedInterruption()
        with self.assertRaises(SimulatedInterruption):
            stopped.fit(resume='none', interrupt_callback=stop)
        self.assertEqual(stopped.manager.recover()['next_epoch'], 1)
        resumed = self.build('interrupted'); resumed.fit()
        for name, tensor in baseline.model.state_dict().items():
            torch.testing.assert_close(tensor, resumed.model.state_dict()[name], rtol=0, atol=0)
        for a, b in zip(baseline.history, resumed.history):
            self.assertEqual({k:v for k,v in a.items() if k != 'seconds'}, {k:v for k,v in b.items() if k != 'seconds'})
        self.assertEqual(resumed.global_step, 4)
        self.assertEqual([r['epoch'] for r in resumed.history], [0, 1])
        for key, values in baseline.optimizer.state_dict()['state'].items():
            for field, value in values.items():
                torch.testing.assert_close(value,resumed.optimizer.state_dict()['state'][key][field],rtol=0,atol=0)
        self.assertEqual(baseline.early_stopping,resumed.early_stopping)

    def test_training_metrics_are_five_class_and_saved(self):
        trainer = self.build('metrics'); trainer.fit(max_epochs=1)
        row = trainer.manager.recover()['history'][0]
        for key in ('training_accuracy', 'training_macro_f1', 'validation_accuracy', 'validation_macro_f1'):
            self.assertGreaterEqual(row[key], 0); self.assertLessEqual(row[key], 1)
        saved = json.loads((trainer.experiment.path/'performance/history.json').read_text())
        self.assertEqual(saved, trainer.history)
        self.assertIn('training_macro_f1', (trainer.experiment.path/'performance/history.csv').read_text())

    def test_real_architecture_corrupted_latest_recovers_then_continues(self):
        trainer = self.build('corrupt'); trainer.fit()
        Path(trainer.manager.discover()['latest']).write_bytes(b'corruption fixture')
        config = copy.deepcopy(self.config); config['max_epochs'] = 3
        resumed = self.build('corrupt', config); resumed.fit()
        self.assertTrue(resumed.manager.recovery_notes)
        self.assertEqual([r['epoch'] for r in resumed.history], [0, 1, 2])

    def test_optimizer_change_resume_rejected(self):
        trainer = self.build('incompatible'); trainer.fit(max_epochs=1)
        config = copy.deepcopy(self.config); config['optimizer']['learning_rate'] *= 2
        with self.assertRaises(ValueError):
            self.build('incompatible', config)

    def test_epoch_extension_and_repeated_resume_no_duplicates(self):
        trainer = self.build('extend'); trainer.fit(max_epochs=1)
        resumed = self.build('extend'); resumed.fit()
        resumed.fit()
        self.assertEqual([r['epoch'] for r in resumed.history], [0, 1])

    def test_quality_flags_export_with_stable_ids_and_embeddings(self):
        trainer = self.build('flags'); trainer.fit(max_epochs=1)
        export = export_inference(trainer.model, trainer.validation_loader, trainer.experiment,
                    metadata_fields=['raw_rail_contact_samples','filter_boundary_flag','quality_flagged'],
                    intermediate_layers=[dict(name='blocks.2.2',pooling='mean_last')])
        manifest, frame, features, root = load_export(export)
        self.assertEqual(features.shape, (10, 64))
        self.assertEqual(frame.quality_flagged.sum(), 4)
        self.assertEqual(frame.epoch_id.tolist(), [r['epoch_id'] for r in self.validation.metadata])
        self.assertEqual(np.load(root/manifest['feature_files']['blocks.2.2']).shape, (10, 64))

    def test_duplicate_export_metadata_fields_rejected(self):
        trainer = self.build('duplicate-fields'); trainer.fit(max_epochs=1)
        with self.assertRaises(ValueError):
            export_inference(trainer.model,trainer.validation_loader,trainer.experiment,metadata_fields=['epoch_id'])

    def test_analysis_modules_do_not_import_training_entrypoints(self):
        for name in ('phase3_analysis.py','cnn_inference.py'):
            tree=ast.parse((PROJECT/'src/sleep_research'/name).read_text())
            imports=[n.module for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)]
            self.assertFalse(set(imports)&{'trainer','cnn_baseline','synthetic'})
            self.assertFalse(any(isinstance(n,ast.Attribute) and n.attr=='fit' for n in ast.walk(tree)))

    def test_test_evaluation_rejected_before_forward(self):
        trainer = self.build('test-gate')
        trainer.validation_loader = DataLoader(FlaggedSynthetic('test', 10, 3000), batch_size=5)
        with self.assertRaises(PermissionError):
            trainer.fit(max_epochs=1)
        self.assertEqual(trainer.manager.recover()['next_epoch'], 0)


if __name__ == '__main__':
    unittest.main()
