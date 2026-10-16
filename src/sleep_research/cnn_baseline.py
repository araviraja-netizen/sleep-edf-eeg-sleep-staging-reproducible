"""Terminal entry point for the authorized Protocol A CNN baseline."""

import argparse
import copy
import json
import os
from pathlib import Path
import resource
import shutil
import time
import uuid

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
import yaml

from models.cnn1d import CNN1D, architecture_details
from .checkpoints import seed_everything
from .common import fingerprint, json_write, now
from .experiments import Experiment, frozen_dataset_contract
from .real_data import ResearchEpochDataset
from .trainer import Trainer, SimulatedInterruption

PROJECT = Path(__file__).resolve().parents[2]


def verify_environment():
    if not Path('/Volumes/Mac/datasets/sleep-edfx').is_dir():
        raise FileNotFoundError('Dataset missing: /Volumes/Mac/datasets/sleep-edfx')
    if not os.access('/Volumes/Mac/envs/dl-m4/bin/python', os.X_OK):
        raise FileNotFoundError('Existing executable Python missing')


def read_config(project=PROJECT):
    config = yaml.safe_load((Path(project) / 'configs/cnn1d_baseline.yaml').read_text())
    if config['protocol'] != 'A' or config['synthetic'] or config['loss'] != 'ordinary_cross_entropy':
        raise ValueError('This baseline requires real Protocol A and ordinary cross entropy')
    if config['precision'] != 'float32' or config['selection_metric'] != 'validation_macro_f1':
        raise ValueError('Use float32 and validation macro-F1 model selection')
    return config


def make_model(config):
    return CNN1D(**{k: v for k, v in config['architecture'].items() if k != 'name'})


def make_trainer(project, config, train, validation, run_id=None, contract=None):
    seed_everything(config['seed'])
    model = make_model(config)
    if config['optimizer']['name'] != 'AdamW':
        raise ValueError('Baseline optimizer must be AdamW')
    opt = config['optimizer']
    optimizer = torch.optim.AdamW(model.parameters(), lr=opt['learning_rate'], weight_decay=opt['weight_decay'])
    if config['scheduler']['name'] != 'none':
        raise ValueError('Initial baseline uses a fixed learning rate (no scheduler)')
    experiment = Experiment(project, 'cnn', config, contract or frozen_dataset_contract(project, 'A'), run_id)
    trainer = Trainer(model, optimizer, None, train, validation, experiment, config)
    return trainer


def datasets(project, config):
    verify_environment()
    train = ResearchEpochDataset(project, 'train', config['max_open_recordings'])
    validation = ResearchEpochDataset(project, 'validation', config['max_open_recordings'])
    if len(train) != 134047 or len(validation) != 30266:
        raise ValueError('Frozen Protocol A train/validation counts changed')
    if set(train.index.subject_id) & set(validation.index.subject_id):
        raise ValueError('Subject leakage')
    return train, validation


def save_run_provenance(trainer, train):
    root = trainer.experiment.path / 'reproducibility'
    # These creation-time artifacts remain immutable across resume/extension.
    if (root / 'architecture.json').exists():
        return
    json_write(root / 'architecture.json', architecture_details(trainer.model))
    counts = np.bincount(train.index.label.to_numpy(dtype=int), minlength=5)
    json_write(root / 'training_class_weights.json', dict(class_counts=counts.tolist(),
               formula='N_train/(5*class_count)', weights=(len(train)/(5*counts)).tolist(),
               applied_to_loss=False, computed_from='frozen Protocol A training labels only'))
    state = train.state
    json_write(root / 'protocol_and_normalization.json', dict(protocol='A',
               normalization=train.parameters, wake_trimming=state['pipeline_identity']['wake_trimming'],
               data_root=state['processed_root'], training_epochs=len(train), validation_epochs=30266,
               test_epochs_locked=31156, all_protocol_a_flagged_epochs_retained=2235,
               full_dataset_qualified_counts=dict(train_subjects=54, validation_subjects=12),
               memory_policy='read-only recording memmaps, max128, workers0; no entire normalized EEG copy'))
    source = root / 'source_snapshot'
    for folder in ('src', 'scripts', 'configs'):
        for path in (trainer.experiment.project / folder).rglob('*'):
            if path.is_file() and path.suffix in ('.py', '.yaml', '.toml') and '__pycache__' not in path.parts:
                target = source / path.relative_to(trainer.experiment.project)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, target)


def smoke(project, config):
    tick = time.monotonic()
    config = copy.deepcopy(config)
    config['device'] = 'mps' if torch.backends.mps.is_available() else 'cpu'
    config['max_epochs'] = 2
    config['progress_batches'] = 0
    config['development_smoke_subset'] = True
    train, validation = datasets(project, config)
    selection = np.random.default_rng(config['seed'])
    train_indices = np.sort(selection.choice(len(train), 128, replace=False)).tolist()
    val_indices = np.sort(selection.choice(len(validation), 64, replace=False)).tolist()
    small_train, small_val = Subset(train, train_indices), Subset(validation, val_indices)
    seed_everything(config['seed'])
    model = make_model(config).to(config['device'])
    batch = next(iter(DataLoader(small_train, batch_size=config['batch_size'])))
    values, labels = batch['eeg'], batch['label']
    assert values.shape == (16, 1, 3000) and values.dtype == torch.float32
    assert torch.isfinite(values).all() and ((labels >= 0) & (labels < 5)).all()
    row = train.index.iloc[train_indices[0]]
    array, _, _ = train._recording(str(row.recording_id))
    expected = ((array[int(row.array_row)].astype(np.float64) - train.parameters['mean_microvolts']) / train.parameters['std_microvolts']).astype(np.float32)
    np.testing.assert_array_equal(values[0, 0].numpy(), expected)
    logits = model(values.to(config['device']))
    loss = torch.nn.functional.cross_entropy(logits, labels.to(config['device']))
    loss.backward()
    assert logits.shape == (16, 5) and torch.isfinite(logits).all() and torch.isfinite(loss)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    contract = frozen_dataset_contract(project, 'A')
    contract['development_subset'] = dict(train_epoch_ids=train.index.iloc[train_indices].epoch_id.tolist(),
                                        validation_epoch_ids=validation.index.iloc[val_indices].epoch_id.tolist())
    run_id = 'real-cnn-smoke-' + uuid.uuid4().hex[:8]
    trainer = make_trainer(project, config, small_train, small_val, run_id, contract)
    def interrupt(epoch, batch_index, step):
        if epoch == 1 and batch_index == 1:
            raise SimulatedInterruption('Deliberate real-CNN smoke interruption')
    try:
        trainer.fit(resume='none', interrupt_callback=interrupt)
    except SimulatedInterruption:
        pass
    recovered = trainer.manager.recover('auto')
    assert recovered['completed_epoch'] == 0 and recovered['next_epoch'] == 1
    resumed = make_trainer(project, config, small_train, small_val, run_id, contract)
    resumed.fit(resume='auto')
    assert [r['epoch'] for r in resumed.history] == [0, 1] and resumed.global_step == 16
    result = dict(successful=True, generated_at=now(), device=config['device'], mps_built=torch.backends.mps.is_built(),
                  mps_available=torch.backends.mps.is_available(), finite_forward_backward=True,
                  normalization_exact_match=True, input_shape=list(values.shape), output_shape=list(logits.shape),
                  initial_loss=float(loss.detach()), labels=labels.tolist(), run_id=run_id,
                  training_samples=128, validation_samples=64, completed_epoch=1,
                  recovered_next_epoch=1, committed_history=resumed.history,
                  real_smoke_metrics_are_not_full_validation_results=True,
                  no_mps_bitwise_guarantee=True, elapsed_seconds=time.monotonic()-tick)
    json_write(Path(project)/'results/phase3_smoke_validation.json', result)
    train.close(); validation.close()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('smoke', 'train'))
    parser.add_argument('--project', type=Path, default=PROJECT)
    parser.add_argument('--run-id')
    parser.add_argument('--resume', default='auto')
    parser.add_argument('--max-epochs', type=int)
    args = parser.parse_args()
    verify_environment()
    config = read_config(args.project)
    if args.command == 'smoke':
        print(json.dumps(smoke(args.project, config), indent=2))
        return
    if config['device'] != 'mps' or not torch.backends.mps.is_available():
        raise SystemExit('Full real training requires available MPS. Run this command in the local VS Code Terminal; no CPU fallback.')
    evidence = json.loads((args.project/'results/phase3_smoke_validation.json').read_text())
    if not evidence['successful'] or evidence['device'] != 'mps':
        raise SystemExit('A successful MPS real-batch smoke test is required before full training')
    if args.max_epochs is not None:
        config['max_epochs'] = args.max_epochs
    train, validation = datasets(args.project, config)
    trainer = make_trainer(args.project, config, train, validation, args.run_id)
    save_run_provenance(trainer, train)
    json_write(args.project/'results/phase3_current_run.json', dict(run_id=trainer.experiment.run_id,
               experiment_path=str(trainer.experiment.path.relative_to(args.project)), synthetic=False))
    print('RUN_ID=' + trainer.experiment.run_id, flush=True)
    try:
        trainer.fit(resume=args.resume)
    finally:
        json_write(trainer.experiment.path/'reproducibility/resource_usage.json',
                   dict(process_high_water_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                        mps_allocated_bytes=torch.mps.current_allocated_memory(),
                        mps_driver_allocated_bytes=torch.mps.driver_allocated_memory(),
                        note='macOS ru_maxrss bytes; mapped pages/unified-memory accounting may overlap'))
        train.close(); validation.close()
    print(json.dumps(dict(run_id=trainer.experiment.run_id, latest_completed_epoch=trainer.next_epoch-1,
                          completed_epochs=trainer.next_epoch, status=trainer.experiment.metadata['status']), indent=2))


if __name__ == '__main__':
    main()
