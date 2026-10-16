"""Load/export an existing real CNN checkpoint without importing the trainer."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from models.cnn1d import CNN1D
from .common import json_write
from .experiments import Experiment
from .inference import load_model, export_inference
from .explanations import export_explanations
from .real_data import ResearchEpochDataset

PROJECT = Path(__file__).resolve().parents[2]
QUALITY_FIELDS = ['raw_rail_contact_samples', 'filter_boundary_flag', 'quality_flagged']


def real_experiment(project, run_id):
    experiment = Experiment(project, 'cnn', run_id=run_id)
    if experiment.metadata['synthetic'] or experiment.metadata['configuration'].get('development_smoke_subset'):
        raise ValueError('Select a full real Protocol A baseline run, not synthetic/smoke artifacts')
    if experiment.metadata['dataset_contract']['protocol_name'] != 'A':
        raise ValueError('Only Protocol A baseline supported')
    return experiment


def model_for(experiment):
    return CNN1D(**{k:v for k,v in experiment.metadata['configuration']['architecture'].items() if k != 'name'})


def export_validation(project, run_id, device='mps', checkpoint='best'):
    experiment = real_experiment(project, run_id)
    config = experiment.metadata['configuration']
    dataset = ResearchEpochDataset(project, 'validation', config['max_open_recordings'])
    loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=False, num_workers=0)
    try:
        analysis = config['analysis']
        model = model_for(experiment)
        export = export_inference(model, loader, experiment, checkpoint=checkpoint, device=device,
                    penultimate_layer=analysis['penultimate_layer'], intermediate_layers=analysis['intermediate_layers'],
                    max_feature_dimension=analysis['max_feature_dimension'], metadata_fields=QUALITY_FIELDS)
        model, state, path = load_model(model, experiment, checkpoint, device)
        count = config['explanations']['max_samples']
        if count < 1 or count > 8:
            raise ValueError('Bound representative explanations to 1..8 epochs')
        # First validation example of each true stage, chosen without scores/confidence.
        indices = [int(np.flatnonzero(dataset.index.label.to_numpy() == label)[0]) for label in range(5)]
        samples = [dataset[i] for i in indices[:count]]
        batch = torch.utils.data.default_collate(samples)
        explanations = export_explanations(model, batch, experiment.path/'explanations'/export.name,
                     path, conv_layer=config['explanations']['conv_layer'], max_samples=count)
        reference = dict(run_id=run_id, export=str(export.relative_to(project)),
                         explanations=str(explanations.relative_to(project)),
                         checkpoint=str(path.relative_to(project)), completed_epoch=state['completed_epoch'],
                         synthetic=False, split='validation', frozen_test_evaluated=False)
        json_write(Path(project)/'results/phase3_validation_export.json', reference)
        return reference
    finally:
        dataset.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('load-best','export-validation'))
    parser.add_argument('--project', type=Path, default=PROJECT)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--device', choices=('cpu','mps'), default='mps')
    args = parser.parse_args()
    if args.command == 'load-best':
        experiment = real_experiment(args.project,args.run_id)
        model, state, path = load_model(model_for(experiment),experiment,'best',args.device)
        print(json.dumps(dict(checkpoint=str(path), completed_epoch=state['completed_epoch'],
                              parameters=sum(p.numel() for p in model.parameters()), model_loaded=True),indent=2))
    else:
        print(json.dumps(export_validation(args.project,args.run_id,args.device),indent=2))


if __name__ == '__main__':
    main()
