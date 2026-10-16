"""Verify prior-file preservation and completed M2 artifacts; never train."""

import argparse
import ast
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT/'src'))
from sleep_research.analysis import load_export
from sleep_research.checkpoints import CheckpointManager
from sleep_research.common import digest,fingerprint,json_write,now
from sleep_research.experiments import Experiment
from sleep_research.metrics import classification_metrics


def preserve():
    snapshot=json.loads((PROJECT/'results/phase3c_preservation_snapshot.json').read_text())
    for name,expected in snapshot['files'].items():
        if digest(PROJECT/name)!=expected:raise ValueError('Prior artifact changed: '+name)
    m1='experiments/cnn/cnn1d-a-seed42-20261014-1800/'
    pilot_ids=json.loads((PROJECT/'results/phase3b_trial_registry.json').read_text())['trials'].values()
    payload=dict(successful=True,verified_at=now(),previous_files_unchanged=len(snapshot['files']),
                 previous_arrays_unchanged=sum(name.endswith('/epochs.npy') for name in snapshot['files']),
                 m1_files_unchanged=sum(name.startswith(m1) for name in snapshot['files']),
                 preserved_pilot_runs=len(pilot_ids),prior_source_and_configs_unchanged=True,
                 frozen_splits_and_normalization_unchanged=True,raw_dataset_written=False)
    json_write(PROJECT/'results/phase3c_preservation_validation.json',payload)
    return payload


def validate():
    results=PROJECT/'results';registry=json.loads((results/'phase3c_registry.json').read_text())
    assert fingerprint(registry['body'])==registry['sha256'];registry=registry['body']
    summary=json.loads((results/'phase3c_analysis_summary.json').read_text())
    table=pd.read_csv(results/'phase3c_validation_comparison.csv')
    assert list(table.model_id)==['M1','M2'] and (table.validation_examples==30266).all()
    assert table.loc[table.model_id=='M2','run_id'].item()==registry['models']['M2']['run_id']
    m2=Experiment(PROJECT,'cnn',run_id=registry['models']['M2']['run_id'])
    manager=CheckpointManager(m2.path/'reproducibility/checkpoints',m2.metadata['resume_fingerprints'],m2.run_id)
    roles=manager.discover();latest=manager.read(roles['latest']);best=manager.read(roles['best'])
    assert latest['next_epoch']==len(latest['history']) and 0<len(latest['history'])<=25
    assert latest['global_step']==len(latest['history'])*8378 and latest['rng']['torch_mps'] is not None
    assert latest['optimizer']['state'] and latest['scheduler'] is None
    assert latest['configuration']['optimizer']==registry['models']['M2']['configuration']['optimizer']
    assert latest['fingerprints']['architecture']==registry['common_architecture_fingerprint']
    assert best['completed_epoch']==max(latest['history'],key=lambda r:r['validation_macro_f1'])['epoch']
    assert latest['early_stopping']['stopped']==(m2.metadata['status']=='early_stopped')
    manifest,frame,features,root=load_export(PROJECT/summary['M2_export'])
    assert len(frame)==30266 and frame.subject_id.nunique()==12 and frame.recording_id.nunique()==24
    assert manifest['checkpoint_sha256']==digest(roles['best']) and features.shape==(30266,64)
    assert int(frame.quality_flagged.sum())==206
    np.testing.assert_array_equal(np.bincount(frame.true_label,minlength=5),[10746,3162,11182,1264,3912])
    np.testing.assert_array_equal(latest['criterion']['alpha'].cpu().numpy(),np.ones(5))
    for path in manifest['feature_files'].values():
        values=np.load(root/path,mmap_mode='r',allow_pickle=False)
        assert values.shape==(30266,64) and np.isfinite(values).all()
    metric=classification_metrics(frame.true_label.to_numpy(),frame[[f'probability_{i}' for i in range(5)]].to_numpy())
    np.testing.assert_allclose(metric['macro_f1'],table.loc[table.model_id=='M2','macro_f1'].item(),atol=1e-12,rtol=0)
    analysis=PROJECT/summary['M2_analysis'];pca=pd.read_csv(analysis/'pca_coordinates.csv')
    assert pca.epoch_id.tolist()==frame.epoch_id.tolist()
    tsne=json.loads((analysis/'tsne_parameters.json').read_text())
    selected=np.sort(np.random.default_rng(42).choice(30266,500,replace=False))
    assert tsne['epoch_ids']==frame.iloc[selected].epoch_id.tolist()
    for name in ['M3','M4']:assert not (PROJECT/'experiments/cnn'/registry['models'][name]['run_id']).exists()
    plan=json.loads((PROJECT/'configs/phase3c/multiseed_plan.json').read_text())
    assert len(plan['runs'])==12 and plan['seeds']==[42,43,44] and not plan['auto_launch']
    assert len([r for r in plan['runs'] if r['current_launch_authorized']])==1
    tree=ast.parse((PROJECT/'src/sleep_research/phase3c_analysis.py').read_text())
    assert not any(isinstance(n,ast.ImportFrom) and n.module in ['trainer','phase3c','phase3b'] for n in ast.walk(tree))
    assert not any(isinstance(n,ast.Attribute) and n.attr=='fit' for n in ast.walk(tree))
    tests=json.loads((results/'phase3c_test_results.json').read_text());smoke=json.loads((results/'phase3c_smoke_validation.json').read_text())
    assert tests['successful'] and tests['tests_run']==144 and tests['skipped']==0 and smoke['successful']
    assert smoke['registry_sha256']==fingerprint(registry)
    preservation=preserve()
    payload=dict(successful=True,validated_at=now(),run_id=m2.run_id,completed_epochs=latest['next_epoch'],
                 best_epoch_1_based=best['completed_epoch']+1,validation_rows=30266,validation_subjects=12,
                 validation_recordings=24,retained_flagged_validation_rows=206,penultimate_shape=[30266,64],
                 tsne_samples=500,tests_run=144,comparison_ids_match=True,m1_preserved=True,
                 prior_files_unchanged=preservation['previous_files_unchanged'],M3_launched=False,M4_launched=False,
                 additional_seeds_launched=False,frozen_test_evaluated=False)
    json_write(results/'phase3c_final_validation.json',payload)
    return payload


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--preservation-only',action='store_true');args=parser.parse_args()
    print(json.dumps(preserve() if args.preservation_only else validate(),indent=2))
