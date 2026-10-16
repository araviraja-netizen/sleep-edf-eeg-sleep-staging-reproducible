"""End-to-end real artifact checks, without training or frozen-test inference."""

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT/'src'))
from sleep_research.analysis import load_export
from sleep_research.checkpoints import CheckpointManager
from sleep_research.common import digest,json_write,now
from sleep_research.experiments import Experiment
from sleep_research.metrics import classification_metrics


if __name__ == '__main__':
    results=PROJECT/'results'
    reference=json.loads((results/'phase3_validation_export.json').read_text())
    analysis_reference=json.loads((results/'phase3_analysis_summary.json').read_text())
    experiment=Experiment(PROJECT,'cnn',run_id=reference['run_id'])
    assert not experiment.metadata['synthetic']
    assert experiment.metadata['status'] in ('completed','early_stopped')
    config=experiment.metadata['configuration']
    assert config['device']=='mps' and config['protocol']=='A' and config['precision']=='float32'
    assert config['loss']=='ordinary_cross_entropy' and config['optimizer']['name']=='AdamW'
    manager=CheckpointManager(experiment.path/'reproducibility/checkpoints',
                               experiment.metadata['resume_fingerprints'],experiment.run_id)
    roles=manager.discover()
    latest,best=[manager.read(roles[name]) for name in ('latest','best')]
    history=latest['history']
    assert len(history)==latest['next_epoch'] and 0<len(history)<=25
    assert [r['epoch'] for r in history]==list(range(len(history)))
    assert latest['global_step']==len(history)*int(np.ceil(134047/config['batch_size']))
    for row in history:
        for key in ('train_loss','training_accuracy','training_macro_f1','validation_loss','validation_accuracy','validation_macro_f1','learning_rate'):
            assert np.isfinite(row[key])
    assert best['completed_epoch']==max(history,key=lambda r:r['validation_macro_f1'])['epoch']
    assert latest['scheduler'] is None and latest['rng']['torch_mps'] is not None
    assert latest['optimizer']['state']
    assert latest['early_stopping']['stopped']==(experiment.metadata['status']=='early_stopped')
    manifest,frame,features,root=load_export(PROJECT/reference['export'])
    assert not manifest['synthetic'] and manifest['splits']==['validation']
    assert manifest['checkpoint_sha256']==digest(roles['best'])
    assert len(frame)==30266 and frame.subject_id.nunique()==12 and frame.recording_id.nunique()==24
    assert np.array_equal(np.bincount(frame.true_label,minlength=5),[10746,3162,11182,1264,3912])
    frozen=pd.read_csv(results/'subject_splits.csv')
    assert set(frame.subject_id)==set(frozen.loc[frozen.split=='validation','subject_id'])
    assert not set(frame.subject_id)&set(frozen.loc[frozen.split!='validation','subject_id'])
    flagged=(frame.raw_rail_contact_samples>0)|(frame.filter_boundary_flag>0)
    assert int(flagged.sum())==206 and np.array_equal(flagged.astype(int),frame.quality_flagged)
    assert features.shape==(30266,64)
    for path in manifest['feature_files'].values():
        values=np.load(root/path,mmap_mode='r',allow_pickle=False)
        assert values.shape==(30266,64) and np.isfinite(values).all()
    metrics=classification_metrics(frame.true_label.to_numpy(),frame[[f'probability_{i}' for i in range(5)]].to_numpy())
    selected=history[best['completed_epoch']]
    np.testing.assert_allclose(metrics['macro_f1'],selected['validation_macro_f1'],atol=1e-12,rtol=0)
    np.testing.assert_allclose(metrics['accuracy'],selected['validation_accuracy'],atol=1e-12,rtol=0)
    analysis=PROJECT/analysis_reference['analysis']
    saved=json.loads((analysis/'metrics.json').read_text())
    for key in ('accuracy','macro_f1','balanced_accuracy','cohen_kappa'):
        np.testing.assert_allclose(saved[key],metrics[key],atol=1e-14,rtol=0)
    quality=json.loads((analysis/'quality_subset_metrics.json').read_text())
    assert quality['flagged']['sample_count']==206 and quality['flag_free']['sample_count']==30060
    assert quality['all']['sample_count']==30266 and quality['no_exclusion_training']
    for name,included in [('flagged',flagged),('flag_free',~flagged)]:
        subset=frame.loc[included]
        expected=classification_metrics(subset.true_label.to_numpy(),subset[[f'probability_{i}' for i in range(5)]].to_numpy())
        assert expected=={k:quality[name][k] for k in expected}
    subjects=json.loads((analysis/'subject_metrics.json').read_text())
    assert len(subjects)==12 and sum(r['sample_count'] for r in subjects.values())==30266
    pca=pd.read_csv(analysis/'pca_coordinates.csv')
    assert pca.epoch_id.tolist()==frame.epoch_id.tolist()
    tsne=json.loads((analysis/'tsne_parameters.json').read_text())
    expected_ids=frame.iloc[np.sort(np.random.default_rng(42).choice(len(frame),500,replace=False))].epoch_id.tolist()
    assert tsne['epoch_ids']==expected_ids
    explanations=PROJECT/reference['explanations']
    explanation_manifest=json.loads((explanations/'manifest.json').read_text())
    assert explanation_manifest['checkpoint_sha256']==manifest['checkpoint_sha256'] and not explanation_manifest['synthetic']
    with np.load(explanations/'explanations.npz',allow_pickle=False) as arrays:
        ids=arrays['epoch_ids'].tolist()
        assert len(ids)==5 and len(set(ids))==5 and set(ids)<=set(frame.epoch_id)
        subset=frame.set_index('epoch_id').loc[ids]
        assert subset.true_label.tolist()==list(range(5))
        assert arrays['target_labels'].tolist()==subset.predicted_label.tolist()
        for name in ('signed_saliency','absolute_saliency','grad_cam_1d'):
            assert arrays[name].shape==(5,1,3000) and np.isfinite(arrays[name]).all()
        assert (arrays['grad_cam_1d']>=0).all() and (arrays['grad_cam_1d']<=1).all()
    notebook=json.loads((results/'phase3_notebook_execution.json').read_text())
    tests=json.loads((results/'phase3_test_results.json').read_text())
    assert notebook['successful'] and notebook['image_outputs']>=10 and notebook['error_outputs']==0
    assert tests['successful'] and tests['tests_run']==118 and tests['skipped']==0
    preservation=json.loads((results/'phase3_preservation_validation.json').read_text())
    assert preservation['successful'] and preservation['previous_arrays_unchanged']==153
    result=dict(successful=True,validated_at=now(),run_id=experiment.run_id,
                completed_epochs=len(history),best_epoch_1_based=best['completed_epoch']+1,
                validation_rows=30266,validation_subjects=12,validation_recordings=24,
                flagged_validation_rows=206,embedding_shape=[30266,64],tsne_rows=500,
                explanation_rows=5,checkpoint_bound_artifacts=True,metrics_recomputed_equal=True,
                frozen_test_evaluated=False,synthetic=False,previous_arrays_unchanged=153)
    json_write(results/'phase3_final_validation.json',result)
    print(json.dumps(result,indent=2))
