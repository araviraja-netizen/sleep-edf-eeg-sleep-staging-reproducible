"""Audit completed M1/M2/M4 validation artifacts without training or test inference."""
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT/'src'))
from sleep_research.analysis import load_export
from sleep_research.checkpoints import CheckpointManager
from sleep_research.common import digest,json_write,now
from sleep_research.experiments import Experiment,model_fingerprint
from sleep_research.cnn_inference import model_for
from sleep_research.metrics import classification_metrics
from sleep_research.phase3d_analysis import registration,verify_validation_rows


def validate(project=PROJECT):
    project=Path(project);results=project/'results';body=registration(project)
    canonical=pd.read_csv(results/'protocol_a_index.csv').query("split == 'validation'").reset_index(drop=True)
    flags=pd.read_csv(results/'quality_flagged_epochs.csv').query("split == 'validation' and wake_trimmed_inclusion == 1")
    refs={name:json.loads((results/file).read_text()) for name,file in [('M1','phase3_validation_export.json'),('M2','phase3c_m2_export.json'),('M4','phase3d_m4_export.json')]}
    summaries={name:json.loads((results/file).read_text()) for name,file in [('M2','phase3c_analysis_summary.json'),('M4','phase3d_analysis_summary.json')]}
    m1_summary=json.loads((results/'phase3_results_summary.json').read_text())
    analyses={name:project/summaries[name][name+'_analysis'] for name in ['M2','M4']}
    rows=[];details={}
    for name,ref in refs.items():
        manifest,frame,features,root=load_export(project/ref['export'])
        verify_validation_rows(manifest,frame,features,canonical,flags)
        exp=Experiment(project,'cnn',run_id=manifest['experiment_id']);cfg=exp.metadata['configuration']
        if exp.metadata['status'] not in ['completed','early_stopped']:raise ValueError('Unfinished '+name)
        if exp.metadata['dataset_contract']!=body['dataset_contract']:raise ValueError('Dataset mismatch '+name)
        model=model_for(exp)
        if sum(p.numel() for p in model.parameters())!=40501 or model_fingerprint(model,cfg['architecture'])!=body['architecture_fingerprint']:
            raise ValueError('Architecture mismatch '+name)
        manager=CheckpointManager(exp.path/'reproducibility/checkpoints',exp.metadata['resume_fingerprints'],exp.run_id)
        roles=manager.discover();states={role:manager.read(path) for role,path in roles.items() if path}
        best=states['best'];latest=states['latest'];history=latest['history']
        assert len(history)==latest['next_epoch'] and latest['global_step']==len(history)*8378
        assert best['completed_epoch']==max(history,key=lambda row:row['validation_macro_f1'])['epoch']
        assert latest['scheduler'] is None and latest['optimizer']['state'] and latest['rng']['torch_mps'] is not None
        assert latest['early_stopping']['stopped']==(exp.metadata['status']=='early_stopped')
        assert manifest['checkpoint_sha256']==digest(roles['best'])
        assert frame.subject_id.nunique()==12 and frame.recording_id.nunique()==24
        if name=='M4':
            assert cfg==dict(body['configuration'],training_class_counts=body['training_counts'],loss_alpha=body['weights_float32'])
            np.testing.assert_array_equal(latest['criterion']['alpha'].cpu().numpy(),body['weights_float32'])
        for state in states.values():
            assert all(torch.isfinite(value).all() for value in state['model'].values())
        for path in manifest['feature_files'].values():
            array=np.load(root/path,mmap_mode='r',allow_pickle=False)
            assert array.shape==(30266,64) and np.isfinite(array).all()
        scores=classification_metrics(frame.true_label.to_numpy(),frame[[f'probability_{i}' for i in range(5)]].to_numpy())
        np.testing.assert_allclose(scores['macro_f1'],history[best['completed_epoch']]['validation_macro_f1'],atol=1e-12,rtol=0)
        np.testing.assert_allclose(scores['accuracy'],history[best['completed_epoch']]['validation_accuracy'],atol=1e-12,rtol=0)
        analysis=analyses.get(name,root/'analysis'/Path(ref['export']).name)
        analysis_manifest=json.loads((analysis/'analysis_manifest.json').read_text())
        assert not analysis_manifest['synthetic'] and not analysis_manifest['final_evaluation'] and not analysis_manifest['training_called']
        assert analysis_manifest['source_manifest_sha256']==digest(project/ref['export']/'manifest.json')
        for relative,sha in analysis_manifest['files'].items():assert digest(analysis/relative)==sha
        saved_scores=json.loads((analysis/'metrics.json').read_text())
        np.testing.assert_allclose(saved_scores['macro_f1'],scores['macro_f1'],atol=1e-12,rtol=0)
        pca=pd.read_csv(analysis/'pca_coordinates.csv');np.testing.assert_array_equal(pca.epoch_id,frame.epoch_id)
        tsne=json.loads((analysis/'tsne_parameters.json').read_text())
        indices=np.sort(np.random.default_rng(42).choice(30266,500,replace=False))
        assert tsne['epoch_ids']==frame.iloc[indices].epoch_id.tolist()
        coords=pd.read_csv(analysis/'tsne_coordinates.csv');np.testing.assert_array_equal(coords.epoch_id,tsne['epoch_ids'])
        assert len(json.loads((analysis/'subject_metrics.json').read_text()))==12
        assert len(list(analysis.glob('hypnogram_*.png')))==24
        if name in ['M1','M4']:
            explanation=project/ref['explanations'];ex=json.loads((explanation/'manifest.json').read_text())
            assert ex['checkpoint_sha256']==manifest['checkpoint_sha256'] and ex['sample_count']==5 and not ex['synthetic']
            for relative,sha in ex['files'].items():assert digest(explanation/relative)==sha
            with np.load(explanation/'explanations.npz',allow_pickle=False) as arrays:
                assert arrays['epoch_ids'].tolist()==ex['epoch_ids']
                for key in ['signed_saliency','absolute_saliency','grad_cam_1d']:
                    assert arrays[key].shape==(5,1,3000) and np.isfinite(arrays[key]).all()
            selected=[frame.loc[frame.true_label==label,'epoch_id'].iloc[0] for label in range(5)]
            assert selected==ex['epoch_ids']
        if name in summaries:
            recorded=next(row for row in summaries[name]['models'] if row['model_id']==name)
            for metric in ['accuracy','macro_f1','balanced_accuracy','cohen_kappa']:
                np.testing.assert_allclose(recorded[metric],scores[metric],atol=1e-12,rtol=0)
        row=dict(model_id=name,run_id=exp.run_id,optimizer=cfg['optimizer']['name'],
                 loss='weighted_cross_entropy' if name=='M4' else 'ordinary_cross_entropy',seed=42,
                 completed_epochs=len(history),best_epoch_1_based=best['completed_epoch']+1,
                 training_seconds=exp.metadata['total_training_attempt_seconds'],validation_epochs=len(frame),
                 **{key:scores[key] for key in ['accuracy','macro_f1','balanced_accuracy','cohen_kappa']})
        for stage,value in scores['classes'].items():
            for key in ['precision','recall','f1']:row[stage.lower()+'_'+key]=value[key]
        rows.append(row)
        details[name]=dict(run_id=exp.run_id,analysis=str(analysis.relative_to(project)),export=ref['export'],
                          checkpoint_roles=roles,completed_epochs=len(history),best_epoch_1_based=best['completed_epoch']+1,
                          all_feature_shapes=[30266,64],pca_rows=30266,tsne_rows=500,flagged_validation_epochs=206,
                          retained_run_bytes=sum(f.stat().st_size for f in exp.path.rglob('*') if f.is_file()))
    pd.DataFrame(rows).to_csv(results/'phase3d_completed_models_validation_comparison.csv',index=False)
    c=json.loads((results/'phase3c_registry.json').read_text())['body']
    assert not (project/'experiments/cnn'/c['models']['M3']['run_id']).exists()
    assert not (project/'experiments/cnn'/c['models']['M4']['run_id']).exists()
    assert details['M4']['run_id']==body['run_id']
    assert len(list((results/'phase3d_figures').glob('M1_M4_hypnogram_*.png')))==24
    payload=dict(successful=True,verified_at=now(),models=rows,artifacts=details,validation_ids_identical=True,
                 validation_rows=30266,validation_subjects=12,validation_recordings=24,
                 best_latest_previous_checksums_verified=True,optimizer_rng_history_verified=True,
                 representations_analysis_explanations_verified=True,training_called=False,frozen_test_evaluated=False,
                 M3_launched=False,additional_seeds_launched=False,
                 note='Audit of existing manually trained M4 and preserved M1/M2; no model forward pass or new training.')
    json_write(results/'phase3d_completion_validation.json',payload)
    return payload

if __name__=='__main__':
    output=validate();print(json.dumps(output,indent=2))
