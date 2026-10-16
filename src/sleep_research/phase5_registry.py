"""Phase 5 immutable selection and authorization; no training or EEG access."""
import json
from pathlib import Path
import sys
import yaml
import numpy as np
import pandas as pd
from .common import digest, fingerprint, json_write, now
from .experiments import Experiment, frozen_dataset_contract, software_versions, model_fingerprint
from .checkpoints import CheckpointManager
from .phase4_registry import envelope, sealed, read_registry, active_keys
from .analysis import load_export
from .cnn_inference import model_for
from models.advanced_sleep import make_advanced

PROJECT = Path(__file__).resolve().parents[2]
PRIMARY = ['M1', 'm5_ce_s42', 'm5_ce_s43', 'm5_ce_s44', 'm7_ce_s42', 'm7_ce_s43', 'm7_ce_s44']
SECONDARY = ['M4', 'm5_wce_s42']
KEYS = PRIMARY + SECONDARY


def environment():
    if not Path('/Volumes/Mac/datasets/sleep-edfx').is_dir() or not Path('/Volumes/Mac/envs/dl-m4/bin/python').is_file():
        raise FileNotFoundError('Existing dataset/interpreter required; stop without installing')
    if Path(sys.executable).resolve() != Path('/Volumes/Mac/envs/dl-m4/bin/python').resolve():
        raise RuntimeError('Use the existing dl-m4 interpreter')


def verify_files(project, files):
    project = Path(project).resolve()
    for relative, value in files.items():
        expected = value['sha256'] if isinstance(value, dict) else value
        path = (project / relative).resolve()
        if not path.is_relative_to(project) or not path.is_file() or digest(path) != expected:
            raise ValueError('Protected artifact changed: ' + relative)


def audit(project=PROJECT):
    """Validate historical source set without treating additive Phase5 files as Phase4 drift."""
    project = Path(project); environment()
    snapshot = json.loads((project/'results/phase5_preservation_snapshot.json').read_text())
    verify_files(project, snapshot['files'])
    if fingerprint(snapshot['phase4_source_files']) != snapshot['phase4_source_identity']:
        raise ValueError('Historical source identity differs')
    body = read_registry(project, check_source=False)
    if body['source_identity'] != snapshot['phase4_source_identity']:
        raise ValueError('Wrong campaign source snapshot')
    state = envelope(project/'results/phase4_campaign_state.json')
    keys = active_keys(body, envelope(project/'results/phase4_selection.json'))
    if state['status'] != 'completed' or len(keys) != 15:
        raise ValueError('All 15 completed experiments required')
    canonical = pd.read_csv(project/'results/protocol_a_index.csv').query("split == 'validation'").reset_index(drop=True)
    for key in keys:
        row = body['runs'][key]; cfg = row['configuration']
        exp = Experiment(project, cfg['family'], run_id=row['run_id'])
        marker = json.loads((exp.path/'reproducibility/phase4_complete.json').read_text())
        result_path = exp.path/'performance/phase4_result.json'
        artifacts = exp.path/'reproducibility/phase4_artifacts.json'
        if digest(result_path) != marker['result_sha256'] or digest(artifacts) != marker['artifact_manifest_sha256']:
            raise ValueError('Completion marker mismatch: '+key)
        verify_files(exp.path, json.loads(artifacts.read_text())['files'])
        result = json.loads(result_path.read_text())
        if exp.metadata['configuration'] != cfg or result['configuration_sha256'] != row['configuration_sha256']:
            raise ValueError('Experiment configuration changed')
        manager = CheckpointManager(exp.path/'reproducibility/checkpoints', exp.metadata['resume_fingerprints'], exp.run_id)
        states = {role:manager.read(path) for role,path in manager.discover().items() if path}
        latest,best = states['latest'],states['best']
        if not (latest['next_epoch'] >= cfg['max_epochs'] or latest['early_stopping']['stopped']):
            raise ValueError('Training incomplete: '+key)
        manifest,frame,features,_ = load_export(project/result['export'])
        if manifest['checkpoint_sha256'] != digest(manager.discover()['best']) or features.shape != (30266,64):
            raise ValueError('Best checkpoint/export mismatch')
        align(frame, canonical)
        if result['best_epoch_1_based'] != best['completed_epoch']+1:
            raise ValueError('Best epoch differs')
    return dict(successful=True,verified_at=now(),completed_runs=len(keys),prior_files_verified=len(snapshot['files']),
                processed_recordings=153,training_called=False,frozen_test_evaluated=False)


def align(frame, canonical):
    if len(frame) != len(canonical) or frame.epoch_id.duplicated().any():
        raise ValueError('Missing/duplicate evaluation epochs')
    for column in ['epoch_id','subject_id','recording_id','night','original_epoch_index','onset_seconds','split']:
        if frame[column].astype(str).tolist() != canonical[column].astype(str).tolist():
            raise ValueError('Evaluation target mismatch: '+column)
    np.testing.assert_array_equal(frame.true_label.to_numpy(), canonical.label.to_numpy())


def validate_subjects(index, assignments):
    if index.epoch_id.duplicated().any():raise ValueError('Duplicate targets')
    if set(assignments.split) != {'train','validation','test'} or assignments.subject_id.duplicated().any():
        raise ValueError('Invalid subject assignments')
    mapping = assignments.set_index('subject_id').split.to_dict()
    if any(mapping.get(str(r.subject_id)) != str(r.split) for r in index.itertuples()):
        raise ValueError('Subject leakage or missing subject')
    if any(g.subject_id.nunique()!=1 or g.split.nunique()!=1 or g.night.nunique()!=1 for _,g in index.groupby('recording_id',observed=True)):
        raise ValueError('Recording grouping conflict')


def selected_models(project):
    project = Path(project); body = read_registry(project, check_source=False); rows=[]
    for key in KEYS:
        if key in ['M1','M4']:
            ref = json.loads((project/'results'/('phase3_validation_export.json' if key=='M1' else 'phase3d_m4_export.json')).read_text())
            family='cnn';run=ref['run_id'];validation=ref['export'];explanations=ref['explanations']
        else:
            row=body['runs'][key];family=row['configuration']['family'];run=row['run_id']
            result=json.loads((project/'experiments'/family/run/'performance/phase4_result.json').read_text())
            validation=result['export'];explanations=result['explanations']
        exp=Experiment(project,family,run_id=run);cfg=exp.metadata['configuration']
        manager=CheckpointManager(exp.path/'reproducibility/checkpoints',exp.metadata['resume_fingerprints'],run)
        checkpoint=Path(manager.discover()['best']);state=manager.read(checkpoint)
        model=make_advanced(cfg['architecture']) if key not in ['M1','M4'] else model_for(exp)
        if model_fingerprint(model,cfg['architecture'])!=exp.metadata['resume_fingerprints']['architecture']:
            raise ValueError('Selected architecture differs')
        manifest,_,_,_=load_export(project/validation)
        if manifest['checkpoint_sha256']!=digest(checkpoint):raise ValueError('Validation did not use selected best')
        rows.append(dict(key=key,model_id=key if key in ['M1','M4'] else key.split('_')[0].upper(),
            role='primary' if key in PRIMARY else 'secondary',seed=cfg['seed'],family=family,run_id=run,
            checkpoint=str(checkpoint.relative_to(project)),checkpoint_sha256=digest(checkpoint),
            best_epoch_1_based=state['completed_epoch']+1,parameters=sum(p.numel() for p in model.parameters()),
            configuration=cfg,configuration_sha256=fingerprint(cfg),resume_fingerprints=exp.metadata['resume_fingerprints'],
            validation_export=validation,validation_export_sha256=digest(project/validation/'manifest.json'),
            validation_explanations=explanations,experiment_metadata_sha256=digest(exp.path/'reproducibility/experiment.json'),
            training_seconds=exp.metadata.get('total_training_attempt_seconds')))
    return rows


def source_files(project):
    project=Path(project)
    return {str(p.relative_to(project)):digest(p) for folder in ['src','scripts','tests'] for p in sorted((project/folder).rglob('*.py'))}


def prepare(project=PROJECT):
    project=Path(project);proof=audit(project)
    index=pd.read_csv(project/'results/protocol_a_index.csv');assignments=pd.read_csv(project/'results/subject_splits.csv')
    validate_subjects(index,assignments)
    test=index.query("split == 'test'").reset_index(drop=True)
    if len(test)!=31156 or test.subject_id.nunique()!=12 or test.recording_id.nunique()!=22:
        raise ValueError('Frozen test metadata differs')
    body=dict(schema=1,protocol='A',channel='EEG Fpz-Cz',sampling_hz=100,classes=['Wake','N1','N2','N3','REM'],
        dataset_contract=frozen_dataset_contract(project,'A'),models=selected_models(project),
        test_subjects=sorted(test.subject_id.unique().tolist()),test_epochs=len(test),test_recordings=22,
        test_epoch_ids_sha256=fingerprint(test.epoch_id.tolist()),protocol_index_sha256=digest(project/'results/protocol_a_index.csv'),
        source_files=source_files(project),software=software_versions(),
        selection='Completed validation only; no test selection, no ensembles or threshold/calibration fitting; argmax five logits',
        primary_groups={'M1':['M1'],'M5':['m5_ce_s42','m5_ce_s43','m5_ce_s44'],'M7':['m7_ce_s42','m7_ce_s43','m7_ce_s44']},
        primary_comparisons=[['M5','M1'],['M7','M1'],['M7','M5']],
        secondary_comparisons=[['M4','M1'],['m5_wce_s42','m5_ce_s42']],
        m8_status='exploratory; no final test predictions; proposed further validation ablations deferred, not a primary-model prerequisite',
        inference=dict(device='mps',batch_size=16,chunk_epochs=2048,num_workers=0,max_open_recordings=2,penultimate_width=64,
            centered_window=5,context='Original Protocol A contiguous indices only; stop at gaps; same subject/recording/night/split; masks and zero padding; no label inputs',
            no_context_models=['M1','M4'],rule='One prediction per central target, in canonical order; include all quality flags'),
        statistics=dict(seed=20261016,resamples=10000,confidence=0.95,primary_simultaneous_confidence=1-0.05/3,
            procedure='Paired cluster bootstrap: sample 12 subjects with replacement; keep all epochs and nights; same draws for all models and seeds; pool weighted subject confusion counts; average metrics across fixed seeds (no ensemble)',
            intervals=['accuracy','macro_f1','n1_f1','n3_f1','balanced_accuracy','cohen_kappa'],
            missing_class='Five-class macro-F1 always includes five classes with zero for undefined F1; subject balanced accuracy averages present true classes; undefined specificity/AUC/AP/kappa is null; report class support',
            variability='Seed sample SD ddof=1; null for single seed. Subject bootstrap conditional on fitted seeds; do not mix seed SD with subject CI; no epoch-IID tests or significance from point estimates'),
        analysis=dict(calibration_bins=10,tsne_max_samples=500,tsne_perplexity=30,tsne_iterations=500,seed=42,
            optional_umap=False,explanations='Reuse checkpoint-linked validation explanations; no outcome-selected test explanations or retraining',
            representation='Per-model PCA and uniform seeded t-SNE; centroid scatter ratio, descriptive; geometries are not comparable coordinate systems'),
        additional_training='None required for the predefined primary comparison. M8 confirmation is deferred and blocked once this plan is frozen.',
        training_called=False,frozen_test_evaluated=False)
    body['plan_id']='phase5-'+fingerprint(body)[:12]
    path=project/'results/phase5_evaluation_plan.json'
    if path.exists():
        if envelope(path)!=body:raise ValueError('Immutable plan exists; never adapt after test access')
    else:sealed(path,body)
    config=project/'configs/phase5_final_evaluation.yaml';serialized=yaml.safe_dump(body,sort_keys=False)
    if config.exists() and config.read_text()!=serialized:raise ValueError('Immutable YAML differs')
    if not config.exists():config.write_text(serialized)
    json_write(project/'results/phase5_project_validation.json',proof)
    return body


def plan(project=PROJECT):
    project=Path(project);body=envelope(project/'results/phase5_evaluation_plan.json')
    if yaml.safe_load((project/'configs/phase5_final_evaluation.yaml').read_text())!=body:raise ValueError('Final YAML changed')
    verify_files(project,body['source_files'])
    if frozen_dataset_contract(project,'A')!=body['dataset_contract'] or digest(project/'results/protocol_a_index.csv')!=body['protocol_index_sha256']:
        raise ValueError('Frozen data changed')
    if software_versions()!=body['software']:raise ValueError('Evaluation software differs from registration')
    for row in body['models']:verify_identity(project,row)
    return body


def verify_identity(project,row):
    project=Path(project);exp=Experiment(project,row['family'],run_id=row['run_id'])
    if exp.metadata['configuration']!=row['configuration'] or exp.metadata['resume_fingerprints']!=row['resume_fingerprints']:
        raise ValueError('Model identity changed')
    if digest(exp.path/'reproducibility/experiment.json')!=row['experiment_metadata_sha256'] or digest(project/row['checkpoint'])!=row['checkpoint_sha256']:
        raise ValueError('Selected checkpoint/metadata changed')
    manager=CheckpointManager(exp.path/'reproducibility/checkpoints',row['resume_fingerprints'],row['run_id'])
    if Path(manager.discover()['best']).resolve()!=(project/row['checkpoint']).resolve():raise ValueError('Best checkpoint pointer changed')
    state=manager.read(project/row['checkpoint'])
    if state['completed_epoch']+1!=row['best_epoch_1_based']:raise ValueError('Selected epoch differs')
    return exp


def freeze(project,reviewed_sha256):
    project=Path(project);body=plan(project);sha=fingerprint(body)
    if reviewed_sha256!=sha:raise PermissionError('Review the exact plan SHA-256 before freezing')
    evidence=json.loads((project/'results/phase5_preparation_validation.json').read_text())
    if not evidence['successful'] or evidence['plan_sha256']!=sha:raise ValueError('Passing preparation evidence required')
    optional=project/'results/phase5_optional_validation/status.json'
    if optional.exists() and json.loads(optional.read_text())['status']!='completed':raise ValueError('Complete all explicitly started optional validation runs before freezing')
    report=project/'results/phase5_preregistered_evaluation_plan.md'
    record=dict(plan_id=body['plan_id'],plan_sha256=sha,review_report_sha256=digest(report),reviewed_at=now(),
                source_fingerprint=fingerprint(body['source_files']),final_evaluation_authorized=False)
    path=project/'results/phase5_final_freeze.json'
    if path.exists():
        old=envelope(path)
        if old['plan_sha256']!=sha or old['review_report_sha256']!=digest(report):raise ValueError('Frozen plan/report changed')
        return old
    sealed(path,record);return record


def authorize(body,freeze_record,explicit,sha):
    if not explicit or sha!=fingerprint(body):raise PermissionError('Explicit --authorize-final-evaluation and exact --plan-sha256 required')
    if not freeze_record or freeze_record['plan_sha256']!=sha:raise PermissionError('Review and freeze plan before test access')
    return {'plan_id':body['plan_id'],'plan_sha256':sha,'explicit_final_evaluation':True}
