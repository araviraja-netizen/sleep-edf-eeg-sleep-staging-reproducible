"""Optional M8 validation confirmation definitions; manually authorized Terminal only.

Deferred proposals do not block the primary final plan. Once frozen, no additional
validation training from this command is permitted, avoiding test-informed revisions.
"""
import copy
import json
from pathlib import Path
import torch
from .common import fingerprint,json_write,now
from .phase5_registry import PROJECT,envelope,plan
from .phase4_registry import configuration


def definitions(project=PROJECT):
    rows=[]
    for ablation in ['control','no_multiscale','no_context']:
        for seed in [42,43,44]:
            # Start from the exact earlier controlled configuration; extend only budget/seed.
            cfg=configuration(project,'M8','cross_entropy',ablation=ablation)
            cfg.update(max_epochs=25,seed=seed,phase='5',stage='optional_m8_validation_confirmation')
            key=f'm8_{ablation}_s{seed}';run='phase5-validation-'+key+'-'+fingerprint(cfg)[:8]
            rows.append(dict(key=key,run_id=run,configuration=cfg,configuration_sha256=fingerprint(cfg)))
    return rows


def train(project=PROJECT,explicit=False):
    project=Path(project)
    if not explicit:raise PermissionError('Deferred M8 campaign requires --authorize-additional-validation-training')
    if (project/'results/phase5_final_freeze.json').exists():raise PermissionError('Final plan is frozen; do not modify models after finalization')
    if not torch.backends.mps.is_available():raise RuntimeError('MPS required; no full CPU fallback')
    from .phase4_campaign import build_trainer,real_datasets
    from .experiments import frozen_dataset_contract
    from .inference import export_inference
    from torch.utils.data import DataLoader
    from .analysis import load_export
    registry=definitions(project);contract=frozen_dataset_contract(project,'A')
    home=project/'results/phase5_optional_validation';home.mkdir(exist_ok=True)
    registered=home/'registry.json'
    if registered.exists() and json.loads(registered.read_text())!=registry:raise ValueError('Optional training registry changed')
    if not registered.exists():json_write(registered,registry)
    json_write(home/'status.json',dict(status='running',registered_runs=len(registry)))
    for row in registry:
        cfg=row['configuration'];trainset,val=real_datasets(project,cfg)
        try:
            trainer=build_trainer(project,cfg,contract,trainset,val,row['run_id'])
            # Existing trainer commits each epoch and repeats an interrupted epoch on auto resume.
            trainer.fit(resume='auto')
            with trainer.experiment.lock():
                ref=home/(row['key']+'.json')
                if ref.exists():load_export(project/json.loads(ref.read_text())['export']);continue
                path=export_inference(trainer.model,DataLoader(val,batch_size=16,shuffle=False,num_workers=0),trainer.experiment,device='mps',streaming=True,
                    intermediate_layers=[dict(name='intermediate',pooling='flatten')],metadata_fields=['quality_flagged','raw_rail_contact_samples','filter_boundary_flag','context_mask','context_epoch_ids','missing_context_count'])
                json_write(ref,dict(export=str(path.relative_to(project)),completed_at=now(),frozen_test_evaluated=False))
        finally:trainset.close();val.close();torch.mps.empty_cache()
    json_write(home/'status.json',dict(status='completed',registered_runs=len(registry),completed_at=now()))
