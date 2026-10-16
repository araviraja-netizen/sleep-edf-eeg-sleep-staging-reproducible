"""Registered full-budget M2 confirmation using the existing shared infrastructure."""

import argparse
import copy
import json
from pathlib import Path
import resource
import uuid

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Subset

from .cnn_baseline import verify_environment,read_config,datasets,make_model
from .checkpoints import CheckpointManager
from .common import digest,fingerprint,json_write,now
from .experiments import Experiment,frozen_dataset_contract,model_fingerprint
from .inference import export_inference
from .objectives import ResearchLoss,class_weights
from .phase3b import build_trial,provenance
from .trainer import SimulatedInterruption

PROJECT=Path(__file__).resolve().parents[2]
BASELINE='cnn1d-a-seed42-20261014-1800'


def configuration(project,model_id,seed=42):
    if model_id not in ('M1','M2','M3','M4') or seed not in (42,43,44):
        raise ValueError('Use registered M1..M4 and seeds42/43/44')
    cfg=copy.deepcopy(read_config(project))
    cfg.update(version='phase3c-v1',phase='3C',stage='full',model_id=model_id,
               baseline_run_id=BASELINE,seed=seed,progress_batches=1000)
    cfg['optimizer']=dict(name='AdamW' if model_id in ('M1','M4') else 'Adam',learning_rate=.001,weight_decay=.01,
                          betas=[.9,.999],eps=1e-8)
    # Torch accepts list betas; the exact JSON-serializable values are registered.
    cfg['loss']=dict(name='weighted_cross_entropy' if model_id in ('M3','M4') else 'cross_entropy')
    cfg.update(batch_size=16,max_epochs=25,precision='float32',device='mps',protocol='A',num_workers=0,
               early_stopping=dict(patience=8,min_delta=0.0),scheduler=dict(name='none'),selection_metric='validation_macro_f1')
    return cfg


def authorized_m2(config):
    if config.get('phase')!='3C' or config.get('model_id')!='M2' or config.get('seed')!=42 or config.get('stage')!='full':
        raise PermissionError('This execution stage authorizes only full M2 seed42; M3/M4/additional seeds stay unlaunched')
    expected=dict(name='Adam',learning_rate=.001,weight_decay=.01,betas=[.9,.999],eps=1e-8)
    if config['optimizer']!=expected or config['loss']!=dict(name='cross_entropy'):
        raise ValueError('M2 optimizer/loss differs from registration')
    if (config['batch_size'],config['max_epochs'],config['early_stopping'],config['scheduler'],config['device'])!=(16,25,dict(patience=8,min_delta=0.),dict(name='none'),'mps'):
        raise ValueError('Primary full-budget settings changed')


def register(project=PROJECT):
    project=Path(project);verify_environment()
    path=project/'results/phase3c_registry.json'
    if path.exists():return read_registry(project)
    contract=frozen_dataset_contract(project,'A')
    baseline=Experiment(project,'cnn',run_id=BASELINE)
    if contract!=baseline.metadata['dataset_contract']:raise ValueError('Frozen data contract differs from M1')
    base=read_config(project);model=make_model(base)
    architecture=model_fingerprint(model,base['architecture'])
    if architecture!=baseline.metadata['resume_fingerprints']['architecture'] or sum(p.numel() for p in model.parameters())!=40501:
        raise ValueError('M1 architecture changed')
    index=pd.read_csv(project/'results/protocol_a_index.csv')
    counts=np.bincount(index.loc[index.split=='train','label'].to_numpy(dtype=int),minlength=5).tolist()
    if sum(counts)!=134047:raise ValueError('Training census changed')
    configs={name:configuration(project,name) for name in ('M1','M2','M3','M4')}
    manager=CheckpointManager(baseline.path/'reproducibility/checkpoints',baseline.metadata['resume_fingerprints'],BASELINE)
    baseline_roles=manager.discover()
    retained={role:dict(path=str(Path(baseline_roles[role]).relative_to(project)),sha256=digest(baseline_roles[role]),
                       completed_epoch=manager.read(baseline_roles[role])['completed_epoch']) for role in ('best','latest')}
    rows={}
    for name,cfg in configs.items():
        run_id=BASELINE if name=='M1' else 'phase3c-'+name.lower()+'-s42-'+uuid.uuid4().hex[:8]
        rows[name]=dict(configuration=cfg,configuration_sha256=fingerprint(cfg),run_id=run_id,
                       initial_status='existing_completed' if name=='M1' else 'registered_unlaunched',
                       launch_authorized=name=='M2',architecture_fingerprint=architecture,dataset_contract=contract,
                       training_counts=counts,loss_alpha=(class_weights(counts).tolist() if name in ('M3','M4') else [1.]*5))
    body=dict(schema=1,registered_at=now(),baseline=BASELINE,models=rows,common_architecture_fingerprint=architecture,
              cnn_source_sha256=digest(project/'src/models/cnn1d.py'),shared_dataset_contract=contract,
              training_examples=134047,validation_examples=30266,locked_test_examples=31156,subjects=[54,12,12],
              class_weights_method='N_train/(5*n_c), all frozen training labels only',weighted_loss_reduction='sum(w_y*CE)/sum(w_y)',
              evaluation='five-class macro-F1 for selection; fixed class conventions from sleep_research.metrics',
              baseline_immutable_checkpoints=retained,planned_seeds=[42,43,44],only_new_full_run_authorized='M2 seed42')
    json_write(path,dict(body=body,sha256=fingerprint(body)))
    for name,row in rows.items():
        json_write(project/f'configs/phase3c/{name}_seed42.json',row)
    render_registry(project,body)
    return body


def read_registry(project=PROJECT):
    project=Path(project);envelope=json.loads((project/'results/phase3c_registry.json').read_text());body=envelope['body']
    if fingerprint(body)!=envelope['sha256']:raise ValueError('Registry checksum mismatch')
    if digest(project/'src/models/cnn1d.py')!=body['cnn_source_sha256']:raise ValueError('CNN source changed')
    if frozen_dataset_contract(project,'A')!=body['shared_dataset_contract']:raise ValueError('Frozen dataset changed')
    for name,row in body['models'].items():
        if fingerprint(row['configuration'])!=row['configuration_sha256'] or row['configuration']!=configuration(project,name):
            raise ValueError('Registered settings changed')
        if json.loads((project/f'configs/phase3c/{name}_seed42.json').read_text())!=row:raise ValueError('Configuration copy changed')
    for row in body['baseline_immutable_checkpoints'].values():
        if digest(project/row['path'])!=row['sha256']:raise ValueError('M1 checkpoint modified')
    return body


def render_registry(project,body):
    lines=['# Phase 3C — registered full-budget comparison',f"Registered before new training: {body['registered_at']}. Registry SHA-256: `{fingerprint(body)}`.",'',
           '| Model | Optimizer | Loss | Seed42 run ID | Registered status | New launch authorized |',
           '| --- | --- | --- | --- | --- | --- |']
    for name,row in body['models'].items():
        cfg=row['configuration'];lines.append(f"| {name} | {cfg['optimizer']['name']} | {cfg['loss']['name']} | {row['run_id']} | {row['initial_status']} | {row['launch_authorized']} |")
    lines+=['','M1 seed42 is the preserved original completed run, not a new experiment. Its original config/source/software and exact checkpoints remain authoritative. New M1 configurations are templates for future seeds43/44 only; seed42 must never be retrained.',
            'All cells share the exact40501-parameter CNN, Fpz-Cz100Hz/3000samples, frozen Protocol A, train-only normalization, class mapping and54/12/12 subjects. Train134047, validation30266, test31156locked. No quality exclusions or preprocessing regeneration. All configurations use LR0.001, batch16, float32, max25, patience8, seed42, no scheduler and validation macro-F1 selection.',
            'Adam uses coupled L2 regularization, while AdamW uses decoupled weight decay. Numeric decay0.01 is matched; M1 vs M2 estimates this optimizer/regularization combination, not an isolated change in adaptivity. Installed defaults are captured; betas0.9/0.999 and eps1e-8 are explicit in new configs.',
            f"Training counts [W,N1,N2,N3,R]: {body['models']['M3']['training_counts']}; weighted alpha: {body['models']['M3']['loss_alpha']}. Weights are N_train/(5*n_c) from all134047 training labels; weighted CE divides by sum of target weights. M1/M2 use ordinary CE.",
            'The common architecture/data fingerprints and all four immutable registered configurations are in `phase3c_registry.json` and `configs/phase3c/`. Same early-stop/selection definition does not require equal realized epoch counts. Each best checkpoint is selected on the full validation cohort.',
            'M1 uses the original Torch CE implementation; M2 uses the tested equivalent ordinary CE through the reusable objective interface. Numeric and gradient equivalence are tested. Software/source differences and non-bitwise MPS behavior remain reproducibility limitations.',
            '**Only M2 seed42 may be launched. M3/M4 and additional seeds remain prepared and unlaunched. Frozen test evaluation and advanced architectures remain outside scope.**']
    (Path(project)/'results/phase3c_experiment_registry.md').write_text('\n\n'.join(lines[:2])+'\n\n'+'\n'.join(lines[2:])+'\n')


def smoke(project=PROJECT):
    body=read_registry(project);config=copy.deepcopy(body['models']['M2']['configuration']);authorized_m2(config)
    if not torch.backends.mps.is_available():raise RuntimeError('MPS unavailable; run the documented local Terminal command')
    train,val=datasets(project,config)
    try:
        rng=np.random.default_rng(42);ti=np.sort(rng.choice(len(train),128,False)).tolist();vi=np.sort(rng.choice(len(val),64,False)).tolist()
        contract=copy.deepcopy(body['shared_dataset_contract']);contract['smoke_epoch_ids']={'train':train.index.iloc[ti].epoch_id.tolist(),'validation':val.index.iloc[vi].epoch_id.tolist()}
        config.update(stage='smoke',development_smoke_subset=True,max_epochs=2,progress_batches=0)
        run_id='phase3c-smoke-'+uuid.uuid4().hex[:8];counts=body['models']['M2']['training_counts']
        small_train,small_val=Subset(train,ti),Subset(val,vi)
        trainer=build_trial(project,config,small_train,small_val,counts,contract,run_id)
        batch=next(iter(trainer.train_loader));x=batch['eeg'].to('mps');y=batch['label'].to('mps')
        assert x.shape==(16,1,3000) and x.dtype==torch.float32 and bool(torch.isfinite(x).all())
        row=train.index.set_index('epoch_id').loc[batch['epoch_id'][0]]
        raw,_,_=train._recording(str(row.recording_id))
        expected=((raw[int(row.array_row)].astype(np.float64)-train.parameters['mean_microvolts'])/train.parameters['std_microvolts']).astype(np.float32)
        np.testing.assert_array_equal(x[0,0].cpu().numpy(),expected)
        logits=trainer.model(x);loss=trainer.criterion(logits,y)
        torch.testing.assert_close(loss,torch.nn.functional.cross_entropy(logits,y),atol=1e-6,rtol=1e-6)
        loss.backward();assert all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in trainer.model.parameters())
        def interrupt(epoch,batch,step):
            if epoch==1 and batch==0:raise SimulatedInterruption('M2 real-data interruption fixture')
        try:trainer.fit(resume='none',interrupt_callback=interrupt)
        except SimulatedInterruption:pass
        state=trainer.manager.recover();assert state['next_epoch']==1 and state['completed_epoch']==0
        restored=build_trial(project,config,small_train,small_val,counts,contract,run_id);restored.fit(resume='auto')
        assert restored.next_epoch==2 and restored.global_step==16 and [r['epoch'] for r in restored.history]==[0,1]
        result=dict(successful=True,generated_at=now(),device='mps',mps_available=True,input_shape=[16,1,3000],
                    finite_logits_loss_gradients=True,normalization_exact=True,ordinary_ce_equivalent=True,
                    recovered_next_epoch=1,completed_epochs=2,run_id=run_id,
                    registry_sha256=fingerprint(body),training_examples=128,validation_examples=64,
                    smoke_metrics_are_not_full_validation_results=True,mps_bitwise_guarantee=False)
        json_write(Path(project)/'results/phase3c_smoke_validation.json',result);return result
    finally:train.close();val.close()


def train_m2(project=PROJECT,resume='auto'):
    project=Path(project);verify_environment();body=read_registry(project)
    cfg=copy.deepcopy(body['models']['M2']['configuration']);authorized_m2(cfg)
    if not torch.backends.mps.is_available():
        raise RuntimeError('MPS unavailable. Local command: PYTHONPATH=src /Volumes/Mac/envs/dl-m4/bin/python -m sleep_research.phase3c train-m2 --resume auto')
    evidence=json.loads((project/'results/phase3c_smoke_validation.json').read_text())
    tests=json.loads((project/'results/phase3c_test_results.json').read_text())
    if not evidence['successful'] or evidence['registry_sha256']!=fingerprint(body) or not tests['successful'] or tests['skipped']:
        raise RuntimeError('Current registered M2 smoke and all-phase tests must pass')
    train,val=datasets(project,cfg);row=body['models']['M2']
    try:
        trainer=build_trial(project,cfg,train,val,row['training_counts'],body['shared_dataset_contract'],row['run_id'])
        provenance(trainer,train,row['training_counts'])
        root=trainer.experiment.path
        if not (root/'reproducibility/phase3c_registration.json').exists():json_write(root/'reproducibility/phase3c_registration.json',dict(registry_sha256=fingerprint(body),registration=body))
        json_write(project/'results/phase3c_m2_run.json',dict(run_id=row['run_id'],experiment=str(root.relative_to(project)),model_id='M2',seed=42))
        print('M2_RUN_ID='+row['run_id'],flush=True)
        try:trainer.fit(resume=resume)
        finally:
            json_write(root/'reproducibility/phase3c_resources.json',dict(process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                        mps_allocated_bytes_end=torch.mps.current_allocated_memory(),mps_driver_bytes_end=torch.mps.driver_allocated_memory(),
                        note='macOS RSS bytes; end MPS readings are not peak memory; shared/mapped accounting overlaps'))
        refpath=project/'results/phase3c_m2_export.json'
        if refpath.exists():return json.loads(refpath.read_text())
        export=export_inference(trainer.model,trainer.validation_loader,trainer.experiment,device='mps',
                   penultimate_layer='embedding',intermediate_layers=cfg['analysis']['intermediate_layers'],max_feature_dimension=512,
                   metadata_fields=['raw_rail_contact_samples','filter_boundary_flag','quality_flagged'])
        result=dict(run_id=row['run_id'],export=str(export.relative_to(project)),completed_epochs=trainer.next_epoch,
                    status=trainer.experiment.metadata['status'],synthetic=False,frozen_test_evaluated=False)
        json_write(refpath,result);print(json.dumps(result,indent=2));return result
    finally:train.close();val.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=['register','smoke','train-m2'])
    parser.add_argument('--project',type=Path,default=PROJECT);parser.add_argument('--resume',default='auto');args=parser.parse_args()
    if args.command=='register':print(json.dumps({'registered':list(register(args.project)['models'])}))
    elif args.command=='smoke':print(json.dumps(smoke(args.project),indent=2))
    else:train_m2(args.project,args.resume)
