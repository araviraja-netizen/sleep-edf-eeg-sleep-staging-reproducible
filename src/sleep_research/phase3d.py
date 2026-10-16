"""Terminal-first M4 preparation and short smoke checks; full runs require explicit train."""
import argparse
import contextlib
import copy
import json
from pathlib import Path
import resource
import sys
import traceback
import uuid

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Subset
import yaml

from .cnn_baseline import verify_environment,datasets,make_model
from .common import digest,fingerprint,json_write,now
from .experiments import Experiment,frozen_dataset_contract,model_fingerprint
from .objectives import class_weights
from .phase3b import build_trial,provenance
from .phase3c import configuration as previous_configuration,read_registry as previous_registry
from .trainer import SimulatedInterruption
from .checkpoints import capture_rng

PROJECT=Path(__file__).resolve().parents[2]
COUNTS=[43782,15249,47664,9302,18050]
LABELS=['Wake','N1','N2','N3','REM']


def expected_configuration(project):
    cfg=previous_configuration(project,'M4')
    cfg.update(version='phase3d-v1',phase='3D')
    return cfg


def read_registry(project=PROJECT):
    project=Path(project)
    envelope=json.loads((project/'results/phase3d_registry.json').read_text());body=envelope['body']
    if fingerprint(body)!=envelope['sha256']:raise ValueError('Registry checksum mismatch')
    config=yaml.safe_load((project/'configs/phase3d_m4.yaml').read_text())
    if config!=body['configuration'] or config!=expected_configuration(project):raise ValueError('Registered M4 settings changed')
    old=previous_registry(project)
    if fingerprint(old)!=body['phase3c_registry_sha256']:raise ValueError('Previous registry changed')
    if body['dataset_contract']!=old['shared_dataset_contract']:raise ValueError('Frozen contract differs')
    counts=np.bincount(pd.read_csv(project/'results/protocol_a_index.csv').query("split == 'train'").label,minlength=5).tolist()
    if counts!=COUNTS or counts!=body['training_counts']:raise ValueError('Training class census changed')
    np.testing.assert_array_equal(body['weights_float32'],class_weights(counts).tolist())
    for relative,sha in body['protected_checkpoints'].items():
        if digest(project/relative)!=sha:raise ValueError('Prior checkpoint changed')
    return body


def prepare(project=PROJECT):
    project=Path(project);verify_environment()
    target=project/'results/phase3d_registry.json'
    if target.exists():return read_registry(project)
    old=previous_registry(project);cfg=expected_configuration(project)
    model=make_model(cfg)
    assert sum(p.numel() for p in model.parameters())==40501
    assert model_fingerprint(model,cfg['architecture'])==old['common_architecture_fingerprint']
    counts=np.bincount(pd.read_csv(project/'results/protocol_a_index.csv').query("split == 'train'").label,minlength=5).tolist()
    if counts!=COUNTS:raise ValueError('Training census differs')
    protected={}
    for name in ['M1','M2']:
        exp=Experiment(project,'cnn',run_id=old['models'][name]['run_id'])
        for file in (exp.path/'reproducibility/checkpoints').glob('*.pt'):
            protected[str(file.relative_to(project))]=digest(file)
    body=dict(schema=1,registered_at=now(),run_id='phase3d-m4-s42-'+uuid.uuid4().hex[:8],
              initial_status='registered_unlaunched_terminal_only',configuration=cfg,
              dataset_contract=old['shared_dataset_contract'],architecture_fingerprint=old['common_architecture_fingerprint'],
              phase3c_registry_sha256=fingerprint(old),phase3c_planned_M4=old['models']['M4']['run_id'],
              baseline_run_id=old['baseline'],training_counts=counts,label_order=LABELS,
              weights_float64=(sum(counts)/(5*np.array(counts,dtype=float))).tolist(),weights_float32=class_weights(counts).tolist(),
              weights_formula='N_train/(5*n_c)',protected_checkpoints=protected,
              execution='manual local Terminal only; Codex short smoke authorized; no full training launched',
              frozen_test_evaluated=False)
    (project/'configs/phase3d_m4.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
    json_write(target,dict(body=body,sha256=fingerprint(body)))
    (project/'results/phase3d_experiment_registry.md').write_text(
        '# Phase 3D — registered M4, unlaunched\n\nRegistered '+body['registered_at']+'; registry SHA256 `'+fingerprint(body)+'`.\n\n'
        'Run ID: `'+body['run_id']+'`. Full training is reserved for the regular local Terminal. No full-run checkpoint or measured M4 result exists at preparation. The distinct Phase3D ID preserves the earlier Phase3C planned M4 reservation.\n\n'
        'Exact M1 CNN:40501 parameters; Fpz-Cz100Hz/3000 samples; ProtocolA;134047 training,30266 validation,31156 locked test epochs;54/12/12 subjects. AdamW lr0.001 decay0.01 betas0.9/0.999 eps1e-8, batch16,float32,MPS,seed42,max25,patience8,no scheduler. Selection: highest five-class validation macro-F1. Only intended M1 change is weighted training loss. M1 and M2 remain preserved.\n\n'
        'Counts [Wake,N1,N2,N3,REM]: `'+str(counts)+'`. Inverse weights N/(5*n_c), derived only from frozen training labels: `'+str(body['weights_float64'])+'`; operational float32: `'+str(body['weights_float32'])+'`. Weighted CE divides by sum of target weights. Ordinary validation CE is separately retained.\n\n'
        'Frozen hashes, architecture fingerprint, protected checkpoints and all hyperparameters: `phase3d_registry.json`. Source/software/config/RNG/optimizer/history are saved by the existing experiment/checkpoint framework when manually launched. No M3, further seed, preprocessing change or final test evaluation is authorized. See `phase3d_training_instructions.md`.\n')
    return read_registry(project)


class Tee:
    def __init__(self,console,file):self.console,self.file=console,file
    def write(self,value):
        self.console.write(value);self.file.write(value);self.flush();return len(value)
    def flush(self):self.console.flush();self.file.flush()
    def isatty(self):return False


def train(project=PROJECT,resume='auto'):
    project=Path(project);verify_environment();body=read_registry(project);cfg=body['configuration']
    if not torch.backends.mps.is_available():raise RuntimeError('MPS unavailable; full CPU fallback is forbidden')
    evidence=json.loads((project/'results/phase3d_smoke_validation.json').read_text())
    tests=json.loads((project/'results/phase3d_test_results.json').read_text())
    if not evidence['successful'] or evidence['registry_sha256']!=fingerprint(body) or not tests['successful'] or tests['skipped']:
        raise RuntimeError('Registered smoke and all-phase tests must pass first')
    # Detect code/config drift after preparation; repeat tests and smoke to refresh evidence.
    current=code_identity(project)
    if evidence['source_identity']!=current or tests.get('source_identity')!=current:
        raise RuntimeError('Source/tests/config changed since verification; rerun tests and short smoke')
    checkpoint_root=project/'experiments/cnn'/body['run_id']/'reproducibility/checkpoints'
    if resume=='none' and (any(checkpoint_root.glob('*.pt')) or any(checkpoint_root.glob('index*.json'))):
        raise FileExistsError('--resume none preserves existing checkpoints; use --resume auto')
    train_data,val=datasets(project,cfg)
    try:
        trainer=build_trial(project,cfg,train_data,val,body['training_counts'],body['dataset_contract'],body['run_id'])
        provenance(trainer,train_data,body['training_counts'])
        root=trainer.experiment.path;logs=root/'logs';logs.mkdir(exist_ok=True)
        registration=root/'reproducibility/phase3d_registration.json'
        if not registration.exists():json_write(registration,dict(body=body,sha256=fingerprint(body)))
        with (logs/'training.log').open('a',buffering=1) as handle, contextlib.redirect_stdout(Tee(sys.stdout,handle)),contextlib.redirect_stderr(Tee(sys.stderr,handle)):
            print(now(),'M4_RUN_ID='+body['run_id'],'resume='+str(resume),'status='+trainer.experiment.metadata['status'],flush=True)
            try:
                try:
                    trainer.fit(resume=resume)
                finally:
                    resources=dict(process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                        mps_allocated_bytes_end=torch.mps.current_allocated_memory(),mps_driver_bytes_end=torch.mps.driver_allocated_memory(),
                        recorded_at=now(),training_attempts=len(trainer.experiment.metadata['training_attempts']),
                        note='macOS RSS bytes; process high-water, not isolated GPU peak; MPS end readings overlap unified memory')
                    json_write(root/('reproducibility/phase3d_resources_attempt_'+uuid.uuid4().hex[:12]+'.json'),resources)
                    json_write(root/'reproducibility/phase3d_resources.json',resources)
                print('Training status:',trainer.experiment.metadata['status'],flush=True)
                from .phase3d_analysis import export_validation,compare
                export_validation(project)
                compare(project)
                print('M4 validation export, explanations and comparison complete.',flush=True)
            except BaseException:
                traceback.print_exc();raise
    finally:train_data.close();val.close()


def code_identity(project=PROJECT):
    paths=[]
    for folder,pattern in [('src','*.py'),('tests','*.py'),('scripts','*.py')]:
        paths.extend((Path(project)/folder).rglob(pattern))
    paths.append(Path(project)/'configs/phase3d_m4.yaml')
    return fingerprint({str(f.relative_to(project)):digest(f) for f in sorted(paths)})


def smoke(project=PROJECT):
    verify_environment()
    body=read_registry(project);config=copy.deepcopy(body['configuration'])
    if not torch.backends.mps.is_available():raise RuntimeError('MPS unavailable; run the documented local Terminal command')
    train,val=datasets(project,config)
    try:
        rng=np.random.default_rng(42);ti=np.sort(rng.choice(len(train),128,False)).tolist();vi=np.sort(rng.choice(len(val),64,False)).tolist()
        contract=copy.deepcopy(body['dataset_contract']);contract['smoke_epoch_ids']={'train':train.index.iloc[ti].epoch_id.tolist(),'validation':val.index.iloc[vi].epoch_id.tolist()}
        config.update(stage='smoke',development_smoke_subset=True,max_epochs=2,progress_batches=0)
        run_id='phase3d-smoke-'+uuid.uuid4().hex[:8];counts=body['training_counts']
        small_train,small_val=Subset(train,ti),Subset(val,vi)
        trainer=build_trial(project,config,small_train,small_val,counts,contract,run_id)
        batch=next(iter(trainer.train_loader));x=batch['eeg'].to('mps');y=batch['label'].to('mps')
        assert x.shape==(16,1,3000) and x.dtype==torch.float32 and bool(torch.isfinite(x).all())
        row=train.index.set_index('epoch_id').loc[batch['epoch_id'][0]]
        raw,_,_=train._recording(str(row.recording_id))
        expected=((raw[int(row.array_row)].astype(np.float64)-train.parameters['mean_microvolts'])/train.parameters['std_microvolts']).astype(np.float32)
        np.testing.assert_array_equal(x[0,0].cpu().numpy(),expected)
        logits=trainer.model(x);loss=trainer.criterion(logits,y)
        reference=torch.nn.functional.cross_entropy(logits,y,weight=trainer.criterion.alpha)
        torch.testing.assert_close(loss,reference,atol=1e-6,rtol=1e-6)
        actual_grad=torch.autograd.grad(loss,logits,retain_graph=True)[0]
        expected_grad=torch.autograd.grad(reference,logits,retain_graph=True)[0]
        torch.testing.assert_close(actual_grad,expected_grad,atol=1e-6,rtol=1e-5)
        loss.backward();assert all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in trainer.model.parameters())
        def interrupt(epoch,batch,step):
            if epoch==1 and batch==0:raise SimulatedInterruption('M4 real-data interruption fixture')
        try:trainer.fit(resume='none',interrupt_callback=interrupt)
        except SimulatedInterruption:pass
        state=trainer.manager.recover();assert state['next_epoch']==1 and state['completed_epoch']==0
        restored=build_trial(project,config,small_train,small_val,counts,contract,run_id)
        restored.restore('auto')
        assert restored.next_epoch==state['next_epoch'] and restored.global_step==state['global_step']
        assert restored.early_stopping==state['early_stopping'] and restored.history==state['history']
        current_rng=capture_rng(True)
        assert current_rng['python']==state['rng']['python'] and current_rng['numpy']==state['rng']['numpy']
        for key in ['torch_cpu','torch_mps']:
            torch.testing.assert_close(current_rng[key],state['rng'][key],rtol=0,atol=0)
        assert restored.optimizer.state_dict()['param_groups']==state['optimizer']['param_groups']
        for parameter,values in state['optimizer']['state'].items():
            for key,value in values.items():
                torch.testing.assert_close(restored.optimizer.state_dict()['state'][parameter][key].cpu(),value.cpu(),rtol=0,atol=0)
        restored.fit(resume='auto')
        assert restored.next_epoch==2 and restored.global_step==16 and [r['epoch'] for r in restored.history]==[0,1]
        result=dict(successful=True,generated_at=now(),device='mps',mps_available=True,input_shape=[16,1,3000],
                    finite_logits_loss_gradients=True,normalization_exact=True,weighted_ce_value_and_gradient_equivalent=True,optimizer_rng_epoch_early_stop_history_restored=True,
                    recovered_next_epoch=1,completed_epochs=2,run_id=run_id,
                    registry_sha256=fingerprint(body),training_examples=128,validation_examples=64,
                    smoke_metrics_are_not_full_validation_results=True,mps_bitwise_guarantee=False,source_identity=code_identity(project))
        json_write(Path(project)/'results/phase3d_smoke_validation.json',result);return result
    finally:train.close();val.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['prepare','validate','smoke','train','status'])
    parser.add_argument('--project',type=Path,default=PROJECT)
    parser.add_argument('--resume',default='auto',help='auto, none, or absolute compatible checkpoint path')
    args=parser.parse_args()
    if args.command=='prepare':result=prepare(args.project)
    elif args.command=='validate':result=dict(run_id=read_registry(args.project)['run_id'],valid=True,full_training_launched=False)
    elif args.command=='smoke':result=smoke(args.project)
    elif args.command=='train':train(args.project,args.resume);return
    else:
        from .phase3d_analysis import status
        result=status(args.project)
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
