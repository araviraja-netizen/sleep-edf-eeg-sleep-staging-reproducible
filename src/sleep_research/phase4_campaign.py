"""Manual-Terminal-only sequential Phase4 campaign using the existing Trainer."""
import argparse
import contextlib
import copy
import gc
import hashlib
import json
from pathlib import Path
import resource
import shutil
import sys
import traceback
import fcntl

import numpy as np
import pandas as pd
import torch

from models.advanced_sleep import make_advanced
from .phase4_registry import PROJECT,environment,settings,configuration,read_registry,register,source_identity,envelope,sealed,select_shortlist,active_keys
from .phase4_data import ContextDataset
from .real_data import ResearchEpochDataset
from .experiments import Experiment,model_fingerprint,software_versions
from .objectives import ResearchLoss,build_optimizer
from .checkpoints import seed_everything,CheckpointManager
from .trainer import Trainer
from .common import digest,fingerprint,json_write,now
from .phase4_analysis import complete_run,verify_run,run_root,training_complete,progress,regenerate
from .phase4_orchestration import CampaignEngine


class Tee:
    def __init__(self,console,file):self.console,self.file=console,file
    def write(self,value):self.console.write(value);self.file.write(value);self.flush();return len(value)
    def flush(self):self.console.flush();self.file.flush()
    def isatty(self):return False


def build_trainer(project,cfg,contract,train,val,run_id):
    seed_everything(cfg['seed']);model=make_advanced(cfg['architecture'])
    exp=Experiment(project,cfg['family'],cfg,contract,run_id)
    return Trainer(model,build_optimizer(model.parameters(),cfg['optimizer']),None,train,val,exp,cfg,
                   criterion=ResearchLoss(cfg['loss'],cfg['training_class_counts']))


def real_datasets(project,cfg):
    train=ContextDataset(ResearchEpochDataset(project,'train',cfg['max_open_recordings']),cfg['architecture']['window'])
    try:val=ContextDataset(ResearchEpochDataset(project,'validation',cfg['max_open_recordings']),cfg['architecture']['window'])
    except BaseException:train.close();raise
    if len(train)!=134047 or len(val)!=30266 or set(train.index.subject_id)&set(val.index.subject_id):
        train.close();val.close();raise ValueError('Frozen central target count/grouping differs')
    return train,val


def one_run(project,body,row,resume='auto'):
    cfg=row['configuration'];root=run_root(project,row)
    if (root/'reproducibility/phase4_complete.json').exists():return verify_run(project,row)
    train,val=real_datasets(project,cfg);trainer=None
    try:
        trainer=build_trainer(project,cfg,body['dataset_contract'],train,val,row['run_id'])
        if trainer.fingerprints['architecture']!=row['architecture_fingerprint']:raise ValueError('Model differs from registered architecture')
        repro=root/'reproducibility'
        if not (repro/'phase4_registration.json').exists():
            json_write(repro/'phase4_registration.json',dict(campaign_id=body['campaign_id'],configuration=row,registry_sha256=fingerprint(body)))
            json_write(repro/'context_mapping.json',dict(window=cfg['architecture']['window'],input_encoding='[window,3001]: EEG[:3000],binary mask[-1]',
                **{split:dict(targets=len(ds),neighbors_sha256=hashlib.sha256(ds.neighbors.tobytes()).hexdigest(),
                              central_ids_sha256=fingerprint(ds.index.epoch_id.tolist()),missing_context_slots=int((ds.neighbors<0).sum())) for split,ds in [('train',train),('validation',val)]}))
            for folder in ['src','configs']:
                for source in (Path(project)/folder).rglob('*'):
                    if source.is_file() and source.suffix in ['.py','.yaml','.json','.toml'] and '__pycache__' not in source.parts:
                        target=repro/'source_snapshot'/source.relative_to(project);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
        logs=root/'logs';logs.mkdir(exist_ok=True)
        with (logs/'training.log').open('a',buffering=1) as logfile,contextlib.redirect_stdout(Tee(sys.stdout,logfile)),contextlib.redirect_stderr(Tee(sys.stderr,logfile)):
            print(now(),'RUN',row['key'],row['run_id'],'resume',resume,flush=True)
            try:
                if not training_complete(trainer.experiment,trainer.manager,cfg):trainer.fit(resume=resume)
                else:
                    trainer.restore('auto') # Repair materialized histories after final-commit power failure, without retraining.
                    if trainer.experiment.metadata['status'] not in ['completed','early_stopped']:
                        recovered=trainer.manager.recover();trainer.experiment.update(status='early_stopped' if recovered['early_stopping']['stopped'] else 'completed')
                committed=trainer.manager.recover('auto')
                reported=trainer.experiment.metadata.get('total_training_attempt_seconds',0.)
                if committed.get('completed_training_seconds',0.)>reported:
                    trainer.experiment.update(total_training_attempt_seconds=committed['completed_training_seconds'],
                        timing_reconstruction_note='Committed epoch time recovered; unrecorded interrupted-attempt wall time is unknown, so this is a lower bound.')
                json_write(repro/'phase4_resources.json',dict(process_high_water_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                    mps_allocated_bytes_end=torch.mps.current_allocated_memory(),mps_driver_bytes_end=torch.mps.driver_allocated_memory(),
                    note='Sequential-process cumulative RSS high-water; MPS end readings, not peak GPU memory'))
                result=complete_run(project,body,row);print('VERIFIED',row['key'],'macro-F1',result['macro_f1'],flush=True)
                return result
            except BaseException:traceback.print_exc();raise
    finally:
        train.close();val.close();del trainer;gc.collect()
        if torch.backends.mps.is_available():torch.mps.empty_cache()


def choose(project,body,primary):
    path=Path(project)/'results/phase4_selection.json';selected=select_shortlist(primary)
    hashes={row['key']:digest(run_root(project,body['runs'][row['key']])/'performance/phase4_result.json') for row in primary}
    if path.exists():
        decision=envelope(path)
        if decision['campaign_id']!=body['campaign_id'] or decision['initial_result_sha256']!=hashes or [r['key'] for r in decision['selected']]!=[r['key'] for r in selected]:raise ValueError('Saved selection evidence changed')
        return decision
    decision=dict(campaign_id=body['campaign_id'],selected_at=now(),ranking=body['ranking'],initial_evidence=primary,selected=selected,
                  initial_result_sha256=hashes,validation_selection_bias=True)
    sealed(path,decision);return decision


def ready(project,body):
    environment()
    if not torch.backends.mps.is_available():raise RuntimeError('MPS required; no full CPU fallback')
    if software_versions()!=body['software']:raise RuntimeError('Registered software changed')
    tests=json.loads((Path(project)/'results/phase4_test_results.json').read_text())
    smoke=json.loads((Path(project)/'results/phase4_smoke_validation.json').read_text())
    for evidence in [tests,smoke]:
        if not evidence['successful'] or evidence['source_identity']!=body['source_identity']:raise RuntimeError('Tests and smoke must pass for final registered source')
    if tests['skipped']:raise RuntimeError('Skipped tests cannot authorize this campaign')
    if shutil.disk_usage(project).free<body['settings']['minimum_free_gib']*2**30:raise RuntimeError('Insufficient registered free SSD budget')


def run(project=PROJECT,resume='auto',key=None):
    project=Path(project);body=read_registry(project);ready(project,body)
    if resume=='none':
        if any(run_root(project,row).exists() for row in body['runs'].values()):raise FileExistsError('Existing campaign runs preserved; use --resume auto')
    elif resume!='auto' and key is None:raise ValueError('Explicit checkpoint resume requires run-one --key; campaign uses auto/none')
    logdir=project/'results/phase4_logs';logdir.mkdir(exist_ok=True)
    with (logdir/(body['campaign_id']+'.lock')).open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with (logdir/(body['campaign_id']+'.log')).open('a',buffering=1) as logfile,contextlib.redirect_stdout(Tee(sys.stdout,logfile)),contextlib.redirect_stderr(Tee(sys.stderr,logfile)):
            print(now(),'CAMPAIGN',body['campaign_id'],'resume',resume,flush=True)
            engine=CampaignEngine(body,project/'results/phase4_campaign_state.json',
                run=lambda k:one_run(project,body,body['runs'][k],resume),
                verify=lambda k:verify_run(project,body['runs'][k]),
                has_result=lambda k:(run_root(project,body['runs'][k])/'reproducibility/phase4_complete.json').exists(),
                selection=lambda primary:choose(project,body,primary))
            if key:
                decision=envelope(project/'results/phase4_selection.json') if (project/'results/phase4_selection.json').exists() else None
                keys=active_keys(body,decision)
                if key not in keys:raise PermissionError('Run is not yet eligible under registered selection')
                for predecessor in keys[:keys.index(key)]:verify_run(project,body['runs'][predecessor])
                engine.execute(key)
            else:
                try:
                    engine.run();regenerate(project)
                    # Execute the prepared analysis-only notebook using the same installed kernel.
                    import subprocess
                    subprocess.run([sys.executable,str(project/'scripts/validate_phase4_notebook.py')],cwd=project,check=True)
                    print('Campaign complete:15 verified runs; final reports and analysis notebook saved.',flush=True)
                except BaseException as exc:
                    if engine.state['status']=='completed':
                        engine.state.update(status='interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed',current='final_analysis')
                        engine.save()
                    raise


def status(project=PROJECT):
    project=Path(project);body=read_registry(project,check_source=False);statepath=project/'results/phase4_campaign_state.json'
    state=envelope(statepath) if statepath.exists() else dict(status='registered_unlaunched',current=None,runs={})
    rows=[]
    for key,row in body['runs'].items():
        node=state.get('runs',{}).get(key,dict(status='conditional' if row['kind']=='conditional_repeat' else 'pending'))
        info=dict(key=key,status=node['status'],completed_epochs=0,best_macro_f1=None)
        root=run_root(project,row);metadata=root/'reproducibility/experiment.json'
        if metadata.exists():
            exp=Experiment(project,row['configuration']['family'],run_id=row['run_id'])
            if 'resume_fingerprints' in exp.metadata:
                manager=CheckpointManager(root/'reproducibility/checkpoints',exp.metadata['resume_fingerprints'],row['run_id']);checkpoint=manager.recover('auto')
                if checkpoint:info.update(completed_epochs=checkpoint['next_epoch'],best_macro_f1=checkpoint['early_stopping']['best_score'],early_stopping=checkpoint['early_stopping'])
        rows.append(info)
    return dict(campaign_id=body['campaign_id'],status=state['status'],current=state.get('current'),completed=sum(r['status']=='completed' for r in rows),
                total_active_planned=15,runs=rows,log=str(project/'results/phase4_logs'/(body['campaign_id']+'.log')),
                note='Checkpoint epochs are authoritative; stale running status after reboot does not prove a live process.',frozen_test_evaluated=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=['register','run','run-one','status'])
    parser.add_argument('--project',type=Path,default=PROJECT);parser.add_argument('--resume',default='auto');parser.add_argument('--key');args=parser.parse_args()
    if args.command=='register':result=dict(campaign_id=register(args.project)['campaign_id'],full_training_launched=False)
    elif args.command=='status':result=status(args.project)
    else:
        if args.command=='run-one' and not args.key:parser.error('run-one requires --key')
        run(args.project,args.resume,args.key if args.command=='run-one' else None);return
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
