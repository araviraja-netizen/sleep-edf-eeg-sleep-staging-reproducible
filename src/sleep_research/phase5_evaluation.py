"""Terminal-only, explicitly authorized resumable inference using the shared exporter.

No training imports. Original experiment files are read-only: checkpoint/export pins
and chunk exports live in separately copied, plan-bound evaluation experiments.
"""
import contextlib
import csv
import fcntl
import json
import os
from pathlib import Path
import shutil
import tempfile
import time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader,Subset
from .common import atomic_file,digest,fingerprint,json_write,now
from .phase5_registry import PROJECT,plan,envelope,authorize,verify_identity,align,audit
from .phase5_data import FinalEpochDataset,FinalContextDataset,validate_saved_context
from .experiments import Experiment
from .analysis import load_export
from .inference import export_inference
from .cnn_inference import model_for
from models.advanced_sleep import make_advanced

EXTRAS=['raw_rail_contact_samples','filter_boundary_flag','quality_flagged','context_epoch_ids','context_mask','missing_context_count']


def workspace(project,body):return Path(project)/'experiments/final_evaluation'/body['plan_id']


def authorization(project,body,explicit,sha):
    record=envelope(Path(project)/'results/phase5_final_freeze.json') if (Path(project)/'results/phase5_final_freeze.json').exists() else None
    token=authorize(body,record,explicit,sha)
    if digest(Path(project)/'results/phase5_preregistered_evaluation_plan.md')!=record['review_report_sha256']:
        raise ValueError('Reviewed report changed')
    return token


def snapshot_experiment(project,body,row):
    original=verify_identity(project,row);home=workspace(project,body)
    destination=home/'experiments'/row['family']/row['run_id'];destination.parent.mkdir(parents=True,exist_ok=True)
    if not destination.exists():
        staging=Path(tempfile.mkdtemp(prefix='.snapshot-',dir=destination.parent))
        try:
            shutil.copytree(original.path/'reproducibility',staging/'reproducibility')
            for folder in ['performance','representations','analysis','explanations']:(staging/folder).mkdir()
            json_write(staging/'reproducibility/phase5_binding.json',dict(plan_sha256=fingerprint(body),checkpoint_sha256=row['checkpoint_sha256'],original_experiment=str(original.path)))
            os.replace(staging,destination)
        finally:
            if staging.exists():shutil.rmtree(staging)
    binding=json.loads((destination/'reproducibility/phase5_binding.json').read_text())
    if binding['plan_sha256']!=fingerprint(body) or binding['checkpoint_sha256']!=row['checkpoint_sha256']:
        raise ValueError('Evaluation snapshot belongs to another plan/checkpoint')
    exp=Experiment(home,row['family'],run_id=row['run_id'])
    if exp.metadata!=original.metadata:raise ValueError('Snapshot model metadata differs')
    checkpoint=destination/'reproducibility/checkpoints'/Path(row['checkpoint']).name
    if digest(checkpoint)!=row['checkpoint_sha256']:raise ValueError('Snapshot checkpoint corrupt')
    return exp


def checked_export(path,canonical,checkpoint_sha,allow_synthetic=False):
    manifest,frame,features,root=load_export(path,final_evaluation=True)
    align(frame,canonical)
    if manifest['checkpoint_sha256']!=checkpoint_sha or manifest['splits']!=['test'] or (manifest['synthetic'] and not allow_synthetic) or features.shape!=(len(canonical),64):
        raise ValueError('Final export identity/shape mismatch')
    for field in EXTRAS:
        if field not in frame:raise ValueError('Missing quality/context metadata')
    return manifest,frame,features,root


def find_chunk(experiment,canonical,checkpoint_sha):
    """Adopt an export committed before interruption but not yet referenced.

    Unpublished staging/representation orphans are ignored. A manifest is the shared
    exporter's commit point; any corrupt committed export stops recovery.
    """
    matches=[]
    for path in sorted((experiment.path/'performance/exports').glob('test-*/manifest.json')):
        manifest,frame,features,_=load_export(path.parent,final_evaluation=True)
        if frame.epoch_id.tolist()==canonical.epoch_id.tolist():
            checked_export(path.parent,canonical,checkpoint_sha,experiment.metadata['synthetic']);matches.append(path.parent)
    if len(matches)>1:raise ValueError('Duplicate committed chunks; refusing ambiguous recovery')
    return matches[0] if matches else None


def chunk_export(model,dataset,experiment,row,chunk_size=2048,batch_size=16,device='mps',interrupt_after=None):
    refs=[];folder=experiment.path/'performance/phase5_chunks';folder.mkdir(exist_ok=True)
    for number,start in enumerate(range(0,len(dataset),chunk_size)):
        stop=min(start+chunk_size,len(dataset));canonical=dataset.index.iloc[start:stop].reset_index(drop=True)
        pointer=folder/f'{number:06d}.json'
        if pointer.exists():
            ref=json.loads(pointer.read_text());path=experiment.path/ref['export']
            if digest(path/'manifest.json')!=ref['manifest_sha256']:raise ValueError('Committed chunk pointer checksum differs')
        else:
            seconds=None
            path=find_chunk(experiment,canonical,row['checkpoint_sha256'])
            if path is None:
                tick=time.monotonic()
                loader=DataLoader(Subset(dataset,list(range(start,stop))),batch_size=batch_size,shuffle=False,num_workers=0)
                path=export_inference(model,loader,experiment,device=device,checkpoint='best',penultimate_layer='embedding',
                    intermediate_layers=[dict(name='intermediate',pooling='flatten')] if row['model_id'] not in ['M1','M4'] else [],
                    metadata_fields=EXTRAS,streaming=True,final_evaluation=True)
                seconds=time.monotonic()-tick
            checked_export(path,canonical,row['checkpoint_sha256'],experiment.metadata['synthetic'])
            ref=dict(export=str(path.relative_to(experiment.path)),manifest_sha256=digest(path/'manifest.json'),start=start,stop=stop,inference_export_seconds=seconds)
            json_write(pointer,ref)
        if (ref['start'],ref['stop'])!=(start,stop):raise ValueError('Chunk offsets differ')
        checked_export(path,canonical,row['checkpoint_sha256'],experiment.metadata['synthetic']);refs.append(ref)
        print(now(),row['key'],'committed targets',stop,'/',len(dataset),flush=True)
        if interrupt_after is not None and len(refs)==interrupt_after:raise InterruptedError('Synthetic export interruption fixture')
    return refs


def merge_chunks(experiment,row,refs,canonical,plan_id):
    """Publish one conventional shared-analysis export; repeat calls are read-only."""
    export_id='final-'+plan_id;target=experiment.path/'performance/exports'/export_id
    if (target/'manifest.json').exists():
        _,frame,_,_=checked_export(target,canonical,row['checkpoint_sha256'],experiment.metadata['synthetic'])
        validate_saved_context(frame,canonical,1 if row['model_id'] in ['M1','M4'] else 5);return target
    records=[];features=[];first=None
    for ref in refs:
        path=experiment.path/ref['export']
        if digest(path/'manifest.json')!=ref['manifest_sha256']:raise ValueError('Chunk manifest changed')
        manifest,frame,values,root=checked_export(path,canonical.iloc[ref['start']:ref['stop']].reset_index(drop=True),row['checkpoint_sha256'],experiment.metadata['synthetic'])
        if first is None:first=manifest
        records.append(frame);features.append({key:np.load(root/name,mmap_mode='r',allow_pickle=False) for key,name in manifest['feature_files'].items()})
    frame=pd.concat(records,ignore_index=True);align(frame,canonical)
    validate_saved_context(frame,canonical,1 if row['model_id'] in ['M1','M4'] else 5)
    staging=Path(tempfile.mkdtemp(prefix='.final-merge-',dir=experiment.path));rep_target=experiment.path/'representations'/export_id
    try:
        perf=staging/'performance';rep=staging/'representations';perf.mkdir();rep.mkdir()
        frame.to_csv(perf/'predictions.csv',index=False)
        mapping=frame[['epoch_id','subject_id','recording_id','original_epoch_index','split']].copy();mapping.insert(0,'feature_row',np.arange(len(frame)))
        mapping.to_csv(rep/'feature_epoch_mapping.csv',index=False)
        shutil.copyfile(experiment.path/first['history_snapshot'],perf/'history.json')
        feature_files={}
        for key in first['feature_files']:
            name='penultimate.npy' if key=='penultimate' else 'intermediate-'+key.replace('.','_')+'.npy'
            values=np.lib.format.open_memmap(rep/name,mode='w+',dtype=np.float32,shape=(len(frame),features[0][key].shape[1]));offset=0
            for chunk in features:values[offset:offset+len(chunk[key])]=chunk[key];offset+=len(chunk[key])
            if offset!=len(frame):raise ValueError('Feature merge incomplete')
            values.flush();del values
            feature_files[key]=str((rep_target/name).relative_to(experiment.path))
        files={str((target/f.name).relative_to(experiment.path)):digest(f) for f in perf.iterdir()}
        files.update({str((rep_target/f.name).relative_to(experiment.path)):digest(f) for f in rep.iterdir()})
        manifest=dict(first,export_id=export_id,created_at=now(),sample_count=len(frame),
            predictions=str((target/'predictions.csv').relative_to(experiment.path)),feature_files=feature_files,
            feature_mapping=str((rep_target/'feature_epoch_mapping.csv').relative_to(experiment.path)),
            history_snapshot=str((target/'history.json').relative_to(experiment.path)),files=files,
            phase5_plan_id=plan_id,source_chunks=refs)
        json_write(perf/'manifest.json',manifest)
        # Only incomplete, plan-owned representation orphans can be replaced.
        if rep_target.exists():shutil.rmtree(rep_target)
        os.replace(rep,rep_target);target.parent.mkdir(parents=True,exist_ok=True);os.replace(perf,target)
    finally:shutil.rmtree(staging,ignore_errors=True)
    checked_export(target,canonical,row['checkpoint_sha256'],experiment.metadata['synthetic']);return target


def run(project=PROJECT,explicit=False,sha=None):
    body=plan(project);access=authorization(project,body,explicit,sha) # Before EEG loader, model forward or artifact access.
    if not torch.backends.mps.is_available():raise RuntimeError('Final inference requires local MPS; no silent CPU fallback')
    audit(project) # Recheck all prior experiment files and153 arrays immediately before test EEG access.
    home=workspace(project,body);home.mkdir(parents=True,exist_ok=True)
    with (home/'.evaluation.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        log=home/'evaluation.log'
        class Tee:
            def __init__(self,console,file):self.console,self.file=console,file
            def write(self,value):self.console.write(value);self.file.write(value);self.flush();return len(value)
            def flush(self):self.console.flush();self.file.flush()
        import sys
        with log.open('a',buffering=1) as handle,contextlib.redirect_stdout(Tee(sys.stdout,handle)),contextlib.redirect_stderr(Tee(sys.stderr,handle)):
            dataset=None
            try:
                dataset=FinalEpochDataset(project,body,access);results={}
                for row in body['models']:
                    print(now(),'MODEL',row['key'],'checkpoint',row['checkpoint_sha256'],flush=True)
                    json_write(home/'status.json',dict(status='inference',current=row['key'],completed=list(results),updated_at=now(),plan_sha256=fingerprint(body)))
                    exp=snapshot_experiment(project,body,row)
                    data=dataset if row['model_id'] in ['M1','M4'] else FinalContextDataset(dataset,access,body['inference']['centered_window'])
                    model=model_for(exp) if row['model_id'] in ['M1','M4'] else make_advanced(row['configuration']['architecture'])
                    with exp.lock():
                        refs=chunk_export(model,data,exp,row,body['inference']['chunk_epochs'],body['inference']['batch_size'])
                        export=merge_chunks(exp,row,refs,dataset.index,body['plan_id'])
                    results[row['key']]=dict(export=str(export.relative_to(Path(project))),manifest_sha256=digest(export/'manifest.json'),checkpoint_sha256=row['checkpoint_sha256'],inference_export_seconds=sum(r.get('inference_export_seconds') or 0 for r in refs),timing_complete=all(r.get('inference_export_seconds') is not None for r in refs))
                    json_write(home/'exports.json',dict(plan_sha256=fingerprint(body),models=results))
                    del model;torch.mps.empty_cache()
                from .phase5_analysis import regenerate
                regenerate(project,explicit=True,sha=sha)
                json_write(home/'status.json',dict(status='completed',completed=list(results),updated_at=now(),plan_sha256=fingerprint(body)))
                return dict(successful=True,models=len(results),plan_id=body['plan_id'],training_called=False)
            except BaseException:
                import traceback;traceback.print_exc()
                json_write(home/'status.json',dict(status='interrupted_or_failed',updated_at=now(),plan_sha256=fingerprint(body)))
                raise
            finally:
                if dataset is not None:dataset.close()
