"""Bounded optimizer-first, loss-second pilot on the unchanged Phase 3 CNN."""

import argparse
import copy
import json
from pathlib import Path
import resource
import shutil
import time
import uuid

import numpy as np
import torch
from torch.utils.data import DataLoader,Subset
import yaml

from .cnn_baseline import verify_environment,make_model,read_config,datasets
from .checkpoints import seed_everything
from .common import digest,fingerprint,json_write,now
from .experiments import Experiment,frozen_dataset_contract,model_fingerprint
from .inference import export_inference
from .objectives import ResearchLoss,build_optimizer,class_weights
from .trainer import Trainer,SimulatedInterruption

PROJECT=Path(__file__).resolve().parents[2]


def settings(project=PROJECT):
    return yaml.safe_load((Path(project)/'configs/phase3b_optimizer_search.yaml').read_text())


def registered_candidate(search,optimizer):
    candidate=next((c for c in search['optimizer_candidates'] if c['optimizer']==optimizer),None)
    if candidate is None:raise ValueError('Retained optimizer configuration is no longer registered')
    return candidate


def validate_architecture(project,config,model):
    previous=Experiment(project,'cnn',run_id=config['baseline_run_id']).metadata
    if config['architecture']!=previous['configuration']['architecture'] or sum(p.numel() for p in model.parameters() if p.requires_grad)!=40501:
        raise ValueError('Phase 3B must preserve the exact 40501-parameter M1 architecture')
    if model_fingerprint(model,config['architecture'])!=previous['resume_fingerprints']['architecture']:
        raise ValueError('M1 architecture source/modules changed')


def frozen_subsets(project,train,validation,search):
    pilot=search['pilot'];path=Path(project)/pilot['subset_path']
    if path.exists():
        saved=json.loads(path.read_text())
        if saved['selection_seed']!=pilot['selection_seed'] or saved['training_count']!=pilot['training_samples'] or saved['validation_count']!=pilot['validation_samples']:
            raise ValueError('Pilot subset cannot change; use a separately registered future pilot')
    else:
        rng=np.random.default_rng(pilot['selection_seed'])
        selected={name:np.sort(rng.choice(len(data),pilot[key],replace=False)).tolist()
                  for name,data,key in [('train',train,'training_samples'),('validation',validation,'validation_samples')]}
        saved=dict(selection_seed=pilot['selection_seed'],training_count=pilot['training_samples'],validation_count=pilot['validation_samples'],
                   indices=selected,epoch_ids={name:data.index.iloc[selected[name]].epoch_id.tolist() for name,data in [('train',train),('validation',validation)]},
                   frozen_subject_split_sha256=train.state['split_manifest_sha256'],selection='uniform without replacement; no label or quality-flag stratification')
        json_write(path,saved)
    if saved['frozen_subject_split_sha256']!=train.state['split_manifest_sha256']:
        raise ValueError('Frozen split changed')
    for name,data in [('train',train),('validation',validation)]:
        indices=saved['indices'][name]
        if len(set(indices))!=len(indices) or data.index.iloc[indices].epoch_id.tolist()!=saved['epoch_ids'][name]:
            raise ValueError('Frozen pilot indices/epoch IDs changed')
    return saved


def trial_config(project,search,candidate,loss,stage,seed):
    config=copy.deepcopy(read_config(project))
    config.update(version='phase3b-v1',phase='3B',baseline_run_id=search['baseline_run_id'],
                  seed=int(seed),optimizer=candidate['optimizer'],loss=dict(loss),stage=stage,
                  progress_batches=0,track_training_metrics=True)
    for key in ('device','protocol','precision','batch_size','num_workers','max_open_recordings','analysis'):
        config[key]=copy.deepcopy(search[key])
    if stage in ('optimizer_pilot','loss_pilot','smoke'):
        config['max_epochs']=search['pilot']['max_epochs']
        config['development_smoke_subset']=True  # Reject use through the full Phase 3 inference entrypoint.
    elif stage=='full':
        config['max_epochs']=search['full_run_proposal']['max_epochs']
    else:
        raise ValueError('Unknown experiment stage')
    config['early_stopping']=copy.deepcopy(search['full_run_proposal']['early_stopping'])
    config['scheduler']=copy.deepcopy(search['full_run_proposal']['scheduler'])
    return config


def build_trial(project,config,train,validation,counts,contract,run_id=None):
    seed_everything(config['seed'])
    model=make_model(config)
    validate_architecture(project,config,model)
    optimizer=build_optimizer(model.parameters(),config['optimizer'])
    if config['scheduler']['name']!='none':
        raise ValueError('Initial controlled comparisons keep the M1 constant-rate schedule')
    criterion=ResearchLoss(config['loss'],counts)
    config=copy.deepcopy(config)
    config['training_class_counts']=list(map(int,counts))
    config['loss_alpha']=criterion.alpha.tolist()
    experiment=Experiment(project,'cnn',config,contract,run_id)
    return Trainer(model,optimizer,None,train,validation,experiment,config,criterion=criterion)


def provenance(trainer,train,counts):
    root=trainer.experiment.path/'reproducibility'
    if (root/'phase3b_provenance.json').exists():
        return
    from models.cnn1d import architecture_details
    json_write(root/'architecture.json',architecture_details(trainer.model))
    json_write(root/'phase3b_provenance.json',dict(training_class_counts=list(map(int,counts)),
               inverse_frequency_weights=class_weights(counts).tolist(),actual_loss_alpha=trainer.criterion.alpha.detach().cpu().tolist(),
               weights_fitted_on='all 134047 frozen Protocol A training labels; never validation/test',
               normalization=train.parameters,prototype='M1 architecture unchanged, fresh seed-controlled initialization; no warm start',
               loss_reduction='weighted CE: sum(alpha_y*CE)/sum(alpha_y); focal: mean(alpha_y*(1-p_y)^gamma*CE)',
               validation_loss='ordinary unweighted CE for comparability; validation_objective_loss stored separately',
               optimizer_defaults=trainer.optimizer.defaults,
               acquisition_and_preprocessing_unchanged=True,quality_flags_preserved=True))
    for folder in ('src','configs'):
        for source in (trainer.experiment.project/folder).rglob('*'):
            if source.is_file() and source.suffix in ('.py','.yaml','.toml') and '__pycache__' not in source.parts:
                target=root/'source_snapshot'/source.relative_to(trainer.experiment.project)
                target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)


def run_trial(project,search,candidate,loss,stage,seed,train,validation,counts,subsets,run_id=None,resume='none'):
    config=trial_config(project,search,candidate,loss,stage,seed)
    contract=frozen_dataset_contract(project,'A')
    selected_train,selected_val=train,validation
    if stage!='full':
        contract['pilot_subsets_sha256']=fingerprint(subsets)
        contract['pilot_counts']=[subsets['training_count'],subsets['validation_count']]
        selected_train=Subset(train,subsets['indices']['train']);selected_val=Subset(validation,subsets['indices']['validation'])
    trainer=build_trial(project,config,selected_train,selected_val,counts,contract,
                        run_id or f'phase3b-{stage}-{candidate["id"]}-{loss["name"]}-s{seed}-{uuid.uuid4().hex[:8]}')
    result_path=trainer.experiment.path/'performance/phase3b_result.json'
    if result_path.exists():
        result=json.loads(result_path.read_text())
        from .analysis import load_export
        load_export(Path(project)/result['export'])
        trainer.manager.read(result['checkpoint_best'])
        return result
    provenance(trainer,train,counts)
    print('START',trainer.experiment.run_id,flush=True)
    tick=time.monotonic();trainer.fit(resume=resume)
    seconds=time.monotonic()-tick
    export=export_inference(trainer.model,trainer.validation_loader,trainer.experiment,device=config['device'],
              penultimate_layer='embedding',intermediate_layers=config['analysis']['intermediate_layers'],
              max_feature_dimension=512,metadata_fields=['raw_rail_contact_samples','filter_boundary_flag','quality_flagged'])
    result=dict(run_id=trainer.experiment.run_id,candidate_id=candidate['id'],optimizer=candidate['optimizer'],
                loss=loss,seed=seed,stage=stage,training_count=len(selected_train),validation_count=len(selected_val),
                export=str(export.relative_to(project)),wall_training_seconds=trainer.experiment.metadata['total_training_attempt_seconds'],
                last_attempt_training_seconds=seconds,
                completed_epochs=trainer.next_epoch,checkpoint_best=trainer.manager.discover()['best'],
                process_high_water_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                no_mps_bitwise_guarantee=True,synthetic=False,frozen_test_evaluated=False)
    json_write(trainer.experiment.path/'performance/phase3b_result.json',result)
    print('FINISH',trainer.experiment.run_id,seconds,flush=True)
    return result


def smoke(project,search,train,validation,counts):
    candidate=search['optimizer_candidates'][0]
    config=trial_config(project,search,candidate,dict(name='weighted_cross_entropy'),'smoke',search['seed'])
    config['max_epochs']=2
    contract=frozen_dataset_contract(project,'A');contract['smoke_epoch_ids']={}
    selected={}
    for name,data in [('train',train),('validation',validation)]:
        ids=np.random.default_rng(42).choice(len(data),32,replace=False).tolist()
        selected[name]=Subset(data,ids);contract['smoke_epoch_ids'][name]=data.index.iloc[ids].epoch_id.tolist()
    run_id='phase3b-smoke-'+uuid.uuid4().hex[:8]
    trainer=build_trial(project,config,selected['train'],selected['validation'],counts,contract,run_id)
    batch=next(iter(trainer.train_loader))
    forward_checks=[]
    # All implemented variants/optimizers receive a real-batch finite update on MPS.
    for opt_spec in [c['optimizer'] for c in search['optimizer_candidates'][::2]] + [dict(name='RMSprop',learning_rate=.001,momentum=.9,weight_decay=.01)]:
        for loss_spec in [dict(name='cross_entropy'),dict(name='weighted_cross_entropy'),dict(name='focal',gamma=2),dict(name='class_balanced_focal',gamma=2,beta=.9999)]:
            seed_everything(42);model=make_model(config).to(config['device'])
            criterion=ResearchLoss(loss_spec,counts).to(config['device']);optimizer=build_optimizer(model.parameters(),opt_spec)
            optimizer.zero_grad();logits=model(batch['eeg'].to(config['device']));loss=criterion(logits,batch['label'].to(config['device']))
            loss.backward()
            if not all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters()):
                raise ValueError('Nonfinite smoke gradients')
            optimizer.step()
            if not all(bool(torch.isfinite(p).all()) for p in model.parameters()):
                raise ValueError('Nonfinite smoke update')
            forward_checks.append(dict(optimizer=opt_spec['name'],loss=loss_spec['name'],finite=True,scalar=float(loss.detach())))
    def stop(epoch,batch,step):
        if epoch==1 and batch==0:
            raise SimulatedInterruption('Phase 3B real weighted-loss resume smoke')
    try:
        trainer.fit(resume='none',interrupt_callback=stop)
    except SimulatedInterruption:
        pass
    state=trainer.manager.recover();assert state['next_epoch']==1
    restored=build_trial(project,config,selected['train'],selected['validation'],counts,contract,run_id)
    restored.fit(resume='auto');assert restored.next_epoch==2 and restored.global_step==4
    result=dict(successful=True,device=config['device'],mps_available=torch.backends.mps.is_available(),checks=forward_checks,
                run_id=run_id,weighted_loss_resume_next_epoch=1,history_epochs=[r['epoch'] for r in restored.history],
                generated_at=now(),scope='32 real training/32 real validation epochs; infrastructure only')
    json_write(Path(project)/'results/phase3b_smoke_validation.json',result)
    return result


def pilot(project=PROJECT):
    verify_environment();search=settings(project)
    if search['device']!='mps' or not torch.backends.mps.is_available():
        raise RuntimeError('Real pilot requires available MPS; no package changes or long CPU fallback')
    candidates=search['optimizer_candidates'];losses=search['loss_candidates'];bounds=search['pilot']
    if len(candidates)>bounds['max_optimizer_trials'] or len(losses)>bounds['max_loss_trials'] or len(candidates)+len(losses)>bounds['max_total_trials']:
        raise ValueError('Pilot exceeds registered bounded trial budget')
    if bounds['training_samples']>8192 or bounds['validation_samples']>4096 or bounds['max_epochs']>3:
        raise ValueError('Current authorization is a bounded 8192/4096,3-epoch pilot')
    evidence=json.loads((Path(project)/'results/phase3b_test_results.json').read_text())
    if not evidence['successful'] or evidence['skipped']:
        raise RuntimeError('All-phase automated checks must pass before pilot')
    base=read_config(project);train,validation=datasets(project,base)
    results_path=Path(project)/'results/phase3b_pilot_runs.json'
    registry_path=Path(project)/'results/phase3b_trial_registry.json'
    registry=json.loads(registry_path.read_text()) if registry_path.exists() else dict(configuration_sha256=fingerprint(search),trials={})
    if registry['configuration_sha256']!=fingerprint(search):raise ValueError('Registered pilot configuration changed')
    try:
        counts=np.bincount(train.index.label.to_numpy(dtype=int),minlength=5)
        subsets=frozen_subsets(project,train,validation,search)
        smoke_path=Path(project)/'results/phase3b_smoke_validation.json'
        if not smoke_path.exists() or not json.loads(smoke_path.read_text())['successful']:
            smoke(project,search,train,validation,counts)
        measured=[]
        def save():
            json_write(results_path,dict(runs=measured,status='partial',seed=search['seed'],
                       subset_sha256=fingerprint(subsets),full_runs_launched=0))
        def registered_trial(candidate,loss,stage):
            key=f'{stage}:{candidate["id"]}:{loss["name"]}:{search["seed"]}'
            if key not in registry['trials']:
                registry['trials'][key]='phase3b-'+candidate['id']+'-'+loss['name']+'-s'+str(search['seed'])+'-'+uuid.uuid4().hex[:8]
                json_write(registry_path,registry)
            return run_trial(project,search,candidate,loss,stage,search['seed'],train,validation,counts,subsets,registry['trials'][key],'auto')
        for candidate in candidates:
            measured.append(registered_trial(candidate,dict(name='cross_entropy'),'optimizer_pilot'));save()
        from .analysis import load_export
        from .metrics import classification_metrics
        def selection_score(result):
            _,frame,_,_=load_export(Path(project)/result['export'])
            return classification_metrics(frame.true_label.to_numpy(),frame[[f'probability_{i}' for i in range(5)]].to_numpy())['macro_f1']
        winner=max(measured,key=selection_score)  # Strict primary metric; stable config order breaks exact ties.
        candidate=next(c for c in candidates if c['id']==winner['candidate_id'])
        json_write(Path(project)/'results/phase3b_optimizer_selection.json',dict(selected_candidate=candidate,
                   selected_run_id=winner['run_id'],validation_macro_f1=selection_score(winner),
                   stage='optimizer_pilot',selection_rule='highest best-epoch validation macro-F1; exact tie: registered order',
                   inference='provisional one-seed short subset pilot; not a full-run recommendation on its own'))
        for loss in losses:
            measured.append(registered_trial(candidate,loss,'loss_pilot'));save()
        json_write(results_path,dict(runs=measured,status='complete',seed=search['seed'],
                   subset_sha256=fingerprint(subsets),full_runs_launched=0))
    finally:
        train.close();validation.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['pilot','resume','train-one'])
    parser.add_argument('--project',type=Path,default=PROJECT)
    parser.add_argument('--run-id');parser.add_argument('--candidate');parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--loss',choices=['cross_entropy','weighted_cross_entropy','focal','class_balanced_focal'],default='cross_entropy')
    args=parser.parse_args();project=args.project.resolve()
    if args.command=='pilot':
        pilot(project);return
    verify_environment();search=settings(project)
    if not torch.backends.mps.is_available():raise RuntimeError('MPS required for real training')
    base=read_config(project);train,validation=datasets(project,base)
    try:
        counts=np.bincount(train.index.label.to_numpy(dtype=int),minlength=5)
        subsets=frozen_subsets(project,train,validation,search)
        if args.command=='resume':
            if not args.run_id:raise ValueError('Resume requires one existing run ID')
            experiment=Experiment(project,'cnn',run_id=args.run_id)
            cfg=experiment.metadata['configuration']
            if cfg.get('phase')!='3B':raise PermissionError('Never resume/modify baseline M1 through Phase 3B')
            candidate=registered_candidate(search,cfg['optimizer'])
            run_trial(project,search,candidate,cfg['loss'],cfg['stage'],cfg['seed'],train,validation,counts,subsets,args.run_id,'auto')
        else:
            if args.seed not in search['repeat_seeds']:raise ValueError('Use registered repeated seeds42,43,44')
            candidate=next((c for c in search['optimizer_candidates'] if c['id']==args.candidate),None)
            if candidate is None:raise ValueError('Select one registered candidate ID')
            if not (project/'results/phase3b_experiment_plan.md').exists():raise RuntimeError('Present the computational plan before full runs')
            loss=dict(name=args.loss)
            if 'focal' in args.loss:loss['gamma']=2.0
            if args.loss=='class_balanced_focal':loss['beta']=.9999
            run_trial(project,search,candidate,loss,'full',args.seed,train,validation,counts,subsets,args.run_id,'none')
    finally:
        train.close();validation.close()


if __name__=='__main__':main()
