"""Immutable campaign registrations and pure scheduling decisions; no trainer."""
import copy
import json
from pathlib import Path
import sys
import yaml
import numpy as np
import pandas as pd

from models.advanced_sleep import make_advanced
from .common import digest,fingerprint,json_write,now
from .experiments import frozen_dataset_contract,model_fingerprint,Experiment,software_versions
from .objectives import class_weights

PROJECT=Path(__file__).resolve().parents[2]
COUNTS=[43782,15249,47664,9302,18050]
FAMILIES={'M5':'cnn_lstm','M6':'attention','M7':'transformer','M8':'attention'}


def environment():
    if not Path('/Volumes/Mac/datasets/sleep-edfx').is_dir() or not Path('/Volumes/Mac/envs/dl-m4/bin/python').is_file():raise FileNotFoundError('Existing dataset/interpreter missing')
    if Path(sys.executable).resolve()!=Path('/Volumes/Mac/envs/dl-m4/bin/python').resolve():raise RuntimeError('Use existing dl-m4 Python')


def settings(project=PROJECT):return yaml.safe_load((Path(project)/'configs/phase4_campaign.yaml').read_text())


def source_identity(project=PROJECT):
    project=Path(project);paths=[]
    for folder in ['src','scripts','tests']:paths.extend((project/folder).rglob('*.py'))
    paths.append(project/'configs/phase4_campaign.yaml')
    return fingerprint({str(f.relative_to(project)):digest(f) for f in sorted(paths)})


def variant(model,ablation=None):
    return dict(name='AdvancedSleepModel',model_id=model,window=5,multiscale=ablation!='no_multiscale',context=ablation!='no_context')


def configuration(project,model,loss,seed=42,ablation=None,batch_size=None):
    spec=settings(project)
    if model not in spec['models'] or loss not in spec['losses'] or seed not in [42,43,44]:raise ValueError('Unregistered candidate')
    if ablation not in [None,*spec['ablations']] or (ablation and (model!='M8' or seed!=42 or loss!=spec['ablation_loss'])):raise ValueError('Invalid ablation')
    cfg={key:copy.deepcopy(spec[key]) for key in ['protocol','device','precision','max_epochs','early_stopping','optimizer','scheduler','selection_metric','num_workers','max_open_recordings']}
    cfg.update(version='phase4-v1',phase='4',stage='ablation' if ablation else 'full',synthetic=False,family=FAMILIES[model],
               model_id=model,seed=seed,architecture=variant(model,ablation),loss=dict(name=loss),batch_size=batch_size or spec['batch_size'],
               progress_batches=1000,track_training_metrics=True,training_class_counts=COUNTS,
               loss_alpha=class_weights(COUNTS).tolist() if loss=='weighted_cross_entropy' else [1.]*5,
               context_protocol='contiguous Protocol A neighbors; masked zero padding; center target; future EEG offline',
               analysis=copy.deepcopy(spec['analysis']),ablation=ablation)
    if ablation:cfg['max_epochs']=spec['ablation_epochs']
    return cfg


def envelope(path):
    value=json.loads(Path(path).read_text())
    if fingerprint(value['body'])!=value['sha256']:raise ValueError('Checksum mismatch: '+str(path))
    return value['body']


def sealed(path,body):json_write(path,dict(body=body,sha256=fingerprint(body)))


def register(project=PROJECT):
    project=Path(project);environment();spec=settings(project);source=source_identity(project)
    bench=json.loads((project/'results/phase4_benchmarks.json').read_text())
    if not bench['successful'] or bench['source_identity']!=source:raise ValueError('Benchmark final source before registering')
    contract=frozen_dataset_contract(project,'A');baseline=Experiment(project,'cnn',run_id='cnn1d-a-seed42-20261014-1800')
    if contract!=baseline.metadata['dataset_contract']:raise ValueError('Frozen data differs from M1')
    counts=np.bincount(pd.read_csv(project/'results/protocol_a_index.csv').query("split == 'train'").label,minlength=5).tolist()
    if counts!=COUNTS:raise ValueError('Frozen training class counts changed')
    campaign='phase4-'+fingerprint(dict(source=source,bench=digest(project/'results/phase4_benchmarks.json'),settings=spec))[:12]
    pointer=project/'results/phase4_registry.json'
    if pointer.exists():
        old=envelope(pointer)
        if old['campaign_id']==campaign:return read_registry(project)
        for row in old['runs'].values():
            if (project/'experiments'/row['configuration']['family']/row['run_id']).exists():raise ValueError('A full campaign run exists; registration cannot change')
    runs={};initial=[];conditional=[];ablations=[]
    for model in spec['models']:
        batch=bench['models'][model]['batch_size']
        for loss in spec['losses']:
            for seed in [42,*spec['repeat_seeds']]:
                key=model.lower()+'_'+('wce' if loss=='weighted_cross_entropy' else 'ce')+'_s'+str(seed)
                cfg=configuration(project,model,loss,seed,batch_size=batch);primary=seed==42
                runs[key]=dict(key=key,run_id=campaign+'-'+key,configuration=cfg,configuration_sha256=fingerprint(cfg),
                    architecture_fingerprint=model_fingerprint(make_advanced(cfg['architecture']),cfg['architecture']),
                    parameters=sum(p.numel() for p in make_advanced(cfg['architecture']).parameters()),
                    kind='initial' if primary else 'conditional_repeat',parent_key=None if primary else key.rsplit('_s',1)[0]+'_s42')
                (initial if primary else conditional).append(key)
    for ablation in spec['ablations']:
        key='m8_ablation_'+ablation+'_s42';batch=bench['models']['M8']['batch_size']
        cfg=configuration(project,'M8',spec['ablation_loss'],ablation=ablation,batch_size=batch)
        runs[key]=dict(key=key,run_id=campaign+'-'+key,configuration=cfg,configuration_sha256=fingerprint(cfg),
                      architecture_fingerprint=model_fingerprint(make_advanced(cfg['architecture']),cfg['architecture']),
                      parameters=sum(p.numel() for p in make_advanced(cfg['architecture']).parameters()),kind='ablation',parent_key=None)
        ablations.append(key)
    protected={}
    for name,ref in [('M1','phase3_validation_export.json'),('M4','phase3d_m4_export.json')]:
        reference=json.loads((project/'results'/ref).read_text());exp=Experiment(project,'cnn',run_id=reference['run_id'])
        for f in (exp.path/'reproducibility/checkpoints').glob('*.pt'):protected[str(f.relative_to(project))]=digest(f)
    body=dict(schema=1,campaign_id=campaign,registered_at=now(),settings=spec,dataset_contract=contract,source_identity=source,
              software=software_versions(),benchmark_sha256=digest(project/'results/phase4_benchmarks.json'),runs=runs,
              initial_keys=initial,conditional_keys=conditional,ablation_keys=ablations,protected_checkpoints=protected,
              actual_planned_runs=15,reserved_configurations=len(runs),training_counts=COUNTS,weights=class_weights(COUNTS).tolist(),
              ranking=spec['ranking'],repeat_selection='top2 initial configurations; seeds43/44; exact ties N1,param,time,key',
              ablation_comparison='three fresh seed42 CE runs, equal3-epoch full-dataset budgets: control,no_multiscale,no_context',
              frozen_test_evaluated=False,terminal_execution_only=True)
    directory=project/'configs/phase4'/campaign;directory.mkdir(parents=True,exist_ok=True)
    for key,row in runs.items():
        target=directory/(key+'.json')
        if target.exists() and json.loads(target.read_text())!=row:raise ValueError('Immutable config collision')
        if not target.exists():json_write(target,row)
    immutable=directory/'registry.json'
    if immutable.exists():raise ValueError('Immutable registry collision')
    sealed(immutable,body);sealed(pointer,body)
    lines=['# Phase4 pre-registered campaign',f"Campaign `{campaign}`; registered {body['registered_at']}. Immutable registry: `{immutable.relative_to(project)}`.",'',
           'Eight seed42 primary runs, then four conditional repeats (two shortlisted configurations ×seeds43/44), then three matched bounded M8 CE ablation/control runs.27 immutable configurations reserve all16 possible repeat choices; only4 repeats execute.15 actual training runs. No full training launched during preparation.','',
           '| Key | Model | Loss | Seed | Max epochs | Batch | Parameters | Role |','| --- | --- | --- | ---: | ---: | ---: | ---: | --- |']
    for key,row in runs.items():
        cfg=row['configuration'];lines.append(f"| {key} | {cfg['model_id']} | {cfg['loss']['name']} | {cfg['seed']} | {cfg['max_epochs']} | {cfg['batch_size']} | {row['parameters']} | {row['kind']} |")
    lines+=['','Frozen ProtocolA train134047/validation30266 targets,54/12/12 subjects. Test31156 locked. All central validation IDs and quality flags preserved. Weights N_train/(5*n_c), counts '+str(COUNTS)+'; float32 weights '+str(body['weights'])+'.',
            'Rank initial runs by macro-F1 descending, exact ties N1F1 descending, parameters ascending, measured training time ascending, key ascending. Selection requires all8 verified exports and is sealed before repeat training. Candidate configurations (architecture+loss) are selected; both losses of one architecture may qualify.',
            'All full runs: AdamW lr0.001,weight_decay0.01,batch16 unless measured predefined OOM fallback8,max25,patience8,float32,MPS,no scheduler. No architecture or learning-rate adaptation after registration. Ablations use equal3-epoch full-dataset budgets and CE, freshseed42; compare with the separate bounded M8 control, not the25-epoch M8 result.',
            'Registry and each config are immutable. Preparation source revisions create distinct preserved registration directories only while no full run exists. Current pointer may be replaced before launch; source/config/software drift after launch is rejected. Previous M1/M4 checkpoints are protected by hashes.']
    (project/'results/phase4_experiment_registry.md').write_text('\n'.join(lines)+'\n')
    pending=dict(campaign_id=campaign,status='registered_unlaunched',current=None,updated_at=now(),
                 runs={key:dict(status='conditional' if row['kind']=='conditional_repeat' else 'pending') for key,row in runs.items()})
    sealed(project/'results/phase4_campaign_state.json',pending)
    (project/'results/phase4_campaign_status.md').write_text('# Phase4 campaign status\n\nRegistered, unlaunched.15 active runs planned;27 reserved configurations. Full training requires the manual Terminal command.\n\n| Key | Status |\n| --- | --- |\n'+'\n'.join('| '+k+' | '+v['status']+' |' for k,v in pending['runs'].items())+'\n')
    return body


def read_registry(project=PROJECT,check_source=True):
    project=Path(project);body=envelope(project/'results/phase4_registry.json')
    if envelope(project/'configs/phase4'/body['campaign_id']/'registry.json')!=body:raise ValueError('Immutable registry differs')
    if settings(project)!=body['settings']:raise ValueError('Campaign configuration changed')
    if check_source and source_identity(project)!=body['source_identity']:raise ValueError('Registered source changed; never silently adapt a running campaign')
    if frozen_dataset_contract(project,'A')!=body['dataset_contract']:raise ValueError('Frozen dataset changed')
    for row in body['runs'].values():
        cfg=project/'configs/phase4'/body['campaign_id']/(row['key']+'.json')
        if json.loads(cfg.read_text())!=row or fingerprint(row['configuration'])!=row['configuration_sha256']:raise ValueError('Registered experiment changed')
    for path,sha in body['protected_checkpoints'].items():
        if digest(project/path)!=sha:raise ValueError('Previous baseline checkpoint changed')
    return body


def select_shortlist(rows):
    if len(rows)!=8 or len({r['key'] for r in rows})!=8:raise ValueError('Require all8 distinct completed primary configurations')
    for row in rows:
        if not all(np.isfinite(row[k]) for k in ['macro_f1','n1_f1','parameters','training_seconds']):raise ValueError('Nonfinite ranking evidence')
    return sorted(rows,key=lambda r:(-r['macro_f1'],-r['n1_f1'],r['parameters'],r['training_seconds'],r['key']))[:2]


def active_keys(body,selection=None):
    keys=list(body['initial_keys'])
    if selection is not None:
        for row in selection['selected']:
            for seed in [43,44]:keys.append(row['key'].rsplit('_s',1)[0]+'_s'+str(seed))
        keys+=body['ablation_keys']
    return keys
