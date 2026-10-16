"""Portable orchestration around the unchanged research implementation.

Creates NEW independently fingerprinted reproduction workspaces. Never edits or
pretends to resume the author's historical signed experiment identities.
Full real training/evaluation is MPS-only and must be invoked in a local Terminal.
"""
from pathlib import Path
import argparse, copy, json, os, shutil, sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import pandas as pd
import torch
import yaml
from sleep_research.common import fingerprint,digest,json_write
from sleep_research.experiments import Experiment,frozen_dataset_contract,source_revision
from sleep_research.checkpoints import CheckpointManager,seed_everything

def workspace(path):
 p=Path(path).resolve()
 if not (p/'REPRODUCTION_WORKSPACE.json').is_file():raise ValueError('Initialize a separate reproduction workspace first')
 return p
def initialize(path,dataset_root):
 p=Path(path).resolve();data=Path(dataset_root).resolve()
 if not data.is_dir():raise FileNotFoundError('User-provided Sleep-EDF directory is missing')
 if p==ROOT or p==data or p.is_relative_to(data) or data.is_relative_to(p):raise ValueError('Use a separate workspace outside the read-only raw dataset and public source tree')
 if p.exists() and any(p.iterdir()):raise FileExistsError('Workspace must be new/empty; existing work is never replaced')
 p.mkdir(parents=True,exist_ok=True)
 for folder in ['src','scripts','tests']:
  shutil.copytree(ROOT/folder,p/folder,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
 for folder in ['results','configs','notebooks','checkpoints','data/processed']:(p/folder).mkdir(parents=True,exist_ok=True)
 for f in (ROOT/'configs').glob('*'):
  if f.is_file():
   text=f.read_text().replace('${WORKSPACE}',str(p)).replace('${DATASET_ROOT}',str(data)).replace('${PYTHON}',sys.executable)
   (p/'configs'/f.name).write_text(text)
 for f in (ROOT/'evidence/splits').iterdir():shutil.copyfile(f,p/'results'/f.name)
 for name in ['README.md','.gitignore','pyproject.toml']:shutil.copyfile(ROOT/name,p/name)
 (p/'AGENTS.md').write_text('Independent reproduction workspace. Preserve frozen subjects and raw data. Never tune using the consumed test cohort.\n')
 # Only the generated workspace test-driver changes: scientific source and tests stay byte-identical.
 driver=p/'scripts/run_phase2_tests.py'
 driver.write_text('import sys\nfrom pathlib import Path\nsys.path.insert(0,'+repr(str(ROOT/'release_tools'))+')\nfrom pilot_tests import main\nraise SystemExit(main(Path(__file__).resolve().parents[1]))\n')
 json_write(p/'REPRODUCTION_WORKSPACE.json',dict(schema=1,raw_dataset_root=str(data),source_root=str(ROOT),
     python=sys.executable,scientific_source=source_revision(p)['source_sha256'],historical_identity_reuse=False))
 return p

def assert_source(p):
 marker=json.loads((p/'REPRODUCTION_WORKSPACE.json').read_text())
 if marker['scientific_source']!=source_revision(p)['source_sha256']:raise ValueError('Reproduction scientific source changed')
 if source_revision(ROOT)['source_sha256']!=marker['scientific_source']:raise ValueError('Public source differs from initialized workspace')

def config(key):
 values=json.loads((ROOT/'configs/registered_models.json').read_text())['configurations']
 if key not in values:raise ValueError('Unregistered model key')
 cfg=copy.deepcopy(values[key]);cfg.update(portable_reproduction=True)
 return cfg
def construct(cfg):
 if cfg['architecture']['name']=='CNN1D':
  from models.cnn1d import CNN1D
  return CNN1D(**{k:v for k,v in cfg['architecture'].items() if k!='name'})
 from models.advanced_sleep import make_advanced
 return make_advanced(cfg['architecture'])

def train_one(p,key,resume='auto'):
 # No actual real-data call is made during repository preparation or --help checks.
 if not torch.backends.mps.is_available():raise RuntimeError('Real reproduction training requires MPS; no CPU fallback')
 assert_source(p)
 if (p/'results/portable_final_freeze.json').exists():raise PermissionError('Final plan is frozen; no additional development training')
 from sleep_research.real_data import ResearchEpochDataset
 from sleep_research.phase4_data import ContextDataset
 from sleep_research.trainer import Trainer
 from sleep_research.objectives import ResearchLoss,build_optimizer
 from sleep_research.inference import export_inference
 from sleep_research.analysis import analyze_export
 cfg=config(key);train=ResearchEpochDataset(p,'train',cfg['max_open_recordings']);val=ResearchEpochDataset(p,'validation',cfg['max_open_recordings'])
 try:
  if len(train)!=134047 or len(val)!=30266 or set(train.index.subject_id)&set(val.index.subject_id):raise ValueError('Frozen counts/subjects changed')
  counts=np.bincount(train.index.label.to_numpy(dtype=int),minlength=5).tolist()
  if counts!=[43782,15249,47664,9302,18050]:raise ValueError('Training-only class counts differ')
  if cfg['architecture']['name']!='CNN1D':train,val=ContextDataset(train,5),ContextDataset(val,5)
  seed_everything(cfg['seed']);model=construct(cfg);contract=frozen_dataset_contract(p,'A')
  criterion=None if key=='M1' else ResearchLoss(cfg['loss'] if isinstance(cfg['loss'],dict) else {'name':'cross_entropy'},counts)
  run_id='reproduction-'+key;family=cfg['family'];exp=Experiment(p,family,cfg,contract,run_id)
  trainer=Trainer(model,build_optimizer(model.parameters(),cfg['optimizer']),None,train,val,exp,cfg,criterion=criterion)
  complete=exp.path/'performance/portable_result.json'
  if complete.exists():
   result=json.loads(complete.read_text());analyze_export(p/result['export'],tsne_max_samples=500,tsne_perplexity=30,tsne_iterations=500,seed=42)
   trainer.manager.read(trainer.manager.discover()['best']);return result
  trainer.fit(resume=resume)
  extras=['raw_rail_contact_samples','filter_boundary_flag','quality_flagged']
  if cfg['architecture']['name']!='CNN1D':extras+=['context_epoch_ids','context_mask','missing_context_count']
  export=export_inference(model,trainer.validation_loader,exp,device='mps',metadata_fields=extras,streaming=True)
  analysis=analyze_export(export,tsne_max_samples=500,tsne_perplexity=30,tsne_iterations=500,seed=42)
  metric=json.loads((analysis/'metrics.json').read_text())
  best=trainer.manager.read(trainer.manager.discover()['best'])
  result=dict(key=key,run_id=run_id,family=family,export=str(export.relative_to(p)),
      parameters=sum(v.numel() for v in model.parameters() if v.requires_grad),
      macro_f1=metric['macro_f1'],n1_f1=metric['classes']['N1']['f1'],
      training_seconds=trainer.completed_training_seconds,best_epoch=best['completed_epoch']+1,
      frozen_test_evaluated=False,new_reproduction_not_historical=True)
  json_write(complete,result);return result
 finally:train.close();val.close()

def campaign(p,resume):
 from sleep_research.phase4_orchestration import CampaignEngine
 from sleep_research.phase4_registry import select_shortlist
 data=json.loads((ROOT/'configs/registered_models.json').read_text())
 spec=yaml.safe_load((p/'configs/phase4_campaign.yaml').read_text())
 if spec!=yaml.safe_load((ROOT/'configs/phase4_campaign.yaml').read_text()):raise ValueError('Workspace campaign settings differ from the fixed public protocol')
 initial=[f'{m.lower()}_{loss}_s42' for m in ['M5','M6','M7','M8'] for loss in ['ce','wce']]
 configs=data['configurations'];runs={}
 for key in configs:
  if not key.startswith('m'):continue
  kind='ablation' if 'ablation' in key else ('initial' if key in initial else 'conditional_repeat')
  runs[key]=dict(key=key,kind=kind)
 body=dict(campaign_id='portable-'+fingerprint(dict(contract=frozen_dataset_contract(p,'A'),configs=configs))[:12],
     registered_configurations=configs,runs=runs,initial_keys=initial,ablation_keys=[k for k in configs if 'ablation' in k])
 path=p/'results/portable_campaign_registry.json'
 if path.exists() and json.loads(path.read_text())!=body:raise ValueError('Registered reproduction configuration changed')
 if not path.exists():json_write(path,body)
 def result_path(key):return p/'experiments'/config(key)['family']/('reproduction-'+key)/'performance/portable_result.json'
 def verify(key):
  result=json.loads(result_path(key).read_text());cfg=config(key)
  exp=Experiment(p,cfg['family'],run_id='reproduction-'+key)
  manager=CheckpointManager(exp.path/'reproducibility/checkpoints',exp.metadata['resume_fingerprints'],exp.run_id);manager.read(manager.discover()['best'])
  from sleep_research.analysis import load_export
  load_export(p/result['export']);return result
 def select(measured):
  # Replay original pre-test shortlist; fresh validation outcomes are not allowed to revise this final roster.
  selection=dict(selected=[{'key':k} for k in data['historical_shortlist']],historical_replay=True,
      note='Original validation-selected configurations fixed for independent reproduction; not a new claim of selection on this machine.')
  json_write(p/'results/portable_historical_shortlist_replay.json',selection);return selection
 engine=CampaignEngine(body,p/'results/portable_campaign_state.json',lambda key:train_one(p,key,resume),verify,
     lambda key:result_path(key).exists(),select)
 return engine.run()

def make_final_plan(p):
 from sleep_research.phase5_registry import PRIMARY,SECONDARY
 assert_source(p);models=[]
 canonical=pd.read_csv(p/'results/protocol_a_index.csv').query("split == 'test'").reset_index(drop=True)
 for key in PRIMARY+SECONDARY:
  cfg=config(key);exp=Experiment(p,cfg['family'],run_id='reproduction-'+key)
  manager=CheckpointManager(exp.path/'reproducibility/checkpoints',exp.metadata['resume_fingerprints'],exp.run_id)
  latest=manager.read(manager.discover()['latest']);best=manager.read(manager.discover()['best'])
  if latest['next_epoch']<cfg['max_epochs'] and not latest['early_stopping']['stopped']:raise ValueError('Training not complete: '+key)
  models.append(dict(key=key,model_id='M1' if key=='M1' else ('M4' if key=='M4' else cfg['model_id']),family=cfg['family'],
      run_id=exp.run_id,checkpoint_sha256=digest(manager.discover()['best']),configuration=cfg))
 body=dict(plan_id='portable-final',models=models,dataset_contract=frozen_dataset_contract(p,'A'),
     source_sha256=source_revision(p)['source_sha256'],test_epochs=len(canonical),test_subjects=sorted(canonical.subject_id.unique()),
     test_epoch_ids_sha256=fingerprint(canonical.epoch_id.tolist()),inference={'max_open_recordings':128},
     primary_groups={'M1':['M1'],'M5':['m5_ce_s42','m5_ce_s43','m5_ce_s44'],'M7':['m7_ce_s42','m7_ce_s43','m7_ce_s44']},
     statistics={'resamples':10000,'seed':20261016,'confidence':.95,'simultaneous_confidence':1-.05/3},
     note='Independent reproduction under original fixed roster/protocol; not the historical plan or original checkpoint identities.')
 if len(canonical)!=31156:raise ValueError('Canonical final target count differs')
 path=p/'results/portable_final_plan.json'
 if path.exists() and json.loads(path.read_text())!=body:raise ValueError('Final plan is immutable once created')
 if not path.exists():json_write(path,body)
 return dict(plan_sha256=fingerprint(body),plan_file=str(path),instruction='Review this plan, then freeze with its exact digest before explicit final evaluation.')

def final_evaluate(p,explicit,sha):
 from sleep_research.phase5_registry import authorize
 from sleep_research.phase5_data import FinalEpochDataset,FinalContextDataset
 from sleep_research.phase5_evaluation import chunk_export,merge_chunks
 body=json.loads((p/'results/portable_final_plan.json').read_text());frozen=json.loads((p/'results/portable_final_freeze.json').read_text())
 access=authorize(body,frozen,explicit,sha) # Gate before dataset/model access.
 assert_source(p)
 if frozen_dataset_contract(p,'A')!=body['dataset_contract'] or source_revision(p)['source_sha256']!=body['source_sha256']:raise ValueError('Frozen data/source changed')
 if not torch.backends.mps.is_available():raise RuntimeError('Final real inference requires MPS; no CPU fallback')
 exports={};home=p/'experiments/final_evaluation'/body['plan_id']
 for row in body['models']:
  original=Experiment(p,row['family'],run_id=row['run_id'])
  original_manager=CheckpointManager(original.path/'reproducibility/checkpoints',original.metadata['resume_fingerprints'],original.run_id)
  if digest(original_manager.discover()['best'])!=row['checkpoint_sha256']:raise ValueError('Frozen best checkpoint differs')
  dest=home/'experiments'/row['family']/row['run_id']
  if not dest.exists():
   shutil.copytree(original.path/'reproducibility',dest/'reproducibility')
   for name in ['performance','representations','analysis','explanations']:(dest/name).mkdir()
  exp=Experiment(home,row['family'],run_id=row['run_id']);base=FinalEpochDataset(p,body,access)
  data=base if row['model_id'] in ['M1','M4'] else FinalContextDataset(base,access,5)
  try:
   with exp.lock():
    refs=chunk_export(construct(row['configuration']),data,exp,row,chunk_size=2048,batch_size=16,device='mps')
    exports[row['key']]=str(merge_chunks(exp,row,refs,data.index.reset_index(drop=True),body['plan_id']).relative_to(p))
  finally:data.close()
 json_write(p/'results/portable_final_exports.json',dict(plan_sha256=fingerprint(body),exports=exports));return exports

def main():
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('command',choices=['init','inspect','preprocess-pilot','preprocess-all','verify-splits','normalization',
     'train','campaign','checkpoints','plan-final','freeze-final','final-evaluate'])
 parser.add_argument('--workspace',type=Path,required=True);parser.add_argument('--dataset-root',type=Path)
 parser.add_argument('--key',default='M1');parser.add_argument('--resume',default='auto');parser.add_argument('--run',type=Path)
 parser.add_argument('--reviewed-plan-sha256');parser.add_argument('--plan-sha256');parser.add_argument('--authorize-final-evaluation',action='store_true')
 args=parser.parse_args()
 if args.command=='init':print(initialize(args.workspace,args.dataset_root));return
 p=workspace(args.workspace);assert_source(p)
 if args.command=='inspect':
  from sleep_edf.__main__ import run
  run(p/'configs/phase1.toml');return
 if args.command.startswith('preprocess-'):
  from sleep_edf.phase2 import run_phase2
  run_phase2(p,'pilot' if args.command.endswith('pilot') else 'all');return
 if args.command=='verify-splits':
  from sleep_edf.phase2 import load_inventory
  from sleep_edf.splits import freeze_splits
  assignments,freeze=freeze_splits(p/'results',load_inventory(p));print(json.dumps(freeze,indent=2));return
 if args.command=='normalization':
  # This step is already performed by original Phase 2 consolidation; expose verification, not a second fit.
  saved=json.loads((p/'results/normalization.json').read_text())
  if saved['fit_split']!='train' or saved['protocols']['A']['epoch_count']!=134047:raise ValueError('Training normalization contract differs')
  print(json.dumps(saved,indent=2));return
 if args.command=='train':result=train_one(p,args.key,args.resume)
 elif args.command=='campaign':result=campaign(p,args.resume)
 elif args.command=='plan-final':result=make_final_plan(p)
 elif args.command=='freeze-final':
  body=json.loads((p/'results/portable_final_plan.json').read_text())
  if args.reviewed_plan_sha256!=fingerprint(body):raise PermissionError('Supply the exact reviewed plan digest')
  target=p/'results/portable_final_freeze.json';record={'plan_sha256':fingerprint(body)}
  if target.exists() and json.loads(target.read_text())!=record:raise ValueError('Existing frozen plan differs')
  json_write(target,record);result=record
 elif args.command=='final-evaluate':result=final_evaluate(p,args.authorize_final_evaluation,args.plan_sha256)
 else:
  if args.run is None:raise ValueError('--run requires an experiment folder')
  meta=json.loads((args.run/'reproducibility/experiment.json').read_text());manager=CheckpointManager(args.run/'reproducibility/checkpoints',meta['resume_fingerprints'],meta['experiment_id'])
  result={k:None if v is None else dict(path=str(v),completed_epoch=manager.read(v)['completed_epoch']) for k,v in manager.discover().items()}
 print(json.dumps(result,indent=2))

if __name__=='__main__':main()
