"""One short synthetic CPU checkpoint/resume smoke using actual research components."""
from pathlib import Path
import argparse,copy,json,sys,tempfile
import torch
from pipeline import ROOT,initialize,config,construct
from sleep_research.synthetic import SyntheticEpochs,synthetic_contract
from sleep_research.experiments import Experiment
from sleep_research.objectives import build_optimizer
from sleep_research.trainer import Trainer
from sleep_research.checkpoints import seed_everything

def smoke(output):
 output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True);torch.set_num_threads(1)
 with tempfile.TemporaryDirectory(prefix='synthetic-smoke-',dir=output) as folder:
  root=Path(folder);raw=root/'empty_fixture';raw.mkdir();project=initialize(root/'workspace',raw)
  seed_everything(42);cfg=config('M1');cfg.update(device='cpu',synthetic=True,max_epochs=1,batch_size=5,progress_batches=0)
  train=SyntheticEpochs('train',10,3000,42);val=SyntheticEpochs('validation',10,3000,43)
  exp=Experiment(project,'cnn',cfg,synthetic_contract(train,val),'synthetic-release-smoke')
  model=construct(cfg);trainer=Trainer(model,build_optimizer(model.parameters(),cfg['optimizer']),None,train,val,exp,cfg)
  trainer.fit();first=trainer.global_step;trainer.fit(resume='auto')
  if trainer.global_step!=first or len(trainer.history)!=1:raise ValueError('Completed resume duplicated training')
  roles=trainer.manager.discover();states={role:trainer.manager.read(path) for role,path in roles.items() if path}
  models=[]
  for key in ['m5_ce_s42','m6_ce_s42','m7_ce_s42','m8_ce_s42','m8_ablation_no_multiscale_s42','m8_ablation_no_context_s42']:
   candidate=construct(config(key));x=torch.randn(2,5,3001);x[:,:,-1]=1
   logits=candidate(x);loss=logits.square().mean();loss.backward()
   if logits.shape!=(2,5) or not torch.isfinite(logits).all():raise ValueError('Synthetic model forward failed')
   models.append({'key':key,'parameters':sum(p.numel() for p in candidate.parameters()),'finite_forward_backward':True})
  report=dict(successful=True,synthetic_only=True,real_data_read=False,real_test_inference_called=False,
      cnn_parameters=40501,completed_epochs=1,training_samples=10,validation_samples=10,
      checkpoint_roles_verified=list(states),resume_repeated_no_steps=True,advanced_models=models)
  (output/'smoke.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);smoke(p.parse_args().output)
