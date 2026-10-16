"""Registered controls and exact-architecture synthetic CPU recovery checks."""

import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch

from sleep_research.phase3c import configuration,authorized_m2,read_registry
from sleep_research.cnn_baseline import make_model
from sleep_research.checkpoints import seed_everything
from sleep_research.experiments import Experiment,model_fingerprint
from sleep_research.objectives import build_optimizer,ResearchLoss,class_weights
from sleep_research.synthetic import SyntheticEpochs,synthetic_contract
from sleep_research.trainer import Trainer,SimulatedInterruption

PROJECT=Path(__file__).resolve().parents[1]


class RegistrationTests(unittest.TestCase):
    def test_four_models_have_same_architecture_dataset_and_budgets(self):
        registry=read_registry(PROJECT)
        self.assertEqual(list(registry['models']),['M1','M2','M3','M4'])
        for name,row in registry['models'].items():
            cfg=row['configuration'];model=make_model(cfg)
            self.assertEqual(sum(p.numel() for p in model.parameters() if p.requires_grad),40501)
            self.assertEqual(model_fingerprint(model,cfg['architecture']),registry['common_architecture_fingerprint'])
            self.assertEqual(row['dataset_contract'],registry['shared_dataset_contract'])
            self.assertEqual((cfg['batch_size'],cfg['max_epochs'],cfg['early_stopping']['patience']),(16,25,8))
            self.assertEqual(cfg['scheduler']['name'],'none');self.assertEqual(cfg['optimizer']['learning_rate'],.001)
            self.assertEqual(cfg['optimizer']['weight_decay'],.01)

    def test_only_m2_seed42_authorized(self):
        for name in ['M1','M2','M3','M4']:
            for seed in [42,43,44]:
                cfg=configuration(PROJECT,name,seed)
                if name=='M2' and seed==42:authorized_m2(cfg)
                else:
                    with self.assertRaises(PermissionError):authorized_m2(cfg)

    def test_changed_optimizer_and_budget_rejected(self):
        cfg=configuration(PROJECT,'M2');cfg['optimizer']['learning_rate']=.01
        with self.assertRaises(ValueError):authorized_m2(cfg)
        cfg=configuration(PROJECT,'M2');cfg['max_epochs']=26
        with self.assertRaises(ValueError):authorized_m2(cfg)

    def test_weights_use_only_full_training_census(self):
        registry=read_registry(PROJECT);index=pd.read_csv(PROJECT/'results/protocol_a_index.csv')
        counts=np.bincount(index.loc[index.split=='train','label'],minlength=5)
        self.assertEqual(int(counts.sum()),134047)
        for name in ['M3','M4']:
            np.testing.assert_array_equal(registry['models'][name]['training_counts'],counts)
            np.testing.assert_array_equal(registry['models'][name]['loss_alpha'],class_weights(counts).tolist())

    def test_registry_tampering_fails_before_any_dataset_load(self):
        with tempfile.TemporaryDirectory(dir=PROJECT/'results') as tmp:
            root=Path(tmp);(root/'results').mkdir()
            envelope=json.loads((PROJECT/'results/phase3c_registry.json').read_text())
            envelope['body']['models']['M2']['configuration']['optimizer']['learning_rate']=.5
            (root/'results/phase3c_registry.json').write_text(json.dumps(envelope))
            with self.assertRaisesRegex(ValueError,'Registry checksum'):read_registry(root)

    def test_m1_is_reused_and_unlaunched_ids_are_distinct(self):
        registry=read_registry(PROJECT);rows=registry['models']
        self.assertEqual(rows['M1']['run_id'],'cnn1d-a-seed42-20261014-1800')
        self.assertEqual(len({r['run_id'] for r in rows.values()}),4)
        for name in ['M3','M4']:
            self.assertFalse(rows[name]['launch_authorized'])
            self.assertFalse((PROJECT/'experiments/cnn'/rows[name]['run_id']).exists())


class M2ResumeTests(unittest.TestCase):
    def test_m2_cnn_cpu_interrupt_and_corrupt_latest_recovery(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory(dir=PROJECT/'results') as tmp:
            root=Path(tmp);cfg=configuration(PROJECT,'M2');cfg.update(device='cpu',max_epochs=2,batch_size=5,progress_batches=0,synthetic=True)
            train=SyntheticEpochs('train',10,3000);val=SyntheticEpochs('validation',10,3000)
            def build(run_id):
                seed_everything(42);model=make_model(cfg)
                exp=Experiment(root,'cnn',cfg,synthetic_contract(train,val),run_id)
                return Trainer(model,build_optimizer(model.parameters(),cfg['optimizer']),None,train,val,exp,cfg,
                               criterion=ResearchLoss(cfg['loss'],[2]*5))
            baseline=build('baseline');baseline.fit(resume='none')
            stopped=build('interrupted')
            def stop(epoch,batch,step):
                if epoch==1 and batch==0:raise SimulatedInterruption()
            with self.assertRaises(SimulatedInterruption):stopped.fit(resume='none',interrupt_callback=stop)
            self.assertEqual(stopped.manager.recover()['next_epoch'],1)
            restored=build('interrupted');restored.fit(resume='auto')
            Path(restored.manager.discover()['latest']).write_bytes(b'corrupt latest test fixture')
            recovered=build('interrupted');recovered.fit(resume='auto')
            self.assertTrue(recovered.manager.recovery_notes)
            self.assertEqual([r['epoch'] for r in recovered.history],[0,1])
            self.assertEqual(recovered.global_step,4)
            for key,value in baseline.model.state_dict().items():torch.testing.assert_close(value,recovered.model.state_dict()[key],rtol=0,atol=0)
            for key,values in baseline.optimizer.state_dict()['state'].items():
                for field,value in values.items():torch.testing.assert_close(value,recovered.optimizer.state_dict()['state'][key][field],rtol=0,atol=0)
            self.assertEqual([{k:v for k,v in row.items() if k!='seconds'} for row in baseline.history],
                             [{k:v for k,v in row.items() if k!='seconds'} for row in recovered.history])


if __name__=='__main__':unittest.main()
