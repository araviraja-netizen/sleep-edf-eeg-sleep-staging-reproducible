"""Optimizer/loss math, failure cases and synthetic CPU recovery evidence."""

import copy
import ast
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
import yaml

from sleep_research.objectives import ResearchLoss,build_optimizer,class_weights
from sleep_research.checkpoints import IncompatibleCheckpoint,seed_everything
from sleep_research.experiments import Experiment
from sleep_research.synthetic import SyntheticEpochs,SyntheticTinyCNN,synthetic_contract
from sleep_research.trainer import Trainer,SimulatedInterruption,evaluate_model
from sleep_research.phase3b import settings,trial_config,validate_architecture,frozen_subsets,registered_candidate
from sleep_research.cnn_baseline import make_model

PROJECT=Path(__file__).resolve().parents[1]
COUNTS=[43782,15249,47664,9302,18050]


class LossTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1);torch.manual_seed(42)
        self.logits=torch.randn(17,5,dtype=torch.float64,requires_grad=True)
        self.labels=torch.arange(17)%5

    def test_ce_matches_torch_value_and_gradient(self):
        for name in ['cross_entropy','weighted_cross_entropy']:
            criterion=ResearchLoss(dict(name=name),COUNTS).double()
            actual=criterion(self.logits,self.labels)
            expected=F.cross_entropy(self.logits,self.labels,weight=criterion.alpha if name.startswith('weighted') else None)
            torch.testing.assert_close(actual,expected,rtol=1e-14,atol=1e-14)
            a=torch.autograd.grad(actual,self.logits,retain_graph=True)[0]
            b=torch.autograd.grad(expected,self.logits,retain_graph=True)[0]
            torch.testing.assert_close(a,b,rtol=1e-13,atol=1e-14)

    def test_gamma_zero_matches_ce(self):
        actual=ResearchLoss(dict(name='focal',gamma=0),COUNTS).double()(self.logits,self.labels)
        torch.testing.assert_close(actual,F.cross_entropy(self.logits,self.labels))

    def test_focal_formula_and_finite_difference_gradient(self):
        criterion=ResearchLoss(dict(name='focal',gamma=2),COUNTS).double()
        logp=self.logits.log_softmax(1)[torch.arange(17),self.labels]
        expected=(-(1-logp.exp()).square()*logp).mean()
        torch.testing.assert_close(criterion(self.logits,self.labels),expected)
        self.assertTrue(torch.autograd.gradcheck(criterion,(self.logits,self.labels),eps=1e-6,atol=1e-5))

    def test_focal_downweights_easy_examples(self):
        easy=torch.tensor([[8.,0,0,0,0]]);hard=torch.tensor([[0.,8,0,0,0]]);labels=torch.tensor([0])
        focal=ResearchLoss(dict(name='focal',gamma=2),COUNTS)
        self.assertLess(float(focal(easy,labels)/F.cross_entropy(easy,labels)),.001)
        self.assertGreater(float(focal(hard,labels)/F.cross_entropy(hard,labels)),.99)

    def test_effective_number_weights_and_gamma_zero_formula(self):
        counts=np.array([3,7,10,20,100]);beta=.9
        expected=(1-beta)/(1-beta**counts);expected=5*expected/expected.sum()
        np.testing.assert_allclose(class_weights(counts,'effective_number',beta),expected,rtol=1e-6)
        criterion=ResearchLoss(dict(name='class_balanced_focal',gamma=0,beta=beta),counts).double()
        logp=self.logits.log_softmax(1)[torch.arange(17),self.labels]
        torch.testing.assert_close(criterion(self.logits,self.labels),(-logp*criterion.alpha[self.labels]).mean())
        np.testing.assert_array_equal(class_weights(counts,'effective_number',0),np.ones(5))

    def test_extreme_logits_have_finite_loss_and_gradient(self):
        for name in ['cross_entropy','weighted_cross_entropy','focal','class_balanced_focal']:
            for gamma in [0,0.5,2]:
                with self.subTest(name=name,gamma=gamma):
                    logits=torch.tensor([[10000.,-10000.,0,0,0],[-10000.,10000.,0,0,0]],requires_grad=True)
                    loss=ResearchLoss(dict(name=name,gamma=gamma),COUNTS)(logits,torch.tensor([0,0]))
                    loss.backward()
                    self.assertTrue(torch.isfinite(loss));self.assertTrue(torch.isfinite(logits.grad).all())

    def test_bad_counts_gamma_beta_and_loss_rejected(self):
        for counts in [[1]*4,[1,2,0,4,5],[1,2,np.nan,4,5],[1,2,3.5,4,5]]:
            with self.assertRaises(ValueError):class_weights(counts)
        for spec in [dict(name='invalid'),dict(name='focal',gamma=-1),dict(name='focal',gamma=float('nan')),dict(name='class_balanced_focal',beta=1)]:
            with self.assertRaises(ValueError):ResearchLoss(spec,COUNTS)

    def test_bad_labels_shapes_and_nonfinite_logits_rejected(self):
        criterion=ResearchLoss(dict(name='focal'),COUNTS)
        for labels in [torch.tensor([-1]),torch.tensor([5]),torch.tensor([1.])]:
            with self.assertRaises(ValueError):criterion(torch.zeros(1,5),labels)
        with self.assertRaises(ValueError):criterion(torch.zeros(1,4),torch.tensor([0]))
        with self.assertRaises(ValueError):criterion(torch.full((1,5),float('nan')),torch.tensor([0]))


class OptimizerResumeTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.tmp=tempfile.TemporaryDirectory(dir=PROJECT/'results');self.root=Path(self.tmp.name)
        self.train=SyntheticEpochs('train',15);self.val=SyntheticEpochs('validation',10)
        self.config=yaml.safe_load((PROJECT/'configs/infrastructure_demo.yaml').read_text())
        self.config.update(max_epochs=2,batch_size=5,track_training_metrics=True)
        self.config['early_stopping']['patience']=8

    def tearDown(self):self.tmp.cleanup()

    def build(self,run,optimizer,loss,counts=COUNTS):
        cfg=copy.deepcopy(self.config);cfg['optimizer']=optimizer;cfg['loss']=loss;cfg['scheduler']={'name':'none'}
        seed_everything(42);model=SyntheticTinyCNN()
        experiment=Experiment(self.root,'cnn',cfg,synthetic_contract(self.train,self.val),run)
        return Trainer(model,build_optimizer(model.parameters(),optimizer),None,self.train,self.val,experiment,cfg,criterion=ResearchLoss(loss,counts))

    def test_all_four_optimizers_make_finite_updates_and_record_defaults(self):
        for name in ['AdamW','Adam','SGD','RMSprop']:
            spec=dict(name=name,learning_rate=.01,weight_decay=.01)
            if name in ('SGD','RMSprop'):spec['momentum']=.9
            model=torch.nn.Linear(2,5);opt=build_optimizer(model.parameters(),spec)
            F.cross_entropy(model(torch.ones(4,2)),torch.arange(4)).backward();opt.step()
            self.assertTrue(all(torch.isfinite(p).all() for p in model.parameters()))
            self.assertEqual(opt.param_groups[0]['lr'],.01)

    def test_invalid_optimizer_lr_and_sgd_momentum(self):
        for spec in [dict(name='other',learning_rate=.1),dict(name='Adam',learning_rate=0),dict(name='SGD',learning_rate=.1),dict(name='Adam',learning_rate=float('inf'))]:
            with self.assertRaises(ValueError):build_optimizer(torch.nn.Linear(2,5).parameters(),spec)

    def test_weighted_focal_and_cb_resume_equal_cpu_weights_optimizer_history(self):
        for name,loss in [('SGD',dict(name='weighted_cross_entropy')),('Adam',dict(name='focal',gamma=2)),('RMSprop',dict(name='class_balanced_focal',beta=.9999,gamma=2))]:
            with self.subTest(name=name):
                spec=dict(name=name,learning_rate=.001,weight_decay=.01)
                if name in ('SGD','RMSprop'):spec['momentum']=.9
                base=self.build(name+'base',spec,loss);base.fit(resume='none')
                stopped=self.build(name+'stop',spec,loss)
                def stop(epoch,batch,step):
                    if epoch==1 and batch==0:raise SimulatedInterruption()
                with self.assertRaises(SimulatedInterruption):stopped.fit(resume='none',interrupt_callback=stop)
                restored=self.build(name+'stop',spec,loss);restored.fit()
                for key,value in base.model.state_dict().items():torch.testing.assert_close(value,restored.model.state_dict()[key],rtol=0,atol=0)
                for key,values in base.optimizer.state_dict()['state'].items():
                    for field,value in values.items():torch.testing.assert_close(value,restored.optimizer.state_dict()['state'][key][field],rtol=0,atol=0)
                self.assertEqual([{k:v for k,v in r.items() if k!='seconds'} for r in base.history],
                                 [{k:v for k,v in r.items() if k!='seconds'} for r in restored.history])
                self.assertIn('criterion',restored.manager.recover())

    def test_changed_class_weights_reject_resume_even_same_shapes(self):
        spec=dict(name='AdamW',learning_rate=.001)
        trainer=self.build('weights',spec,dict(name='weighted_cross_entropy'));trainer.fit(max_epochs=1)
        other=self.build('weights',spec,dict(name='weighted_cross_entropy'),counts=[1,1,1,1,1])
        with self.assertRaises(IncompatibleCheckpoint):other.fit()

    def test_validation_keeps_standard_ce_and_globally_reduced_objective(self):
        spec=dict(name='Adam',learning_rate=.001)
        trainer=self.build('evaluation',spec,dict(name='weighted_cross_entropy'))
        result=evaluate_model(trainer.model,DataLoader(self.val,batch_size=3),criterion=trainer.criterion)
        with torch.no_grad():
            batch=next(iter(DataLoader(self.val,batch_size=10)));logits=trainer.model(batch['eeg'])
        self.assertAlmostEqual(result['loss'],float(F.cross_entropy(logits,batch['label'])),places=6)
        self.assertAlmostEqual(result['objective_loss'],float(trainer.criterion(logits,batch['label'])),places=6)

    def test_test_selection_for_custom_loss_is_blocked(self):
        trainer=self.build('testgate',dict(name='Adam',learning_rate=.001),dict(name='focal',gamma=2))
        trainer.validation_loader=DataLoader(SyntheticEpochs('test',5),batch_size=5)
        with self.assertRaises(PermissionError):trainer.fit()
        self.assertEqual(trainer.manager.recover()['completed_epoch'],-1)


class ProtocolTests(unittest.TestCase):
    def test_resume_keeps_registered_candidate_identity(self):
        search=settings(PROJECT)
        for candidate in search['optimizer_candidates']:
            self.assertEqual(registered_candidate(search,candidate['optimizer'])['id'],candidate['id'])
        spec=copy.deepcopy(search['optimizer_candidates'][0]['optimizer']);spec['learning_rate']=.5
        with self.assertRaises(ValueError):registered_candidate(search,spec)

    def test_fixed_architecture_and_registered_seeds_bounds(self):
        search=settings(PROJECT)
        cfg=trial_config(PROJECT,search,search['optimizer_candidates'][0],dict(name='focal',gamma=2),'optimizer_pilot',42)
        validate_architecture(PROJECT,cfg,make_model(cfg))
        self.assertEqual(search['repeat_seeds'],[42,43,44]);self.assertEqual(len(search['optimizer_candidates']),6)
        self.assertEqual(cfg['max_epochs'],3)
        changed=copy.deepcopy(cfg);changed['architecture']['dropout']=.4
        with self.assertRaises(ValueError):validate_architecture(PROJECT,changed,make_model(changed))

    def test_pilot_subset_identity_is_frozen_and_tampering_rejected(self):
        from types import SimpleNamespace
        class FixtureDataset(SimpleNamespace):
            def __len__(self):return len(self.index)
        with tempfile.TemporaryDirectory(dir=PROJECT/'results') as tmp:
            root=Path(tmp);(root/'results').mkdir()
            train=FixtureDataset(index=pd.DataFrame({'epoch_id':[f'TRAIN:{i}' for i in range(20)]}),state={'split_manifest_sha256':'fixture'})
            val=FixtureDataset(index=pd.DataFrame({'epoch_id':[f'VALIDATION:{i}' for i in range(10)]}))
            search=settings(PROJECT);search['pilot'].update(training_samples=5,validation_samples=4)
            first=frozen_subsets(root,train,val,search);self.assertEqual(first,frozen_subsets(root,train,val,search))
            index=first['indices']['train'][0];train.index.loc[index,'epoch_id']='changed'
            with self.assertRaises(ValueError):frozen_subsets(root,train,val,search)

    def test_three_seed_summary_matches_sample_sd_and_rejects_duplicate_seed(self):
        from sleep_research.repeated_seeds import summarize_seeds,METRICS
        frame=pd.DataFrame([dict(configuration_group='synthetic-fixture',seed=seed,**{key:value for key in METRICS}) for seed,value in zip([42,43,44],[.4,.5,.6])])
        result=summarize_seeds(frame).iloc[0]
        self.assertEqual(result.independent_seed_count,3);self.assertTrue(result.at_least_three_seeds)
        self.assertAlmostEqual(result.macro_f1_mean,.5);self.assertAlmostEqual(result.macro_f1_sample_sd,.1)
        frame.loc[2,'seed']=42
        with self.assertRaises(ValueError):summarize_seeds(frame)

    def test_analysis_has_no_trainer_import_or_fit_call(self):
        for name in ['phase3b_analysis.py','repeated_seeds.py']:
            tree=ast.parse((PROJECT/'src/sleep_research'/name).read_text())
            imports=[node.module for node in ast.walk(tree) if isinstance(node,ast.ImportFrom)]
            self.assertFalse(set(imports)&{'trainer','phase3b','cnn_baseline'})
            self.assertFalse(any(isinstance(node,ast.Attribute) and node.attr=='fit' for node in ast.walk(tree)))


if __name__=='__main__':unittest.main()
