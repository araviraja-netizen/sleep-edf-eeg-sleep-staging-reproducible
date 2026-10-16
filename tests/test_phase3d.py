"""M4 controls, weighted gradient/recovery and Terminal command safety evidence."""
import ast
import copy
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from sleep_research.phase3d import expected_configuration,read_registry,COUNTS,LABELS,Tee,train
from sleep_research.phase3d_analysis import status,completed_experiment,registration,compare
from sleep_research.cnn_baseline import make_model
from sleep_research.checkpoints import seed_everything
from sleep_research.experiments import Experiment,model_fingerprint
from sleep_research.objectives import build_optimizer,ResearchLoss,class_weights
from sleep_research.synthetic import SyntheticEpochs,synthetic_contract
from sleep_research.trainer import Trainer,SimulatedInterruption,evaluate_model

PROJECT=Path(__file__).resolve().parents[1]
configuration=lambda project,model:expected_configuration(project)


class M4Controls(unittest.TestCase):
    def test_registered_configuration_and_counts(self):
        b=read_registry(PROJECT);cfg=b['configuration'];model=make_model(cfg)
        self.assertEqual(sum(p.numel() for p in model.parameters()),40501)
        self.assertEqual(model_fingerprint(model,cfg['architecture']),b['architecture_fingerprint'])
        self.assertEqual(b['label_order'],['Wake','N1','N2','N3','REM'])
        self.assertEqual(b['training_counts'],COUNTS)
        np.testing.assert_array_equal(b['weights_float32'],class_weights(COUNTS))
        np.testing.assert_array_equal(b['weights_float64'],134047/(5*np.array(COUNTS)))
        self.assertEqual((cfg['model_id'],cfg['seed'],cfg['device'],cfg['max_epochs']),('M4',42,'mps',25))
        self.assertEqual(cfg['optimizer']['name'],'AdamW')
        self.assertEqual(cfg['loss']['name'],'weighted_cross_entropy')

    def test_configuration_drift_rejected(self):
        body=read_registry(PROJECT)
        import yaml
        with tempfile.TemporaryDirectory(dir=PROJECT/'results') as temp:
            root=Path(temp);(root/'results').mkdir();(root/'configs').mkdir()
            (root/'results/phase3d_registry.json').write_text((PROJECT/'results/phase3d_registry.json').read_text())
            cfg=copy.deepcopy(body['configuration']);cfg['optimizer']['learning_rate']=.01
            (root/'configs/phase3d_m4.yaml').write_text(yaml.safe_dump(cfg))
            with self.assertRaisesRegex(ValueError,'settings changed'):read_registry(root)

    def test_m4_run_id_is_distinct_from_preserved_reservation(self):
        b=read_registry(PROJECT)
        self.assertNotEqual(b['run_id'],b['phase3c_planned_M4'])

    def test_status_is_read_only_and_reports_unstarted(self):
        before=set((PROJECT/'experiments/cnn').iterdir())
        with tempfile.TemporaryDirectory(dir=PROJECT/'results') as temp,patch('sleep_research.phase3d_analysis.registration',return_value=read_registry(PROJECT)):
            result=status(Path(temp))
        self.assertEqual(result['completed_epochs'],0);self.assertIsNone(result['best_validation_macro_f1'])
        self.assertFalse(result['full_training_launched'])
        self.assertEqual(before,set((PROJECT/'experiments/cnn').iterdir()))

    def test_analysis_before_training_refuses_results(self):
        with tempfile.TemporaryDirectory(dir=PROJECT/'results') as temp,patch('sleep_research.phase3d_analysis.registration',return_value=read_registry(PROJECT)):
            with self.assertRaisesRegex(RuntimeError,'not been trained'):compare(Path(temp))

    def test_full_training_refuses_cpu_before_constructing_trainer(self):
        with patch('sleep_research.phase3d.torch.backends.mps.is_available',return_value=False),patch('sleep_research.phase3d.build_trial') as build:
            with self.assertRaisesRegex(RuntimeError,'CPU fallback'):train(PROJECT)
            build.assert_not_called()

    def test_resume_none_existing_checkpoint_is_rejected_without_mutation(self):
        from sleep_research.common import fingerprint,json_write
        body=read_registry(PROJECT)
        with tempfile.TemporaryDirectory(dir=PROJECT/'results') as temp:
            root=Path(temp);directory=root/'experiments/cnn'/body['run_id']/'reproducibility/checkpoints';directory.mkdir(parents=True)
            checkpoint=directory/'preserved.pt';checkpoint.write_bytes(b'preserved-checkpoint-fixture')
            json_write(root/'results/phase3d_smoke_validation.json',dict(successful=True,registry_sha256=fingerprint(body),source_identity='fixture'))
            json_write(root/'results/phase3d_test_results.json',dict(successful=True,skipped=0,source_identity='fixture'))
            with patch('sleep_research.phase3d.read_registry',return_value=body),patch('sleep_research.phase3d.code_identity',return_value='fixture'),patch('sleep_research.phase3d.torch.backends.mps.is_available',return_value=True),patch('sleep_research.phase3d.datasets') as loader:
                with self.assertRaises(FileExistsError):train(root,resume='none')
                loader.assert_not_called()
            self.assertEqual(checkpoint.read_bytes(),b'preserved-checkpoint-fixture')

    def test_analysis_transitive_modules_do_not_import_training(self):
        seen=set()
        def visit(module):
            if module in seen:return
            seen.add(module)
            path=PROJECT/'src/sleep_research'/f'{module}.py'
            if not path.exists():return
            tree=ast.parse(path.read_text())
            for node in ast.walk(tree):
                if isinstance(node,ast.ImportFrom) and node.level==1 and node.module:
                    self.assertNotIn(node.module,['trainer','phase3d','phase3c','phase3b','cnn_baseline'])
                    visit(node.module)
                if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute):self.assertNotEqual(node.func.attr,'fit')
        visit('phase3d_analysis')

    def test_persistent_logger_flushes_console_and_file(self):
        console=io.StringIO();file=io.StringIO();tee=Tee(console,file)
        print('M4 run status interrupted',file=tee)
        self.assertEqual(console.getvalue(),file.getvalue());self.assertIn('interrupted',file.getvalue())

    def test_terminal_cli_validate_and_status(self):
        for command in [('sleep_research.phase3d','validate'),('sleep_research.phase3d_analysis','status')]:
            result=subprocess.run(['/Volumes/Mac/envs/dl-m4/bin/python','-m',*command],cwd=PROJECT,
                env=dict(os.environ,PYTHONPATH=str(PROJECT/'src')),capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr);payload=json.loads(result.stdout)
            if command[1]=='validate':self.assertFalse(payload['full_training_launched'])
            else:self.assertEqual(payload['run_id'],read_registry(PROJECT)['run_id'])


class M4Math(unittest.TestCase):
    def test_registered_weights_value_and_gradient(self):
        torch.manual_seed(42);x=torch.randn(11,5,dtype=torch.float64,requires_grad=True);y=torch.arange(11)%5
        loss=ResearchLoss(dict(name='weighted_cross_entropy'),COUNTS).double()
        a=loss(x,y);b=torch.nn.functional.cross_entropy(x,y,weight=torch.tensor(class_weights(COUNTS),dtype=torch.float64))
        torch.testing.assert_close(a,b,rtol=1e-14,atol=1e-14)
        torch.testing.assert_close(torch.autograd.grad(a,x,retain_graph=True)[0],torch.autograd.grad(b,x)[0],rtol=1e-13,atol=1e-14)

    def test_validation_records_unweighted_ce_and_weighted_objective_separately(self):
        from models.cnn1d import CNN1D
        model=CNN1D();data=SyntheticEpochs('validation',11,3000);loader=DataLoader(data,batch_size=4)
        loss=ResearchLoss(dict(name='weighted_cross_entropy'),COUNTS)
        result=evaluate_model(model,loader,'cpu',criterion=loss)
        logits=torch.tensor(result['logits']);labels=torch.tensor(result['labels'])
        self.assertAlmostEqual(result['loss'],float(torch.nn.functional.cross_entropy(logits,labels)),places=6)
        self.assertAlmostEqual(result['objective_loss'],float(loss(logits,labels)),places=6)

    def test_test_split_refused_for_weighted_validation_before_forward(self):
        from models.cnn1d import CNN1D
        model=CNN1D();loss=ResearchLoss(dict(name='weighted_cross_entropy'),COUNTS)
        with patch.object(model,'forward',side_effect=AssertionError('must not forward test')) as forward:
            with self.assertRaises(PermissionError):evaluate_model(model,DataLoader(SyntheticEpochs('test',5,3000),batch_size=5),'cpu',criterion=loss)
            forward.assert_not_called()


class M4ResumeTests(unittest.TestCase):
    def test_m4_cnn_cpu_interrupt_and_corrupt_latest_recovery(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory(dir=PROJECT/'results') as tmp:
            root=Path(tmp);cfg=configuration(PROJECT,'M4');cfg.update(device='cpu',max_epochs=2,batch_size=5,progress_batches=0,synthetic=True)
            train=SyntheticEpochs('train',10,3000);val=SyntheticEpochs('validation',10,3000)
            def build(run_id):
                seed_everything(42);model=make_model(cfg)
                exp=Experiment(root,'cnn',cfg,synthetic_contract(train,val),run_id)
                return Trainer(model,build_optimizer(model.parameters(),cfg['optimizer']),None,train,val,exp,cfg,
                               criterion=ResearchLoss(cfg['loss'],COUNTS))
            baseline=build('baseline');baseline.fit(resume='none')
            stopped=build('interrupted')
            def stop(epoch,batch,step):
                if epoch==1 and batch==0:raise SimulatedInterruption()
            with self.assertRaises(SimulatedInterruption):stopped.fit(resume='none',interrupt_callback=stop)
            self.assertEqual(stopped.manager.recover()['next_epoch'],1)
            explicit=str(stopped.manager.recovered_path)
            preview=build('interrupted');preview.restore(explicit)
            self.assertEqual(preview.next_epoch,1)
            self.assertEqual(preview.early_stopping,stopped.manager.read(explicit)['early_stopping'])
            restored=build('interrupted');restored.fit(resume='auto')
            Path(restored.manager.discover()['latest']).write_bytes(b'corrupt latest test fixture')
            recovered=build('interrupted');recovered.fit(resume='auto')
            self.assertTrue(recovered.manager.recovery_notes)
            self.assertEqual([r['epoch'] for r in recovered.history],[0,1])
            self.assertEqual(recovered.global_step,4)
            self.assertEqual(baseline.early_stopping,recovered.early_stopping)
            a=baseline.manager.recover();b=recovered.manager.recover()
            self.assertEqual(a['rng']['python'],b['rng']['python'])
            self.assertEqual(a['rng']['numpy'],b['rng']['numpy'])
            torch.testing.assert_close(a['rng']['torch_cpu'],b['rng']['torch_cpu'],rtol=0,atol=0)
            self.assertEqual(a['next_epoch'],b['next_epoch'])
            for key,value in baseline.model.state_dict().items():torch.testing.assert_close(value,recovered.model.state_dict()[key],rtol=0,atol=0)
            for key,values in baseline.optimizer.state_dict()['state'].items():
                for field,value in values.items():torch.testing.assert_close(value,recovered.optimizer.state_dict()['state'][key][field],rtol=0,atol=0)
            self.assertEqual([{k:v for k,v in row.items() if k!='seconds'} for row in baseline.history],
                             [{k:v for k,v in row.items() if k!='seconds'} for row in recovered.history])




class FullValidationGateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from sleep_research.analysis import load_export
        reference=json.loads((PROJECT/'results/phase3_validation_export.json').read_text())
        cls.manifest,cls.frame,cls.features,_=load_export(PROJECT/reference['export'])
        cls.canonical=pd.read_csv(PROJECT/'results/protocol_a_index.csv').query("split == 'validation'").reset_index(drop=True)
        cls.flags=pd.read_csv(PROJECT/'results/quality_flagged_epochs.csv').query("split == 'validation' and wake_trimmed_inclusion == 1")

    def test_exact_complete_validation_reference_passes(self):
        from sleep_research.phase3d_analysis import verify_validation_rows
        verify_validation_rows(self.manifest,self.frame,self.features,self.canonical,self.flags)

    def test_reordered_missing_wrong_labels_and_test_split_fail_closed(self):
        from sleep_research.phase3d_analysis import verify_validation_rows
        for mutation in ['reorder','missing','labels','test']:
            manifest=copy.deepcopy(self.manifest);frame=self.frame.copy()
            if mutation=='reorder':frame=frame.iloc[::-1].reset_index(drop=True)
            elif mutation=='missing':frame=frame.iloc[:-1]
            elif mutation=='labels':frame.loc[0,'true_label']=(int(frame.loc[0,'true_label'])+1)%5
            else:manifest['splits']=['test']
            with self.subTest(mutation=mutation),self.assertRaises((ValueError,AssertionError)):
                verify_validation_rows(manifest,frame,self.features,self.canonical,self.flags)

    def test_quality_flags_cannot_be_silently_removed(self):
        from sleep_research.phase3d_analysis import verify_validation_rows
        frame=self.frame.copy();frame['quality_flagged']=0
        with self.assertRaises(AssertionError):verify_validation_rows(self.manifest,frame,self.features,self.canonical,self.flags)


    def test_comparison_report_wiring_uses_temporary_m1_reference_fixture_only(self):
        # Reuse M1 twice to check report/plot plumbing with zero deltas. This does
        # not estimate M4 performance; all fixture outputs are deleted immediately.
        from types import SimpleNamespace
        from sleep_research.checkpoints import CheckpointManager
        from sleep_research.common import json_write
        import shutil
        baseline=Experiment(PROJECT,'cnn',run_id=read_registry(PROJECT)['baseline_run_id'])
        manager=CheckpointManager(baseline.path/'reproducibility/checkpoints',baseline.metadata['resume_fingerprints'],baseline.run_id)
        reference=json.loads((PROJECT/'results/phase3_validation_export.json').read_text())
        with tempfile.TemporaryDirectory(prefix='phase3d-analysis-wiring-fixture-',dir=PROJECT/'results') as temp:
            root=Path(temp);results=root/'results';results.mkdir()
            fixture_id='SYNTHETIC_ANALYSIS_WIRING_ONLY_NOT_M4_RESULTS'
            fixture=SimpleNamespace(run_id=fixture_id,path=root/'experiments/cnn'/fixture_id,metadata=copy.deepcopy(baseline.metadata))
            fixture.path.joinpath('reproducibility').mkdir(parents=True)
            json_write(fixture.path/'reproducibility/phase3d_resources.json',dict(process_peak_rss_bytes=0,note='temporary analysis wiring fixture'))
            shutil.copyfile(baseline.path/'reproducibility/architecture.json',fixture.path/'reproducibility/architecture.json')
            json_write(results/'phase3_validation_export.json',dict(export=str(PROJECT/reference['export'])))
            json_write(results/'phase3d_m4_export.json',dict(run_id=fixture_id,export=str(PROJECT/reference['export'])))
            for name in ['protocol_a_index.csv','quality_flagged_epochs.csv']:shutil.copyfile(PROJECT/'results'/name,results/name)
            analysis=fixture.path/'analysis/REFERENCE_FIXTURE';analysis.mkdir(parents=True)
            class FixtureManager:
                def __init__(self,*args):pass
                def discover(self):return manager.discover()
                def read(self,path):
                    state=manager.read(path)
                    for row in state['history']:row.update(validation_n1_f1=0.,validation_n3_f1=0.)
                    return state
            body=copy.deepcopy(read_registry(PROJECT));body['run_id']=fixture_id
            with patch('sleep_research.phase3d_analysis.registration',return_value=body),patch('sleep_research.phase3d_analysis.completed_experiment',return_value=fixture),patch('sleep_research.phase3d_analysis.Experiment',side_effect=lambda project,family,run_id:fixture if run_id==fixture_id else baseline),patch('sleep_research.phase3d_analysis.CheckpointManager',FixtureManager),patch('sleep_research.phase3d_analysis.analyze_export',return_value=analysis):
                summary=compare(root)
            self.assertTrue(summary['same_validation_ids']);self.assertFalse(summary['training_called'])
            self.assertTrue(all(value==0 for value in summary['delta_M4_minus_M1'].values()))
            self.assertTrue((results/'phase3d_m4_results.md').exists())
            self.assertEqual(len(list((results/'phase3d_figures').glob('M1_M4_hypnogram_*.png'))),24)
            table=pd.read_csv(results/'phase3d_m1_m4_comparison.csv')
            self.assertEqual(table.validation_examples.tolist(),[30266,30266])
            self.assertEqual(table.iloc[1].loss,'weighted_cross_entropy')

if __name__=='__main__':unittest.main()
