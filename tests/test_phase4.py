"""Mask/boundary/model/streaming/resume/campaign tests; all training is tiny synthetic."""
import ast
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from models.advanced_sleep import make_advanced
from sleep_research.phase4_data import ContextDataset
from sleep_research.phase4_registry import configuration,variant,select_shortlist,active_keys,COUNTS
from sleep_research.phase4_campaign import build_trainer,ready
from sleep_research.phase4_orchestration import CampaignEngine
from sleep_research.phase4_analysis import aligned,normalized_confusion,transition_metrics
from sleep_research.phase4_explanations import context_cam
from sleep_research.synthetic import SyntheticEpochs,synthetic_contract
from sleep_research.trainer import SimulatedInterruption,evaluate_model
from sleep_research.objectives import ResearchLoss,class_weights
from sleep_research.inference import export_inference
from sleep_research.analysis import load_export
from sleep_research.real_data import ResearchEpochDataset
from sleep_research.metrics import classification_metrics
from sleep_research.checkpoints import IncompatibleCheckpoint

PROJECT=Path(__file__).resolve().parents[1]


def synthetic(split='train',count=10):
    base=SyntheticEpochs(split,count,3000);base.index=pd.DataFrame(base.metadata);base.index['label']=base.labels
    return base


class ContextTests(unittest.TestCase):
    def test_preserves_all_targets_and_masks_recording_boundaries(self):
        base=synthetic();data=ContextDataset(base)
        self.assertEqual(len(data),len(base))
        for i in range(len(data)):
            row=data[i];self.assertEqual(row['epoch_id'],base[i]['epoch_id']);self.assertEqual(row['eeg'].shape,(5,3001))
            self.assertEqual(float(row['eeg'][2,-1]),1.)
            ids=json.loads(row['context_epoch_ids']);mask=json.loads(row['context_mask'])
            for slot,neighbor in enumerate(data.neighbors[i]):
                if neighbor<0:
                    self.assertIsNone(ids[slot]);self.assertEqual(mask[slot],0);self.assertTrue(torch.equal(row['eeg'][slot],torch.zeros(3001)))
                else:
                    other=base.index.iloc[neighbor];center=base.index.iloc[i]
                    self.assertEqual((other.subject_id,other.recording_id,other.night),(center.subject_id,center.recording_id,center.night))
                    self.assertEqual(other.original_epoch_index-center.original_epoch_index,slot-2)

    def test_gap_blocks_farther_existing_context(self):
        base=synthetic(count=4)
        for i,epoch in enumerate([10,11,13,14]):
            base.metadata[i].update(subject_id='A',recording_id='A1',original_epoch_index=epoch,onset_seconds=epoch*30,epoch_id='A1:'+str(epoch))
        base.index=pd.DataFrame(base.metadata);base.index['label']=base.labels
        data=ContextDataset(base)
        self.assertEqual(data.neighbors[1].tolist(),[-1,0,1,-1,-1])
        self.assertEqual(data.neighbors[2].tolist(),[-1,-1,2,3,-1])

    def test_neighbor_labels_never_change_inputs(self):
        base=synthetic();data=ContextDataset(base);before=data[0]['eeg'].clone()
        base.labels[:]=(base.labels+3)%5
        torch.testing.assert_close(data[0]['eeg'],before,rtol=0,atol=0)

    def test_mixed_test_duplicate_ambiguous_and_off_grid_metadata_refused(self):
        for mutation in ['test','mixed','duplicate','subject','time','night']:
            base=synthetic()
            if mutation=='test':base.split='test'
            elif mutation=='mixed':base.index.loc[0,'split']='validation'
            elif mutation=='duplicate':base.index.loc[1,'epoch_id']=base.index.loc[0,'epoch_id']
            elif mutation=='subject':base.index.loc[4,'subject_id']='OTHER'
            elif mutation=='time':base.index['onset_seconds']=base.index.onset_seconds.astype(float);base.index.loc[0,'onset_seconds']=.1
            else:base.index.loc[4,'night']=2
            with self.subTest(mutation=mutation),self.assertRaises((ValueError,PermissionError)):ContextDataset(base)

    def test_real_metadata_and_normalized_context_stay_inside_frozen_subjects(self):
        train=ResearchEpochDataset(PROJECT,'train',2);val=ResearchEpochDataset(PROJECT,'validation',2)
        try:
            a,b=ContextDataset(train),ContextDataset(val)
            self.assertEqual((len(a),len(b)),(134047,30266));self.assertFalse(set(a.index.subject_id)&set(b.index.subject_id))
            for data in [a,b]:
                for i in [0,1,len(data)-1]:
                    sample=data[i];torch.testing.assert_close(sample['eeg'][2,:3000],torch.as_tensor(data.base[i]['eeg'][0]),rtol=0,atol=0)
                    for slot,neighbor in enumerate(data.neighbors[i]):
                        if neighbor>=0:self.assertEqual(int(data.index.iloc[neighbor].original_epoch_index),int(data.index.iloc[i].original_epoch_index)+slot-2)
        finally:train.close();val.close()


class ModelTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1)
    def test_all_models_and_ablations_shapes_finite_gradients(self):
        x=torch.randn(3,5,3001);x[:,:,-1]=1;x[:,0]=0
        for model,ablation in [('M5',None),('M6',None),('M7',None),('M8',None),('M8','no_multiscale'),('M8','no_context')]:
            with self.subTest(model=model,ablation=ablation):
                net=make_advanced(variant(model,ablation));y=net(x);self.assertEqual(y.shape,(3,5))
                loss=ResearchLoss(dict(name='weighted_cross_entropy'),COUNTS);value=loss(y,torch.tensor([0,1,3]));value.backward()
                self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters()))
                expected=torch.nn.functional.cross_entropy(y,torch.tensor([0,1,3]),weight=torch.tensor(class_weights(COUNTS)))
                torch.testing.assert_close(value,expected)

    def test_binary_mask_missing_center_nonfinite_and_bad_padding_refused(self):
        for mutation in ['mask','center','nonfinite','padding']:
            x=torch.zeros(2,5,3001);x[:,:,3000]=1
            if mutation=='mask':x[:,0,-1]=.5
            elif mutation=='center':x[:,2,-1]=0
            elif mutation=='nonfinite':x[:,1,2]=float('nan')
            else:x[:,0,-1]=0;x[:,0,4]=1
            with self.assertRaises(ValueError):make_advanced(variant('M7'))(x)

    def test_context_ablation_cannot_use_neighbor_eeg(self):
        net=make_advanced(variant('M8','no_context')).eval();x=torch.randn(2,5,3001);x[:,:,-1]=1
        first=net(x);x[:,[0,1,3,4],:3000]*=100;torch.testing.assert_close(first,net(x),rtol=0,atol=0)

    def test_configurations_are_controlled_and_ablations_equal_budget(self):
        for model in ['M5','M6','M7','M8']:
            a=configuration(PROJECT,model,'cross_entropy');b=configuration(PROJECT,model,'weighted_cross_entropy')
            self.assertEqual(a['architecture'],b['architecture']);self.assertEqual(a['optimizer'],b['optimizer'])
            self.assertEqual((a['max_epochs'],a['batch_size'],a['early_stopping']['patience']),(25,16,8))
            self.assertEqual(a['loss_alpha'],[1.]*5);np.testing.assert_array_equal(b['loss_alpha'],class_weights(COUNTS))
        for ablation in ['control','no_multiscale','no_context']:
            self.assertEqual(configuration(PROJECT,'M8','cross_entropy',ablation=ablation)['max_epochs'],3)

    def test_masked_context_cam_has_original_epoch_mapping(self):
        x=torch.randn(2,5,3001);x[:,:,-1]=1;x[:,0]=0
        for model,layer in [('M5','encoder.cnn.blocks.3.0'),('M8','encoder.branches.0.4')]:
            net=make_advanced(variant(model)).eval();cam=context_cam(net,x,layer)
            self.assertEqual(cam.shape,(2,5,3000));self.assertTrue(np.isfinite(cam).all());self.assertTrue((cam[:,0]==0).all())


class ResumeExportTests(unittest.TestCase):
    def test_all_architectures_exact_cpu_optimizer_rng_history_recovery(self):
        torch.set_num_threads(1)
        for model in ['M5','M6','M7','M8']:
            with self.subTest(model=model),tempfile.TemporaryDirectory(dir=PROJECT/'results') as tmp:
                root=Path(tmp);cfg=configuration(PROJECT,model,'weighted_cross_entropy');cfg.update(device='cpu',synthetic=True,stage='synthetic_test',max_epochs=2,batch_size=5,progress_batches=0)
                base,val=synthetic(),synthetic('validation');train_data,val_data=ContextDataset(base),ContextDataset(val)
                contract=synthetic_contract(base,val)
                def build(run):return build_trainer(root,cfg,contract,train_data,val_data,run)
                direct=build('direct');direct.fit(resume='none');stopped=build('interrupted')
                def stop(epoch,batch,step):
                    if epoch==1 and batch==0:raise SimulatedInterruption()
                with self.assertRaises(SimulatedInterruption):stopped.fit(resume='none',interrupt_callback=stop)
                checkpoint=stopped.manager.recover();self.assertEqual(checkpoint['next_epoch'],1)
                restored=build('interrupted');restored.fit(resume='auto')
                for key,value in direct.model.state_dict().items():torch.testing.assert_close(value,restored.model.state_dict()[key],rtol=0,atol=0)
                for param,values in direct.optimizer.state_dict()['state'].items():
                    for key,value in values.items():torch.testing.assert_close(value,restored.optimizer.state_dict()['state'][param][key],rtol=0,atol=0)
                self.assertEqual(direct.early_stopping,restored.early_stopping)
                self.assertEqual([{k:v for k,v in r.items() if k!='seconds'} for r in direct.history],[{k:v for k,v in r.items() if k!='seconds'} for r in restored.history])
                torch.testing.assert_close(direct.manager.recover()['rng']['torch_cpu'],restored.manager.recover()['rng']['torch_cpu'],rtol=0,atol=0)
                latest=Path(restored.manager.discover()['latest']);latest.write_bytes(b'corrupt fixture')
                recovered=build('interrupted');recovered.fit(resume='auto');self.assertTrue(recovered.manager.recovery_notes)
                for key,value in direct.model.state_dict().items():torch.testing.assert_close(value,recovered.model.state_dict()[key],rtol=0,atol=0)

    def test_streamed_export_matches_ids_features_and_class_probability_mapping(self):
        with tempfile.TemporaryDirectory(dir=PROJECT/'results') as tmp:
            root=Path(tmp);cfg=configuration(PROJECT,'M6','cross_entropy');cfg.update(device='cpu',synthetic=True,max_epochs=1,batch_size=5,progress_batches=0)
            a,b=synthetic(),synthetic('validation');train,val=ContextDataset(a),ContextDataset(b)
            trainer=build_trainer(root,cfg,synthetic_contract(a,b),train,val,'stream');trainer.fit(resume='none')
            exports=[]
            for streaming in [False,True]:
                export=export_inference(trainer.model,trainer.validation_loader,trainer.experiment,device='cpu',streaming=streaming,
                    intermediate_layers=[dict(name='intermediate',pooling='flatten')],metadata_fields=['context_epoch_ids','context_mask'])
                manifest,frame,features,_=load_export(export);exports.append((frame,features.copy()))
                self.assertEqual(frame.epoch_id.tolist(),b.index.epoch_id.tolist());self.assertEqual(features.shape,(10,64))
            pd.testing.assert_frame_equal(exports[0][0],exports[1][0]);np.testing.assert_array_equal(exports[0][1],exports[1][1])
            self.assertFalse(list(trainer.experiment.path.glob('.feature-spool-*')))

    def test_test_loader_is_gated_before_forward(self):
        net=make_advanced(variant('M6'))
        with patch.object(net,'forward',side_effect=AssertionError('No test forward')) as forward:
            with self.assertRaises(PermissionError):evaluate_model(net,DataLoader(synthetic('test'),batch_size=5))
            forward.assert_not_called()


class CampaignTests(unittest.TestCase):
    def body(self):
        initial=[f'm{i}_{loss}_s42' for i in range(5,9) for loss in ['ce','wce']]
        conditional=[key.rsplit('_s',1)[0]+'_s'+str(seed) for key in initial for seed in [43,44]]
        ablations=['m8_ablation_'+a+'_s42' for a in ['control','no_multiscale','no_context']]
        return dict(campaign_id='synthetic-only',initial_keys=initial,conditional_keys=conditional,ablation_keys=ablations,
                    runs={key:dict(kind='initial' if key in initial else 'ablation' if key in ablations else 'conditional_repeat') for key in initial+conditional+ablations})
    def test_interrupt_restart_skips_verified_previous_runs_and_runs_exact_budget(self):
        with tempfile.TemporaryDirectory(dir=PROJECT/'results') as tmp:
            body=self.body();done={};calls=[];stop=[True]
            def run(key):
                calls.append(key)
                if len(done)==2 and stop[0]:stop[0]=False;raise KeyboardInterrupt()
                done[key]=dict(key=key,macro_f1=.5+body['initial_keys'].index(key)*.01 if key in body['initial_keys'] else .5,n1_f1=.2,parameters=100,training_seconds=1)
            def verify(key):return done[key]
            def selection(rows):return dict(selected=select_shortlist(rows))
            def engine():return CampaignEngine(body,Path(tmp)/'state.json',run,verify,lambda k:k in done,selection)
            first=engine()
            with self.assertRaises(KeyboardInterrupt):first.run()
            self.assertEqual(first.state['status'],'interrupted');engine().run()
            self.assertEqual(len(done),15);self.assertEqual(calls.count(body['initial_keys'][0]),1)
            self.assertEqual(calls.count(body['initial_keys'][1]),1);self.assertEqual(len(calls),16)
            engine().run();self.assertEqual(len(calls),16)
    def test_corrupt_completed_run_stops_without_retraining_or_silent_skip(self):
        with tempfile.TemporaryDirectory(dir=PROJECT/'results') as tmp:
            body=self.body();called=[]
            def verify(key):raise ValueError('Corrupt completed artifact')
            engine=CampaignEngine(body,Path(tmp)/'state.json',lambda k:called.append(k),verify,lambda k:True,lambda x:None)
            with self.assertRaises(ValueError):engine.run()
            self.assertEqual(called,[]);self.assertEqual(engine.state['status'],'failed')
    def test_shortlist_tie_rule_and_missing_duplicate_results(self):
        rows=[dict(key=str(i),macro_f1=.7,n1_f1=.3,parameters=100+i,training_seconds=1) for i in range(8)]
        self.assertEqual([r['key'] for r in select_shortlist(rows)],['0','1'])
        rows[4]['n1_f1']=.4;self.assertEqual(select_shortlist(rows)[0]['key'],'4')
        for bad in [rows[:-1],rows[:7]+[rows[0]]]:
            with self.assertRaises(ValueError):select_shortlist(bad)
    def test_full_campaign_no_cpu_fallback(self):
        with patch('sleep_research.phase4_campaign.torch.backends.mps.is_available',return_value=False):
            with self.assertRaisesRegex(RuntimeError,'MPS required'):ready(PROJECT,{})


class AnalysisTests(unittest.TestCase):
    def test_same_id_gate_rejects_reordered_and_missing_validation(self):
        ref=json.loads((PROJECT/'results/phase3_validation_export.json').read_text());_,frame,_,_=load_export(PROJECT/ref['export'])
        canonical=pd.read_csv(PROJECT/'results/protocol_a_index.csv').query("split == 'validation'").reset_index(drop=True)
        aligned(frame,canonical)
        for altered in [frame.iloc[::-1],frame.iloc[:-1]]:
            with self.assertRaises((ValueError,AssertionError)):aligned(altered,canonical)
    def test_transition_definition_cannot_bridge_gaps(self):
        frame=pd.DataFrame(dict(subject_id=['A']*4,recording_id=['A1']*4,night=[1]*4,original_epoch_index=[1,2,4,5],true_label=[0,1,2,2],predicted_label=[0,1,2,2]))
        for i in range(5):frame['probability_'+str(i)]=(frame.true_label==i).astype(float)
        result=transition_metrics(frame);self.assertEqual(result['transition']['epochs'],1);self.assertEqual(result['stable']['epochs'],1)
        self.assertTrue(np.isfinite(normalized_confusion(np.zeros((5,5)))).all())
    def test_analysis_transitive_imports_do_not_import_trainer_or_campaign(self):
        seen=set()
        def visit(name):
            if name in seen:return
            seen.add(name);path=PROJECT/'src/sleep_research'/(name+'.py')
            if not path.exists():return
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node,ast.ImportFrom) and node.level==1 and node.module:
                    self.assertNotIn(node.module,['trainer','phase4_campaign','phase4_smoke']);visit(node.module)
                if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute):self.assertNotEqual(node.func.attr,'fit')
        visit('phase4_analysis')


    def test_final_report_pipeline_requires_all_runs_and_writes_only_temporary_fixture_results(self):
        # Isolated report-wiring fixture; saved M1/M4 references are read only.
        # It does not estimate advanced-model performance or create project results.
        import shutil
        from sleep_research.phase4_analysis import regenerate
        body=CampaignTests().body();decision=dict(selected=[dict(key=k) for k in body['initial_keys'][:2]])
        ref=json.loads((PROJECT/'results/phase3_validation_export.json').read_text());_,real,_,_=load_export(PROJECT/ref['export'])
        metric=classification_metrics(real.true_label.to_numpy(),real[[f'probability_{i}' for i in range(5)]].to_numpy())
        transitions=dict(transition=dict(epochs=1,metrics=metric),stable=dict(epochs=1,metrics=metric))
        def fake_verify(project,row):
            key=next(key for key,value in body['runs'].items() if value is row)
            return dict(key=key,model_id=key.split('_')[0].upper(),kind=row['kind'],seed=int(key.rsplit('_s',1)[1]),
                loss='weighted_cross_entropy' if '_wce_' in key else 'cross_entropy',parameters=100,training_seconds=1.,
                completed_epochs=3,ablation=key.split('ablation_')[1].rsplit('_s',1)[0] if 'ablation_' in key else None,
                export=ref['export'],metrics=metric,transition_metrics=transitions,n1_f1=metric['classes']['N1']['f1'],n3_f1=metric['classes']['N3']['f1'],
                **{k:metric[k] for k in ['accuracy','macro_f1','balanced_accuracy','cohen_kappa']})
        with tempfile.TemporaryDirectory(prefix='phase4-report-wiring-synthetic-',dir=PROJECT/'results') as tmp:
            root=Path(tmp);out=root/'results';out.mkdir()
            shutil.copyfile(PROJECT/'results/protocol_a_index.csv',out/'protocol_a_index.csv')
            for filename in ['phase3_validation_export.json','phase3d_m4_export.json']:
                reference=json.loads((PROJECT/'results'/filename).read_text());reference['export']=str(PROJECT/reference['export'])
                (out/filename).write_text(json.dumps(reference))
            with patch('sleep_research.phase4_analysis.read_registry',return_value=body),patch('sleep_research.phase4_analysis.envelope',return_value=decision),patch('sleep_research.phase4_analysis.verify_run',side_effect=fake_verify):
                result=regenerate(root)
            self.assertEqual(result['measured_runs'],15);self.assertFalse(result['training_called'])
            table=pd.read_csv(out/'phase4_models_comparison.csv');self.assertEqual(len(table),17)
            self.assertEqual(len(pd.read_csv(out/'phase4_classwise_results.csv')),85)
            seeds=pd.read_csv(out/'phase4_multiseed_summary.csv');self.assertEqual(len(seeds),2)
            np.testing.assert_allclose(seeds.macro_f1_sample_sd.to_numpy(),np.zeros(2),atol=1e-14,rtol=0)
            for name in ['phase4_final_comparison.md','phase4_ablation_results.md','phase4_research_contribution_analysis.md']:
                self.assertTrue((out/name).exists())

if __name__=='__main__':unittest.main()
