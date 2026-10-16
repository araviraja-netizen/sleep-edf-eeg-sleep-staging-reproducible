"""Authorization, preserved identity, cluster uncertainty and synthetic export recovery."""
import ast,copy,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sleep_research.common import fingerprint,digest
from sleep_research.phase5_registry import authorize,align,validate_subjects,verify_files,KEYS,PRIMARY,SECONDARY
from sleep_research.phase5_data import FinalEpochDataset,FinalContextDataset,require_access,validate_saved_context
from sleep_research.phase5_statistics import from_confusion,detailed_metrics,subject_bootstrap,cluster_matrices
from sleep_research.phase5_evaluation import chunk_export,merge_chunks,find_chunk,checked_export
from sleep_research.phase5_training import definitions
from sleep_research.metrics import classification_metrics,confusion_matrix
from sleep_research.synthetic import SyntheticEpochs,synthetic_contract
from sleep_research.experiments import Experiment
from sleep_research.trainer import Trainer
from models.cnn1d import CNN1D
P=Path(__file__).resolve().parents[1]


def metadata(split='test',n=15):
    data=SyntheticEpochs(split,n,3000);data.index=pd.DataFrame(data.metadata);data.index['label']=data.labels
    for row in data.metadata:row.update(raw_rail_contact_samples=0,filter_boundary_flag=0,quality_flagged=0,context_mask='[1]',context_epoch_ids=json.dumps([row['epoch_id']]),missing_context_count=0)
    data.index=pd.DataFrame(data.metadata);data.index['label']=data.labels
    return data


def frame(n=30):
    data=metadata(n=n);df=data.index.rename(columns={'label':'true_label'}).copy();df['predicted_label']=(df.true_label+(np.arange(n)%4==0).astype(int))%5
    for i in range(5):df['probability_'+str(i)]=np.where(df.predicted_label==i,.8,.05)
    return df


class AuthorizationTests(unittest.TestCase):
    def test_explicit_flag_exact_digest_and_freeze_required(self):
        body={'plan_id':'synthetic-only','models':['locked']};sha=fingerprint(body);frozen={'plan_sha256':sha}
        for explicit,value,record in [(False,sha,frozen),(True,'bad',frozen),(True,sha,None),(True,sha,{'plan_sha256':'bad'})]:
            with self.assertRaises(PermissionError):authorize(body,record,explicit,value)
        self.assertTrue(authorize(body,frozen,True,sha)['explicit_final_evaluation'])
    def test_gate_before_any_test_loader(self):
        for access in [None,{},dict(explicit_final_evaluation=False)]:
            with patch('sleep_research.phase5_data.EpochDataset.__init__',side_effect=AssertionError('Do not touch loader')) as load:
                with self.assertRaises(PermissionError):FinalEpochDataset(P,{},access)
                load.assert_not_called()
    def test_no_real_test_run_without_authorization(self):
        from sleep_research.phase5_evaluation import run
        body={'plan_id':'synthetic-only'}
        with patch('sleep_research.phase5_evaluation.plan',return_value=body),patch('sleep_research.phase5_evaluation.FinalEpochDataset',side_effect=AssertionError('No real test access')) as loader:
            with self.assertRaises(PermissionError):run(P,False,fingerprint(body))
            loader.assert_not_called()
    def test_freeze_is_distinct_from_authorization_and_rejects_incomplete_optional_training(self):
        from sleep_research.phase5_registry import freeze
        from sleep_research.common import json_write
        with tempfile.TemporaryDirectory(prefix='phase5-synthetic-freeze-',dir=P/'results') as tmp:
            root=Path(tmp);(root/'results').mkdir();body={'plan_id':'synthetic-only','source_files':{}};sha=fingerprint(body)
            (root/'results/phase5_preregistered_evaluation_plan.md').write_text('Synthetic review only')
            json_write(root/'results/phase5_preparation_validation.json',dict(successful=True,plan_sha256=sha))
            with patch('sleep_research.phase5_registry.plan',return_value=body):
                with self.assertRaises(PermissionError):freeze(root,'not-reviewed')
                json_write(root/'results/phase5_optional_validation/status.json',dict(status='running'))
                with self.assertRaises(ValueError):freeze(root,sha)
                json_write(root/'results/phase5_optional_validation/status.json',dict(status='completed'))
                record=freeze(root,sha);self.assertFalse(record['final_evaluation_authorized'])
                self.assertEqual(freeze(root,sha),record)
                (root/'results/phase5_preregistered_evaluation_plan.md').write_text('Mutated report')
                with self.assertRaises(ValueError):freeze(root,sha)
    def test_checkpoint_selection_rejects_altered_identity_without_modifying_original(self):
        from sleep_research.phase5_registry import selected_models,verify_identity
        row=selected_models(P)[0];verify_identity(P,row)
        for field,value in [('checkpoint_sha256','bad'),('best_epoch_1_based',999),('experiment_metadata_sha256','bad')]:
            bad=copy.deepcopy(row);bad[field]=value
            with self.assertRaises(ValueError):verify_identity(P,bad)
    def test_no_cpu_fallback_for_real_final_inference(self):
        from sleep_research.phase5_evaluation import run
        with patch('sleep_research.phase5_evaluation.plan',return_value={}),patch('sleep_research.phase5_evaluation.authorization',return_value={}),patch('torch.backends.mps.is_available',return_value=False),patch('sleep_research.phase5_evaluation.FinalEpochDataset',side_effect=AssertionError('No CPU test access')) as loader:
            with self.assertRaisesRegex(RuntimeError,'MPS'):run(P,True,'synthetic')
            loader.assert_not_called()
    def test_frozen_selection_has_only_predefined_primary_secondary_seeds(self):
        self.assertEqual(len(KEYS),9);self.assertEqual(PRIMARY,['M1','m5_ce_s42','m5_ce_s43','m5_ce_s44','m7_ce_s42','m7_ce_s43','m7_ce_s44'])
        self.assertEqual(SECONDARY,['M4','m5_wce_s42']);self.assertFalse(any('m8' in k for k in KEYS))
    def test_checkpoint_integrity_rejects_mutation_and_path_escape(self):
        with tempfile.TemporaryDirectory(dir=P/'results') as tmp:
            root=Path(tmp);p=root/'checkpoint.pt';p.write_bytes(b'original');files={'checkpoint.pt':digest(p)};verify_files(root,files)
            p.write_bytes(b'corrupt')
            with self.assertRaises(ValueError):verify_files(root,files)
            with self.assertRaises(ValueError):verify_files(root,{'../escape':'bad'})
    def test_subject_isolation_duplicate_and_cross_split_rejected(self):
        df=metadata().index;assign=pd.DataFrame({'subject_id':list(df.subject_id.unique())+['train','validation'],'split':['test']*df.subject_id.nunique()+['train','validation']})
        validate_subjects(df,assign)
        for bad in [pd.concat([df,df.iloc[:1]],ignore_index=True),df.assign(split='train')]:
            with self.assertRaises(ValueError):validate_subjects(bad,assign)
    def test_same_target_gate_rejects_missing_reordering_and_label_change(self):
        original=frame();canonical=original.rename(columns={'true_label':'label'})
        align(original,canonical)
        for bad in [original.iloc[:-1],original.iloc[::-1],original.assign(true_label=(original.true_label+1)%5),original.assign(subject_id='leak')]:
            with self.assertRaises((ValueError,AssertionError)):align(bad,canonical)


class ContextMetricTests(unittest.TestCase):
    def test_authorized_context_preserves_test_metadata_and_stops_at_gaps(self):
        data=metadata(n=4)
        for i,epoch in enumerate([10,11,13,14]):data.metadata[i].update(subject_id='S',recording_id='S1',night=1,original_epoch_index=epoch,onset_seconds=30*epoch,epoch_id='S1:'+str(epoch))
        data.index=pd.DataFrame(data.metadata);data.index['label']=data.labels
        access=dict(explicit_final_evaluation=True,plan_sha256='synthetic-only')
        ds=FinalContextDataset(data,access)
        self.assertEqual(ds.neighbors[1].tolist(),[-1,0,1,-1,-1]);self.assertEqual(ds.neighbors[2].tolist(),[-1,-1,2,3,-1])
        for i in range(len(ds)):
            sample=ds[i];self.assertEqual(sample['split'],'test');self.assertEqual(sample['epoch_id'],data.index.iloc[i].epoch_id)
            self.assertTrue(np.all(sample['eeg'].numpy()[np.array(json.loads(sample['context_mask']))==0]==0))
        data.index.loc[0,'split']='train'
        with self.assertRaises(PermissionError):FinalContextDataset(data,access)
    def test_saved_context_and_quality_corruption_rejected(self):
        df=frame();canonical=df.rename(columns={'true_label':'label'})
        validate_saved_context(df,canonical,1)
        for column,value in [('context_mask','[0]'),('context_epoch_ids','[null]'),('missing_context_count',1),('quality_flagged',1)]:
            altered=df.copy();altered.loc[0,column]=value
            with self.assertRaises(ValueError):validate_saved_context(altered,canonical,1)
    def test_metrics_perfect_wrong_and_absent_classes(self):
        df=frame();m,_=detailed_metrics(df);c=from_confusion(confusion_matrix(df.true_label.to_numpy(),df.predicted_label.to_numpy()))
        for key in ['accuracy','macro_f1','balanced_accuracy','cohen_kappa']:self.assertAlmostEqual(m[key],c[key])
        self.assertEqual(m['micro_f1'],m['accuracy']);self.assertGreaterEqual(m['multiclass_brier_score'],0)
        df=frame(5);df['true_label']=0;df['predicted_label']=0
        for i in range(5):df['probability_'+str(i)]=float(i==0)
        m,_=detailed_metrics(df);self.assertEqual(m['macro_f1'],.2);self.assertEqual(m['balanced_accuracy'],1)
        self.assertIsNone(m['cohen_kappa']);self.assertIsNone(m['classes']['Wake']['specificity']);self.assertIsNone(m['classes']['N1']['roc_auc'])
    def test_bootstrap_identical_models_zero_difference_and_deterministic_subject_draws(self):
        df=frame();frames={'a':df,'b':df.copy()};groups={'group':['a','b']}
        first=subject_bootstrap(frames,groups,[['group','a']],resamples=100,seed=3)
        second=subject_bootstrap(frames,groups,[['group','a']],resamples=100,seed=3)
        self.assertEqual(first,second);self.assertEqual(first['subject_count'],df.subject_id.nunique())
        for d in first['paired_differences']:self.assertEqual(d['estimate'],0);self.assertEqual(d['percentile_interval']['lower'],0);self.assertEqual(d['percentile_interval']['upper'],0)
        with self.assertRaises(ValueError):cluster_matrices({'a':df,'b':df.iloc[::-1]})
        with self.assertRaises(ValueError):subject_bootstrap({'a':df.assign(subject_id='only')},{},[],resamples=10)
    def test_cluster_bootstrap_matches_manual_subject_resampling(self):
        df=frame();subjects,matrices=cluster_matrices({'a':df});B=120;seed=9
        got=subject_bootstrap({'a':df},{},[],resamples=B,seed=seed)
        weights=np.random.default_rng(seed).multinomial(len(subjects),np.full(len(subjects),1/len(subjects)),size=B)
        manual=[from_confusion(np.tensordot(w,matrices['a'],axes=(0,0)))['macro_f1'] for w in weights]
        np.testing.assert_allclose([got['estimates']['a']['macro_f1']['lower'],got['estimates']['a']['macro_f1']['upper']],np.quantile(manual,[.025,.975]))
    def test_optional_ablations_are_equal_full_budgets_and_seed_complete(self):
        rows=definitions(P);self.assertEqual(len(rows),9)
        for row in rows:
            cfg=row['configuration'];self.assertEqual(cfg['max_epochs'],25);self.assertEqual(cfg['early_stopping']['patience'],8);self.assertEqual(cfg['loss']['name'],'cross_entropy')
        self.assertEqual({r['configuration']['seed'] for r in rows},{42,43,44})


class ResumeTests(unittest.TestCase):
    def test_synthetic_test_export_interruption_orphan_adoption_idempotence_and_corruption(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory(prefix='phase5-synthetic-',dir=P/'results') as tmp:
            root=Path(tmp);train=SyntheticEpochs('train',10,3000);val=SyntheticEpochs('validation',10,3000)
            cfg=dict(architecture={'name':'CNN1D'},device='cpu',seed=42,batch_size=5,max_epochs=1,early_stopping={'patience':8},progress_batches=0,num_workers=0,synthetic=True)
            model=CNN1D();exp=Experiment(root,'cnn',cfg,synthetic_contract(train,val),'synthetic-final-export')
            trainer=Trainer(model,torch.optim.AdamW(model.parameters()),None,train,val,exp,cfg);trainer.fit(resume='none')
            checkpoint=Path(trainer.manager.discover()['best']);row=dict(key='synthetic-only',model_id='M1',checkpoint_sha256=digest(checkpoint))
            dataset=metadata(n=15)
            with self.assertRaises(InterruptedError):chunk_export(model,dataset,exp,row,chunk_size=5,batch_size=5,device='cpu',interrupt_after=1)
            # Simulate power loss between a committed exporter publication and chunk pointer commit.
            pointers=exp.path/'performance/phase5_chunks';(pointers/'000000.json').unlink()
            with patch('sleep_research.phase5_evaluation.export_inference',wraps=__import__('sleep_research.inference',fromlist=['export_inference']).export_inference) as exporter:
                refs=chunk_export(model,dataset,exp,row,5,5,'cpu');self.assertEqual(exporter.call_count,2)
            merged=merge_chunks(exp,row,refs,dataset.index,'synthetic-only')
            manifest,df,features,_=checked_export(merged,dataset.index,row['checkpoint_sha256'],allow_synthetic=True);self.assertEqual(len(df),15);self.assertEqual(features.shape,(15,64))
            before=digest(merged/'manifest.json')
            with patch('sleep_research.phase5_evaluation.export_inference',side_effect=AssertionError('Resume must not infer')):
                again=chunk_export(model,dataset,exp,row,5,5,'cpu');merge_chunks(exp,row,again,dataset.index,'synthetic-only')
            self.assertEqual(before,digest(merged/'manifest.json'))
            first=exp.path/refs[0]['export'];(first/'predictions.csv').write_text('corrupt')
            with self.assertRaises(ValueError):chunk_export(model,dataset,exp,row,5,5,'cpu')
    def test_analysis_transitive_imports_do_not_train(self):
        seen=set()
        def visit(name):
            if name in seen:return
            seen.add(name);p=P/'src/sleep_research'/(name+'.py')
            if not p.exists():return
            for node in ast.walk(ast.parse(p.read_text())):
                if isinstance(node,ast.ImportFrom) and node.level==1 and node.module:
                    self.assertNotIn(node.module,['trainer','phase4_campaign','phase5_training']);visit(node.module)
                if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute):self.assertNotEqual(node.func.attr,'fit')
        visit('phase5_analysis')

if __name__=='__main__':unittest.main()
