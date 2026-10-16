"""Additional public-interface failure tests; no real data access."""
from pathlib import Path
import copy,json,shutil,tempfile,unittest
from unittest.mock import patch
import numpy as np
from verify import metrics,verify,ROOT
from pipeline import initialize,train_one,config,construct,campaign

class IndependentMetricTests(unittest.TestCase):
 def test_perfect_scores_and_all_five_classes(self):
  m,c=metrics(np.eye(5,dtype=int)*4)
  self.assertEqual(m['macro_f1'],1);self.assertEqual(m['accuracy'],1)
  np.testing.assert_array_equal(c['f1'],np.ones(5))
 def test_absent_class_stays_in_macro_denominator(self):
  a=np.zeros((5,5));a[0,0]=10;m,c=metrics(a)
  self.assertEqual(m['macro_f1'],.2);self.assertEqual(m['balanced_accuracy'],1)
  self.assertTrue(np.isnan(m['cohen_kappa']))
 def test_negative_empty_nonfinite_wrong_shape_rejected(self):
  for a in [np.zeros((5,5)),np.eye(4),-np.eye(5),np.full((5,5),np.nan)]:
   with self.subTest(shape=a.shape),self.assertRaises(ValueError):metrics(a)
 def test_scalar_and_vectorized_count_metrics_match(self):
  a=np.arange(25).reshape(5,5)+1;b=a.T
  vector=metrics(np.stack([a,b]))[0]
  for i,m in enumerate([a,b]):
   for k,v in metrics(m)[0].items():self.assertAlmostEqual(v,vector[k][i],places=14)

class PublicInterfaceTests(unittest.TestCase):
 def test_missing_dataset_stops_before_output(self):
  with tempfile.TemporaryDirectory() as t:
   root=Path(t)
   with self.assertRaises(FileNotFoundError):initialize(root/'output',root/'missing')
   self.assertFalse((root/'output').exists())
 def test_existing_workspace_not_overwritten(self):
  with tempfile.TemporaryDirectory() as t:
   root=Path(t);raw=root/'raw';raw.mkdir();out=root/'out';out.mkdir();(out/'keep').write_text('keep')
   with self.assertRaises(FileExistsError):initialize(out,raw)
   self.assertEqual((out/'keep').read_text(),'keep')
 def test_raw_directory_cannot_be_used_for_outputs(self):
  with tempfile.TemporaryDirectory() as t:
   root=Path(t)
   with self.assertRaises(ValueError):initialize(root/'generated',root)
 def test_no_cpu_fallback_before_data_access(self):
  with patch('torch.backends.mps.is_available',return_value=False):
   with self.assertRaisesRegex(RuntimeError,'no CPU fallback'):train_one(Path('/nonexistent'),'M1')
 def test_actual_model_parameter_counts(self):
  expected={'M1':40501,'m5_ce_s42':65717,'m6_ce_s42':57589,'m7_ce_s42':107893,'m8_ce_s42':44629,
      'm8_ablation_no_multiscale_s42':31861,'m8_ablation_no_context_s42':19413}
  for key,count in expected.items():
   with self.subTest(key=key):self.assertEqual(sum(p.numel() for p in construct(config(key)).parameters() if p.requires_grad),count)
 def test_invalid_model_key_rejected(self):
  with self.assertRaises(ValueError):config('best_test_seed')
 def test_campaign_replays_exact_historical_fifteen_run_roster(self):
  from sleep_research.phase4_registry import active_keys
  observed={}
  class FakeEngine:
   def __init__(self,body,path,run,verify,has_result,selection):
    self.body,self.selection=body,selection
   def run(self):
    selected=self.selection([]);observed['keys']=active_keys(self.body,selected);return selected
  with tempfile.TemporaryDirectory() as t:
   p=Path(t);(p/'results').mkdir();(p/'configs').mkdir()
   (p/'configs/phase4_campaign.yaml').write_text((ROOT/'configs/phase4_campaign.yaml').read_text())
   with patch('sleep_research.phase4_orchestration.CampaignEngine',FakeEngine),patch('pipeline.frozen_dataset_contract',return_value={'synthetic_fixture':True}),patch('pipeline.train_one',side_effect=AssertionError('Never train in roster test')):
    selection=campaign(p,'auto')
   self.assertEqual([r['key'] for r in selection['selected']],['m5_ce_s42','m7_ce_s42'])
   self.assertEqual(len(observed['keys']),15)
   self.assertEqual(len(set(observed['keys'])),15)
   self.assertTrue(all(k not in observed['keys'] for k in ['m8_ce_s43','m8_wce_s44']))

class ReducedEvidenceTests(unittest.TestCase):
 def fixture(self,root,subjects=False,splits=False):
  for name in ['evidence','provenance','results/publication_tables']:
   shutil.copytree(ROOT/name,root/name,dirs_exist_ok=True)
  if not subjects:(root/'evidence/subject_confusions.json').unlink()
  if not splits:shutil.rmtree(root/'evidence/splits')
 def test_auto_without_restricted_evidence_checks_aggregate_only(self):
  with tempfile.TemporaryDirectory() as t:
   root=Path(t);self.fixture(root)
   report=verify(root,root/'out')
   self.assertTrue(report['successful'])
   self.assertFalse(report['subject_bootstrap_verified'])
   self.assertFalse(report['frozen_split_verified'])
   self.assertGreater(report['numeric_checks'],100)
   self.assertNotIn('paired subject-bootstrap intervals',report['coverage'])
 def test_explicit_full_requires_restricted_evidence(self):
  with tempfile.TemporaryDirectory() as t:
   root=Path(t);self.fixture(root)
   with self.assertRaisesRegex(FileNotFoundError,'Full verification requires'):verify(root,root/'out','full')
 def test_aggregate_scope_ignores_subject_file(self):
  with tempfile.TemporaryDirectory() as t:
   root=Path(t);self.fixture(root,subjects=True,splits=True)
   (root/'evidence/subject_confusions.json').write_text('deliberately unreadable')
   report=verify(root,root/'out','aggregate')
   self.assertFalse(report['subject_bootstrap_verified'])
   self.assertTrue(report['frozen_split_verified'])
 def test_incomplete_split_bundle_rejected(self):
  with tempfile.TemporaryDirectory() as t:
   root=Path(t);self.fixture(root,splits=True)
   (root/'evidence/splits/subject_splits.csv').unlink()
   with self.assertRaisesRegex(ValueError,'Incomplete frozen split bundle'):verify(root,root/'out')
 def test_reduced_run_removes_stale_paired_output(self):
  with tempfile.TemporaryDirectory() as t:
   root=Path(t);self.fixture(root);out=root/'out';out.mkdir()
   stale=out/'recalculated_paired_comparisons.csv';stale.write_text('not recalculated')
   verify(root,out,'aggregate');self.assertFalse(stale.exists())
 def test_aggregate_mismatch_rejected_without_subject_data(self):
  with tempfile.TemporaryDirectory() as t:
   root=Path(t);self.fixture(root)
   path=root/'results/publication_tables/10_all_seed_test_metrics.csv'
   # Corrupt a guaranteed reported value without depending on CSV formatting.
   import csv
   rows=list(csv.DictReader(path.open()));rows[0]['accuracy']='0'
   with path.open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
   with self.assertRaisesRegex(ValueError,'Numerical mismatch'):verify(root,root/'out')

class PublicationGuardTests(unittest.TestCase):
 def test_content_approval_does_not_authorize_staging(self):
  import prepare_git
  with tempfile.TemporaryDirectory() as t:
   path=Path(t)/'approval.json'
   path.write_text(json.dumps(dict(authors_and_citation_complete=True,code_license_approved=True,
       subject_confusion_counts_release_approved=True,frozen_split_identifiers_release_approved=True,
       publication_inventory_approved=False)))
   args=['prepare_git.py','--stage','--reviewed-manifest-sha256','0'*64,'--approval-record',str(path)]
   with patch('sys.argv',args),patch('prepare_git.subprocess.run',side_effect=AssertionError('No Git before approval')) as git:
    with self.assertRaises(PermissionError):prepare_git.main()
    git.assert_not_called()
 def test_each_sensitive_evidence_category_requires_approval(self):
  import prepare_git
  for missing in ['subject_confusion_counts_release_approved','frozen_split_identifiers_release_approved']:
   with self.subTest(missing=missing),tempfile.TemporaryDirectory() as t:
    record=dict(authors_and_citation_complete=True,code_license_approved=True,subject_confusion_counts_release_approved=True,
        frozen_split_identifiers_release_approved=True,publication_inventory_approved=True)
    record[missing]=False;path=Path(t)/'approval.json';path.write_text(json.dumps(record))
    args=['prepare_git.py','--stage','--reviewed-manifest-sha256','0'*64,'--approval-record',str(path)]
    with patch('sys.argv',args),patch('prepare_git.subprocess.run',side_effect=AssertionError('No Git before approval')) as git:
     with self.assertRaises(PermissionError):prepare_git.main()
     git.assert_not_called()

if __name__=='__main__':unittest.main()
