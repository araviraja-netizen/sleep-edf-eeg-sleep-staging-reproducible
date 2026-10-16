"""Portable CPU suite using unchanged historical tests and explicit data-required skips."""
from pathlib import Path
import argparse,importlib,io,json,os,sys,tempfile,time,unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tests'),str(ROOT/'release_tools')]

SAFE={
 'test_phase1':{'FilenameTests','AnnotationTests'},
 'test_phase2':{'WakeTests','SplitTests','FilterTests','StorageTests','NormalizationTests'},
 'test_phase2_5':{'CheckpointTests','TrainerTests','ArtifactAnalysisTests','MetricsRepresentationTests'},
 'test_phase3':{'CNNTests','CNNResumeTests'},
 'test_phase3b':{'LossTests','OptimizerResumeTests','ProtocolTests'},
 'test_phase3d':{'M4Math'},
 'test_phase4':{'ContextTests','ModelTests','ResumeExportTests','CampaignTests','AnalysisTests'},
 'test_phase5':{'AuthorizationTests','ContextMetricTests','ResumeTests'},
 'test_phase5_pipeline':{'FinalAnalysisPipelineTests'},
}
REQUIRES={
 'test_phase2.SplitTests.test_frozen_manifests_not_rewritten_and_tampering_rejected',
 'test_phase4.ContextTests.test_real_metadata_and_normalized_context_stay_inside_frozen_subjects',
 'test_phase4.CampaignTests.test_full_campaign_no_cpu_fallback',
 'test_phase3b.ProtocolTests.test_fixed_architecture_and_registered_seeds_bounds',
 'test_phase4.AnalysisTests.test_same_id_gate_rejects_reordered_and_missing_validation',
 'test_phase4.AnalysisTests.test_final_report_pipeline_requires_all_runs_and_writes_only_temporary_fixture_results',
 'test_phase5.AuthorizationTests.test_checkpoint_selection_rejects_altered_identity_without_modifying_original',
}
def flatten(suite):
 for case in suite:
  if isinstance(case,unittest.TestSuite):yield from flatten(case)
  else:yield case

def run(output):
 output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True)
 with tempfile.TemporaryDirectory(prefix='release-cpu-',dir=output) as directory:
  temp=Path(directory);raw=temp/'empty_raw_fixture';raw.mkdir()
  os.environ['MPLCONFIGDIR']=str(temp/'matplotlib')
  import torch
  torch.set_num_threads(1)
  from pipeline import initialize
  workspace=initialize(temp/'workspace',raw)
  records=[];suite=unittest.TestSuite()
  for file in sorted((ROOT/'tests').glob('test*.py')):
   module=importlib.import_module(file.stem)
   for name in ['P','PROJECT']:
    if hasattr(module,name):setattr(module,name,workspace)
   for test in flatten(unittest.defaultTestLoader.loadTestsFromModule(module)):
    identifier=test.id();cls=type(test).__name__
    permitted=cls in SAFE.get(file.stem,set()) and identifier not in REQUIRES
    if permitted:
     suite.addTest(test);records.append(dict(test=identifier,classification='CPU/synthetic core'))
    else:
     reason='Requires user-provided EEG and/or excluded historical processed/experiment artifacts; not a lightweight CPU check'
     def skip(reason=reason):raise unittest.SkipTest(reason)
     suite.addTest(unittest.FunctionTestCase(skip,description=identifier));records.append(dict(test=identifier,classification='additional inputs required',reason=reason))
  import test_release
  suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(test_release))
  log=io.StringIO();tick=time.monotonic();result=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
  summary=dict(successful=result.wasSuccessful(),tests_discovered=result.testsRun,passed=result.testsRun-len(result.skipped)-len(result.failures)-len(result.errors),
      failures=len(result.failures),errors=len(result.errors),additional_inputs_required=len(result.skipped),
      elapsed_seconds=time.monotonic()-tick,raw_data_read=False,full_training_called=False,
      historical_test_source_unchanged=True,classifications=records)
  (output/'core_tests.log').write_text(log.getvalue());(output/'core_tests.json').write_text(json.dumps(summary,indent=2)+'\n')
  print(json.dumps({k:v for k,v in summary.items() if k!='classifications'},indent=2))
  if not result.wasSuccessful():print(log.getvalue())
  return 0 if result.wasSuccessful() else 1

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);raise SystemExit(run(p.parse_args().output))
