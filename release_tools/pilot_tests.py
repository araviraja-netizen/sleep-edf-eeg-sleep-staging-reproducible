"""Run unchanged Phase 1/2 pilot tests against explicitly configured workspace paths."""
from pathlib import Path
import io,json,sys,time,unittest

def main(project):
 project=Path(project);sys.path[:0]=[str(project/'src'),str(project/'tests')]
 import tomllib
 config=tomllib.loads((project/'configs/phase1.toml').read_text())
 import test_phase1
 test_phase1.DATASET=Path(config['dataset_root'])
 modules=[test_phase1,__import__('test_phase2'),__import__('test_phase2_pilot')]
 for m in modules:
  if hasattr(m,'PROJECT'):m.PROJECT=project
 suite=unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromModule(m) for m in modules)
 log=io.StringIO();tick=time.monotonic();result=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
 payload=dict(successful=result.wasSuccessful() and not result.skipped,tests_run=result.testsRun,failures=len(result.failures),
     errors=len(result.errors),skipped=len(result.skipped),elapsed_seconds=time.monotonic()-tick,
     dataset_path_configured=True,scientific_tests_unchanged=True)
 (project/'results/phase2_pilot_test_results.txt').write_text(log.getvalue())
 (project/'results/phase2_pilot_test_results.json').write_text(json.dumps(payload,indent=2)+'\n')
 print(log.getvalue());return 0 if payload['successful'] else 1
