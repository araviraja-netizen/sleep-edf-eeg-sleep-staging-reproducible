"""All historical/new tests; preserve legacy result writes and historical evidence."""
import io,json,os,sys,tempfile,time,unittest
from pathlib import Path
from unittest.mock import patch
P=Path(__file__).resolve().parents[1];sys.path.insert(0,str(P/'src'))
os.environ.setdefault('MPLCONFIGDIR',str(P/'results/.phase5_matplotlib'))
if __name__=='__main__':
    tick=time.monotonic();suite=unittest.defaultTestLoader.discover(str(P/'tests'),pattern='test*.py');log=io.StringIO()
    import test_phase2_full
    original=test_phase2_full.atomic_json
    with tempfile.TemporaryDirectory(prefix='phase5-legacy-evidence-',dir=P/'results') as tmp:
        def redirect(path,value):
            return original(Path(tmp)/Path(path).name if Path(path).name in ['phase2_manifest_validation.json','phase2_normalization_validation.json'] else path,value)
        with patch.object(test_phase2_full,'atomic_json',side_effect=redirect):result=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
    (P/'results/phase5_test_results.txt').write_text(log.getvalue())
    from sleep_research.common import json_write,now,fingerprint
    from sleep_research.phase5_registry import source_files
    json_write(P/'results/phase5_test_results.json',dict(successful=result.wasSuccessful() and not result.skipped,tests_run=result.testsRun,
        failures=len(result.failures),errors=len(result.errors),skipped=len(result.skipped),elapsed_seconds=time.monotonic()-tick,
        python_executable=sys.executable,executed_at=now(),legacy_result_writes_redirected=True,real_test_inference_called=False,
        synthetic_test_split_exports_only=True,source_files_fingerprint=fingerprint(source_files(P))))
    print(log.getvalue());raise SystemExit(0 if result.wasSuccessful() and not result.skipped else 1)
