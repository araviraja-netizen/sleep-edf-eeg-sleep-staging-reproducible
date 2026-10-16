"""All-phase tests with separate evidence and read-only legacy-result preservation."""

import argparse
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT / "results/.phase3b_matplotlib"))

if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--infrastructure-only",action="store_true")
    parser.add_argument("--previous-only",action="store_true")
    args=parser.parse_args()
    tick=time.monotonic()
    pattern="test_phase3b.py" if args.infrastructure_only else "test*.py"
    suite=unittest.defaultTestLoader.discover(str(PROJECT/"tests"),pattern=pattern)
    if args.previous_only:
        selected=unittest.TestSuite()
        for pattern in ("test_phase1.py","test_phase2.py","test_phase2_full.py","test_phase2_pilot.py","test_phase2_5.py","test_phase3.py"):
            selected.addTests(unittest.defaultTestLoader.discover(str(PROJECT/"tests"),pattern=pattern))
        suite=selected
    log=io.StringIO()
    if args.infrastructure_only:
        result=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
    else:
        import test_phase2_full
        original=test_phase2_full.atomic_json
        with tempfile.TemporaryDirectory(prefix="phase3b-legacy-evidence-",dir=PROJECT/"results") as tmp:
            def redirected(path,value):
                if Path(path).name in ("phase2_manifest_validation.json","phase2_normalization_validation.json"):
                    return original(Path(tmp)/Path(path).name,value)
                return original(path,value)
            with patch.object(test_phase2_full,"atomic_json",side_effect=redirected):
                result=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
    prefix="phase3b_previous" if args.previous_only else ("phase3b_unit" if args.infrastructure_only else "phase3b")
    (PROJECT/f"results/{prefix}_test_results.txt").write_text(log.getvalue())
    payload={"tests_run":result.testsRun,"failures":len(result.failures),"errors":len(result.errors),"skipped":len(result.skipped),
             "successful":result.wasSuccessful(),"elapsed_seconds":time.monotonic()-tick,"python_executable":sys.executable,
             "legacy_test_code_modified":False,"legacy_result_writes_redirected":not args.infrastructure_only}
    (PROJECT/f"results/{prefix}_test_results.json").write_text(json.dumps(payload,indent=2)+"\n")
    print(log.getvalue())
    raise SystemExit(0 if result.wasSuccessful() and not result.skipped else 1)
