"""Separate Phase 2 test artifacts; preserves all Phase 1 test results."""

import argparse
import io
import json
from pathlib import Path
import sys
import time
import unittest

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()
    tick = time.monotonic()
    if args.pilot:
        suite = unittest.TestSuite(unittest.defaultTestLoader.discover(str(PROJECT / "tests"), pattern=pattern)
                                   for pattern in ("test_phase1.py", "test_phase2.py", "test_phase2_pilot.py"))
    else:
        suite = unittest.defaultTestLoader.discover(str(PROJECT / "tests"))
    log = io.StringIO()
    result = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
    prefix = "phase2_pilot" if args.pilot else "phase2"
    (PROJECT / f"results/{prefix}_test_results.txt").write_text(log.getvalue())
    payload = {"tests_run": result.testsRun, "failures": len(result.failures), "errors": len(result.errors),
               "skipped": len(result.skipped), "successful": result.wasSuccessful(),
               "elapsed_seconds": time.monotonic() - tick, "python_executable": sys.executable}
    (PROJECT / f"results/{prefix}_test_results.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(log.getvalue())
    raise SystemExit(0 if result.wasSuccessful() and not result.skipped else 1)
