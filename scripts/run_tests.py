"""Run unittest with a persistent log and machine-readable result."""

import io
import json
from pathlib import Path
import sys
import time
import unittest

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))


def main():
    tick = time.monotonic()
    suite = unittest.defaultTestLoader.discover(str(PROJECT / "tests"))
    log = io.StringIO()
    result = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
    results = PROJECT / "results"
    results.mkdir(exist_ok=True)
    (results / "test_results.txt").write_text(log.getvalue())
    sys.stdout.write(log.getvalue())
    payload = {"tests_run": result.testsRun, "failures": len(result.failures), "errors": len(result.errors),
               "skipped": len(result.skipped), "successful": result.wasSuccessful(),
               "elapsed_seconds": time.monotonic() - tick, "python_executable": sys.executable,
               "command": f"{sys.executable} scripts/run_tests.py"}
    (results / "test_results.json").write_text(json.dumps(payload, indent=2) + "\n")
    raise SystemExit(0 if result.wasSuccessful() and not result.skipped else 1)


if __name__ == "__main__":
    main()
