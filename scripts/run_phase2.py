"""Execute an explicitly gated Phase 2 pilot or all-SC preprocessing pass."""

import argparse
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
from sleep_edf.phase2 import run_phase2

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--pilot", action="store_true")
    group.add_argument("--all", action="store_true")
    args = parser.parse_args()
    run_phase2(PROJECT, "pilot" if args.pilot else "all")
