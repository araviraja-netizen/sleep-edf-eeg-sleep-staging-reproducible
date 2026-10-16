"""Read-only dataset quality review; new summaries and training-subject plots only."""

import os
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
cache = PROJECT / "results/.phase2_5_matplotlib"
os.environ.setdefault("MPLCONFIGDIR", str(cache))
from sleep_research.quality_review import review_quality, quality_waveforms

if __name__ == "__main__":
    summary = review_quality(PROJECT)
    paths = quality_waveforms(PROJECT, summary)
    print(summary["protocols"])
    print('Training waveform plots:',len(paths))
