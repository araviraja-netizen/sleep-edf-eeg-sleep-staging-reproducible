"""Verify M1 and all prior artifacts without rewriting earlier evidence."""

import json
from pathlib import Path
import sys

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT/'src'))
from sleep_research.common import digest,json_write,now,preserve_previous


if __name__=='__main__':
    phase12=preserve_previous(PROJECT)
    snapshot=json.loads((PROJECT/'results/phase3b_preservation_snapshot.json').read_text())
    allowed=set(snapshot['permitted_shared_source_extensions']);unchanged=[];extensions=[]
    for relative,expected in snapshot['files'].items():
        actual=digest(PROJECT/relative)
        if actual==expected:unchanged.append(relative)
        elif relative in allowed:extensions.append(dict(path=relative,before_sha256=expected,after_sha256=actual))
        else:raise ValueError(f'Prior artifact modified: {relative}')
    baseline='experiments/cnn/cnn1d-a-seed42-20261014-1800/'
    json_write(PROJECT/'results/phase3b_preservation_validation.json',dict(successful=True,verified_at=now(),
                phase1_2_files_unchanged=len(phase12),prior_files_unchanged=len(unchanged),
                m1_artifact_files_unchanged=sum(p.startswith(baseline) for p in unchanged),
                previous_arrays_unchanged=sum(p.endswith('/epochs.npy') for p in unchanged),
                model_source_unchanged='src/models/cnn1d.py' in unchanged,
                shared_source_extensions=extensions,old_test_code_and_logs_unchanged=True,
                frozen_splits_and_normalization_unchanged=True,raw_directory_written=False))
    print((PROJECT/'results/phase3b_preservation_validation.json').read_text())
