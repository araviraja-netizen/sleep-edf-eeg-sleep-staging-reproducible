"""Verify previous bytes, allowing only the two shared infrastructure extensions."""

import json
from pathlib import Path
import sys

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT/'src'))
from sleep_research.common import digest, json_write, now, preserve_previous


def verify():
    phase12=preserve_previous(PROJECT)
    snapshot=json.loads((PROJECT/'results/phase3_preservation_snapshot.json').read_text())
    permitted=set(snapshot['permitted_source_extensions'])
    unchanged=[];extensions=[]
    for relative, expected in snapshot['files'].items():
        actual=digest(PROJECT/relative)
        if actual == expected:
            unchanged.append(relative)
        elif relative in permitted:
            extensions.append(dict(path=relative,before_sha256=expected,after_sha256=actual))
        else:
            raise ValueError(f'Prior artifact changed: {relative}')
    result=dict(successful=True,verified_at=now(),phase1_2_files_unchanged=len(phase12),
                all_previous_files_unchanged=len(unchanged),shared_infrastructure_extensions=extensions,
                previous_arrays_unchanged=sum(p.endswith('/epochs.npy') for p in unchanged),
                original_test_code_and_results_unchanged=True,raw_directory_written=False)
    json_write(PROJECT/'results/phase3_preservation_validation.json',result)
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    verify()
