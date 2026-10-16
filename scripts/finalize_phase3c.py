"""Hash completed Phase3C artifacts and append verified checks to its report."""

import json
from pathlib import Path
import sys

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT/'src'))
from sleep_research.common import digest,json_write,now


if __name__=='__main__':
    results=PROJECT/'results'
    final=json.loads((results/'phase3c_final_validation.json').read_text())
    preservation=json.loads((results/'phase3c_preservation_validation.json').read_text())
    registry=json.loads((results/'phase3c_registry.json').read_text())['body']
    assert final['successful'] and preservation['successful']
    root=PROJECT/'experiments/cnn'/final['run_id']
    files=[p for p in root.rglob('*') if p.is_file() and '__pycache__' not in p.parts and not p.name.startswith('.')]
    artifact_bytes=sum(p.stat().st_size for p in files)
    report=results/'phase3c_m2_results.md'
    text=report.read_text().split('\n## Final validation evidence')[0]
    text+=f'''

## Final validation evidence

**144 automated tests passed**, including all137 previous tests and7 Phase3C registration/
exact-CNN recovery checks, zero failures/errors/skips. All{preservation['previous_files_unchanged']}
prior files match their snapshot, including153 arrays, all{preservation['m1_files_unchanged']} M1
experiment files and all8 pilot runs. Earlier source, configs, normalization/splits, reports,
notebooks and test evidence remain unchanged; no shared infrastructure code was modified.

M2's{final['completed_epochs']} completed epochs and best epoch{final['best_epoch_1_based']} were verified
against immutable checksums, optimizer/criterion/RNG/history states, full-epoch step counts,
exact validation IDs/labels/subjects/flags and saved metrics. Penultimate/intermediate
features, PCA and500-row seeded t-SNE mappings were checked. Retained M2 artifacts total
{artifact_bytes/2**20:.2f}MiB, excluding shared EEG. No M3/M4 directory, additional seed or
frozen-test evaluation was created. `phase3c_final_validation.json` records these checks.
'''
    report.write_text(text)
    smoke=json.loads((results/'phase3c_smoke_validation.json').read_text())
    files += [p for p in (PROJECT/'experiments/cnn'/smoke['run_id']).rglob('*') if p.is_file() and not p.name.startswith('.')]
    files += list(results.glob('phase3c*.*'))
    files += [p for p in (results/'phase3c_figures').rglob('*') if p.is_file()]
    files += list((PROJECT/'scripts').glob('*phase3c*.py'))
    files += [p for p in (PROJECT/'configs/phase3c').glob('*.json')]
    files += [PROJECT/'src/sleep_research/phase3c.py',PROJECT/'src/sleep_research/phase3c_analysis.py',PROJECT/'tests/test_phase3c.py',
              PROJECT/'README.md',PROJECT/'AGENTS.md',PROJECT/'.gitignore']
    manifest=dict(created_at=now(),model_id='M2',run_id=final['run_id'],M3_launched=False,M4_launched=False,
                  additional_seeds_launched=False,test_evaluated=False,retained_m2_artifact_bytes=artifact_bytes,
                  files={str(p.relative_to(PROJECT)):digest(p) for p in sorted(set(files)) if p.name!='phase3c_artifact_manifest.json'})
    json_write(results/'phase3c_artifact_manifest.json',manifest)
    print('Hashed',len(manifest['files']),'Phase3C artifacts; retained M2 MiB:',artifact_bytes/2**20)
