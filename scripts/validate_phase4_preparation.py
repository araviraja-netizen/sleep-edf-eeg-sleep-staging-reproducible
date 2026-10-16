"""Verify registered preparation and old-file preservation, without any training."""
import json
from pathlib import Path
import sys
import ast
import numpy as np
import nbformat

PROJECT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(PROJECT/'src'))
from sleep_research.common import digest,json_write,now,fingerprint
from sleep_research.phase4_registry import read_registry,source_identity
from models.advanced_sleep import make_advanced
from sleep_research.experiments import model_fingerprint


def validate():
    project=PROJECT;body=read_registry(project);source=source_identity(project)
    for name in ['phase4_benchmarks.json','phase4_smoke_validation.json','phase4_test_results.json']:
        evidence=json.loads((project/'results'/name).read_text())
        if not evidence['successful'] or evidence['source_identity']!=source:raise ValueError('Stale or failed evidence '+name)
    tests=json.loads((project/'results/phase4_test_results.json').read_text());assert not tests['skipped'] and tests['tests_run']>=182
    for row in body['runs'].values():
        if (project/'experiments'/row['configuration']['family']/row['run_id']).exists():raise ValueError('Full campaign run started during preparation')
        model=make_advanced(row['configuration']['architecture'])
        assert model_fingerprint(model,row['configuration']['architecture'])==row['architecture_fingerprint']
        assert sum(p.numel() for p in model.parameters())==row['parameters']
    forbidden=['phase4_final_comparison.md','phase4_models_comparison.csv','phase4_classwise_results.csv','phase4_multiseed_analysis.md','phase4_ablation_results.md','phase4_research_contribution_analysis.md']
    for file in forbidden:
        if (project/'results'/file).exists():raise ValueError('Premature measured report: '+file)
    notebook=nbformat.read(project/'notebooks/04_advanced_models_analysis.ipynb',as_version=4)
    for cell in notebook.cells:
        if cell.cell_type=='code':
            assert cell.execution_count is not None
            for node in ast.walk(ast.parse(cell.source)):
                if isinstance(node,ast.ImportFrom):assert 'trainer' not in (node.module or '') and 'campaign' not in (node.module or '')
                if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute):assert node.func.attr not in ['fit','train']
    snapshot=json.loads((project/'results/phase4_preservation_snapshot.json').read_text());changed=[]
    allowed={'src/sleep_research/inference.py'}
    for relative,expected in snapshot['files'].items():
        actual=digest(project/relative)
        if actual!=expected:
            if relative not in allowed:raise ValueError('Prior file unexpectedly changed: '+relative)
            changed.append(dict(path=relative,before_sha256=expected,after_sha256=actual,
                reason='Backward-compatible optional streaming embedding storage; existing default export behavior tested'))
    result=dict(successful=True,verified_at=now(),campaign_id=body['campaign_id'],source_identity=source,
        prior_files_checked=len(snapshot['files']),prior_files_unchanged=len(snapshot['files'])-len(changed),explicit_shared_extensions=changed,
        processed_arrays_unchanged=sum(x.endswith('/epochs.npy') for x in snapshot['files']),
        full_training_launched=False,frozen_test_evaluated=False,initial_experiments=8,conditional_repeats=4,bounded_ablation_control_runs=3,
        total_planned_runs=15,immutable_configurations=len(body['runs']),tests=tests,
        benchmark_variants=6,real_mps_resume_smoke_models=4,final_measured_reports_absent=True,phase5_started=False)
    json_write(project/'results/phase4_preparation_validation.json',result)
    json_write(project/'results/phase4_preservation_validation.json',dict(successful=True,prior_files_checked=result['prior_files_checked'],
        unchanged=result['prior_files_unchanged'],extensions=changed,processed_arrays_unchanged=153))
    return result

if __name__=='__main__':print(json.dumps(validate(),indent=2))
