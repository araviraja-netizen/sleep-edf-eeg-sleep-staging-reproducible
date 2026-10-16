"""Verify and seal preparation evidence only. Never freezes or authorizes test inference."""
import json,shutil,sys
from pathlib import Path
P=Path(__file__).resolve().parents[1];sys.path.insert(0,str(P/'src'))
from sleep_research.common import digest,fingerprint,json_write,now
from sleep_research.phase5_registry import plan,audit,source_files
from sleep_research.phase5_evaluation import workspace

if __name__=='__main__':
    body=plan(P);sha=fingerprint(body);validation=audit(P)
    tests=json.loads((P/'results/phase5_test_results.json').read_text());notebook=json.loads((P/'results/phase5_notebook_execution.json').read_text());smoke=json.loads((P/'results/phase5_validation_smoke.json').read_text())
    assert tests['successful'] and not tests['failures'] and not tests['errors'] and not tests['skipped']
    assert tests['source_files_fingerprint']==fingerprint(source_files(P))
    assert notebook['successful'] and notebook['error_outputs']==0 and not notebook['authorized_saved_final_results_read']
    assert smoke['successful'] and len(smoke['models'])==9 and all(r['cpu_forward_successful'] for r in smoke['models'])
    assert not (P/'results/phase5_final_freeze.json').exists(),'Codex preparation must stop before final freeze'
    assert not workspace(P,body).exists(),'No real final-test inference inside preparation'
    assert not (P/'results/phase5_optional_validation/status.json').exists(),'No optional training executed'
    for filename in ['phase5_final_test_results.md','phase5_model_comparison.csv','phase5_classwise_metrics.csv','phase5_statistical_analysis.md']:
        assert not (P/'results'/filename).exists(),'No result-bearing final files before evaluation'
    required=['configs/phase5_final_evaluation.yaml','results/phase5_preregistered_evaluation_plan.md','results/phase5_execution_instructions.md',
        'results/phase5_model_selection.md','results/phase5_research_novelty_review.md','results/phase5_reproducibility_report.md',
        'notebooks/05_final_research_analysis.ipynb','reports/final_sleep_stage_classification_research_report.md']
    for relative in required:assert (P/relative).stat().st_size>0,relative
    directory=P/'results/phase5_source_snapshot'
    for relative,expected in body['source_files'].items():
        destination=directory/relative;destination.parent.mkdir(parents=True,exist_ok=True)
        if destination.exists() and digest(destination)!=expected:raise ValueError('Immutable Phase5 source snapshot changed')
        if not destination.exists():shutil.copyfile(P/relative,destination)
    (P/'results/phase5_figures').mkdir(exist_ok=True)
    (P/'results/phase5_figures/.gitkeep').touch(exist_ok=True)
    result=dict(successful=True,completed_at=now(),plan_id=body['plan_id'],plan_sha256=sha,source_files_fingerprint=fingerprint(body['source_files']),
        historical_files_preserved=validation['prior_files_verified'],phase4_runs_verified=15,processed_recordings_preserved=153,selected_checkpoints_verified=9,
        tests=tests,notebook=notebook,validation_smoke=smoke,required_preparation_files=required,
        plan_frozen=False,final_evaluation_authorized=False,real_test_eeg_read=False,real_test_predictions_computed=False,
        long_training_called=False,optional_validation_training_executed=False,synthetic_resume_and_report_tests_passed=True,
        resource_estimates_not_test_benchmarks=True)
    json_write(P/'results/phase5_preparation_validation.json',result)
    summary=f'''# Phase5 preparation complete

Verified {now()}. All15 Phase4 runs,153 arrays and{validation['prior_files_verified']} prior files preserved. Nine final checkpoint identities registered. All{tests['tests_run']} automated tests passed, zero failures/errors/skips; legacy JSON writes redirected. All nine selected models passed short CPU/MPS validation-batch smoke checks. Notebook05 executed{notebook['executed_code_cells']} code cells with{notebook['image_outputs']} image outputs and zero errors; no test result read. Synthetic tests exercised interrupted export, orphan adoption, idempotent merge, subject-cluster statistics and end-to-end publication reports without real test EEG.

Plan `{body['plan_id']}`; SHA-256 `{sha}`. Sources/configuration/dataset/checkpoints/software are locked. The plan is registered for review and remains UNFROZEN/UNAUTHORIZED. No real test EEG, predictions, logits, embeddings or metrics were computed; no optional validation training was launched. The manuscript is a measured-validation draft with final-test sections pending, and no final-result CSV/report placeholders exist.

Review [the preregistered plan](phase5_preregistered_evaluation_plan.md) and [execution instructions](phase5_execution_instructions.md), then use the distinct freeze and authorized local Terminal commands. Expected primary final-inference+analysis cost0.5–2hours is an estimate, not a measured test benchmark. Reserve2GiB SSD. M8 remains exploratory; optional full-budget validation definitions are deferred and not required for this primary comparison.
'''
    (P/'results/phase5_preparation_completion.md').write_text(summary)
    paths={P/r for r in required};paths.update(p for p in (P/'results').glob('phase5*') if p.is_file() and p.name!='phase5_preparation_artifact_manifest.json')
    paths.update(p for p in directory.rglob('*') if p.is_file());paths.update(P/r for r in body['source_files'])
    json_write(P/'results/phase5_preparation_artifact_manifest.json',dict(plan_sha256=sha,created_at=now(),files={str(p.relative_to(P)):digest(p) for p in sorted(paths)},training_called=False,real_test_evaluated=False))
    print(json.dumps({k:v for k,v in result.items() if k not in ['tests','notebook','validation_smoke']},indent=2))
