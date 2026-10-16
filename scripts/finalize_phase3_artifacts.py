"""Publish small reviewable copies and hash the completed Phase 3 artifacts."""

import json
from pathlib import Path
import shutil
import sys

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT/'src'))
from sleep_research.common import digest,json_write,now


if __name__ == '__main__':
    results=PROJECT/'results'
    analysis_reference=json.loads((results/'phase3_analysis_summary.json').read_text())
    export_reference=json.loads((results/'phase3_validation_export.json').read_text())
    run_reference=json.loads((results/'phase3_current_run.json').read_text())
    run=PROJECT/run_reference['experiment_path']
    analysis=PROJECT/analysis_reference['analysis']
    figures=results/'phase3_figures';figures.mkdir(exist_ok=True)
    for folder in (analysis,PROJECT/export_reference['explanations']):
        for file in folder.glob('*.png'):
            shutil.copyfile(file,figures/file.name)
    for name,target in [('metrics.json','phase3_validation_metrics.json'),
                        ('subject_metrics.json','phase3_subject_metrics.json'),
                        ('quality_subset_metrics.json','phase3_quality_subset_metrics.json'),
                        ('tsne_parameters.json','phase3_tsne_parameters.json')]:
        shutil.copyfile(analysis/name,results/target)
    shutil.copyfile(run/'reproducibility/architecture.json',results/'phase3_architecture.json')
    shutil.copyfile(run/'reproducibility/training_class_weights.json',results/'phase3_training_class_weights.json')
    json_write(results/'phase3_figures/provenance.json',dict(checkpoint_sha256=analysis_reference['checkpoint_sha256'],
               source_analysis=analysis_reference['analysis'],source_explanations=export_reference['explanations'],
               synthetic=False,split='validation',files={f.name:digest(f) for f in figures.glob('*.png')}))
    paths=[]
    smoke_reference=json.loads((results/'phase3_smoke_validation.json').read_text())
    smoke_run=PROJECT/'experiments/cnn'/smoke_reference['run_id']
    for folder in [PROJECT/'src/models',PROJECT/'src/sleep_research',run,smoke_run,figures]:
        paths += [p for p in folder.rglob('*') if p.is_file() and '__pycache__' not in p.parts and not p.name.startswith('.')]
    paths += list(results.glob('phase3*.*'))
    paths += list((PROJECT/'scripts').glob('*phase3*.py'))
    paths += [PROJECT/'tests/test_phase3.py',PROJECT/'configs/cnn1d_baseline.yaml',
              PROJECT/'notebooks/03_cnn_training_analysis.ipynb',PROJECT/'README.md',PROJECT/'AGENTS.md',PROJECT/'.gitignore']
    paths=sorted(set(p for p in paths if p.name != 'phase3_artifact_manifest.json'))
    json_write(results/'phase3_artifact_manifest.json',dict(created_at=now(),synthetic=False,
               run_id=run_reference['run_id'],checkpoint_sha256=analysis_reference['checkpoint_sha256'],
               files={str(p.relative_to(PROJECT)):digest(p) for p in paths},
               note='Large experiment weights/predictions/arrays stay ignored in Git; small result copies and executed notebook are reviewable.'))
    print('Phase 3 manifest:',len(paths),'files; reviewable figures:',len(list(figures.glob('*.png'))))
