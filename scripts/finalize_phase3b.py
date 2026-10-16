"""Validate measured pilot artifacts and publish small reviewable evidence."""

import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT/'src'))
from sleep_research.analysis import load_export
from sleep_research.checkpoints import CheckpointManager
from sleep_research.common import digest,fingerprint,json_write,now
from sleep_research.experiments import Experiment
from sleep_research.metrics import classification_metrics


if __name__=='__main__':
    results=PROJECT/'results'
    measured=json.loads((results/'phase3b_pilot_runs.json').read_text())
    summary=json.loads((results/'phase3b_analysis_summary.json').read_text())
    table=pd.read_csv(results/'phase3b_validation_comparison.csv')
    subsets=json.loads((results/'phase3b_pilot_subsets.json').read_text())
    assert measured['status']=='complete' and len(measured['runs'])==len(table)==8
    assert measured['full_runs_launched']==0 and set(table.seed)=={42}
    assert (table.completed_epochs==3).all() and (table.training_epochs==8192).all() and (table.validation_epochs==4096).all()
    canonical=pd.read_csv(results/'protocol_a_index.csv').set_index('epoch_id')
    expected=canonical.loc[subsets['epoch_ids']['validation']]
    assert set(expected.split)=={'validation'} and expected.subject_id.nunique()==12
    train=canonical.loc[subsets['epoch_ids']['train']]
    assert set(train.split)=={'train'} and train.subject_id.nunique()==54
    assert not set(train.subject_id)&set(expected.subject_id)
    quality=pd.read_csv(results/'quality_flagged_epochs.csv').set_index('epoch_id')
    full_counts=json.loads((results/'phase3_training_class_weights.json').read_text())['class_counts']
    baseline=Experiment(PROJECT,'cnn',run_id='cnn1d-a-seed42-20261014-1800')
    paths=[];flagged=None;artifact_bytes=0
    for result in measured['runs']:
        root=PROJECT/'experiments/cnn'/result['run_id'];experiment=Experiment(PROJECT,'cnn',run_id=result['run_id'])
        cfg=experiment.metadata['configuration'];contract=experiment.metadata['dataset_contract']
        assert cfg['architecture']==baseline.metadata['configuration']['architecture']
        assert cfg['training_class_counts']==full_counts
        assert contract['pilot_subsets_sha256']==fingerprint(subsets)==measured['subset_sha256']
        for key in ['preprocessing','normalization','subject_split','epoch_manifest','protocol','class_mapping']:
            assert contract[key]==baseline.metadata['dataset_contract'][key]
        manager=CheckpointManager(root/'reproducibility/checkpoints',experiment.metadata['resume_fingerprints'],result['run_id'])
        roles=manager.discover();latest=manager.read(roles['latest']);best=manager.read(roles['best'])
        assert latest['completed_epoch']==2 and latest['global_step']==1536 and latest['scheduler'] is None
        assert [r['epoch'] for r in latest['history']]==[0,1,2] and 'criterion' in latest and latest['rng']['torch_mps'] is not None
        np.testing.assert_array_equal(latest['criterion']['alpha'].cpu().numpy(),np.asarray(cfg['loss_alpha'],dtype=np.float32))
        manifest,frame,features,_=load_export(PROJECT/result['export'])
        assert manifest['checkpoint_sha256']==digest(roles['best']) and frame.epoch_id.tolist()==subsets['epoch_ids']['validation']
        assert manifest['completed_epoch']==best['completed_epoch'] and features.shape==(4096,64)
        assert np.load(root/manifest['feature_files']['blocks.2.2'],mmap_mode='r').shape==(4096,64)
        for key in ['subject_id','recording_id','night','original_epoch_index','onset_seconds','split']:
            np.testing.assert_array_equal(frame[key],expected[key])
        np.testing.assert_array_equal(frame.true_label,expected.label)
        for row in frame.itertuples():
            saved=quality.loc[row.epoch_id] if row.epoch_id in quality.index else None
            rail=0 if saved is None else int(saved.raw_rail_contact_samples)
            edge=0 if saved is None else int(saved.filter_boundary_flag)
            assert (row.raw_rail_contact_samples,row.filter_boundary_flag,row.quality_flagged)==(rail,edge,int(rail>0 or edge>0))
        if flagged is None:flagged=int(frame.quality_flagged.sum())
        assert int(frame.quality_flagged.sum())==flagged
        metrics=classification_metrics(frame.true_label.to_numpy(),frame[[f'probability_{i}' for i in range(5)]].to_numpy())
        row=table[table.run_id==result['run_id']].iloc[0]
        np.testing.assert_allclose(metrics['macro_f1'],row.macro_f1,atol=1e-12,rtol=0)
        np.testing.assert_allclose(metrics['macro_f1'],latest['history'][best['completed_epoch']]['validation_macro_f1'],atol=1e-12,rtol=0)
        analysis=PROJECT/summary['analyses'][result['run_id']]
        assert not list(analysis.glob('hypnogram_*.png'))
        assert len(pd.read_csv(analysis/'pca_coordinates.csv'))==4096
        tsne=json.loads((analysis/'tsne_parameters.json').read_text())
        selected=np.sort(np.random.default_rng(42).choice(4096,200,replace=False))
        assert tsne['epoch_ids']==frame.iloc[selected].epoch_id.tolist()
        destination=results/'phase3b_figures'/(result['candidate_id']+'_'+result['loss']['name'])
        destination.mkdir(exist_ok=True)
        for name in ['confusion_matrix.png','classwise_confusion.png','objective_learning_curves.png','pca.png','tsne.png']:
            shutil.copyfile(analysis/name,destination/name)
        json_write(destination/'provenance.json',dict(checkpoint_sha256=manifest['checkpoint_sha256'],
                   source_analysis=str(analysis.relative_to(PROJECT)),files={p.name:digest(p) for p in destination.glob('*.png')}))
        run_files=[p for p in root.rglob('*') if p.is_file() and not p.name.startswith('.') and '__pycache__' not in p.parts]
        paths+=run_files;artifact_bytes+=sum(p.stat().st_size for p in run_files)
    assert len(table[table.stage=='optimizer_pilot'])==6 and len(table[table.stage=='loss_pilot'])==2
    selected=json.loads((results/'phase3b_optimizer_selection.json').read_text())
    assert table[table.stage=='optimizer_pilot'].sort_values('macro_f1',ascending=False,kind='stable').iloc[0].run_id==selected['selected_run_id']
    assert set(table[table.stage=='loss_pilot'].candidate_id)=={selected['selected_candidate']['id']}
    proposals=json.loads((results/'phase3b_full_run_proposals.json').read_text())
    assert proposals['full_runs_launched']==0 and all(not p['measured'] and not p['auto_launch'] and p['seeds']==[42,43,44] for p in proposals['configurations'])
    assert len(pd.read_csv(results/'phase3b_full_validation_comparison.csv'))==0
    assert len(pd.read_csv(results/'phase3b_repeated_seed_summary.csv'))==0
    tests=json.loads((results/'phase3b_test_results.json').read_text());notebook=json.loads((results/'phase3b_notebook_execution.json').read_text())
    preserve=json.loads((results/'phase3b_preservation_validation.json').read_text())
    assert tests['successful'] and tests['tests_run']==137 and tests['skipped']==0
    assert notebook['successful'] and notebook['image_outputs']==7 and notebook['error_outputs']==0
    assert preserve['successful'] and preserve['m1_artifact_files_unchanged']==124 and preserve['previous_arrays_unchanged']==153
    validation=dict(successful=True,validated_at=now(),pilot_trials=8,completed_epochs_per_trial=3,
                    seed=42,training_count=8192,validation_count=4096,training_subjects=54,validation_subjects=12,
                    retained_flagged_validation_examples=flagged,full_runs_launched=0,
                    pilot_run_artifact_bytes=artifact_bytes,pilot_training_attempt_seconds=float(table.training_seconds.sum()),
                    measured_class_weights_fit_split='train',m1_preserved=True,test_evaluated=False,
                    tests_run=tests['tests_run'],notebook_executed_code_cells=7,notebook_image_outputs=7)
    json_write(results/'phase3b_final_validation.json',validation)
    # Add measured validation evidence without changing any earlier phase's report.
    report=results/'phase3b_optimizer_comparison.md'
    report.write_text(report.read_text().split('\n## Final validation evidence')[0]+f'''

## Final validation evidence

All{tests['tests_run']} automated tests passed (previous118 plus{tests['tests_run']-118} new), with zero failures/errors/skips.
The executed notebook contains7 code cells,7 PNG outputs and zero errors. Legacy test code
and original result logs remain unchanged; two legacy JSON writes were redirected to SSD
temporary files. The model source,153 arrays, normalization/splits and all124 M1 run files
match the preservation snapshot. Two shared modules gained backward-compatible extensions.

Measured pilot training/validation attempt time: {table.training_seconds.sum():.2f}s total,
excluding setup/smoke/export/analysis/notebook. Retained pilot run artifacts:
{artifact_bytes/2**20:.2f}MiB. Reported pilot RSS is a cumulative process high-water value,
not an independent isolated-memory benchmark for each trial. All4096 validation rows per
trial retain metadata and the same{flagged} flagged examples. Each best/latest checkpoint,
loss state, exported metric/embedding, class confusion and seeded projection was checked.
`phase3b_final_validation.json` records these checks. No full or multi-seed results exist.
''')
    smoke=json.loads((results/'phase3b_smoke_validation.json').read_text())
    paths += [p for p in (PROJECT/'experiments/cnn'/smoke['run_id']).rglob('*') if p.is_file() and not p.name.startswith('.')]
    paths += [p for p in (PROJECT/'src/sleep_research').glob('*.py')]
    paths += list((PROJECT/'scripts').glob('*phase3b*.py'))+list(results.glob('phase3b*.*'))
    paths += [p for p in (results/'phase3b_figures').rglob('*') if p.is_file()]
    paths += [PROJECT/'tests/test_phase3b.py',PROJECT/'configs/phase3b_optimizer_search.yaml',PROJECT/'notebooks/03b_optimizer_analysis.ipynb',
              PROJECT/'src/models/cnn1d.py',PROJECT/'README.md',PROJECT/'AGENTS.md',PROJECT/'.gitignore']
    json_write(results/'phase3b_artifact_manifest.json',dict(generated_at=now(),pilot_only=True,full_runs_launched=0,
                files={str(p.relative_to(PROJECT)):digest(p) for p in sorted(set(paths)) if p.name!='phase3b_artifact_manifest.json'}))
    print(json.dumps(validation,indent=2))
