"""Saved-validation analysis only; no trainer imports or test interface."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .analysis import load_export, analyze_export
from .common import json_write, now, digest
from .metrics import classification_metrics

PROJECT = Path(__file__).resolve().parents[2]


def analyze_validation(export, project=PROJECT):
    manifest, frame, features, root = load_export(export)
    if manifest['synthetic'] or manifest['splits'] != ['validation'] or len(frame) != 30266:
        raise ValueError('A complete real validation export is required')
    # Check against frozen identifiers/labels without loading any frozen-test EEG.
    expected = pd.read_csv(Path(project)/'results/protocol_a_index.csv')
    expected = expected[expected.split == 'validation'].reset_index(drop=True)
    for key in ('epoch_id','subject_id','recording_id','night','original_epoch_index','onset_seconds','split'):
        if not np.array_equal(frame[key].to_numpy(),expected[key].to_numpy()):
            raise ValueError(f'Validation metadata changed: {key}')
    if not np.array_equal(frame.true_label.to_numpy(),expected.label.to_numpy()):
        raise ValueError('Validation labels changed')
    flagged = (frame.raw_rail_contact_samples.to_numpy() > 0) | (frame.filter_boundary_flag.to_numpy() > 0)
    if not np.array_equal(flagged.astype(int),frame.quality_flagged.to_numpy()) or flagged.sum() != 206:
        raise ValueError('Frozen validation quality flags changed')
    existing = pd.read_csv(Path(project)/'results/quality_flagged_epochs.csv')
    existing = existing[(existing.split=='validation') & (existing.wake_trimmed_inclusion==1)]
    if set(existing.epoch_id) != set(frame.loc[flagged,'epoch_id']):
        raise ValueError('Quality epoch IDs disagree with Phase 2.5')
    for row in existing.itertuples():
        actual = frame.loc[frame.epoch_id==row.epoch_id].iloc[0]
        if actual.raw_rail_contact_samples != row.raw_rail_contact_samples or actual.filter_boundary_flag != row.filter_boundary_flag:
            raise ValueError('Quality counts disagree with frozen evidence')
    config = json.loads((root/'reproducibility/configuration.json').read_text())['analysis']
    output = analyze_export(export, tsne_max_samples=config['tsne_max_samples'],
               tsne_perplexity=config['tsne_perplexity'],tsne_iterations=config['tsne_iterations'],seed=config['seed'])
    quality = {}
    for name, included in [('all',np.ones(len(frame),dtype=bool)),('flagged',flagged),('flag_free',~flagged)]:
        subset = frame.loc[included]
        quality[name] = classification_metrics(subset.true_label.to_numpy(),
                             subset[[f'probability_{i}' for i in range(5)]].to_numpy())
        quality[name]['class_counts'] = np.bincount(subset.true_label.to_numpy(),minlength=5).tolist()
    quality.update(descriptive_only=True, no_exclusion_training=True, split='validation',
                   source_checkpoint_sha256=manifest['checkpoint_sha256'],
                   note='Subsets differ in size, subjects and class mix; differences do not establish artifact causality.')
    json_write(output/'quality_subset_metrics.json',quality)
    fig, ax = plt.subplots(figsize=(7,4),constrained_layout=True)
    names = ['all','flagged','flag_free']; x = np.arange(3)
    ax.bar(x-.16,[quality[n]['accuracy'] for n in names],width=.32,label='Accuracy')
    ax.bar(x+.16,[quality[n]['macro_f1'] for n in names],width=.32,label='Macro-F1')
    ax.set(xticks=x,xticklabels=[f'{n}\nn={quality[n]["sample_count"]}' for n in names],ylim=(0,1),
           title='Validation quality subsets — descriptive only',ylabel='Score');ax.legend()
    fig.savefig(output/'quality_subsets.png',dpi=140);plt.close(fig)
    history = json.loads((root/manifest['history_snapshot']).read_text())
    epochs = [r['epoch']+1 for r in history]
    fig, axes = plt.subplots(1,3,figsize=(13,4),constrained_layout=True)
    for ax, train_key, val_key, label in zip(axes[:2],['training_accuracy','training_macro_f1'],
                      ['validation_accuracy','validation_macro_f1'],['Accuracy','Macro-F1']):
        ax.plot(epochs,[r[train_key] for r in history],label='Training (online, dropout active)')
        ax.plot(epochs,[r[val_key] for r in history],label='Validation (eval mode)')
        ax.set(xlabel='Completed epoch',ylabel=label,ylim=(0,1));ax.legend(fontsize=7)
    axes[2].plot(epochs,[r['learning_rate'] for r in history]);axes[2].set(xlabel='Completed epoch',ylabel='Learning rate')
    fig.savefig(output/'accuracy_f1_learning_rate.png',dpi=140);plt.close(fig)
    # Extend the same saved-artifact manifest to include Phase 3 quality/history plots.
    analysis_manifest = json.loads((output/'analysis_manifest.json').read_text())
    analysis_manifest['files'] = {p.name:digest(p) for p in output.iterdir() if p.is_file() and p.name != 'analysis_manifest.json'}
    json_write(output/'analysis_manifest.json',analysis_manifest)
    reference = dict(export=str(Path(export).relative_to(project)),analysis=str(output.relative_to(project)),
                     sample_count=len(frame),subjects=frame.subject_id.nunique(),flagged_epochs=int(flagged.sum()),
                     embedding_shape=list(features.shape),synthetic=False,frozen_test_evaluated=False,
                     generated_at=now(),checkpoint_sha256=manifest['checkpoint_sha256'])
    json_write(Path(project)/'results/phase3_analysis_summary.json',reference)
    return reference


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--export',type=Path,required=True)
    parser.add_argument('--project',type=Path,default=PROJECT)
    args=parser.parse_args()
    print(json.dumps(analyze_validation(args.export.resolve(),args.project.resolve()),indent=2))


if __name__ == '__main__':
    main()
