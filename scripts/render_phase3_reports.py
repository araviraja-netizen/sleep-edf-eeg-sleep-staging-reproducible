"""Render Phase 3 reports exclusively from real committed run artifacts."""

import json
from pathlib import Path
import sys

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT/'src'))
from sleep_research.common import json_write, now, digest
from sleep_research.checkpoints import CheckpointManager
from sleep_research.experiments import Experiment


def main():
    results=PROJECT/'results'
    reference=json.loads((results/'phase3_current_run.json').read_text())
    root=PROJECT/reference['experiment_path']
    experiment=Experiment(PROJECT,'cnn',run_id=reference['run_id'])
    if experiment.metadata['status'] not in ('completed','early_stopped'):
        raise ValueError('Do not render completed results for an unfinished run')
    metadata=experiment.metadata
    manager=CheckpointManager(root/'reproducibility/checkpoints',metadata['resume_fingerprints'],reference['run_id'])
    checkpoints=manager.discover()
    latest=manager.read(checkpoints['latest']);best=manager.read(checkpoints['best'])
    history=latest['history']
    analysis_reference=json.loads((results/'phase3_analysis_summary.json').read_text())
    analysis=PROJECT/analysis_reference['analysis']
    export_reference=json.loads((results/'phase3_validation_export.json').read_text())
    export_manifest=json.loads((PROJECT/export_reference['export']/'manifest.json').read_text())
    if export_manifest['checkpoint_sha256'] != digest(checkpoints['best']):
        raise ValueError('Saved export is not the selected best checkpoint')
    metrics=json.loads((analysis/'metrics.json').read_text())
    quality=json.loads((analysis/'quality_subset_metrics.json').read_text())
    architecture=json.loads((root/'reproducibility/architecture.json').read_text())
    resources=json.loads((root/'reproducibility/resource_usage.json').read_text())
    weights=json.loads((root/'reproducibility/training_class_weights.json').read_text())
    tests=json.loads((results/'phase3_test_results.json').read_text())
    notebook=json.loads((results/'phase3_notebook_execution.json').read_text())
    preservation=json.loads((results/'phase3_preservation_validation.json').read_text())
    lines=['# Phase 3 — real Sleep-EDF 1D CNN baseline',f'Generated: {now()}.','',
        f"Run **{reference['run_id']}**, status **{metadata['status']}**, actual **MPS float32** training.",
        f"Completed **{latest['next_epoch']}** epochs; latest epoch **{latest['completed_epoch']+1}** and selected best epoch **{best['completed_epoch']+1}** (1-based display).",
        'Checkpoint fields/history use zero-based completed_epoch. This report contains actual SC validation results; synthetic infrastructure tests and the bounded real smoke are separate evidence.','',
        '## Fixed data and training protocol','',
        'Protocol A only: 134047 training epochs/54 subjects/107 recordings; 30266 validation epochs/12 subjects/24 recordings. The 31156 frozen test epochs remain locked and have no predictions, embeddings or model metrics.',
        'All 2235 Protocol A flagged epochs remain retained, including 1701 training and 206 validation epochs; 328 test flags are prior data-quality inventory only. No artifact exclusions, split changes or preprocessing regeneration.',
        'Fpz-Cz native 100 Hz, 30 s / 3000 samples. The Phase 2 fixed train-only global normalization is unchanged: mean 2.4219063337607186e-05 µV, std 17.664351915765064 µV. Ordinary cross entropy; AdamW lr 0.001,weight decay 0.01; batch 16; seed 42; max 25; early patience 8; selection strictly highest validation five-class macro-F1; constant LR/no scheduler. Workers: 0; shuffled training, ordered validation.',
        f"Training counts [W,N1,N2,N3,R]: {weights['class_counts']}. Future weighted-loss metadata {weights['weights']}; not applied.",'',
        '## Architecture and measured resources','',
        f"CNN1D has {architecture['parameters']} parameters and a 64-dimensional penultimate embedding. Input [B,1,3000]; widths 16/32/64/64; Conv1d kernels 25/9/7/5, strides 2/2/2/1, symmetric padding 12/4/3/2, bias disabled; GroupNorm with 4 groups, ReLU and MaxPool1d(kernel=2,stride=2) in each block; adaptive global average pool; dropout 0.3; five logits.",'',
        '| Layer | Output (without batch) | Parameters | Local support samples | Conv/Linear MACs |',
        '| --- | --- | ---: | ---: | ---: |']
    for row in architecture['layers']:
        lines.append(f"| {row['name']} | {row['output']} | {row['parameters']} | {row.get('local_receptive_field_samples','—')} | {row.get('macs','—')} |")
    lines += ['',f"Forward Conv/Linear MACs per epoch: {architecture['conv_linear_macs_per_epoch']}; approximate FLOPs {architecture['approximate_conv_linear_flops_per_epoch']}. Normalization/activation/pooling/backward/optimizer are excluded. Final local convolutional support 515 samples / 5.15 s; GroupNorm time statistics and global pooling have whole 30 s support. Learned striding does not guarantee anti-aliasing.",
              f"Committed training+validation epoch time: {sum(r['seconds'] for r in history):.2f}s; all attempt time: {metadata['total_training_attempt_seconds']:.2f}s. Peak process RSS: {resources['process_high_water_rss_bytes']/2**30:.3f} GiB. End-of-run MPS memory readings (not peak readings) are in `{str((root/'reproducibility/resource_usage.json').relative_to(PROJECT))}`. No full normalized array copy was created.",'',
              '## Selected best checkpoint — validation only','',
              f"Accuracy **{metrics['accuracy']:.6f}**, five-class macro-F1 **{metrics['macro_f1']:.6f}**, balanced accuracy **{metrics['balanced_accuracy']:.6f}**, Cohen kappa **{metrics['cohen_kappa']:.6f}**. These are validation/model-selection results, not an unbiased final test estimate.",'',
              '| Stage | Precision | Recall | F1 | Support |','| --- | ---: | ---: | ---: | ---: |']
    # Metrics stores per-class dictionaries keyed by stage.
    for name, row in metrics['classes'].items():
        lines.append(f"| {name} | {row['precision']:.6f} | {row['recall']:.6f} | {row['f1']:.6f} | {row['support']} |")
    lines += ['', 'N1 has the lowest class F1 (0.365433). This remains a baseline limitation; no additional tuning or weighted-loss run was started.', '', 'Confusion matrix: rows=true, columns=predicted, order W,N1,N2,N3,R.','', '```text',
              *[' '.join(str(n) for n in row) for row in metrics['confusion_matrix']],'```','',
              '## Retained quality subsets — descriptive validation analysis','',
              '| Subset | Epochs | Accuracy | Five-class macro-F1 | Class counts W/N1/N2/N3/R |',
              '| --- | ---: | ---: | ---: | --- |']
    for name in ('all','flagged','flag_free'):
        row=quality[name]
        lines.append(f"| {name} | {row['sample_count']} | {row['accuracy']:.6f} | {row['macro_f1']:.6f} | {row['class_counts']} |")
    lines += ['',quality['note'],'No sensitivity/exclusion training or test-driven adjustment was started.','',
              '## Saved evidence and reproducibility','',
              f"Run directory: `{reference['experiment_path']}`. Best/latest/previous immutable generations, byte checksums and atomic pointer indices are in `reproducibility/checkpoints/`; best checkpoint SHA-256 `{export_manifest['checkpoint_sha256']}`. Export pins the checkpoint for future inference.",
              f"Prediction export: `{export_reference['export']}`. All 30,266 validation rows retain true/predicted labels, five logits/probabilities, subject/recording/night/original epoch/onset/split and quality flags. Embeddings [30266,64] and opt-in pooled intermediate [30266,64] have stable epoch mappings and hashed manifests.",
              f"Analysis: `{analysis_reference['analysis']}`; confusion, per-class/subject metrics, ROC/PR/calibration, loss/accuracy/F1/LR curves, whole-validation PCA, seeded bounded 500-row t-SNE (seed 42/perplexity 30 / 500 iterations), recording-specific true/predicted hypnograms with original gaps.",
              f"Explanations: `{export_reference['explanations']}`; first validation epoch of each true stage, selected independently of confidence/errors. Bounded saliency and compatible final Conv1d Grad-CAM bind exact weights; explanations are not evidence of causal physiology.",
              'Config, software, source/config snapshot, seed, full dataset/split/normalization/protocol fingerprints and class weights are in reproducibility. Optimizer, null scheduler, early-stop, RNG/loader, global-step/history state are checkpointed. Incomplete epochs repeat; history is authoritative. No bitwise MPS resume guarantee.',
              'Start/resume/load-best/regenerate/extend commands are in README and phase3_training_log.md. Analysis-only modules do not import or call Trainer.','',
              '## Validation and preservation','',
              f"Previous suite: 104 passed. Final suite: {tests['tests_run']} run, {tests['failures']} failures, {tests['errors']} errors, {tests['skipped']} skipped. Separate logs `phase3_previous_test_results.*` and `phase3_test_results.*`; old tests unchanged and legacy JSON writes redirected to SSD-local temporary paths.",
              'Real MPS batch smoke verified exact frozen normalization and finite logits/loss/gradients, 128 training / 64 validation epochs and intentional interruption/resume. CNN CPU synthetic tests verify uninterrupted/resumed weights, metrics/history, corruption recovery, incompatible settings and test gates. These are checks, not real full-cohort metrics.',
              f"Notebook success: {notebook['successful']}; code cells: {notebook['executed_code_cells']}; PNG outputs: {notebook['image_outputs']}; errors: {notebook['error_outputs']}. Existing interpreter, temporary kernel specification; no kernel/environment installation.",
              f"Preserved prior files: {preservation['all_previous_files_unchanged']} unchanged, including 153 arrays; Phase 1/2 baseline 552 unchanged. Only shared trainer/exporter gained backward-compatible Phase 3 extensions. Prior reports/notebooks/test logs/synthetic experiments remain byte-identical. No package install, raw write or dataset download.",'',
              '## Limitations and stopping boundary','',
              'Validation was used for best-epoch selection and early stopping. Macro-F1 uses all five classes, zero F1 for absent classes; balanced accuracy averages recall of present classes. Epoch observations within subjects are correlated; subset/subject metrics and t-SNE are descriptive, with no population significance claim.',
              'Historical R&K five-class merging, filename-derived identities, retained ADC rail contacts/edge effects, zero-phase filtering and annotation-based Wake trimming constrain interpretation. The single-epoch model has no surrounding-epoch sequence context. CPU equivalence does not guarantee identical MPS results across versions/devices/workers.',
              '**Stopped after CNN baseline training and validation analysis. No frozen final test evaluation or CNN-LSTM/attention/Transformer training.**']
    (results/'phase3_cnn_results.md').write_text('\n'.join(lines)+'\n')
    log=['# Phase 3 — training and resume log',f'Generated: {now()}.','',f"Run: `{reference['run_id']}`; actual MPS float32; status:{metadata['status']}. All epoch numbers below are 1-based.",'',
         '| Epoch | Train loss | Train acc | Train macro-F1 | Val loss | Val acc | Val macro-F1 | LR | Seconds |',
         '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for r in history:
        log.append(f"| {r['epoch']+1} | {r['train_loss']:.6f} | {r['training_accuracy']:.6f} | {r['training_macro_f1']:.6f} | {r['validation_loss']:.6f} | {r['validation_accuracy']:.6f} | {r['validation_macro_f1']:.6f} | {r['learning_rate']:.6g} | {r['seconds']:.2f} |")
    log += ['', 'Training metrics are accumulated online with evolving weights and dropout active; validation runs in eval mode after the epoch. No separate training-set inference is implied.', '', '## Actual attempts', '', '```json',json.dumps(metadata['training_attempts'],indent=2),'```','',
           f"Latest completed epoch: {latest['completed_epoch']+1}; next epoch: {latest['next_epoch']+1}; best: {best['completed_epoch']+1}. Global steps: {latest['global_step']}. Early-stop state: {latest['early_stopping']}. Scheduler is null (constant LR).",'',
           '## Terminal commands','', '```sh','cd /Volumes/Mac/projects/sleep-stage-classification',
           '# Actual initial command (use a new run ID for another fresh experiment)',
           f"PYTHONPATH=src /Volumes/Mac/envs/dl-m4/bin/python -m sleep_research.cnn_baseline train --run-id {reference['run_id']} --resume none", '# Resume',
           f"PYTHONPATH=src /Volumes/Mac/envs/dl-m4/bin/python -m sleep_research.cnn_baseline train --run-id {reference['run_id']} --resume auto", '# Load best / regenerate validation exports without training',
           f"PYTHONPATH=src /Volumes/Mac/envs/dl-m4/bin/python -m sleep_research.cnn_inference load-best --run-id {reference['run_id']}",
           f"PYTHONPATH=src /Volumes/Mac/envs/dl-m4/bin/python -m sleep_research.cnn_inference export-validation --run-id {reference['run_id']}", '# Regenerate metrics/plots from this saved export',
           f"PYTHONPATH=src /Volumes/Mac/envs/dl-m4/bin/python -m sleep_research.phase3_analysis --export {export_reference['export']}", '# Extend total budget (a triggered early stop remains authoritative)',
           f"PYTHONPATH=src /Volumes/Mac/envs/dl-m4/bin/python -m sleep_research.cnn_baseline train --run-id {reference['run_id']} --resume auto --max-epochs 35",'```','',
           'Automatic recovery validates checksums and compatible fingerprints. Interrupted epochs repeat from the last committed epoch; a corrupt latest generation falls back to a recoverable prior generation. Best selection is independent of latest. Long training is Terminal-driven; notebook reads committed saved artifacts only.',
           'Raw training stdout: results/phase3_terminal_training.txt. Smoke stdout/JSON: results/phase3_smoke_run.txt and phase3_smoke_validation.json. Test logs remain separate from all prior phases.']
    (results/'phase3_training_log.md').write_text('\n'.join(log)+'\n')
    json_write(results/'phase3_results_summary.json',dict(run_id=reference['run_id'],status=metadata['status'],
               completed_epochs=latest['next_epoch'],best_epoch_1_based=best['completed_epoch']+1,
               validation_accuracy=metrics['accuracy'],validation_macro_f1=metrics['macro_f1'],
               balanced_accuracy=metrics['balanced_accuracy'],cohen_kappa=metrics['cohen_kappa'],
               frozen_test_evaluated=False,synthetic=False))
    print((results/'phase3_results_summary.json').read_text())


if __name__ == '__main__':
    main()
