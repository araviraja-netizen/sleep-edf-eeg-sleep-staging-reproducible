"""Evidence-backed infrastructure/quality reports; preserves all prior phase files."""

import csv
import json
from pathlib import Path
import sys

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT/'src'))
from sleep_research.checkpoints import CheckpointManager
from sleep_research.common import atomic_file, digest, json_write, now, preserve_previous

RESULTS=PROJECT/'results'


def load(name):
    return json.loads((RESULTS/name).read_text())


def table(headers,rows):
    lines=['| '+' | '.join(headers)+' |','| '+' | '.join('---' for _ in headers)+' |']
    lines.extend('| '+' | '.join(str(value) for value in row)+' |' for row in rows)
    return '\n'.join(lines)


def csv_rows(name):
    with (RESULTS/name).open(newline='') as handle:
        return list(csv.DictReader(handle))


def main():
    baseline=preserve_previous(PROJECT)
    quality=load('artifact_quality_summary.json')
    tests=load('phase2_5_test_results.json')
    notebook=load('phase2_5_notebook_execution.json')
    demo=load('phase2_5_notebook_demo.json')
    split_rows=csv_rows('quality_by_split.csv')
    stage_rows=csv_rows('quality_by_stage.csv')
    subject_rows=csv_rows('quality_by_subject.csv')
    recording_rows=csv_rows('quality_by_recording.csv')
    training_subjects={row['subject_id'] for row in csv_rows('subject_splits.csv') if row['split']=='train'}
    if not tests['successful'] or tests['failures'] or tests['errors'] or tests['skipped'] or not notebook['successful']:
        raise RuntimeError('Infrastructure report requires successful tests and notebook execution')
    flags=quality['protocols']
    subjects_affected={protocol:sum(r['protocol']==protocol and int(r['any_flag'])>0 for r in subject_rows) for protocol in ('A','B')}
    recordings_affected={protocol:sum(r['protocol']==protocol and int(r['any_flag'])>0 for r in recording_rows) for protocol in ('A','B')}
    quality_lines=['# Phase 2.5 — artifact quality review','',f'Generated: {now()} (Asia/Kolkata).','',
        'The frozen SC data, labels, inclusion flags, normalization and subject splits remain unchanged.',
        'This is a descriptive review of the existing numeric flags, not model evaluation or a new exclusion policy.',
        'Waveform selection uses training subjects only; split-level test counts describe data quality, not test performance.','',
        '## Flag totals and overlap','',
        table(['Protocol','Usable epochs','ADC rail','Recording edge','Both flags','Distinct flagged','Flag-free','Subjects with flags','Recordings with flags'],
              [[p,flags[p]['usable_epochs'],flags[p]['rail_epochs'],flags[p]['edge_epochs'],flags[p]['both_flags'],
                flags[p]['any_flag'],flags[p]['flag_free'],subjects_affected[p],recordings_affected[p]] for p in ('A','B')]),'',
        'The reported **12,020 rail-contact epochs and 301 edge flags** refer to full-valid Protocol B.',
        '**16 epochs overlap**, giving **12,305 distinct flagged epochs**. Protocol A retains **2,226 rail epochs**,',
        '**9 edge epochs**, no overlap, and **2,235 distinct flagged epochs**. No flag was converted into an exclusion.','',
        'Rail contacts mean raw calibrated samples numerically equal an EDF physical minimum/maximum (Phase 2 tolerance',
        '1e-8 µV). This can indicate saturation/contact problems but is not by itself a clinical artifact diagnosis.',
        'Edge flags denote epochs touching the first/last 30 seconds of the recording; they are a caution window,',
        'not proof that filter transients last exactly 30 seconds. Stored filtered values need not touch raw ADC rails.','',
        f"Valid supervised epochs contain {flags['B']['rail_contact_samples']:,} rail-contact samples, versus 133,917",
        'over all complete raw epochs in Phase 2. The difference of 154 samples is in annotation-excluded epochs.','',
        '## Distribution by split','',
        table(['Protocol','Split','Usable','Rail','Edge','Both','Distinct flagged','Rail %','Edge %'],
              [[r[k] for k in ('protocol','split','usable_epochs','rail_epochs','edge_epochs','both_flags','any_flag')]+
               [f"{float(r['rail_percent']):.4f}",f"{float(r['edge_percent']):.4f}"] for r in split_rows]),'',
        '## Distribution by true sleep-stage label','',
        table(['Protocol','Stage','Usable','Rail','Edge','Both','Rail %'],
              [[r[k] for k in ('protocol','stage','usable_epochs','rail_epochs','edge_epochs','both_flags')]+[f"{float(r['rail_percent']):.4f}"] for r in stage_rows]),'',
        'Most full-recording rail-contact epochs are Wake: **11,641 / 12,020**. Every non-Wake rail-contact epoch',
        'remains in Protocol A; exterior Wake trimming accounts for the reduced rail burden. This describes the',
        'existing label-based protocol, not a newly optimized artifact rule.','',
        '## Distribution by subject and recording','',
        'Complete tables include unflagged groups and both protocols in `quality_by_subject.csv` and',
        '`quality_by_recording.csv`. Split/stage combinations are in `quality_by_split_stage.csv`.',
        '`quality_flagged_epochs.csv` retains IDs, source index/onset/row, true label, split and flag overlap.',
        'The highest-count training recordings are listed for review only; rankings did not select model exclusions.','',
        table(['Training recording (B)','Subject','Rail epochs','Edge epochs','Both','Distinct flagged'],
              [[r['recording'],r['recording'][:5],r['rail_epochs'],r['edge_epochs'],r['both_flags'],r['any_flag']]
               for r in sorted([r for r in recording_rows if r['protocol']=='B' and r['recording'][:5] in training_subjects],
                               key=lambda r:int(r['rail_epochs']),reverse=True)[:12]]),'',
        '## Representative training-subject waveforms','',
        'Selection: the training epoch with the largest existing contact count for each of the five labels, plus',
        'the first available training edge example and first overlap example. No validation/test waveform was selected.',
        'Each figure overlays raw calibrated and frozen continuous-filtered EEG on its original 30-second interval;',
        'red dashed lines mark EDF physical rails. These extreme descriptive examples are not prevalence estimates.','',
        table(['Example','Epoch ID','Subject','Stage','Rail-contact samples','Edge'],
              [[name,row['epoch_id'],row['subject_id'],('Wake','N1','N2','N3','REM')[int(row['label'])],
                row['raw_rail_contact_samples'],row['filter_boundary_flag']] for name,row in quality['training_examples'].items()]),'',
        'Figures: `results/quality_review_figures/`; exact paths: `quality_review_figures.json`.',
        'Contact-density bands are descriptive only: '+str(quality['rail_contact_density_epoch_counts_B'])+'.','',
        '## Proposed sensitivity analyses — not applied','',
        '1. Keep the primary A baseline unchanged. For a later trained model, report validation metrics separately',
        '   for flag-free, rail-only, edge-only and overlap groups, alongside whole-validation performance.',
        '2. Pre-register separate training runs excluding rail-flagged training epochs, edge-flagged training epochs,',
        '   or their union using index masks over the same arrays. Preserve labels, frozen subjects and test membership.',
        '   Keep the existing training normalization fixed to isolate eligibility effects, or pre-register a separately',
        '   refitted training-only normalization variant; do not silently change the primary parameters.',
        '3. Compare whole-validation and flagged-subset behavior without dropping difficult validation samples from',
        '   the primary headline result. Use the recorded binary flags rather than tuning an ADC-density threshold.',
        '4. Freeze all sensitivity choices using training/validation evidence before the explicitly authorized final',
        '   evaluation. Apply that fixed plan to the frozen test cohort once; never optimize an exclusion rule using',
        '   test scores. Current test flag counts do not grant permission to compute model test performance.','',
        'No sensitivity run, relabeling, new normalization, exclusion mask applied to the baseline, or real-data training has occurred.','']
    with atomic_file(RESULTS/'artifact_quality_review.md') as handle:
        handle.write('\n'.join(quality_lines))
    experiments=[]
    # Pin already published exports from the earlier Terminal/notebook executions as well.
    for metadata_path in sorted((PROJECT/'experiments').glob('*/*/reproducibility/experiment.json')):
        metadata=json.loads(metadata_path.read_text());run=metadata_path.parents[1]
        if not metadata['synthetic']:
            raise RuntimeError('Phase 2.5 report refuses an unexpected real-data training run')
        manager=CheckpointManager(run/'reproducibility/checkpoints',metadata['resume_fingerprints'],metadata['experiment_id'])
        state=manager.recover('auto')
        exports=list((run/'performance/exports').glob('*/manifest.json'))
        for export in exports:
            export_manifest=json.loads(export.read_text())
            manager.pin(run/export_manifest['checkpoint'],f"inference-export:{export_manifest['export_id']}")
        experiments.append({'run_id':metadata['experiment_id'],'family':metadata['family'],'synthetic':True,
                            'completed_epoch':state['completed_epoch'],'next_epoch':state['next_epoch'],'global_step':state['global_step'],
                            'history_epochs':[r['epoch'] for r in state['history']],'retained_checkpoints':len(list(manager.directory.glob('*.pt'))),
                            'exports':len(exports),'status':metadata['status'],'seconds':metadata.get('total_training_attempt_seconds',0),
                            'software':metadata['software'],'run_path':str(run.relative_to(PROJECT))})
    experiment_bytes=sum(path.stat().st_size for path in (PROJECT/'experiments').rglob('*') if path.is_file())
    report=['# Phase 2.5 — resumable training and research experiment infrastructure','',f'Generated: {now()} (Asia/Kolkata).','',
        '## Completion and boundaries','',
        '**Infrastructure built and validated. Real Sleep-EDF CNN training has not started.**',
        f"The preservation snapshot verifies **{len(baseline)} prior files**, including all 153 processed EEG arrays,",
        'existing code, configurations, reports, manifests, normalization, frozen splits and both prior notebooks.',
        'New reusable code is isolated in `src/sleep_research`; existing Phase 1/2 code was not changed.',
        'No environment/package installation, upgrade, dataset download or preprocessing regeneration occurred.',
        'All retained model runs use tiny synthetic signals. Their metrics are infrastructure evidence, not Sleep-EDF performance.',
        'No real frozen-test logits, predictions, embeddings or performance metrics were computed.','',
        '## Data quality review','',
        f"Protocol B: {flags['B']['rail_epochs']:,} rail epochs, {flags['B']['edge_epochs']} edge flags, {flags['B']['both_flags']} overlaps;",
        f"{flags['B']['any_flag']:,} distinct flagged epochs. Protocol A: {flags['A']['rail_epochs']:,} rail epochs,",
        f"{flags['A']['edge_epochs']} edge flags, {flags['A']['both_flags']} overlaps. Seven measured training-only waveform examples were saved.",
        'Subject/recording/split/stage distributions and proposed pre-registered sensitivity analyses are in',
        '`results/artifact_quality_review.md` and the quality CSVs. Existing primary data and labels remain unchanged.','',
        '## Checkpoint and resume implementation','',
        'Shared manager: `checkpoints.py`; reusable trainer: `trainer.py`; permanent runs/contracts: `experiments.py`.',
        'Immutable checkpoint generations and byte-hash sidecars are validated with the restricted Torch loader before',
        'an atomic pointer index publishes latest, best and previous together. A checksummed previous index permits',
        'recovery if the current pointer file is damaged. Uncommitted orphan generations are never auto-selected.',
        'A prior recoverable epoch is retained. Rolling roles and the prior index typically retain at most four',
        'generations; checkpoints referenced by permanent exports are additionally pinned for later inference.',
        'Best selection uses strictly highest validation macro-F1 over all five classes. Best/latest roles are independent.',
        'Checkpoint checksums detect corruption; they are integrity evidence rather than authentication.','',
        'Payloads save model/optimizer/scheduler, early-stop counters/best score, completed/next epoch, global step,',
        'Python/global-NumPy/CPU-Torch RNG, relevant CUDA/MPS RNG, loader generators, optional sampler/dataset state,',
        'history, exact config, source revision and architecture/data/split/normalization/class/protocol fingerprints.',
        'A bootstrap checkpoint stores completed=-1 and next=0. Every trained-and-validated completed epoch then',
        'commits scheduler/early-stop/history state. An interruption before that commit repeats the entire epoch',
        'from the prior checkpoint. Materialized history is rebuilt from the checkpoint without duplicates.',
        'Manual rewind establishes a new committed timeline and discards future best state from the abandoned timeline.',
        'Incompatible architecture/config/preprocessing/normalization/classes/subject splits/epoch manifests/protocols',
        'or experiment identity are rejected, rather than silently restarting or adapting model weights.','',
        'CLI resume values: `--resume auto`, `--resume none`, `--resume /absolute/checkpoint/path.pt`.',
        'Epoch budget extension is allowed; other training configuration changes are rejected. A fresh start preserves',
        'existing checkpoints by requiring a new run. Both Terminal and Jupyter invoke the same `Trainer` class.',
        '**Bitwise-identical Apple MPS resumption is not promised.** CPU equivalence is validated only for this',
        'software/configuration fixture. Different backends, versions, workers or custom stochastic transforms may differ.',
        'Custom RNGs need dataset/sampler state callbacks; worker-internal RNG state is not captured. Demo workers=0.','',
        '## Permanent research artifacts','',
        'Runs use `experiments/{cnn,cnn_lstm,attention,transformer}/<unique_run_id>/`; only the cnn family currently',
        'has synthetic demonstration runs. The other family directories are ready for future authorized models.','',
        '- Reproducibility: config, software, seed/device, source hashes/Git revision if available, frozen contract,',
        '  attempt durations/status, optimizer/scheduler/model checkpoints and integrity/fingerprint manifests.',
        '- Performance: committed train/validation history and epoch-ID-aligned true/predicted labels, all five',
        '  raw logits, five probabilities, subject/recording/night/original-index/onset metadata and split membership.',
        '- Representations: penultimate features, row-to-epoch mapping and configurable pooled intermediate features.',
        '  Intermediate tensors are off by default and have a width cap. No entire processed EEG dataset is copied.','',
        'Inference exports pin their source checkpoint so more features/explanations can be generated without retraining.',
        'Optional bounded saliency and compatible Conv1d Grad-CAM are demonstrated on synthetic data. Attention capture',
        'is tested with a compatible MultiheadAttention fixture returning weights; arbitrary Transformer layers may',
        'require an adapter. Unsupported/non-returning attention layers fail explicitly. Explanations are not causal proof.','',
        '## Analysis-only interfaces and final-test gate','',
        '`analysis.py` imports no trainer and reads verified saved exports. It recalculates confusion matrices,',
        'accuracy/precision/recall/F1, balanced accuracy, Cohen kappa, log loss/Brier/ECE, ROC/PR, subject metrics,',
        'learning curves, PCA/t-SNE and predicted-versus-true hypnograms. Original recording boundaries and gaps are respected.',
        'True labels/probabilities and all five logits remain available for recalibration and ROC/PR inputs.',
        'PCA uses centered SVD with recorded components/variance. Exact NumPy/SciPy t-SNE matches the analytic gradient',
        'in a finite-difference test and records seed, perplexity, iterations, subset IDs and KL divergence.',
        'Its quadratic cost is bounded by a saved subsample. Scikit-learn and UMAP are absent; no packages were installed.',
        'The optional UMAP function reports unavailable unless the package already exists.','',
        'Default inference/analysis gates reject test and mixed-test exports before model test performance is computed.',
        '`--final-evaluation` is an explicit future-stage interface; no real-data final-evaluation invocation occurred.',
        'Data-quality counts by test split are descriptive baseline statistics and do not bypass this gate.','',
        '## Validation evidence','',
        f"Final suite: **{tests['tests_run']} tests; {tests['failures']} failures, {tests['errors']} errors, {tests['skipped']} skipped**.",
        'The suite includes all 72 prior tests, plus interruption/continuation, CPU weights/Adam moments/StepLR/RNG',
        'restoration, bootstrap recovery, best/latest independence, corrupted bytes/pointers, pinned retention,',
        'history uniqueness, incompatibility rejection, exact exported IDs/probabilities/features, PCA/t-SNE,',
        'confusion recalculation without training, gap-aware plotting, attention/Grad-CAM/saliency and CLI integration.',
        'Old test code remains unchanged; the two legacy JSON-writing callbacks are redirected to temporary SSD-local',
        'paths so original Phase 2 test evidence is never rewritten. Logs: `results/phase2_5_test_results.txt`.',
        f"Notebook: success={notebook['successful']}; **{notebook['executed_code_cells']} code cells, {notebook['image_outputs']} PNG outputs, {notebook['error_outputs']} errors**.",
        'The notebook uses the exact existing interpreter and a temporary local kernel with explicit cleanup.',
        f"Synthetic notebook interruption was inside epoch {demo['interrupted_epoch']}; it recovered next epoch {demo['recovered_next_epoch']}",
        f"and ended at next epoch {demo['final_next_epoch']} with unique history {demo['history_epochs']}.",
        'Terminal evidence: `phase2_5_terminal_interrupt.txt` and `phase2_5_terminal_resume.txt`.',
        'Notebook evidence: `phase2_5_notebook_execution.json`, `phase2_5_notebook_demo.json` and the executed notebook.','',
        table(['Retained synthetic run','Completed epoch','Next epoch','Global step','Checkpoints','Exports','Attempt seconds'],
              [[run['run_id'],run['completed_epoch'],run['next_epoch'],run['global_step'],run['retained_checkpoints'],run['exports'],f"{run['seconds']:.3f}"] for run in experiments]),'',
        f"Permanent experiment storage: **{experiment_bytes:,} bytes ({experiment_bytes/1024**2:.3f} MiB)**, excluding the existing EEG dataset.",
        'All permanent model artifacts are marked synthetic. Trained synthetic checkpoints are retained for additional inference.','',
        '## Readiness and remaining scope','',
        'Shared epoch-boundary checkpointing, resume, permanent artifact export and analysis-only workflows are ready',
        'for later explicitly authorized model experiments. The primary processed dataset and frozen split are unchanged.',
        'Actual CNN/CNN-LSTM/attention/Transformer architecture selection, full training, scientific comparisons and the',
        'frozen final-evaluation stage remain future work. Artifact sensitivity masks are proposals, not applied changes.',
        '**Stopped after Phase 2.5 infrastructure. No full Sleep-EDF CNN training was started.**','',
        '## References and reproducibility','',
        '- [PyTorch reproducibility](https://docs.pytorch.org/docs/2.14/notes/randomness.html): backend/version reproducibility limits.',
        '- [PyTorch serialization](https://docs.pytorch.org/docs/2.14/notes/serialization.html): state dictionaries and restricted loading.',
        '- [van der Maaten & Hinton, Visualizing Data using t-SNE](https://www.jmlr.org/papers/v9/vandermaaten08a.html), JMLR 9 (2008).',
        '- `configs/infrastructure_demo.yaml`, README commands and `src/sleep_research/` modules.',
        '- `results/phase2_5_preservation_snapshot.json` and `phase2_5_artifact_manifest.json`.',
        '- Prior dataset attribution and experimental protocol remain in the preserved Phase 1/2 reports.','']
    with atomic_file(RESULTS/'phase2_5_infrastructure_report.md') as handle:
        handle.write('\n'.join(report))
    json_write(RESULTS/'phase2_5_validation_summary.json',{'tests':tests,'notebook':notebook,'preserved_files':len(baseline),
              'quality':flags,'runs':experiments,'experiment_storage_bytes':experiment_bytes,'real_eeg_training':False,
              'real_test_performance_computed':False,'dataset_regenerated':False})
    paths=[*PROJECT.glob('src/sleep_research/*.py'),*PROJECT.glob('scripts/*phase2_5*.py'),PROJECT/'scripts/review_artifact_quality.py',
           PROJECT/'configs/infrastructure_demo.yaml',PROJECT/'tests/test_phase2_5.py',PROJECT/'README.md',PROJECT/'AGENTS.md',PROJECT/'.gitignore',
           PROJECT/'notebooks/02_5_checkpoint_resume_demo.ipynb',
           *[path for path in RESULTS.glob('*') if path.is_file() and str(path.relative_to(PROJECT)) not in baseline
             and path.name!='phase2_5_artifact_manifest.json'],
           *RESULTS.glob('quality_review_figures/*.png'),
           *[path for path in (PROJECT/'experiments').rglob('*') if path.is_file() and '.matplotlib' not in path.parts]]
    manifest={str(path.relative_to(PROJECT)):digest(path) for path in sorted(set(paths))}
    json_write(RESULTS/'phase2_5_artifact_manifest.json',manifest)
    print(RESULTS/'artifact_quality_review.md')
    print(RESULTS/'phase2_5_infrastructure_report.md')


if __name__ == '__main__':
    main()
