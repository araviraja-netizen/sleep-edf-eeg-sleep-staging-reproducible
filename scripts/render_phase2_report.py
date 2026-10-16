"""Render evidence-backed Phase 2 reports and a separate final artifact snapshot."""

import csv
from datetime import datetime
import json
from pathlib import Path
import shutil
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
from sleep_edf.annotations import STAGE_NAMES
from sleep_edf.inspection import sha256_file
from sleep_edf.phase2 import preserve_phase1
from sleep_edf.phase2_utils import atomic_json, atomic_writer, timestamp

RESULTS = PROJECT / "results"


def load(name):
    return json.loads((RESULTS / name).read_text())


def table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    lines.extend("| " + " | ".join(str(value) for value in row) + " |" for row in rows)
    return "\n".join(lines)


def gib(value):
    return f"{value / 1024**3:.6f} GiB"


def main():
    preserve_phase1(PROJECT)
    state, frozen = load("phase2_state.json"), load("subject_split_freeze.json")
    tests, pilot, notebook = load("phase2_test_results.json"), load("phase2_pilot.json"), load("phase2_notebook_execution.json")
    normalization_validation = load("phase2_normalization_validation.json")
    manifest_validation = load("phase2_manifest_validation.json")
    exceptions = load("phase2_exceptions.json")
    with (RESULTS / "subject_splits.csv").open(newline="") as handle:
        subject_rows = list(csv.DictReader(handle))
    with (RESULTS / "recording_preprocessing_summary.csv").open(newline="") as handle:
        record_rows = list(csv.DictReader(handle))
    subjects = {split: sorted(r["subject_id"] for r in subject_rows if r["split"] == split)
                for split in ("train", "validation", "test")}
    full = sum(state["counts"]["B"]["all"].values())
    trimmed = sum(state["counts"]["A"]["all"].values())
    removed = full - trimmed
    phase1_paths = set(load("phase2_phase1_snapshot.json"))
    processed_root = PROJECT / state["processed_root"]
    phase2_result_files = [p for p in RESULTS.rglob("*") if p.is_file()
                           and str(p.relative_to(PROJECT)) not in phase1_paths
                           and p.name != "phase2_artifact_manifest.json"]
    result_bytes = sum(p.stat().st_size for p in phase2_result_files)
    array_files = list(processed_root.rglob("epochs.npy"))
    processed_files = [p for p in processed_root.rglob("*") if p.is_file()]
    allocated_bytes = sum(getattr(p.stat(), "st_blocks", 0) * 512 for p in processed_files + phase2_result_files)
    duration = (datetime.fromisoformat(state["completed_at"]) - datetime.fromisoformat(state["run"]["started_at"])).total_seconds()
    ready = (state["status"] == "complete" and not exceptions and pilot["successful"] and tests["successful"]
             and not tests["skipped"] and not tests["failures"] and not tests["errors"] and notebook["successful"]
             and state["leakage"]["subject_overlap"] == 0 and manifest_validation["sample_label_alignment"])
    lines = ["# Phase 2 — Sleep-EDF EEG preprocessing and dataset preparation", "",
             f"Report generated: {timestamp()} (Asia/Kolkata).", "",
             "## Result and scope", "",
             f"**{state['processed_recordings']}/153 SC recordings successfully processed; {len(exceptions)} failures.**",
             f"Protocol B retains **{full:,}** usable epochs. Primary Protocol A retains **{trimmed:,}**,",
             f"removing **{removed:,} Wake epochs**. All non-Wake classes have identical counts between A and B.",
             "All Phase 1 reports, inventories, test logs, provenance, figures and the executed 01 notebook remain",
             "byte-for-byte unchanged against `results/phase2_phase1_snapshot.json`. ST remains separate and unprocessed.",
             "The existing interpreter/dataset were verified; no installation, environment creation, download, or raw EDF modification.",
             "No CNN, model training, model checkpoint, prediction, logit, embedding or model-evaluation output was created.", "",
             "## Frozen subject-independent split", "",
             table(["Split", "Subjects", "% subjects", "Recordings", "A epochs", "B epochs"],
                   [[split, len(subjects[split]), f"{100 * len(subjects[split]) / 78:.3f}",
                     sum(r["split"] == split for r in record_rows),
                     sum(state["counts"]["A"][split].values()), sum(state["counts"]["B"][split].values())]
                    for split in subjects]), "",
             "Seed **42**. Sorted SC subject IDs were permuted with `numpy.random.default_rng(42)` and assigned",
             "54/12/12 in order. Every night of a subject has one split; subject overlap and recording conflicts are zero.",
             "No stratification or test-performance optimization was applied. The test cohort is frozen for future models.", "",
             f"Subject split manifest SHA-256: `{frozen['subject_manifest_sha256']}`.",
             f"Recording assignment manifest SHA-256: `{frozen['recording_manifest_sha256']}`.", ""]
    for split in subjects:
        lines += [f"**{split.capitalize()} subjects:** " + ", ".join(subjects[split]) + ".", ""]
    lines += ["## Continuous EEG preprocessing", "",
              "Native **EEG Fpz-Cz, 100 Hz**. MNE applies EDF calibration to volts; the pipeline converts to microvolts.",
              "Only this channel is read, in 600-second chunks, into one recording buffer. Continuous finite EEG is",
              "filtered before extracting complete half-open `[30*k,30*(k+1))` epochs of exactly **3000 samples**.",
              "Unknown/movement intervals stay in the continuous signal during filtering but supply no supervised labels.",
              "There is no resampling, timestamp shift, per-epoch filter, invented sample or beyond-signal epoch.", "",
              "Filter: `butter(4, [0.3,35], fs=100, btype='bandpass', output='sos')`, followed by",
              "`sosfiltfilt(..., padtype='odd', padlen=27)` in float64. N=4 specifies the fourth-order Butterworth",
              "prototype; band-pass transformation yields an eighth-order one-pass response and forward/backward",
              "application an effective order 16. The zero-phase amplitude response is squared, with approximately",
              "−6.02 dB at both specified cutoffs. No cutoff compensation was silently applied.",
              "Actual SOS coefficients and measured response: `results/filter_response.json`; figure and notebook show both responses.", "",
              "Odd reflection padding is synthetic filter support only. It is never exported as EEG samples. Low-frequency",
              "settling can extend beyond 27 samples, and reflection cannot eliminate all edge effects. Epochs touching the",
              "first/last 30 seconds are flagged and retained; this caution window does not certify exact transient extent.",
              "Historical acquisition filters listed in the EDF/Phase 1 inventory cannot be reversed. Zero-phase filtering",
              "uses future samples, so this preprocessing protocol is offline rather than a causal streaming detector.", "",
              "## Wake protocols and class distributions", "",
              "**A (primary):** locate first/last annotation-and-signal-valid sleep labels 1–4; retain up to 60 original",
              "epochs before and after, capped by the signal. Preserve every valid epoch between them, including internal",
              "Wake. Invalid/movement/unknown positions remain gaps in original epoch indices. No-sleep records are flagged",
              "and excluded from A. **B (secondary):** retain every valid epoch, including all Wake, with the same splits.",
              "No class balancing was performed. Brief early/late sleep annotations can retain long internal Wake stretches.",
              "**Label-based trimming is an offline evaluation protocol, not a deployable sleep-boundary detector.**", "",
              table(["Stage", "Label", "Original valid annotations", "B full epochs", "A trimmed epochs", "B %", "A %"],
                    [[stage, label, state["counts"]["original_annotation_valid"]["all"][stage],
                      state["counts"]["B"]["all"][stage], state["counts"]["A"]["all"][stage],
                      f"{100 * state['counts']['B']['all'][stage] / full:.3f}",
                      f"{100 * state['counts']['A']['all'][stage] / trimmed:.3f}"] for label, stage in enumerate(STAGE_NAMES)]), "",
              f"Original complete signal epochs: **416,424**; annotation-valid epochs: **414,961**;",
              f"signal-valid full epochs: **{full:,}**; trimmed epochs: **{trimmed:,}**.",
              "Per-recording counts and boundaries: `results/recording_preprocessing_summary.csv` and `wake_trimming_report.md`.",
              "Per-subject totals/classes: `results/per_subject_epoch_counts.csv` and `per_subject_class_counts.csv`.",
              "Separate class counts for all/train/validation/test: `results/stage_distribution_comparison.csv`.", "",
              "## Training-only normalization and leakage verification", "",
              "One sample-weighted global population mean/std (ddof=0) is fitted independently for each protocol using",
              "only its training epochs. No validation/test signal enters either fit. The fixed parameters apply unchanged",
              "to validation/test. Filtered unnormalized float32 microvolt EEG is stored once; the loader applies z-scoring",
              "on demand and returns float32 dimensionless samples. There are no duplicate normalized A/B arrays.", "",
              table(["Protocol", "Training subjects", "Training epochs", "Training samples", "Mean µV", "Population std µV", "Normalized train mean", "Normalized train std"],
                    [[protocol, len(state["normalization"]["protocols"][protocol]["training_subjects"]),
                      state["normalization"]["protocols"][protocol]["epoch_count"],
                      state["normalization"]["protocols"][protocol]["sample_count"],
                      f"{state['normalization']['protocols'][protocol]['mean_microvolts']:.12g}",
                      f"{state['normalization']['protocols'][protocol]['std_microvolts']:.12g}",
                      f"{normalization_validation[protocol]['normalized_training_mean']:.12g}",
                      f"{normalization_validation[protocol]['normalized_training_population_std']:.12g}"] for protocol in ("A", "B")]), "",
              "Independent post-normalization scans validate population mean ≈0 and std ≈1 over every stored training",
              "sample in each protocol. Validation/test populations are not forced to zero mean/unit variance.",
              "Evidence: `normalization.json`, `phase2_normalization_validation.json`, and `phase2_leakage_checks.json`.", "",
              "## Signal-quality issues and exceptions", "",
              table(["Metric", "Count"], [[key, value] for key, value in state["quality"].items()]), "",
              "All processed raw and filtered samples are finite. Nonfinite raw/filtered samples would reject a recording",
              "without interpolation; exact constant raw epochs would be excluded while retaining original indices.",
              f"**{state['quality']['usable_epochs_with_rail_contacts']:,} usable epochs** have ADC rail contacts; these remain",
              "flagged rather than being removed by an unapproved amplitude threshold. Numeric sanity checks do not establish",
              "physiological artifact-free data. Review the saved per-recording ranges/variances and these flags in later research.",
              f"Recording-edge flags: **{manifest_validation['recording_edge_flagged_epochs']} epochs**.",
              "SC4362F's final Wake annotation extends 15,120 seconds beyond its 67,980-second PSG. Only supported",
              "in-signal epochs are stored, with no timestamp shift; its unknown tail and other annotation overhang remain evidence.",
              f"No-valid-sleep recordings: **{len(state['no_valid_sleep_recordings'])}**. Failed recordings: **{len(exceptions)}**.",
              "Detailed failure files: `phase2_exceptions.json`, `phase2_pilot_exceptions.json`, `phase2_all_exceptions.json`.", "",
              "## Memory-mappable storage, atomic publication and resume", "",
              f"Processed root: `{state['processed_root']}`.",
              f"Pipeline fingerprint: `{state['pipeline_fingerprint']}`.",
              "Each of the 153 recording directories has one `epochs.npy` array, `epoch_metadata.csv`, and `completion.json`.",
              "Arrays have shape `(usable_full_epochs,3000)`, dtype float32, filtered microvolt units, and no stored normalization.",
              "Completion files bind raw SHA-256, relevant config/code/dependency versions, array/metadata hashes and sizes.",
              "A validated temporary directory is atomically renamed. Resume rechecks raw hashes and completed artifact",
              "hashes, sample finiteness, dtype/shape and metadata; incomplete/corrupt outputs are never silently reused.",
              "Uncommitted temporary directories are ignored; a lock prevents concurrent writers to the same fingerprint.",
              f"The full run reused **{state['run']['resumed_recordings']}** validated pilot outputs and created **{state['run']['new_recordings']}** new recordings.", "",
              "The master epoch manifest carries unique ID, dataset/study, subject, night/recording, original index/onset,",
              "true label, split, both inclusion flags, array path/row, units, preprocessing version/fingerprint and quality flags.",
              "A/B index manifests refer to the same stored rows. `phase2_state.json` is the final consolidation commit",
              "with hashes protecting all indices, normalization and frozen split artifacts. The loader checks metadata",
              "against stored sample rows and opens only a bounded number of read-only memmaps per worker.", "",
              table(["Resource", "Measured usage"], [["One copy of epoch arrays", f"{state['array_storage_bytes']:,} bytes ({gib(state['array_storage_bytes'])})"],
                   ["Processed directory including per-recording metadata", f"{state['processed_directory_bytes']:,} bytes ({gib(state['processed_directory_bytes'])})"],
                   ["Phase 2 result metadata/logs/figures at report time", f"{result_bytes:,} bytes ({gib(result_bytes)})"],
                   ["Preprocessing process high-water RSS", gib(state["process_peak_rss_bytes"])],
                   ["Recording processing before consolidation", f"{state['run']['elapsed_seconds']:.3f} seconds"],
                   ["Full pass including consolidation/normalization", f"{duration:.3f} seconds"],
                   ["Allocated processed + Phase 2 result file blocks at report time", gib(allocated_bytes)],
                   ["SSD free at report time", gib(shutil.disk_usage(PROJECT).free)]]), "",
              "Logical file bytes and allocated blocks differ; filesystem page cache and notebook/test processes are not",
              "included in the preprocessing process RSS. Figure/report artifact sizes can change slightly when rerendered.", "",
              "## Pilot, automated tests and executed notebook", "",
              f"Two-recording pilot: **successful={pilot['successful']}**, B={pilot['full_epochs']:,} epochs, A={pilot['trimmed_epochs']:,}.",
              "The pilot stored samples match independently filtered full-recording references, including rows on either side",
              "of annotation exclusions. Pilot gating and tests passed before processing the remaining SC recordings.",
              f"Final suite: **{tests['tests_run']} tests; {tests['failures']} failures, {tests['errors']} errors, {tests['skipped']} skipped**.",
              "It covers Phase 1 invariants; first/last sleep boundaries and internal Wake; stage exclusions; stable filtering",
              "and zero phase; training-only moments; frozen subject assignment; corruption/interruption detection; every",
              "recording's hashes/metadata/labels/trimming; full manifest uniqueness; normalization across training populations;",
              "and loader behavior for both protocols and all three splits. Logs: `results/phase2_test_results.txt`.",
              f"Notebook execution: **success={notebook['successful']}**, {notebook.get('executed_code_cells', 0)} executed code cells,",
              f"{notebook.get('image_outputs', 0)} PNG outputs, {notebook.get('error_outputs', 0)} error outputs.",
              "The exact existing interpreter runs through a temporary SSD-local kernel specification; no global kernel is installed.",
              "Notebook: `notebooks/02_eeg_preprocessing.ipynb`. Saved raw/filtered EEG, Welch PSD, original/trimmed hypnograms,",
              "and response figures: `results/phase2_figures/`. PSD comparisons use matching epochs, not falsely concatenated gaps.", "",
              "## Readiness for later deep-learning experiments", "",
              ("**The processed A/B datasets are technically ready for later deep-learning training.**" if ready
               else "**Resolve the reported processing/validation issues before training.**"),
              "Frozen cohorts, stable epoch IDs, true labels, onsets, source hashes, quality flags and normalization hashes",
              "support reproducible experiments, prediction/logit/probability joins, embeddings, subject analyses and",
              "sleep-transition analyses. Sequence/transition analysis must respect gaps in original epoch indices.",
              "Future training must save model/optimizer/scheduler, epoch, RNG and sampler state for exact resume and record",
              "dataset/split/config/normalization hashes. That training state is intentionally not fabricated during preprocessing.",
              "Remaining scientific limitations: historical R&K-to-five-class mapping, offline zero-phase/label-based protocols,",
              "ADC rail contacts requiring artifact review, retained recording-edge transients, and filename-derived identities.",
              "ST is not eligible for these initial experiments. No test-performance-driven tuning has occurred.", "",
              "**Stopped after Phase 2. CNN training has not begun.**", "",
              "## Reproducibility references", "",
              "- [Sleep-EDF Expanded](https://physionet.org/content/sleep-edfx/1.0.0/), DOI 10.13026/C2X676; raw attribution in README.",
              "- [SciPy Butterworth](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.butter.html) and",
              "  [SOS forward/backward filtering](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.sosfiltfilt.html).",
              "- `configs/preprocessing.yaml`, `configs/wake_trimming.yaml`, source modules and commands in README.",
              "- Relevant source/dependency snapshot: `phase2_state.json` and processed `pipeline_identity.json`.",
              "- Final small-artifact/code snapshot: `results/phase2_artifact_manifest.json`; array hashes in completion manifests.", ""]
    with atomic_writer(RESULTS / "phase2_preprocessing_report.md") as handle:
        handle.write("\n".join(lines))
    wake_lines = ["# Phase 2 — Wake trimming report", "",
                  "Protocol A is the primary 30-minute context rule; B keeps the full valid recording.",
                  "First/last valid sleep uses labels 1–4 after annotation and signal-validity checks. Up to 60 original",
                  "epochs before and after are retained, with inclusive last+60 (exclusive stop last+61). Internal Wake",
                  "is preserved. Invalid/unknown/movement epochs remain original-grid gaps; samples are never shifted.",
                  "**This is label-based offline evaluation, not a deployable sleep-boundary detector.**", "",
                  f"Full valid: **{full:,}** epochs. Trimmed: **{trimmed:,}**. Removed: **{removed:,} Wake epochs**.",
                  "Every non-Wake class is preserved. No class balancing or test-performance optimization.", "",
                  table(["Recording", "Subject", "Split", "Original valid", "B full", "A trimmed", "Wake removed", "First sleep", "Last sleep", "Start", "Stop exclusive", "No sleep"],
                        [[r[k] for k in ("recording_id", "subject_id", "split", "original_annotation_valid_epochs",
                                        "usable_full_epochs", "usable_trimmed_epochs", "wake_epochs_removed", "first_valid_sleep_epoch",
                                        "last_valid_sleep_epoch", "trim_start_epoch", "trim_stop_epoch_exclusive", "no_valid_sleep")]
                         for r in record_rows]), "",
                  "Full numerical/quality summaries: `recording_preprocessing_summary.csv`. Class distributions by split:",
                  "`stage_distribution_comparison.csv`. Per-subject counts: `per_subject_epoch_counts.csv` and `per_subject_class_counts.csv`.", ""]
    with atomic_writer(RESULTS / "wake_trimming_report.md") as handle:
        handle.write("\n".join(wake_lines))
    # Preserve Phase 1's snapshot; this separate snapshot references large arrays through completions.
    paths = [*PROJECT.glob("src/**/*.py"), *PROJECT.glob("scripts/*.py"), *PROJECT.glob("tests/*.py"),
             *PROJECT.glob("configs/*"), PROJECT / "README.md", PROJECT / "AGENTS.md", PROJECT / ".gitignore", PROJECT / "pyproject.toml",
             PROJECT / "notebooks/02_eeg_preprocessing.ipynb",
             *[p for p in RESULTS.rglob("*") if p.is_file() and str(p.relative_to(PROJECT)) not in phase1_paths
               and p.name != "phase2_artifact_manifest.json"],
             *processed_root.glob("recordings/*/completion.json"), processed_root / "pipeline_identity.json"]
    manifest = {str(path.relative_to(PROJECT)): sha256_file(path) for path in sorted(set(paths))}
    atomic_json(RESULTS / "phase2_artifact_manifest.json", manifest)
    print(RESULTS / "phase2_preprocessing_report.md")
    print(RESULTS / "wake_trimming_report.md")


if __name__ == "__main__":
    main()
