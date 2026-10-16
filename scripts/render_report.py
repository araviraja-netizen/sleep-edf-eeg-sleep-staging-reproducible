"""Render the Phase 1 evidence report from generated artifacts, never raw signals."""

import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
from zoneinfo import ZoneInfo

PROJECT = Path(__file__).resolve().parents[1]
RESULTS = PROJECT / "results"


def load(name):
    return json.loads((RESULTS / name).read_text())


def table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    lines.extend("| " + " | ".join(str(item) for item in row) + " |" for row in rows)
    return "\n".join(lines)


def main():
    summary, provenance, detailed = load("dataset_summary.json"), load("provenance.json"), load("detailed_inspection.json")
    tests, notebook = load("test_results.json"), load("notebook_execution.json")
    kernel_mps = load("mps_check.json") if (RESULTS / "mps_check.json").exists() else provenance["mps"]
    issues, groups = load("discovery_issues.json"), load("subject_groups.json")
    with (RESULTS / "dataset_inventory.csv").open(newline="") as handle:
        inventory = list(csv.DictReader(handle))
    by_id = {row["recording_id"]: row for row in inventory}
    lines = ["# Phase 1 — Sleep-EDF Expanded dataset inspection", "",
             "Automatic Sleep Stage Classification Using Deep Learning on PhysioNet Sleep-EDF Expanded.", "",
             f"Inventory completed: {provenance['completed_at_Asia_Kolkata']} (Asia/Kolkata).",
             f"Report generated: {datetime.now(ZoneInfo('Asia/Kolkata')).isoformat()}.", "",
             "## Scope and environment", "",
             f"Dataset: `{provenance['config']['dataset_root']}`. Interpreter: `{provenance['python_executable']}`.",
             f"Python: `{provenance['python_version'].splitlines()[0]}`. Platform: `{provenance['platform']}`.",
             "Dataset/interpreter existence was verified before project creation. Raw EDFs were read only.",
             "No environments, installations, downloads, wake trimming, filtering, normalization, splitting, or training.",
             f"MPS observed in inventory process: `{provenance['mps']}`.",
             f"MPS observed in executed notebook kernel: `{kernel_mps}`. Phase 1 uses CPU inspection only.",
             "Hardware discovery can differ with sandbox access. No GPU computation or training was performed.", "",
             "## Separate study inventories", "",
             table(["Study", "PSGs", "Hypnograms", "Subjects", "Paired", "Inspected", "Two-night subjects", "One-night subjects"],
                   [[s, summary[s]["psg_files"], summary[s]["hypnogram_files"], summary[s]["subjects"],
                     summary[s]["paired_recordings"], summary[s]["inspected_ok"], summary[s]["subjects_with_two_nights"],
                     summary[s]["subjects_with_one_night"]] for s in ("SC", "ST")]), "",
             "SC is reserved for the initial future training experiments. ST remains a separate inventory.",
             "Study-qualified subject IDs group both nights and distinguish SC and ST subjects with the same numeric suffix.",
             "SC filenames include acquisition letters E, F, and G, as verified in the official RECORDS list; ST uses J.", "",
             "One-night SC groups: " + ", ".join(f"`{s}` ({', '.join(recordings)})" for s, recordings in groups.items()
                                                     if s.startswith("SC") and len(recordings) == 1) + ".",
             "The documented lost SC recordings are subject 13 night 2, subject 36 night 1, and subject 52 night 1.", "",
             "## Pairing, channels, duration, and integrity", "",
             f"Discovery/pairing issues: **{len(issues)}**. Inspection errors: "
             f"**{sum(summary[s]['inspection_errors'] for s in ('SC', 'ST'))}**.",
             f"SHA-256 matches against supplied manifest: **{provenance['checksum_verified_files']} files**; "
             f"mismatches: **{len(provenance['checksum_mismatch'])}**; missing referenced files: "
             f"**{len(provenance['checksum_missing_files'])}**; unverified non-manifest files: "
             f"**{len(provenance['checksum_unverified_files'])}**.",
             "The manifest itself is hashed for provenance, not checked against itself.",
             f"RECORDS missing/unlisted PSGs: **{len(provenance['RECORDS_missing_psgs'])}/"
             f"{len(provenance['RECORDS_unlisted_psgs'])}**.", ""]
    for study in ("SC", "ST"):
        selected = [r for r in inventory if r["study"] == study and r["inspection_status"] == "ok"]
        durations = [float(r["signal_duration_seconds"]) / 3600 for r in selected]
        rates = sorted({float(r["fpz_cz_sample_rate_hz"]) for r in selected})
        lines.append(f"- {study}: Fpz-Cz present in {summary[study]['fpz_cz_available_recordings']} recordings; "
                     f"Pz-Oz present in {summary[study]['pz_oz_available_recordings']}. "
                     f"Fpz-Cz native rates: {rates} Hz. Duration range: "
                     f"{min(durations):.4f}–{max(durations):.4f} hours." if durations else f"- {study}: no successfully inspected recordings.")
    lines += ["", "Native sample rates and acquisition prefilters for every channel are recorded as JSON in the CSV.",
              "EDF file sizes are compared with declared header/data-record sizes. Shared local header times and",
              "zero EDF+ annotation TAL origin are checked before label alignment; timestamps are never shifted.", "",
              "## Five-class counts and distribution", "",
              "Mapping: Wake=0, N1=1, N2=2, N3=3 (original stages 3+4), REM=4.",
              "All valid Wake is retained. Percentages use valid supervised epochs only.", "",
              table(["Class", "Label", "SC epochs", "SC % valid", "ST epochs", "ST % valid"],
                    [[name, label, summary["SC"]["class_epoch_counts"][name],
                      f"{summary['SC']['class_percent_of_valid'][name]:.3f}", summary["ST"]["class_epoch_counts"][name],
                      f"{summary['ST']['class_percent_of_valid'][name]:.3f}"]
                     for label, name in enumerate(("Wake", "N1", "N2", "N3", "REM"))]), "",
              "## Annotation quality and exact epoch boundaries", "",
              "Epochs are half-open `[30*k, 30*(k+1))` intervals on the PSG sample grid, with 3000 native EEG",
              "samples at 100 Hz. No epoch extends beyond the signal. A label requires one grid-aligned segment",
              "covering the entire epoch. Movement/unknown/unrecognized text has no supervised label; overlaps",
              "and malformed timing reject affected epochs. Segment counts and epoch counts are distinct.", ""]
    metric_names = ["complete_signal_epochs", "valid_epochs", "movement_epochs", "unknown_epochs", "unrecognized_epochs",
                    "missing_epochs", "overlap_epochs", "invalid_timing_epochs", "partial_annotation_epochs",
                    "movement_annotation_segments", "unknown_annotation_segments", "unrecognized_annotation_segments",
                    "invalid_timing_annotation_segments", "out_of_signal_annotation_segments",
                    "out_of_signal_supervised_annotation_segments", "out_of_signal_annotation_seconds",
                    "annotation_missing_seconds", "annotation_overlap_seconds", "trailing_signal_seconds"]
    lines += [table(["Metric", "SC", "ST"], [[k, summary["SC"][k], summary["ST"][k]] for k in metric_names]), "",
              "Annotation coverage is the union of finite positive annotation intervals clipped to actual signal",
              "duration, including excluded stages. Missing seconds include partial tails; missing epoch counts",
              "include only complete grid epochs. Outside-signal annotation time is reported but never epochized.", ""]
    anomaly_rows = [r for r in inventory if r["inspection_status"] == "ok" and any(float(r.get(k) or 0) > 0
                    for k in ("missing_epochs", "overlap_epochs", "invalid_timing_annotation_segments",
                              "unrecognized_annotation_segments", "out_of_signal_supervised_annotation_segments"))]
    lines += ["Recordings requiring timing/coverage review (unknown/movement-only exclusions and unknown overhang alone are omitted below):", ""]
    if anomaly_rows:
        lines.append(table(["Recording", "Missing epochs", "Overlap epochs", "Malformed segments", "Unrecognized segments", "Supervised overhang segments"],
                           [[r["recording_id"], r["missing_epochs"], r["overlap_epochs"], r["invalid_timing_annotation_segments"],
                             r["unrecognized_annotation_segments"], r["out_of_signal_supervised_annotation_segments"]] for r in anomaly_rows]))
    else:
        lines.append("None.")
    if "SC4362F" in by_id:
        row = by_id["SC4362F"]
        lines += ["", "SC4362F has a Wake annotation from 52,380 to 83,100 seconds, while its PSG ends at",
                  "67,980 seconds. The 15,120 seconds of Wake beyond the signal and subsequent unknown tail",
                  "are not converted to epochs. Its uniquely annotated complete in-signal Wake remains valid.",
                  "This is annotation/PSG duration disagreement in checksum-matching source files; it does not",
                  "justify shifting annotation times or assuming additional recorded EEG."]
    lines += ["", "## Two detailed SC recordings", "",
              "SC4001E and SC4002E are nights 1 and 2 of subject SC400. The notebook shows a native EEG epoch,",
              "the full untrimmed hypnogram, and stage distribution for each. Numeric scans read 600-second chunks.", ""]
    selected_names = ["signal_duration_seconds", "complete_signal_epochs", "valid_epochs", "Wake_epochs", "N1_epochs",
                      "N2_epochs", "N3_epochs", "REM_epochs", "movement_epochs", "unknown_epochs", "missing_epochs",
                      "out_of_signal_annotation_segments", "out_of_signal_annotation_seconds"]
    lines += [table(["Metric", "SC4001E", "SC4002E"], [[key, by_id["SC4001E"][key], by_id["SC4002E"][key]] for key in selected_names]), "",
              table(["Recording", "Samples scanned", "Nonfinite", "Min µV", "Max µV", "Mean µV", "Std µV", "Equal adjacent pairs"],
                    [[rid, values["samples_scanned"], values["nonfinite_samples"],
                      *[f"{values[k]:.6f}" if values[k] is not None else "unavailable" for k in ("min_uv", "max_uv", "mean_uv", "std_uv")],
                      values["equal_adjacent_sample_pairs"]] for rid, values in detailed.items()]), "",
              "Amplitude and equal-adjacent counts are descriptive, with no artifact rejection threshold imposed.",
              "Finite/amplitude checks and two plots do not certify physiological quality of the full dataset.", "",
              "## Automated tests and notebook execution", "",
              f"Tests: **{tests['tests_run']} run, {tests['failures']} failures, {tests['errors']} errors, "
              f"{tests['skipped']} skipped**; success: **{tests['successful']}**. "
              f"Interpreter: `{tests['python_executable']}`. Runner: standard-library unittest.",
              "Coverage includes strict filename parsing, missing/duplicate pairs, scorer changes, both SC acquisition",
              "variants, mapping and exclusions, long Wake preservation, exact sample grids, half-open boundaries,",
              "gaps, overlaps, off-grid timing, negative/nonfinite durations/onsets, overhang, subject grouping, and",
              "actual-dataset header/channel/origin checks. Full logs: `results/test_results.txt`.", "",
              f"Notebook success: **{notebook['successful']}**; executed code cells: "
              f"**{notebook.get('executed_code_cells', 0)}**; PNG outputs: **{notebook.get('image_outputs', 0)}**; "
              f"error outputs: **{notebook.get('error_outputs', 0)}**.",
              "The exact existing Python ran the kernel through a temporary local kernel spec; no kernel was installed.",
              "Executed notebook: `notebooks/01_dataset_inspection.ipynb`; saved figures: `results/figures/`.", "",
              "## Remaining issues and Phase 2 readiness", ""]
    sc = summary["SC"]
    data_blockers = (issues or provenance["checksum_mismatch"] or provenance["checksum_missing_files"]
                     or provenance["checksum_unverified_files"] or provenance["RECORDS_missing_psgs"]
                     or provenance["RECORDS_unlisted_psgs"] or sc["inspection_errors"]
                     or sc["invalid_timing_annotation_segments"] or sc["overlap_epochs"] or sc["unrecognized_epochs"])
    validation_ok = tests["successful"] and not tests["skipped"] and notebook["successful"]
    lines.append("The SC inventory and validation are ready to inform a separately authorized Phase 2." if not data_blockers and validation_ok
                 else "Resolve the reported SC data/integrity or validation problems before Phase 2.")
    lines += ["", f"- SC missing complete epochs: {sc['missing_epochs']}; unknown exclusions: {sc['unknown_epochs']}; "
              f"movement exclusions: {sc['movement_epochs']}. Their eligibility must remain explicit in future processing.",
              "- Outside-signal annotations are retained as inspection evidence and cannot supply training epochs.",
              "- ST coverage gaps/partial tails are documented separately and do not enter the first SC experiments.",
              ("- MPS is available in the executed notebook kernel; the sandboxed inventory reported it unavailable."
               if kernel_mps["available"] else "- Recheck MPS availability before training; both checks reported it unavailable despite built-in support."),
              "- Artifact review is limited to two SC recordings. Subject grouping is filename-derived; anonymized headers",
              "  do not independently authenticate biological identity. Spreadsheet demographics and ST drug assignments",
              "  have not been interpreted.",
              "- No preprocessing policy or train/validation/test split is implemented. Future evaluation must group",
              "  both nights of a subject to avoid leakage, and document all new choices and fitted transformations.",
              "- The starting directory was not a Git repository. Source hashes provide a snapshot; no commit was made.", "",
              "**Stopped after Phase 1. Phase 2 has not begun.**", "",
              "## Sources and reproducibility artifacts", "",
              "- [PhysioNet Sleep-EDF Expanded](https://physionet.org/content/sleep-edfx/1.0.0/), DOI 10.13026/C2X676.",
              "- [Official recording names](https://physionet.org/content/sleep-edfx/1.0.0/RECORDS).",
              "- [EDF layout](https://www.edfplus.info/specs/edf.html) and [EDF+ timekeeping](https://www.edfplus.info/specs/edfplus.html).",
              "- `results/dataset_inventory.csv`, `dataset_summary.json`, `detailed_inspection.json`, `subject_groups.json`.",
              "- `results/raw_file_manifest.json`, `provenance.json`, `environment_observed.txt`, `artifact_manifest.json`.",
              "- `results/test_results.txt`, `test_results.json`, `notebook_execution.json`, `mps_check.json`.",
              "- Configuration: `configs/phase1.toml`; reproduction commands and metric semantics: `README.md`.", ""]
    (RESULTS / "phase1_dataset_inspection.md").write_text("\n".join(lines))
    artifacts = [*PROJECT.glob("src/**/*.py"), *PROJECT.glob("tests/**/*.py"), *PROJECT.glob("scripts/**/*.py"),
                 *PROJECT.glob("configs/*"), *PROJECT.glob("notebooks/*.ipynb"),
                 PROJECT / "README.md", PROJECT / "AGENTS.md", PROJECT / ".gitignore", PROJECT / "pyproject.toml",
                 *RESULTS.glob("*.csv"), *RESULTS.glob("*.json"), *RESULTS.glob("*.txt"), *RESULTS.glob("*.md"),
                 *RESULTS.glob("figures/*.png")]
    payload = {str(path.relative_to(PROJECT)): hashlib.sha256(path.read_bytes()).hexdigest()
               for path in sorted(set(artifacts)) if path.name != "artifact_manifest.json"}
    (RESULTS / "artifact_manifest.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(RESULTS / "phase1_dataset_inspection.md")


if __name__ == "__main__":
    main()
