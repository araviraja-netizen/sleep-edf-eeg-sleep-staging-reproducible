"""Terminal entrypoint for synthetic resume validation and analysis-only workflows."""

import argparse
import json
import os
from pathlib import Path

import yaml


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    demo = commands.add_parser("demo", help="Synthetic infrastructure smoke training only")
    demo.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[2])
    demo.add_argument("--config", type=Path, default=Path("configs/infrastructure_demo.yaml"))
    demo.add_argument("--run-id")
    demo.add_argument("--resume", default="auto", help="auto, none, or a checksummed checkpoint path")
    demo.add_argument("--max-epochs", type=int)
    demo.add_argument("--interrupt-after-steps", type=int)
    demo.add_argument("--skip-analysis", action="store_true")
    analysis = commands.add_parser("analyze", help="Read saved predictions/features; never train")
    analysis.add_argument("--export", type=Path, required=True)
    analysis.add_argument("--output", type=Path)
    analysis.add_argument("--final-evaluation", action="store_true", help="Explicitly enter the frozen final-evaluation stage")
    analysis.add_argument("--tsne-max-samples", type=int, default=500)
    analysis.add_argument("--seed", type=int, default=42)
    inspect = commands.add_parser("checkpoints", help="Inspect checkpoint roles without training")
    inspect.add_argument("--run", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "analyze":
        os.environ.setdefault("MPLCONFIGDIR", str(args.export.resolve().parents[2] / ".matplotlib"))
        from .analysis import analyze_export
        path = analyze_export(args.export, args.output, args.final_evaluation, args.tsne_max_samples, seed=args.seed)
        print(json.dumps({"analysis": str(path), "training_called": False}))
        return 0
    if args.command == "checkpoints":
        from .checkpoints import CheckpointManager
        metadata = json.loads((args.run / "reproducibility/experiment.json").read_text())
        manager = CheckpointManager(args.run / "reproducibility/checkpoints", metadata["resume_fingerprints"], metadata["experiment_id"])
        recovered = manager.recover("auto")
        print(json.dumps({"roles": manager.discover(), "completed_epoch": recovered["completed_epoch"],
                          "next_epoch": recovered["next_epoch"], "recovery_notes": manager.recovery_notes}, indent=2))
        return 0
    project = args.project.resolve()
    os.environ.setdefault("MPLCONFIGDIR", str(project / "results/.phase2_5_matplotlib"))
    config_path = args.config if args.config.is_absolute() else project / args.config
    config = yaml.safe_load(config_path.read_text())
    if args.max_epochs is not None:
        config["max_epochs"] = args.max_epochs
    run_id = args.run_id
    if run_id is None and args.resume == "auto":
        paths = sorted((project / "experiments" / config["family"]).glob("*/reproducibility/experiment.json"), key=lambda path: path.stat().st_mtime_ns)
        if paths:
            run_id = json.loads(paths[-1].read_text())["experiment_id"]
    elif run_id is None and args.resume not in ("auto", "none"):
        run_id = json.loads(Path(args.resume).with_suffix(Path(args.resume).suffix + ".json").read_text())["experiment_id"]
    from .synthetic import build_demo_trainer
    from .trainer import SimulatedInterruption
    trainer = build_demo_trainer(project, config, run_id)
    print("SYNTHETIC run:", trainer.experiment.run_id, flush=True)
    def interrupt(epoch, batch, step):
        if step == args.interrupt_after_steps:
            raise SimulatedInterruption(f"Synthetic interruption at epoch={epoch}, global_step={step}")
    try:
        history = trainer.fit(resume=args.resume, interrupt_callback=interrupt if args.interrupt_after_steps is not None else None)
    except SimulatedInterruption as exc:
        print(str(exc)); print("Resume the same run with --resume auto; the incomplete epoch will repeat.")
        return 130
    result = {"synthetic": True, "run": str(trainer.experiment.path), "completed_epochs": len(history),
              "next_epoch": trainer.next_epoch, "global_step": trainer.global_step, "checkpoints": trainer.manager.discover()}
    if not args.skip_analysis and history:
        from .inference import export_inference
        from .analysis import analyze_export
        from .explanations import export_explanations
        from .inference import load_model
        analysis_config = config["analysis"]
        export = export_inference(trainer.model, trainer.validation_loader, trainer.experiment,
                                 penultimate_layer=analysis_config["penultimate_layer"],
                                 intermediate_layers=analysis_config["intermediate_layers"], max_feature_dimension=analysis_config["max_feature_dimension"])
        output = analyze_export(export, tsne_max_samples=analysis_config["tsne_max_samples"], tsne_perplexity=analysis_config["tsne_perplexity"],
                                tsne_iterations=analysis_config["tsne_iterations"], seed=analysis_config["seed"])
        model, _, checkpoint = load_model(trainer.model, trainer.experiment)
        explanation = config["explanations"]
        export_explanations(model, next(iter(trainer.validation_loader)), trainer.experiment.path / "explanations" / export.name,
                            checkpoint, conv_layer=explanation["conv_layer"], max_samples=explanation["max_samples"])
        result.update(export=str(export), analysis=str(output))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
