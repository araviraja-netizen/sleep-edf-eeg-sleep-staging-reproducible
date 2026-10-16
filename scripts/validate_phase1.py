"""Run automated tests and execute the notebook using the exact existing Python.

The kernel spec exists only in a temporary project directory during execution.
Nothing is installed in the environment or in the user's Jupyter configuration.
"""

from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import tomllib
from zoneinfo import ZoneInfo

PROJECT = Path(__file__).resolve().parents[1]


def main():
    config = tomllib.loads((PROJECT / "configs/phase1.toml").read_text())
    python = Path(config["python_executable"])
    if not python.is_file() or not Path(config["dataset_root"]).is_dir():
        raise SystemExit("STOP: required Python/dataset missing")
    if python.resolve() != Path(sys.executable).resolve():
        raise SystemExit(f"STOP: run this script with {python}")
    subprocess.run([str(python), str(PROJECT / "scripts/run_tests.py")], cwd=PROJECT, check=True)
    import nbformat
    from nbclient import NotebookClient
    from jupyter_client import AsyncKernelManager
    from jupyter_client.kernelspec import KernelSpecManager

    target = PROJECT / "notebooks/01_dataset_inspection.ipynb"
    notebook = nbformat.read(target, as_version=4)
    tick = time.monotonic()
    result = {"successful": False, "python_executable": str(python), "notebook": str(target.relative_to(PROJECT)),
              "executed_at_Asia_Kolkata": datetime.now(ZoneInfo("Asia/Kolkata")).isoformat()}
    try:
        with tempfile.TemporaryDirectory(prefix="phase1-kernel-", dir=PROJECT / "results") as tmp:
            temporary = Path(tmp)
            kernel_dir = temporary / "kernels/sleep-edf-phase1"
            kernel_dir.mkdir(parents=True)
            argv = [str(python), "-m", "ipykernel_launcher", "-f", "{connection_file}"]
            spec = {"argv": argv, "display_name": "Python (existing dl-m4)", "language": "python"}
            (kernel_dir / "kernel.json").write_text(json.dumps(spec))
            manager = AsyncKernelManager(kernel_name="sleep-edf-phase1",
                                    kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(temporary / "kernels")]))
            env = os.environ.copy()
            env.update({"MPLCONFIGDIR": str(temporary / "matplotlib"),
                        "JUPYTER_RUNTIME_DIR": str(temporary / "runtime"),
                        "IPYTHONDIR": str(temporary / "ipython"),
                        "PYTHONPATH": str(PROJECT / "src")})
            client = NotebookClient(notebook, km=manager, timeout=180,
                                    resources={"metadata": {"path": str(PROJECT)}}, store_widget_state=False)
            client.execute(cleanup_kc=True, env=env)
            nbformat.validate(notebook)
            code = [cell for cell in notebook.cells if cell.cell_type == "code"]
            errors = [output for cell in code for output in cell.outputs if output.output_type == "error"]
            images = sum("image/png" in output.get("data", {}) for cell in code for output in cell.outputs)
            if errors or any(cell.execution_count is None for cell in code) or images < 3:
                raise RuntimeError("Notebook did not execute all code cells and produce three plots")
            nbformat.write(notebook, target)
            (PROJECT / "results/failed_notebook.ipynb").unlink(missing_ok=True)
            result.update({"successful": True, "executed_code_cells": len(code),
                           "image_outputs": images, "error_outputs": len(errors), "kernel_argv": argv,
                           "kernel_cleanup": True})
    except Exception as exc:
        nbformat.write(notebook, PROJECT / "results/failed_notebook.ipynb")
        result["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        result["elapsed_seconds"] = time.monotonic() - tick
        (PROJECT / "results/notebook_execution.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    subprocess.run([str(python), str(PROJECT / "scripts/render_report.py")], cwd=PROJECT, check=True)


if __name__ == "__main__":
    main()
