"""Execute Phase 2 using a temporary SSD-local kernel spec and the existing Python."""

import json
import os
from pathlib import Path
import sys
import tempfile
import time

import nbformat
from nbclient import NotebookClient
from jupyter_client import AsyncKernelManager
from jupyter_client.kernelspec import KernelSpecManager

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
from sleep_edf.phase2_utils import load_config, atomic_json, timestamp
from sleep_edf.phase2 import preserve_phase1

if __name__ == "__main__":
    config = load_config(PROJECT)[0]
    target = PROJECT / "notebooks/02_eeg_preprocessing.ipynb"
    notebook = nbformat.read(target, as_version=4)
    result = {"successful": False, "executed_at": timestamp(), "python_executable": sys.executable}
    tick = time.monotonic()
    try:
        with tempfile.TemporaryDirectory(prefix="phase2-kernel-", dir=PROJECT / "results") as tmp:
            temporary = Path(tmp)
            directory = temporary / "kernels/sleep-edf-phase2"
            directory.mkdir(parents=True)
            argv = [config["python_executable"], "-m", "ipykernel_launcher", "-f", "{connection_file}"]
            (directory / "kernel.json").write_text(json.dumps({"argv": argv, "display_name": "Python (existing dl-m4)", "language": "python"}))
            manager = AsyncKernelManager(kernel_name="sleep-edf-phase2", kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(temporary / "kernels")]))
            env = os.environ.copy()
            env.update({"MPLCONFIGDIR": str(temporary / "matplotlib"), "JUPYTER_RUNTIME_DIR": str(temporary / "runtime"),
                        "IPYTHONDIR": str(temporary / "ipython"), "PYTHONPATH": str(PROJECT / "src")})
            client = NotebookClient(notebook, km=manager, timeout=240, resources={"metadata": {"path": str(PROJECT)}}, store_widget_state=False)
            client.execute(cleanup_kc=True, env=env)
            nbformat.validate(notebook)
            code = [c for c in notebook.cells if c.cell_type == "code"]
            images = sum("image/png" in o.get("data", {}) for c in code for o in c.outputs)
            errors = sum(o.output_type == "error" for c in code for o in c.outputs)
            if errors or images < 6 or any(c.execution_count is None for c in code):
                raise RuntimeError("Incomplete Phase 2 notebook execution/plots")
            nbformat.write(notebook, target)
            (PROJECT / "results/phase2_failed_notebook.ipynb").unlink(missing_ok=True)
            result.update(successful=True, executed_code_cells=len(code), image_outputs=images, error_outputs=errors,
                          kernel_argv=argv, kernel_cleanup=True)
    except Exception as exc:
        nbformat.write(notebook, PROJECT / "results/phase2_failed_notebook.ipynb")
        result["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        result["elapsed_seconds"] = time.monotonic() - tick
        atomic_json(PROJECT / "results/phase2_notebook_execution.json", result)
    preserve_phase1(PROJECT)
    print(json.dumps(result, indent=2))
