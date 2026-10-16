"""Execute the synthetic infrastructure demo with the existing Python and local kernel."""

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

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT/'src'))
from sleep_research.common import json_write, now

if __name__ == "__main__":
    target=PROJECT/'notebooks/02_5_checkpoint_resume_demo.ipynb'
    notebook=nbformat.read(target,as_version=4)
    tick=time.monotonic();result={'successful':False,'executed_at':now(),'python_executable':sys.executable,'synthetic_only':True}
    if Path(sys.executable).resolve()!=Path('/Volumes/Mac/envs/dl-m4/bin/python').resolve():
        raise SystemExit('Use the specified existing Python interpreter')
    try:
        with tempfile.TemporaryDirectory(prefix='phase2_5-kernel-',dir=PROJECT/'results') as tmp:
            temporary=Path(tmp);directory=temporary/'kernels/sleep-edf-phase2-5';directory.mkdir(parents=True)
            argv=[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}']
            (directory/'kernel.json').write_text(json.dumps({'argv':argv,'display_name':'Python (existing dl-m4)','language':'python'}))
            manager=AsyncKernelManager(kernel_name='sleep-edf-phase2-5',kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(temporary/'kernels')]))
            env=dict(os.environ);env.update(MPLCONFIGDIR=str(temporary/'matplotlib'),JUPYTER_RUNTIME_DIR=str(temporary/'runtime'),
                                           IPYTHONDIR=str(temporary/'ipython'),PYTHONPATH=str(PROJECT/'src'))
            client=NotebookClient(notebook,km=manager,timeout=240,resources={'metadata':{'path':str(PROJECT)}},store_widget_state=False)
            client.execute(cleanup_kc=True,env=env)
            nbformat.validate(notebook)
            code=[c for c in notebook.cells if c.cell_type=='code']
            images=sum('image/png' in o.get('data',{}) for c in code for o in c.outputs)
            errors=sum(o.output_type=='error' for c in code for o in c.outputs)
            if errors or images<7 or any(c.execution_count is None for c in code):
                raise RuntimeError('Incomplete infrastructure notebook execution')
            nbformat.write(notebook,target)
            result.update(successful=True,executed_code_cells=len(code),image_outputs=images,error_outputs=errors,kernel_cleanup=True,kernel_argv=argv)
    except Exception as exc:
        nbformat.write(notebook,PROJECT/'results/phase2_5_failed_notebook.ipynb')
        result['error']=f'{type(exc).__name__}: {exc}'
        raise
    finally:
        result['elapsed_seconds']=time.monotonic()-tick
        json_write(PROJECT/'results/phase2_5_notebook_execution.json',result)
    print(json.dumps(result,indent=2))
