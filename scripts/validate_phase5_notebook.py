"""Execute analysis-only notebook; opt-in gate for reading already completed test results."""
import argparse,json,os,sys,tempfile,time
from pathlib import Path
import nbformat
from nbclient import NotebookClient
from jupyter_client import AsyncKernelManager
from jupyter_client.kernelspec import KernelSpecManager
P=Path(__file__).resolve().parents[1];sys.path.insert(0,str(P/'src'))
from sleep_research.common import json_write,now
from sleep_research.phase5_registry import environment
if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--show-authorized-final-results',action='store_true');parser.add_argument('--plan-sha256');args=parser.parse_args()
    environment();tick=time.monotonic();nb=nbformat.read(P/'notebooks/05_final_research_analysis.ipynb',as_version=4)
    result=dict(successful=False,executed_at=now(),python_executable=sys.executable,training_called=False,inference_called=False,
        authorized_saved_final_results_read=args.show_authorized_final_results,real_test_inference_called=False)
    with tempfile.TemporaryDirectory(prefix='phase5-kernel-',dir=P/'results') as tmp:
        root=Path(tmp);kernel=root/'kernels/sleep-edf-phase5';kernel.mkdir(parents=True)
        argv=[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}']
        (kernel/'kernel.json').write_text(json.dumps(dict(argv=argv,display_name='Existing dl-m4',language='python')))
        manager=AsyncKernelManager(kernel_name='sleep-edf-phase5',kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(root/'kernels')]))
        env=dict(os.environ);env.update(MPLCONFIGDIR=str(root/'matplotlib'),JUPYTER_RUNTIME_DIR=str(root/'runtime'),IPYTHONDIR=str(root/'ipython'),PYTHONPATH=str(P/'src'))
        env.pop('SLEEP_PHASE5_SHOW_FINAL',None);env.pop('SLEEP_PHASE5_PLAN_SHA256',None)
        if args.show_authorized_final_results:env.update(SLEEP_PHASE5_SHOW_FINAL='explicitly-authorized',SLEEP_PHASE5_PLAN_SHA256=args.plan_sha256 or '')
        client=NotebookClient(nb,km=manager,timeout=240,resources={'metadata':{'path':str(P)}},store_widget_state=False)
        client.execute(cleanup_kc=True,env=env);nbformat.validate(nb)
        code=[c for c in nb.cells if c.cell_type=='code'];images=sum('image/png' in o.get('data',{}) for c in code for o in c.outputs);errors=sum(o.output_type=='error' for c in code for o in c.outputs)
        if errors or images<1 or any(c.execution_count is None for c in code):raise RuntimeError('Incomplete notebook execution')
        nbformat.write(nb,P/'notebooks/05_final_research_analysis.ipynb')
        result.update(successful=True,executed_code_cells=len(code),image_outputs=images,error_outputs=errors,elapsed_seconds=time.monotonic()-tick,kernel_cleanup=True)
    json_write(P/'results/phase5_notebook_execution.json',result);print(json.dumps(result,indent=2))
