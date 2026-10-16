"""Create/execute only the lightweight reviewer notebook with the selected Python."""
from pathlib import Path
import argparse,json,os,sys,tempfile
import nbformat
from nbclient import NotebookClient
from jupyter_client import AsyncKernelManager
from jupyter_client.kernelspec import KernelSpecManager
ROOT=Path(__file__).resolve().parents[1]

def build():
 nb=nbformat.v4.new_notebook(cells=[
  nbformat.v4.new_markdown_cell('# Reviewer verification: saved Sleep-EDF outcomes\n\nThis notebook recounts approved saved confusion evidence, not EEG or model inference. The author approved inclusion of the cluster counts and frozen splits. Cluster aliases remain pseudonyms, not guaranteed anonymous identifiers. See docs/REVIEWER_VERIFICATION.md.'),
  nbformat.v4.new_code_cell('from pathlib import Path\nimport sys, json\nROOT = next(p for p in [Path.cwd(), Path.cwd().parent, Path.cwd().parent.parent] if (p / "release_tools/verify.py").is_file())\nsys.path.insert(0, str(ROOT / "release_tools"))\nfrom verify import verify\n'),
  nbformat.v4.new_code_cell('result = verify(ROOT, ROOT / "verification_outputs/notebook")\nprint(json.dumps(result, indent=2))'),
  nbformat.v4.new_markdown_cell('All predefined seeds are retained. Seed SD is separate from subject uncertainty. Context/capacity/replication differ across M1 and M5/M7. Higher aggregate performance coexists with lower transition-target macro-F1; no causal mechanism is established.'),
  nbformat.v4.new_code_cell('from IPython.display import Image, display\ndisplay(Image(filename=str(ROOT / "results/publication_figures/16_transition_errors.png")))\nprint((ROOT / "results/publication_tables/03a_primary_display.md").read_text())'),
  nbformat.v4.new_markdown_cell('The original test cohort is consumed. This notebook does not select a seed/model, fit a threshold, recalculate feature projections or perform training. Probability-based and physiological analyses require separately approved additional inputs.')])
 nb.metadata['kernelspec']={'name':'python3','display_name':'Python 3','language':'python'}
 return nb

def main(output):
 output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True);nb=build()
 with tempfile.TemporaryDirectory(prefix='reviewer-kernel-',dir=output) as t:
  temp=Path(t);kernel=temp/'kernels/reviewer';kernel.mkdir(parents=True)
  (kernel/'kernel.json').write_text(json.dumps(dict(argv=[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}'],display_name='Selected research Python',language='python')))
  manager=AsyncKernelManager(kernel_name='reviewer',kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(temp/'kernels')]))
  env=dict(os.environ,MPLCONFIGDIR=str(temp/'matplotlib'),JUPYTER_RUNTIME_DIR=str(temp/'runtime'),IPYTHONDIR=str(temp/'ipython'))
  client=NotebookClient(nb,km=manager,timeout=120,resources={'metadata':{'path':str(ROOT)}},store_widget_state=False)
  client.execute(cleanup_kc=True,env=env)
  code=[c for c in nb.cells if c.cell_type=='code'];errors=sum(o.output_type=='error' for c in code for o in c.outputs)
  images=sum('image/png' in o.get('data',{}) for c in code for o in c.outputs)
  if errors or images<1 or any(c.execution_count is None for c in code):raise ValueError('Incomplete notebook execution')
  nbformat.validate(nb);nbformat.write(nb,ROOT/'notebooks/00_reviewer_verification.ipynb')
  proof=dict(successful=True,executed_code_cells=len(code),image_outputs=images,error_outputs=errors,
      training_called=False,inference_called=False,raw_data_read=False,python_version=sys.version.split()[0],kernel_cleanup=True)
  (output/'notebook_execution.json').write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps(proof,indent=2))

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);main(p.parse_args().output)
