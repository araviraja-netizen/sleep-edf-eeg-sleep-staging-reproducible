"""Saved-artifact analysis only; does not import a trainer or execute a model."""
from pathlib import Path
import argparse,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--export',type=Path);p.add_argument('--workspace',type=Path)
 p.add_argument('--final-statistics',action='store_true');p.add_argument('--final-evaluation',action='store_true');p.add_argument('--output',type=Path,required=True)
 a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
 if a.final_statistics:
  if not a.final_evaluation or a.workspace is None:raise PermissionError('Explicit completed final-evaluation analysis and workspace required')
  from sleep_research.analysis import load_export
  from sleep_research.phase5_statistics import subject_bootstrap
  from sleep_research.common import fingerprint
  body=json.loads((a.workspace/'results/portable_final_plan.json').read_text())
  record=json.loads((a.workspace/'results/portable_final_exports.json').read_text())
  if record['plan_sha256']!=fingerprint(body):raise ValueError('Final export binding differs')
  frames={k:load_export(a.workspace/v,final_evaluation=True)[1] for k,v in record['exports'].items()}
  groups={k+'_fixed_seeds':v for k,v in body['primary_groups'].items()}
  pairs=[['M5_fixed_seeds','M1_fixed_seeds'],['M7_fixed_seeds','M1_fixed_seeds'],['M7_fixed_seeds','M5_fixed_seeds'],['M4','M1'],['m5_wce_s42','m5_ce_s42']]
  result=subject_bootstrap(frames,groups,pairs,**body['statistics'])
  (a.output/'subject_bootstrap.json').write_text(json.dumps(result,indent=2)+'\n');return
 if a.export is None:raise ValueError('--export is required')
 from sleep_research.analysis import analyze_export
 result=analyze_export(a.export,output=a.output,final_evaluation=a.final_evaluation,tsne_max_samples=500,tsne_perplexity=30,tsne_iterations=500,seed=42)
 print(result)

if __name__=='__main__':main()
