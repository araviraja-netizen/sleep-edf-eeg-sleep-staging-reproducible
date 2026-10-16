"""Phase5 commands: prepare/review/freeze; explicit Terminal test run; analysis only."""
import argparse
import json
from pathlib import Path
from .common import fingerprint,json_write
from .phase5_registry import PROJECT,environment,prepare,plan,audit,freeze


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['prepare','validate','review','freeze','run','status','analyze','optional-validation-definitions','optional-validation-run'])
    parser.add_argument('--project',type=Path,default=PROJECT)
    parser.add_argument('--reviewed-plan-sha256');parser.add_argument('--plan-sha256')
    parser.add_argument('--authorize-final-evaluation',action='store_true')
    parser.add_argument('--authorize-additional-validation-training',action='store_true')
    args=parser.parse_args();environment()
    if args.command=='prepare':
        body=prepare(args.project)
        from .phase5_publication import preparation_reports
        preparation_reports(args.project,body)
        result=dict(plan_id=body['plan_id'],plan_sha256=fingerprint(body),frozen=False,test_predictions_computed=False)
    elif args.command=='validate':
        result=audit(args.project);body=plan(args.project);result['selected_checkpoints_verified']=len(body['models']);result['plan_sha256']=fingerprint(body)
        json_write(args.project/'results/phase5_project_validation.json',result)
    elif args.command=='review':
        body=plan(args.project);print((args.project/'results/phase5_preregistered_evaluation_plan.md').read_text())
        result=dict(plan_id=body['plan_id'],plan_sha256=fingerprint(body),frozen=(args.project/'results/phase5_final_freeze.json').exists(),test_predictions_computed=False)
    elif args.command=='freeze':result=freeze(args.project,args.reviewed_plan_sha256)
    elif args.command=='run':
        from .phase5_evaluation import run
        result=run(args.project,args.authorize_final_evaluation,args.plan_sha256)
    elif args.command=='analyze':
        from .phase5_analysis import regenerate
        result=regenerate(args.project,args.authorize_final_evaluation,args.plan_sha256)
    elif args.command=='status':
        body=plan(args.project)
        from .phase5_evaluation import workspace
        path=workspace(args.project,body)/'status.json'
        result=json.loads(path.read_text()) if path.exists() else dict(status='prepared_unlaunched',plan_id=body['plan_id'],frozen=(args.project/'results/phase5_final_freeze.json').exists(),log=str(workspace(args.project,body)/'evaluation.log'))
    elif args.command=='optional-validation-definitions':
        from .phase5_training import definitions
        result=definitions(args.project)
    else:
        from .phase5_training import train
        result=train(args.project,args.authorize_additional_validation_training)
    if args.command in ['run','analyze']:
        import subprocess,sys
        subprocess.run([sys.executable,str(args.project/'scripts/validate_phase5_notebook.py'),'--show-authorized-final-results','--plan-sha256',args.plan_sha256],check=True)
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
