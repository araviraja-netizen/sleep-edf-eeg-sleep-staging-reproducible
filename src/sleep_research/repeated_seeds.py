"""Independent-seed summaries from verified future full-run exports only."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .analysis import load_export
from .common import fingerprint
from .metrics import classification_metrics

METRICS=['macro_f1','n1_f1','n3_f1','accuracy','balanced_accuracy','cohen_kappa']


def summarize_seeds(frame):
    rows=[]
    for group,population in frame.groupby('configuration_group',sort=True):
        if population.seed.duplicated().any():
            raise ValueError('Duplicate seed is not an independent repeat; select one registered run per seed')
        row=dict(configuration_group=group,seeds=','.join(map(str,sorted(population.seed))),
                 independent_seed_count=len(population),at_least_three_seeds=len(population)>=3)
        for key in METRICS:
            values=population[key].to_numpy(dtype=float)
            if not np.isfinite(values).all():raise ValueError('Nonfinite repeated-run metric')
            row.update({key+'_mean':float(values.mean()),key+'_sample_sd':float(values.std(ddof=1)) if len(values)>1 else None,
                        key+'_min':float(values.min()),key+'_max':float(values.max())})
        rows.append(row)
    return pd.DataFrame(rows,columns=['configuration_group','seeds','independent_seed_count','at_least_three_seeds']+
                         [key+'_'+stat for key in METRICS for stat in ['mean','sample_sd','min','max']])


def summarize_project(project):
    project=Path(project);rows=[]
    for path in sorted((project/'experiments/cnn').glob('phase3b-*/performance/phase3b_result.json')):
        result=json.loads(path.read_text())
        if result['stage']!='full':continue
        manifest,frame,_,root=load_export(project/result['export'])
        if manifest['synthetic'] or manifest['splits']!=['validation'] or len(frame)!=30266:
            raise ValueError('Repeated full comparisons require complete real validation exports')
        experiment=json.loads((root/'reproducibility/experiment.json').read_text());config=experiment['configuration']
        group=fingerprint(dict(config={k:v for k,v in config.items() if k not in ('seed','run_id')},
                               dataset_contract=experiment['dataset_contract'],source=experiment['source_revision']))
        metrics=classification_metrics(frame.true_label.to_numpy(),frame[[f'probability_{i}' for i in range(5)]].to_numpy())
        rows.append(dict(configuration_group=group,seed=config['seed'],run_id=result['run_id'],
                         optimizer=config['optimizer']['name'],loss=config['loss']['name'],
                         macro_f1=metrics['macro_f1'],n1_f1=metrics['classes']['N1']['f1'],n3_f1=metrics['classes']['N3']['f1'],
                         accuracy=metrics['accuracy'],balanced_accuracy=metrics['balanced_accuracy'],cohen_kappa=metrics['cohen_kappa']))
    frame=pd.DataFrame(rows,columns=['configuration_group','seed','run_id','optimizer','loss']+METRICS)
    summary=summarize_seeds(frame)
    frame.to_csv(project/'results/phase3b_full_validation_comparison.csv',index=False)
    summary.to_csv(project/'results/phase3b_repeated_seed_summary.csv',index=False)
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project',type=Path,default=Path(__file__).resolve().parents[2]);args=parser.parse_args()
    summary=summarize_project(args.project)
    print('Measured full configuration groups:',len(summary),'; empty output means no full repeats were run.')
