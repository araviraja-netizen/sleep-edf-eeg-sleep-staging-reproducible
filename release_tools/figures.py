"""Redraw aggregate figures from approved saved counts; no EEG, inference or training."""
from pathlib import Path
import argparse,json,sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from verify import verify,load,metrics,ROOT

def main(output):
 output=Path(output);report=verify(ROOT,output);point=load(output/'recalculated_metrics.json')
 projection=load(ROOT/'provenance/frozen_evaluation_projection.json');aggregates=load(ROOT/'evidence/aggregate_confusions.json')
 names=['M1','M5','M7'];colors=['#47759c','#c78a36','#358b73']
 plt.rcParams.update({'font.size':11,'pdf.fonttype':42})
 def save(fig,name):
  fig.savefig(output/(name+'.png'),dpi=300,bbox_inches='tight');fig.savefig(output/(name+'.pdf'),bbox_inches='tight');plt.close(fig)
 fig,ax=plt.subplots(figsize=(7,4))
 for i,mid in enumerate(names):
  keys=projection['primary_groups'][mid];vals=[point[k]['macro_f1'] for k in keys]
  ax.scatter(np.full(len(vals),i),vals,color=colors[i]);ax.hlines(np.mean(vals),i-.15,i+.15,color=colors[i])
 ax.set(xticks=range(3),xticklabels=names,ylabel='Final-test macro-F1',title='All predefined fitted seeds; line = mean');save(fig,'primary_fixed_seeds')
 fig,axes=plt.subplots(1,3,figsize=(12,4),constrained_layout=True)
 for i,mid in enumerate(names):
  keys=projection['primary_groups'][mid];a=np.mean([metrics(aggregates['models'][key])[1]['f1'] for key in keys],axis=0)
  axes[i].bar(aggregates['classes'],a,color=colors[i]);axes[i].set(title=mid,ylabel='Class F1',ylim=(0,1))
 save(fig,'class_f1')
 fig,ax=plt.subplots(figsize=(7,4))
 for i,mid in enumerate(names):
  keys=projection['primary_groups'][mid];v=[np.mean([metrics(aggregates['subsets'][k][s])[0]['macro_f1'] for k in keys]) for s in ['transition','stable']]
  ax.bar(np.arange(2)+(i-1)*.23,v,.23,label=mid,color=colors[i])
 ax.set(xticks=range(2),xticklabels=['Transition','Stable'],ylabel='Descriptive subset macro-F1',ylim=(0,1));ax.legend();save(fig,'transition_stable')
 print(json.dumps({'successful':True,'new_data_generated':False,'output':str(output),'numeric_checks':report['numeric_checks']}))

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);main(p.parse_args().output)
