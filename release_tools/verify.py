"""Independent count-based verification; NumPy + standard library, no EEG/PyTorch.

Recounts the registered published outcomes, never performs inference or selection.
"""
from pathlib import Path
import argparse, csv, hashlib, json, math
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
def load(path):return json.loads(Path(path).read_text())
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def dump(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def rows(path):
 with Path(path).open(newline='') as f:return list(csv.DictReader(f))
def ratios(a,b):return np.divide(a,b,out=np.zeros_like(a,dtype=float),where=b!=0)
def metrics(matrix):
 a=np.asarray(matrix,dtype=float)
 if a.shape[-2:]!=(5,5) or np.any(a<0) or not np.isfinite(a).all() or np.any(a.sum(axis=(-2,-1))<=0):raise ValueError('Require nonempty finite 5x5 confusion counts')
 support=a.sum(-1);pred=a.sum(-2);tp=np.diagonal(a,axis1=-2,axis2=-1)
 precision=ratios(tp,pred);recall=ratios(tp,support);f1=ratios(2*tp,support+pred)
 total=support.sum(-1);accuracy=tp.sum(-1)/total;chance=(support*pred).sum(-1)/total**2
 kappa=np.divide(accuracy-chance,1-chance,out=np.full_like(accuracy,np.nan),where=chance<1)
 values=dict(accuracy=accuracy,macro_f1=f1.mean(-1),n1_f1=f1[...,1],n3_f1=f1[...,3],
     balanced_accuracy=recall.sum(-1)/(support>0).sum(-1),cohen_kappa=kappa)
 return values,dict(precision=precision,recall=recall,f1=f1,support=support)
def close(a,b):
 if a is None:return b is None or (isinstance(b,float) and math.isnan(b))
 return math.isclose(float(a),float(b),rel_tol=2e-12,abs_tol=2e-12)
def assert_same(a,b,where='value'):
 if not close(a,b):raise ValueError(f'Numerical mismatch at {where}: expected {a}, calculated {b}')
def interval(values,confidence):
 values=np.asarray(values);valid=values[np.isfinite(values)];q=(1-confidence)/2
 return dict(lower=float(np.quantile(valid,q)) if len(valid) else None,upper=float(np.quantile(valid,1-q)) if len(valid) else None,defined_replicates=len(valid))

def verify(root=ROOT,output=None,scope="auto"):
 if scope not in {"auto","full","aggregate"}:raise ValueError("Unknown verification scope")
 root=Path(root);output=Path(output or root/'verification_outputs');output.mkdir(parents=True,exist_ok=True)
 projection=load(root/'provenance/frozen_evaluation_projection.json')
 aggregate=load(root/'evidence/aggregate_confusions.json')
 cluster_path=root/'evidence/subject_confusions.json'
 bootstrap_available=cluster_path.is_file() and (root/'evidence/expected_bootstrap.json').is_file()
 split_names=['subject_split_freeze.json','subject_splits.csv','recording_splits.csv']
 split_present=[(root/'evidence/splits'/name).is_file() for name in split_names]
 if any(split_present) and not all(split_present):raise ValueError('Incomplete frozen split bundle; withhold or supply all three files')
 if scope=='full' and (not bootstrap_available or not all(split_present)):
  raise FileNotFoundError('Full verification requires approved subject counts, bootstrap references and complete split manifests; use --scope aggregate for pooled metrics only')
 bootstrap_verified=scope!='aggregate' and bootstrap_available
 clusters=load(cluster_path) if bootstrap_verified else None
 expected=load(root/'evidence/expected_bootstrap.json') if bootstrap_verified else None
 groups={key+'_fixed_seeds':value for key,value in projection['primary_groups'].items()}
 model_names={'M1':'CNN','M5':'CNN-BiLSTM','M7':'CNN-Transformer'}
 labels=aggregate['classes'];point={};class_scores={};samples={};verified=0
 for key,matrix in aggregate['models'].items():
  if clusters is not None and not np.array_equal(np.asarray(clusters['models'][key]).sum(0),matrix):raise ValueError('Subject/aggregate counts disagree: '+key)
  values,stages=metrics(matrix);point[key]={m:float(v) for m,v in values.items()};class_scores[key]=stages
  if int(np.sum(matrix))!=31156:raise ValueError('Target count differs')
 for group,keys in groups.items():
  point[group]={m:float(np.mean([point[k][m] for k in keys])) for m in point[keys[0]]}
 for r in rows(root/'results/publication_tables/10_all_seed_test_metrics.csv'):
  for metric in point[r['key']]:assert_same(r[metric],point[r['key']][metric],r['key']+'/'+metric);verified+=1
 for mid,keys in projection['primary_groups'].items():
  saved=next(r for r in rows(root/'results/publication_tables/03_primary_test_summary.csv') if r['model']==model_names[mid])
  if len(keys)!=int(saved['seeds']):raise ValueError('Seed replication mismatch')
  for metric in point[keys[0]]:
   values=[point[k][metric] for k in keys]
   assert_same(saved[metric+'_mean'],np.mean(values),mid+'/'+metric);verified+=1
   if len(keys)>1:assert_same(saved[metric+'_seed_sd'],np.std(values,ddof=1),mid+'/SD/'+metric);verified+=1
   elif saved[metric+'_seed_sd']:raise ValueError('One-seed SD must be undefined')
  for r in [r for r in rows(root/'results/publication_tables/04_primary_classwise.csv') if r['model']==model_names[mid]]:
   i=labels.index(r['stage'])
   for metric in ['precision','recall','f1']:
    assert_same(r[metric],np.mean([class_scores[k][metric][i] for k in keys]),mid+'/'+r['stage']+'/'+metric);verified+=1
   assert_same(r['support_per_seed'],class_scores[keys[0]]['support'][i]);verified+=1
 for mid,keys in projection['primary_groups'].items():
  for subset in ['transition','stable']:
   scores=[metrics(aggregate['subsets'][key][subset])[0] for key in keys]
   row=next(r for r in rows(root/'results/publication_tables/08_transition_primary.csv') if r['model']==model_names[mid] and r['subset']==subset)
   for metric in ['accuracy','macro_f1']:assert_same(row[metric],np.mean([s[metric] for s in scores]));verified+=1
   assert_same(row['epochs_per_seed'],np.sum(aggregate['subsets'][keys[0]][subset]));verified+=1
 comparisons=[]
 if bootstrap_verified:
  # Same registered cluster order, count weights, fixed seeds and quantile definitions.
  if len(set(clusters['cluster_aliases']))!=12:raise ValueError('Expected twelve distinct reviewed cluster aliases')
  settings=projection['statistics'];count=12
  weights=np.random.default_rng(expected['seed']).multinomial(count,np.full(count,1/count),size=expected['resamples'])
  if hashlib.sha256(weights.tobytes()).hexdigest()!=expected['bootstrap_weights_sha256']:raise ValueError('Registered bootstrap draws differ; use the pinned NumPy version')
  for key,m in clusters['models'].items():
   pooled=np.einsum('bs,sij->bij',weights,np.asarray(m,dtype=np.int64));samples[key]=metrics(pooled)[0]
  for group,keys in groups.items():samples[group]={m:np.mean([samples[k][m] for k in keys],axis=0) for m in samples[keys[0]]}
  for key,metrics_expected in expected['estimates'].items():
   for metric,e in metrics_expected.items():
    assert_same(e['estimate'],point[key][metric],key+'/'+metric);ci=interval(samples[key][metric],expected['confidence'])
    for part in ci:assert_same(e[part],ci[part],key+'/'+metric+'/'+part);verified+=1
  for e in expected['paired_differences']:
   a,b,m=e['a'],e['b'],e['metric'];estimate=point[a][m]-point[b][m]
   ci=interval(samples[a][m]-samples[b][m],expected['confidence']);wide=interval(samples[a][m]-samples[b][m],expected['simultaneous_confidence'])
   assert_same(e['estimate'],estimate,a+' minus '+b)
   for key,v in ci.items():assert_same(e['percentile_interval'][key],v);verified+=1
   for key,v in wide.items():assert_same(e['simultaneous_interval'][key],v);verified+=1
   comparisons.append(dict(a=a,b=b,metric=m,estimate=estimate,lower95=ci['lower'],upper95=ci['upper'],lower_simultaneous=wide['lower'],upper_simultaneous=wide['upper']))
  for r in rows(root/'results/publication_tables/05_paired_comparisons.csv'):
   matched=next(c for c in comparisons if c['a']==r['a'] and c['b']==r['b'] and c['metric']==r['metric'])
   for key in ['estimate','lower95','upper95','lower_simultaneous','upper_simultaneous']:
    if r[key]:assert_same(r[key],matched[key]);verified+=1
 split_verified=all(split_present)
 if split_verified:
  freeze=load(root/'evidence/splits/subject_split_freeze.json')
  if digest(root/'evidence/splits/subject_splits.csv')!=freeze['subject_manifest_sha256'] or digest(root/'evidence/splits/recording_splits.csv')!=freeze['recording_manifest_sha256']:raise ValueError('Frozen split checksum differs')
  allocations=rows(root/'evidence/splits/subject_splits.csv');membership={r['subject_id']:r['split'] for r in allocations}
  if len(membership)!=78 or [sum(r['split']==s for r in allocations) for s in ['train','validation','test']]!=[54,12,12]:raise ValueError('Frozen subjects/allocations differ')
  for r in rows(root/'evidence/splits/recording_splits.csv'):
   if membership[r['subject_id']]!=r['split']:raise ValueError('Night/subject leakage')
 # Regenerate machine-readable reviewer outputs; never overwrite provided evidence.
 dump(output/'recalculated_metrics.json',point)
 dump(output/'recalculated_classwise.json',{k:{n:v.tolist() for n,v in vals.items()} for k,vals in class_scores.items()})
 dump(output/'recalculated_confusions.json',dict(aggregate,
     row_normalized={k:ratios(np.asarray(v,dtype=float),np.asarray(v).sum(1,keepdims=True)).tolist() for k,v in aggregate['models'].items()}))
 if comparisons:
  with (output/'recalculated_paired_comparisons.csv').open('w',newline='') as f:
   w=csv.DictWriter(f,fieldnames=list(comparisons[0]));w.writeheader();w.writerows(comparisons)
 else:
  (output/'recalculated_paired_comparisons.csv').unlink(missing_ok=True)
 report=dict(successful=True,numeric_checks=verified,models=len(aggregate['models']),target_epochs_per_checkpoint=31156,
     scope_requested=scope,subject_bootstrap_verified=bootstrap_verified,frozen_split_verified=split_verified,
     independent_subject_clusters=12 if bootstrap_verified else None,bootstrap_resamples=expected['resamples'] if bootstrap_verified else None,bootstrap_weights_sha256=expected['bootstrap_weights_sha256'] if bootstrap_verified else None,
     dataset_download_required=False,pytorch_required=False,training_called=False,inference_called=False,
     privacy_review_required=True,coverage=['accuracy','five-class macro-F1','all class precision/recall/F1','balanced accuracy','kappa',
     'count and normalized confusion matrices','all fixed seeds and sample SD','transition/stable outcomes']
     +(['paired subject-bootstrap intervals'] if bootstrap_verified else [])+(['frozen split integrity'] if split_verified else []),
     unavailable=['ROC/PR/AP/Brier/ECE require approved probabilities','EEG/hypnograms/embeddings/saliency/attention require excluded artifacts']
     +([] if bootstrap_verified else ['Subject-paired confidence intervals are reported historical results, not independently reconstructed in this scope'])
     +([] if split_verified else ['Exact frozen allocation and night grouping cannot be verified without approved split manifests']))
 dump(output/'verification.json',report);return report

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--output',type=Path);p.add_argument('--scope',choices=['auto','full','aggregate'],default='auto')
 args=p.parse_args();print(json.dumps(verify(args.root,args.output,args.scope),indent=2))
