"""Explicit five-class metrics and paired subject-cluster uncertainty; no trainer."""
import numpy as np
from .metrics import classification_metrics,binary_curves,confusion_matrix,CLASS_NAMES


def from_confusion(matrix):
    a=np.asarray(matrix,dtype=float)
    if a.shape!=(5,5) or not np.isfinite(a).all() or np.any(a<0) or a.sum()<=0:raise ValueError('Nonempty finite 5x5 counts required')
    support=a.sum(1);predicted=a.sum(0);tp=np.diag(a)
    f1=np.divide(2*tp,support+predicted,out=np.zeros(5),where=support+predicted>0)
    recall=np.divide(tp,support,out=np.zeros(5),where=support>0)
    accuracy=tp.sum()/a.sum();chance=np.dot(support,predicted)/a.sum()**2
    return dict(accuracy=float(accuracy),macro_f1=float(f1.mean()),n1_f1=float(f1[1]),n3_f1=float(f1[3]),
                balanced_accuracy=float(recall[support>0].mean()),cohen_kappa=float((accuracy-chance)/(1-chance)) if chance<1 else None)


def detailed_metrics(frame):
    labels=frame.true_label.to_numpy(dtype=np.int64);probs=frame[[f'probability_{i}' for i in range(5)]].to_numpy()
    result=classification_metrics(labels,probs);result['micro_f1']=result['accuracy']
    matrix=np.asarray(result['confusion_matrix']);n=matrix.sum();curves={}
    for i,name in enumerate(CLASS_NAMES):
        tp=matrix[i,i];fn=matrix[i].sum()-tp;fp=matrix[:,i].sum()-tp;tn=n-tp-fn-fp
        c=binary_curves(labels==i,probs[:,i]);curves[name]=c
        result['classes'][name].update(specificity=float(tn/(tn+fp)) if tn+fp else None,negative_support=int(tn+fp),
            roc_auc=c['roc_auc'],average_precision=c['average_precision'],predicted_support=int(matrix[:,i].sum()))
    result['conventions'].update(specificity='TN/(TN+FP) one-vs-rest; null with no negatives',
        micro_f1='Single-label multiclass micro-F1 equals accuracy; denominator all targets',
        auc_ap='One-vs-rest; ROC trapezoids; AP step integral sum(delta recall * precision); null without positive or negative examples',
        calibration='10 equal-width top-label confidence bins, [lo,hi), final bin includes 1; ECE=sum(n_bin/N*abs(acc-conf)); Brier=mean sum of five squared errors (range0..2)')
    return result,curves


def cluster_matrices(frames):
    first=next(iter(frames.values()));subjects=sorted(first.subject_id.astype(str).unique());ids=first.epoch_id.tolist()
    result={}
    for key,frame in frames.items():
        if frame.epoch_id.tolist()!=ids or frame.subject_id.astype(str).tolist()!=first.subject_id.astype(str).tolist() or frame.true_label.tolist()!=first.true_label.tolist():
            raise ValueError('Paired bootstrap requires identical ordered targets/labels/subjects')
        result[key]=np.stack([confusion_matrix(g.true_label.to_numpy(dtype=int),g.predicted_label.to_numpy(dtype=int)) for s in subjects for g in [frame[frame.subject_id.astype(str)==s]]])
    return subjects,result


def subject_bootstrap(frames,groups,comparisons,resamples=10000,seed=20261016,confidence=.95,simultaneous_confidence=1-.05/3):
    if resamples<2 or not 0<confidence<1 or not 0<simultaneous_confidence<1:raise ValueError('Invalid resampling design')
    if any(not keys or len(keys)!=len(set(keys)) or set(keys)-set(frames) for keys in groups.values()):raise ValueError('Invalid fixed seed groups')
    subjects,matrices=cluster_matrices(frames)
    if len(subjects)<2:raise ValueError('At least two subjects required')
    rng=np.random.default_rng(seed);weights=rng.multinomial(len(subjects),np.full(len(subjects),1/len(subjects)),size=resamples)
    stats=list(from_confusion(next(iter(matrices.values())).sum(0)));samples={};point={}
    for key,m in matrices.items():
        point[key]=from_confusion(m.sum(0));samples[key]={k:np.full(resamples,np.nan) for k in stats}
        for b in range(resamples):
            values=from_confusion(np.tensordot(weights[b],m,axes=(0,0)))
            for metric,value in values.items():samples[key][metric][b]=np.nan if value is None else value
    for group,keys in groups.items():
        if group in samples:raise ValueError('Group names must differ from individual run keys')
        point[group]={k:float(np.mean([point[x][k] for x in keys])) if all(point[x][k] is not None for x in keys) else None for k in stats}
        samples[group]={k:np.mean([samples[x][k] for x in keys],axis=0) for k in stats}
    def interval(x,level):
        valid=x[np.isfinite(x)];q=(1-level)/2
        return dict(lower=float(np.quantile(valid,q)) if len(valid) else None,upper=float(np.quantile(valid,1-q)) if len(valid) else None,defined_replicates=len(valid))
    estimates={key:{k:dict(estimate=point[key][k],**interval(x,confidence)) for k,x in metrics.items()} for key,metrics in samples.items()}
    differences=[]
    for a,b in comparisons:
        for metric in stats:
            x=samples[a][metric]-samples[b][metric]
            differences.append(dict(a=a,b=b,metric=metric,estimate=None if point[a][metric] is None or point[b][metric] is None else point[a][metric]-point[b][metric],
                percentile_interval=interval(x,confidence),simultaneous_interval=interval(x,simultaneous_confidence)))
    return dict(subjects=subjects,subject_count=len(subjects),resamples=resamples,seed=seed,confidence=confidence,
        simultaneous_confidence=simultaneous_confidence,estimates=estimates,paired_differences=differences,
        method='Paired subject cluster percentile bootstrap; full nights/epochs retained via confusion-count weights. Fixed seed groups average metrics, never probabilities. Intervals conditional on trained seeds; no significance claim.',
        bootstrap_weights_sha256=__import__('hashlib').sha256(weights.tobytes()).hexdigest())
