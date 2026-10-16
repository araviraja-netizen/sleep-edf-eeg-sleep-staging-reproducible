"""Explicit five-class loss reductions and optimizer factories for Phase 3B."""

import inspect
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .common import fingerprint


def class_weights(counts, kind='inverse_frequency', beta=0.9999):
    counts=np.asarray(counts,dtype=np.float64)
    if counts.shape!=(5,) or not np.isfinite(counts).all() or (counts<=0).any() or (counts!=np.floor(counts)).any():
        raise ValueError('Five positive finite integer training counts required')
    if kind=='inverse_frequency':
        values=counts.sum()/(5*counts)
    elif kind=='effective_number':
        if not math.isfinite(beta) or not 0<=beta<1:
            raise ValueError('Effective-number beta must be in [0,1)')
        values=np.ones(5) if beta==0 else (1-beta)/(-np.expm1(counts*np.log(beta)))
        values=5*values/values.sum()
    else:
        raise ValueError('Unknown class weighting rule')
    return values.astype(np.float32)


class ResearchLoss(nn.Module):
    """CE weighted mean; focal arithmetic mean of alpha_y*(1-p_y)^gamma*(-log p_y).

    Focal p_y always comes from unweighted log-softmax, not exp(-weighted CE).
    CE denominator is sum alpha_y; focal denominator is N. No validation-derived weights.
    """
    def __init__(self, specification, training_counts):
        super().__init__()
        self.specification=dict(specification)
        self.name=specification['name']
        if self.name not in ('cross_entropy','weighted_cross_entropy','focal','class_balanced_focal'):
            raise ValueError('Unknown loss')
        class_weights(training_counts)  # Validate provenance-derived counts for every objective.
        self.gamma=float(specification.get('gamma',2.0))
        if not math.isfinite(self.gamma) or self.gamma<0:
            raise ValueError('Focal gamma must be finite and nonnegative')
        weights=np.ones(5,dtype=np.float32)
        if self.name=='weighted_cross_entropy':
            weights=class_weights(training_counts)
        if self.name=='class_balanced_focal':
            weights=class_weights(training_counts,'effective_number',float(specification.get('beta',0.9999)))
        self.register_buffer('alpha',torch.tensor(weights))

    def normalizer(self, labels):
        return self.alpha[labels].sum() if self.name=='weighted_cross_entropy' else labels.numel()

    def forward(self, logits, labels):
        if logits.ndim!=2 or logits.shape[1]!=5 or labels.ndim!=1 or len(labels)!=len(logits) or not len(labels):
            raise ValueError('Loss expects nonempty [N,5] logits and [N] labels')
        if labels.dtype!=torch.int64 or not bool(((labels>=0)&(labels<5)).all()):
            raise ValueError('Loss labels must be int64 in 0..4')
        if not bool(torch.isfinite(logits).all()):
            raise ValueError('Nonfinite logits')
        log_pt=F.log_softmax(logits,dim=1).gather(1,labels[:,None]).squeeze(1)
        per_sample=-log_pt
        if self.name in ('focal','class_balanced_focal') and self.gamma!=0:
            # -expm1(log p) is accurate near p=1; clamp protects roundoff.
            per_sample=per_sample*(-torch.expm1(log_pt)).clamp_min(torch.finfo(log_pt.dtype).tiny).pow(self.gamma)
        return (self.alpha[labels]*per_sample).sum()/self.normalizer(labels)


def objective_fingerprint(criterion):
    return fingerprint(dict(source=inspect.getsource(type(criterion)),description=repr(criterion),
            specification=criterion.specification,
            buffers={k:v.detach().cpu().tolist() for k,v in criterion.state_dict().items()}))


def build_optimizer(parameters, specification):
    spec=dict(specification);name=spec.pop('name');lr=float(spec.pop('learning_rate'))
    if not math.isfinite(lr) or lr<=0:
        raise ValueError('Positive finite learning rate required')
    constructors={'AdamW':torch.optim.AdamW,'Adam':torch.optim.Adam,
                  'SGD':torch.optim.SGD,'RMSprop':torch.optim.RMSprop}
    if name not in constructors:
        raise ValueError('Unknown optimizer')
    if name=='SGD' and not 0<float(spec.get('momentum',0))<1:
        raise ValueError('SGD comparison requires momentum in (0,1)')
    return constructors[name](parameters,lr=lr,**spec)
