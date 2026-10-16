"""Authorized test adapter; preserve the established loader and context implementation."""
import json
import numpy as np
from sleep_edf.dataset import EpochDataset
from .phase4_data import ContextDataset
from .phase5_registry import validate_subjects
from .common import fingerprint
import pandas as pd


def require_access(access):
    if not access or access.get('explicit_final_evaluation') is not True or not access.get('plan_sha256'):
        raise PermissionError('Final evaluation requires a reviewed, frozen, explicitly authorized plan')


class FinalEpochDataset(EpochDataset):
    def __init__(self,project,body,access):
        require_access(access)
        if access['plan_sha256']!=fingerprint(body) or access['plan_id']!=body['plan_id']:
            raise PermissionError('Authorization belongs to another plan')
        self.split='test'
        super().__init__(project,protocol='A',split='test',normalize=True,max_open_recordings=body['inference']['max_open_recordings'])
        validate_subjects(self.index,pd.read_csv(self.project/'results/subject_splits.csv'))
        if len(self)!=body['test_epochs'] or sorted(self.index.subject_id.astype(str).unique())!=body['test_subjects'] or fingerprint(self.index.epoch_id.tolist())!=body['test_epoch_ids_sha256']:
            raise ValueError('Frozen test population differs')

    def __getitem__(self,index):
        sample=super().__getitem__(index);row=self.index.iloc[index]
        _,metadata,_=self._recording(sample['recording_id']);saved=metadata[int(row.array_row)]
        sample['raw_rail_contact_samples']=int(saved['raw_rail_contact_samples'])
        sample['filter_boundary_flag']=int(saved['filter_boundary_flag'])
        sample['quality_flagged']=int(sample['raw_rail_contact_samples']>0 or sample['filter_boundary_flag']>0)
        sample['context_epoch_ids']=json.dumps([sample['epoch_id']]);sample['context_mask']='[1]';sample['missing_context_count']=0
        return sample


class _ContextMetadata:
    """Feed verified test metadata through unchanged Phase4 boundary construction.

    The legacy implementation permits train/validation only. This proxy presents a
    metadata-only development split during neighbor construction, never EEG or labels.
    Actual access below retains the true test split and exact original sample metadata.
    """
    split='validation'
    def __init__(self,base):
        self.index=base.index.copy();self.index['split']='validation'
    def __len__(self):return len(self.index)
    def __getitem__(self,index):raise AssertionError('Metadata proxy must never read EEG')


class FinalContextDataset(ContextDataset):
    def __init__(self,base,access,window=5):
        require_access(access)
        if getattr(base,'split',None)!='test' or set(base.index.split.astype(str))!={'test'}:
            raise PermissionError('Authorized context requires isolated test subjects')
        super().__init__(_ContextMetadata(base),window)
        self.base=base;self.index=base.index.reset_index(drop=True);self.split='test'


def validate_saved_context(frame,canonical,window):
    """Verify every saved context ID/mask against the full canonical target population."""
    if window==1:
        neighbors=np.arange(len(canonical),dtype=np.int64)[:,None]
    else:
        from types import SimpleNamespace
        construction=ContextDataset(_ContextMetadata(SimpleNamespace(index=canonical)),window)
        neighbors=construction.neighbors
    ids=canonical.epoch_id.to_numpy(dtype=object)
    for i,row in enumerate(frame.itertuples(index=False)):
        mask=neighbors[i]>=0
        expected=[str(ids[n]) if n>=0 else None for n in neighbors[i]]
        if json.loads(row.context_epoch_ids)!=expected or json.loads(row.context_mask)!=mask.astype(int).tolist() or row.missing_context_count!=int((~mask).sum()):
            raise ValueError('Saved context ID/mask/boundary mismatch: '+str(row.epoch_id))
    expected=(frame.raw_rail_contact_samples.to_numpy()>0)|(frame.filter_boundary_flag.to_numpy()>0)
    if not np.array_equal(frame.quality_flagged.to_numpy(),expected.astype(int)):
        raise ValueError('Quality flags differ from raw metadata')
