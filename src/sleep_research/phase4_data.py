"""Same central targets, contiguous within-recording EEG context, no input labels."""
import json
import numpy as np
import torch


class ContextDataset:
    def __init__(self,base,window=5):
        if getattr(base,'split',None) not in ['train','validation']:raise PermissionError('Frozen test context is locked')
        if window<1 or window>9 or window%2!=1:raise ValueError('Use odd window1..9')
        self.base=base;self.index=base.index.reset_index(drop=True);self.split=base.split;self.window=window
        if len(self.index)!=len(base) or self.index.epoch_id.duplicated().any():raise ValueError('Invalid target IDs')
        if set(self.index.split.astype(str))!={self.split}:raise PermissionError('Mixed split context forbidden')
        columns=['subject_id','recording_id','night','original_epoch_index','onset_seconds']
        rows=list(self.index[columns].itertuples(index=False,name=None))
        for _,group in self.index.groupby('recording_id',observed=True):
            if group.subject_id.nunique()!=1 or group.night.nunique()!=1:raise ValueError('Ambiguous recording/subject/night')
        lookup={}
        for i,(subject,recording,night,epoch,onset) in enumerate(rows):
            if not np.isfinite(onset) or epoch!=int(epoch) or epoch<0 or abs(onset-30*epoch)>1e-9:raise ValueError('Original onset/index timing mismatch')
            key=(str(subject),str(recording),int(night),int(epoch))
            if key in lookup:raise ValueError('Duplicate original epoch')
            lookup[key]=i
        self.neighbors=np.full((len(rows),window),-1,dtype=np.int64);middle=window//2
        self.neighbors[:,middle]=np.arange(len(rows))
        for i,(subject,recording,night,epoch,onset) in enumerate(rows):
            for direction in [-1,1]:
                for distance in range(1,middle+1):
                    candidate=lookup.get((str(subject),str(recording),int(night),int(epoch)+direction*distance))
                    if candidate is None:break # Do not jump a gap to a farther existing epoch.
                    self.neighbors[i,middle+direction*distance]=candidate
    def __len__(self):return len(self.index)
    def __getitem__(self,index):
        central=dict(self.base[index]);packet=np.zeros((self.window,3001),dtype=np.float32)
        ids=[];mask=[]
        for slot,neighbor in enumerate(self.neighbors[index]):
            valid=neighbor>=0;mask.append(int(valid))
            if valid:
                sample=central if neighbor==index else self.base[int(neighbor)]
                if sample['split']!=self.split:raise PermissionError('Context split changed')
                packet[slot,:3000]=np.asarray(sample['eeg']).reshape(3000)
                packet[slot,-1]=1;ids.append(sample['epoch_id'])
            else:ids.append(None)
        central.update(eeg=torch.from_numpy(packet),context_epoch_ids=json.dumps(ids),context_mask=json.dumps(mask),
                       missing_context_count=self.window-sum(mask))
        return central
    def close(self):
        if hasattr(self.base,'close'):self.base.close()
