"""Short existing-checkpoint validation-batch smoke; never reads test EEG or trains."""
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from .phase5_registry import PROJECT,selected_models,verify_identity
from .real_data import ResearchEpochDataset
from .phase4_data import ContextDataset
from .cnn_inference import model_for
from models.advanced_sleep import make_advanced
from .inference import load_model
from .common import json_write,now


def smoke(project=PROJECT):
    base=ResearchEpochDataset(project,'validation',2);rows=[]
    try:
        for row in selected_models(project):
            exp=verify_identity(project,row);data=base if row['model_id'] in ['M1','M4'] else ContextDataset(base)
            batch=torch.utils.data.default_collate([data[i] for i in range(2)])
            model=model_for(exp) if row['model_id'] in ['M1','M4'] else make_advanced(row['configuration']['architecture'])
            model,state,checkpoint=load_model(model,exp,'best','cpu')
            with torch.no_grad():cpu=model(batch['eeg']).cpu().numpy()
            mps_checked=False;max_error=None
            if torch.backends.mps.is_available():
                model.to('mps')
                with torch.no_grad():mps=model(batch['eeg'].to('mps')).cpu().numpy()
                np.testing.assert_allclose(mps,cpu,atol=2e-3,rtol=2e-3);mps_checked=True;max_error=float(np.max(np.abs(mps-cpu)))
            if cpu.shape!=(2,5) or not np.isfinite(cpu).all():raise ValueError('Invalid validation logits')
            rows.append(dict(key=row['key'],validation_epoch_ids=batch['epoch_id'],checkpoint_sha256=row['checkpoint_sha256'],cpu_forward_successful=True,mps_checked=mps_checked,max_abs_cpu_mps_logit_difference=max_error))
            del model
            if torch.backends.mps.is_available():torch.mps.empty_cache()
        result=dict(successful=True,executed_at=now(),models=rows,split='validation',real_test_eeg_read=False,training_called=False)
        json_write(Path(project)/'results/phase5_validation_smoke.json',result);return result
    finally:base.close()

if __name__=='__main__':print(__import__('json').dumps(smoke(),indent=2))
