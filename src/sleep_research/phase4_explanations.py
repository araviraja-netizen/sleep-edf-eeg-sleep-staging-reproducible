"""Bounded context saliency/Grad-CAM/attention from retained checkpoints; no trainer."""
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from .common import atomic_file,digest,json_write,guard_evaluation
from .explanations import saliency,attention_weights


def context_cam(model,inputs,layer):
    activation=[]
    hook=dict(model.named_modules())[layer].register_forward_hook(lambda m,i,o:activation.append(o))
    try:
        values=inputs.detach().clone().requires_grad_(True);logits=model(values);targets=logits.argmax(1)
        gradient=torch.autograd.grad(logits[torch.arange(len(values),device=values.device),targets].sum(),activation[0])[0]
        cam=torch.relu((gradient.mean(-1,keepdim=True)*activation[0]).sum(1,keepdim=True))
        cam=torch.nn.functional.interpolate(cam,size=3000,mode='linear',align_corners=False)
        maximum=cam.amax(-1,keepdim=True);cam=cam/maximum.clamp_min(1e-12)
        if len(cam)==len(values):
            result=values.new_zeros(len(values),model.window,3000);result[:,model.window//2]=cam[:,0]
        else:result=cam.reshape(len(values),model.window,3000)
        return (result*values[:,:,-1,None]).detach().cpu().numpy()
    finally:hook.remove()


def save_explanations(model,batch,checkpoint,output):
    guard_evaluation(batch['split']);model.eval();output.mkdir(parents=True,exist_ok=True)
    state=torch.load(checkpoint,map_location='cpu',weights_only=True)
    for key,value in model.state_dict().items():
        if not torch.equal(value.detach().cpu(),state['model'][key]):raise ValueError('Explanation checkpoint weights differ')
    values=batch['eeg'].to(next(model.parameters()).device)
    gradients,targets=saliency(model,values)
    conv='encoder.branches.0.4' if model.model_id=='M8' else 'encoder.cnn.blocks.3.0'
    layer='blocks.1.attention' if model.model_id=='M7' else ('attention' if hasattr(model,'attention') else None)
    arrays=dict(signed_saliency=gradients[:,:,:3000],absolute_saliency=np.abs(gradients[:,:,:3000]),
                grad_cam=context_cam(model,values,conv),normalized_eeg=values[:,:,:3000].detach().cpu().numpy(),
                mask=values[:,:,-1].detach().cpu().numpy(),target_labels=targets,epoch_ids=np.asarray(batch['epoch_id'],dtype=str),
                context_epoch_ids=np.asarray(batch['context_epoch_ids'],dtype=str))
    if layer:arrays['attention']=attention_weights(model,values,layer,max_tokens=16)
    for value in arrays.values():
        if value.dtype.kind in 'fc' and not np.isfinite(value).all():raise ValueError('Nonfinite explanation')
    with atomic_file(output/'explanations.npz',binary=True) as handle:np.savez_compressed(handle,**arrays)
    fig,axes=plt.subplots(len(values),2,figsize=(11,2.2*len(values)),squeeze=False,constrained_layout=True)
    for i in range(len(values)):
        axes[i,0].imshow(arrays['absolute_saliency'][i],aspect='auto');axes[i,0].set(title=batch['epoch_id'][i],ylabel='Context slot',xlabel='Sample')
        axes[i,1].imshow(arrays['grad_cam'][i],aspect='auto',vmin=0,vmax=1);axes[i,1].set(title='Branch-specific Grad-CAM',xlabel='Sample')
    fig.savefig(output/'saliency_grad_cam.png',dpi=130);plt.close(fig)
    if layer:
        weights=arrays['attention'][0].mean(0)
        fig,ax=plt.subplots(figsize=(6,3));ax.imshow(weights,aspect='auto');ax.set(title='Representative mean-head attention',xlabel='Context key slot',ylabel='Query slot')
        fig.savefig(output/'attention.png',dpi=130);plt.close(fig)
    json_write(output/'manifest.json',dict(checkpoint_sha256=digest(checkpoint),epoch_ids=list(batch['epoch_id']),
        context_epoch_ids=list(batch['context_epoch_ids']),conv_layer=conv,attention_layer=layer,
        note='First validation example of each true class, without confidence selection. Predicted-logit gradients; branch-specific Grad-CAM. Attention is descriptive, not causal.',
        files={f.name:digest(f) for f in output.iterdir() if f.is_file() and f.name!='manifest.json'},synthetic=state['configuration'].get('synthetic',False)))
