"""Compact offline central-target models; every context mask is explicit."""
import torch
from torch import nn
from .cnn1d import CNN1D


class EpochEncoder(nn.Module):
    def __init__(self,multiscale=False,single_scale=False):
        super().__init__();self.multiscale=multiscale
        if multiscale:
            kernels=[25] if single_scale else [7,25,75]
            self.branches=nn.ModuleList([nn.Sequential(nn.Conv1d(1,16,k,stride=4,padding=k//2,bias=False),
                nn.GroupNorm(4,16),nn.GELU(),nn.MaxPool1d(4),nn.Conv1d(16,32,7,stride=2,padding=3,bias=False),
                nn.GroupNorm(4,32),nn.GELU(),nn.AdaptiveAvgPool1d(1),nn.Flatten(1)) for k in kernels])
            self.projection=nn.Linear(32*len(kernels),64)
        else:
            self.cnn=CNN1D();del self.cnn.classifier
    def forward(self,x):
        if self.multiscale:return self.projection(torch.cat([branch(x) for branch in self.branches],dim=-1))
        return self.cnn.extract_features(x)


class MaskedBiLSTM(nn.Module):
    """Masked LSTMCell updates skip padding; separate forward/backward hidden32."""
    def __init__(self):
        super().__init__();self.forward_cell=nn.LSTMCell(64,32);self.backward_cell=nn.LSTMCell(64,32)
    def forward(self,tokens,mask):
        outputs=[];center=tokens.shape[1]//2
        for cell,order in [(self.forward_cell,range(tokens.shape[1])),(self.backward_cell,range(tokens.shape[1]-1,-1,-1))]:
            h=tokens.new_zeros(len(tokens),32);c=h.clone();target=None
            for i in order:
                nh,nc=cell(tokens[:,i],(h,c));valid=mask[:,i,None]
                h=torch.where(valid,nh,h);c=torch.where(valid,nc,c)
                if i==center:target=h
            outputs.append(target)
        return torch.cat(outputs,dim=-1)


class ContextBlock(nn.Module):
    def __init__(self):
        super().__init__();self.attention=nn.MultiheadAttention(64,4,dropout=.1,batch_first=True)
        self.norm1=nn.LayerNorm(64);self.norm2=nn.LayerNorm(64)
        self.ff=nn.Sequential(nn.Linear(64,128),nn.GELU(),nn.Dropout(.1),nn.Linear(128,64))
    def forward(self,x,mask):
        a,_=self.attention(x,x,x,key_padding_mask=~mask,need_weights=True,average_attn_weights=False)
        x=self.norm1(x+a);return self.norm2(x+self.ff(x))*mask[:,:,None]


class AdvancedSleepModel(nn.Module):
    """Input [B,T,3001]: normalized EEG3000 plus explicit binary mask scalar.

    Mask is a separate logical input packed into the final scalar for compatibility
    with the shared Trainer EEG-only forward interface. It is never an EEG sample.
    No neighboring sleep labels are accepted. Central slot must always be valid.
    """
    def __init__(self,model_id='M5',window=5,multiscale=True,context=True):
        super().__init__()
        if model_id not in ['M5','M6','M7','M8'] or window<1 or window%2!=1 or window>9:raise ValueError('Invalid architecture/window')
        self.model_id=model_id;self.window=window;self.use_context=context
        self.encoder=EpochEncoder(multiscale=model_id=='M8',single_scale=not multiscale)
        self.position=nn.Parameter(torch.randn(window,64)*.02) if model_id in ['M6','M7','M8'] and context else None
        if model_id=='M5':self.recurrent=MaskedBiLSTM()
        elif model_id=='M7':self.blocks=nn.ModuleList([ContextBlock(),ContextBlock()])
        elif context:self.attention=nn.MultiheadAttention(64,4,dropout=.1,batch_first=True)
        if model_id=='M8' and context:self.gate=nn.Linear(128,64)
        self.norm=nn.LayerNorm(64);self.intermediate=nn.Identity();self.embedding=nn.Identity()
        self.classifier=nn.Sequential(nn.Dropout(.3),nn.Linear(64,5))
    def forward(self,values):
        if values.ndim!=3 or values.shape[1:]!=(self.window,3001) or not bool(torch.isfinite(values).all()):raise ValueError('Expected finite[B,T,3001] context packet')
        mask_values=values[:,:,-1]
        if not bool(((mask_values==0)|(mask_values==1)).all()):raise ValueError('Context mask must be binary')
        mask=mask_values.bool();center=self.window//2
        if not bool(mask[:,center].all()):raise ValueError('Central target cannot be missing')
        eeg=values[:,:,:3000]
        if not bool((eeg.masked_select(~mask[:,:,None].expand_as(eeg))==0).all()):raise ValueError('Missing EEG must be zero padded')
        if not self.use_context:
            central=self.intermediate(self.encoder(eeg[:,center,None,:]));output=central
        else:
            tokens=self.encoder(eeg.reshape(-1,1,3000)).reshape(len(values),self.window,64)*mask[:,:,None]
            central=self.intermediate(tokens[:,center])
            if self.model_id=='M5':output=self.recurrent(tokens,mask)
            else:
                tokens=(tokens+self.position[None])*mask[:,:,None]
                if self.model_id=='M7':
                    for block in self.blocks:tokens=block(tokens,mask)
                    output=tokens[:,center]
                else:
                    attended,_=self.attention(tokens[:,center:center+1],tokens,tokens,key_padding_mask=~mask,
                                              need_weights=True,average_attn_weights=False)
                    attended=attended[:,0]
                    if self.model_id=='M8':
                        gate=torch.sigmoid(self.gate(torch.cat([central,attended],dim=-1)))
                        output=gate*attended+(1-gate)*central
                    else:output=central+attended
        return self.classifier(self.embedding(self.norm(output)))


def make_advanced(architecture):
    return AdvancedSleepModel(**{key:value for key,value in architecture.items() if key!='name'})
