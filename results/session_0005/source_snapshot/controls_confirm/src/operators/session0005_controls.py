"""Causal controls for G; these controls are not new invention candidates."""
import torch
from torch import nn
from src.operators.session0004 import CoGenerationModel


def physical_flux(raw,u,family):
    n=u.shape[1];full=torch.cat([raw[:,2*n+1:2*n+2],u,raw[:,2*n+2:2*n+3]],1)
    result=-raw[:,:n+1].exp()*torch.diff(full,dim=1)*(n+1)
    if family=='H':
        v=raw[:,-1:];result=result+v.clamp_min(0)*full[:,:-1]+v.clamp_max(0)*full[:,1:]
    return result


class GControl(CoGenerationModel):
    def __init__(self,kind='original',family='G'):
        super().__init__(mode='direct' if kind=='direct_matched' else 'feedback',width=64 if kind=='direct_matched' else 48)
        self.kind=kind;self.family=family
        if kind in ['physics_plus_feature','learned_feature']:
            self.feature_projection=nn.Linear(48,16,bias=False)

    def generate(self,h,raw):
        if self.family=='H':
            raw=raw.clone();raw[:,-1]=0.
        return super().generate(h,raw)

    def forward(self,raw):
        h=torch.tanh(self.encode((raw-self.center)/self.scale))
        if self.kind=='static_physics':
            u,j=self.generate(h,raw);r0=j-physical_flux(raw,u,self.family)
        for _ in range(self.depth):
            if self.kind=='direct_matched':h=torch.tanh(self.evolve(h));continue
            if self.kind=='learned_feature':s=self.feature_projection(h)
            elif self.kind=='static_physics':s=r0
            else:
                u,j=self.generate(h,raw);s=j-physical_flux(raw,u,self.family)
                if self.kind=='physics_plus_feature':s=s+self.feature_projection(h)
            h=torch.tanh(self.evolve(h)+self.mismatch(s)+torch.tanh(self.gate(h))*torch.tanh(self.reaction(s)))
        return self.generate(h,raw)


class LearnedJacobi(nn.Module):
    def __init__(self,family='G',steps=3):
        super().__init__();self.family=family;self.alpha=nn.Parameter(torch.zeros(steps))
        self.register_buffer('x',torch.arange(1,16,dtype=torch.float64)/16)
    def forward(self,raw):
        u=raw[:,31:32]*(1-self.x)+raw[:,32:33]*self.x
        k=raw[:,:16].exp();diag=(k[:,:-1]+k[:,1:])*256
        if self.family=='H':diag=diag+raw[:,-1:].abs()*16
        for a in self.alpha:
            j=physical_flux(raw,u,self.family);source=raw[:,16:31]
            if self.family=='G':source=source+raw[:,-1:]*u.exp()
            residual=source-torch.diff(j,dim=1)*16
            u=u+torch.sigmoid(a)*residual/diag
        return u,physical_flux(raw,u,self.family)
