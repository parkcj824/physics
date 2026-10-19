"""Outcome-triggered, fresh-data conditioning controls; no invention claim."""
import torch
from src.operators.session0005_controls import GControl,physical_flux


def normalize(v):return v/torch.sqrt(v.square().mean(1,keepdim=True)+1e-12)


class NormalizedControl(GControl):
    def __init__(self,kind='norm_physical',family='G'):
        super().__init__('physics_plus_feature',family);self.normalized_kind=kind
    def forward(self,raw):
        h=torch.tanh(self.encode((raw-self.center)/self.scale))
        if self.normalized_kind=='norm_static':
            u,j=self.generate(h,raw);r0=normalize(j-physical_flux(raw,u,self.family))
        for _ in range(self.depth):
            s=normalize(self.feature_projection(h))
            if self.normalized_kind=='norm_physical':
                u,j=self.generate(h,raw);s=normalize(s+normalize(j-physical_flux(raw,u,self.family)))
            elif self.normalized_kind=='norm_static':s=normalize(s+r0)
            h=torch.tanh(self.evolve(h)+self.mismatch(s)+torch.tanh(self.gate(h))*torch.tanh(self.reaction(s)))
        return self.generate(h,raw)
