"""Finite-reaction overlap selection and explicit interaction baselines."""
import torch
from torch import nn
import torch.nn.functional as F


def net(i,o,w=32):return nn.Sequential(nn.Linear(i,w),nn.Tanh(),nn.Linear(w,w),nn.Tanh(),nn.Linear(w,o))


class SingletonEncoder(nn.Module):
    def __init__(self):
        super().__init__();self.base=net(2,1);self.response=net(5,1)
    def forward(self,k,lam,sources,x):
        b,m,_=sources.shape;n=len(x);l=lam/k
        basez=torch.stack([l[:,None].expand(b,n),x[None].expand(b,n)],-1)
        base=l[:,None]*x*(1-x)*F.softplus(self.base(basez)[...,0])
        a,c,w=sources.unbind(-1)
        z=torch.stack([l[:,None,None].expand(b,m,n),(l[:,None]*a/k[:,None])[:,:,None].expand(b,m,n),
            c[:,:,None].expand(b,m,n),w[:,:,None].expand(b,m,n),x[None,None].expand(b,m,n)],-1)
        response=(a/k[:,None])[:,:,None]*x*(1-x)*F.softplus(self.response(z)[...,0])
        return base,response


class Interaction(nn.Module):
    def __init__(self,kind='reaction_gate',n=15):
        super().__init__();self.kind=kind;self.n=n
        if kind=='attention':
            self.query=net(3,32);self.key=net(n+5,32);self.value=net(n+5,32);self.decode=net(35,1)
        else:
            count=2 if kind=='volterra3' else 1
            self.kernel=nn.Parameter(torch.zeros(count,3,n,n))
            with torch.no_grad():self.kernel[:,0].fill_(-3.5 if kind.startswith('reaction') else .03)
            if kind=='reaction_gate':self.threshold=nn.Parameter(torch.tensor([-9.21,0.],dtype=torch.float64))

    def forward(self,base,r,sources,k,lam,x):
        b,m,n=r.shape;s=r.sum(1);l=lam/k;context=torch.stack([torch.ones_like(k),l,k.log()],-1)
        if self.kind=='attention':
            active=sources[...,0]>0
            z=torch.cat([sources,r,l[:,None,None].expand(b,m,1),k.log()[:,None,None].expand(b,m,1)],-1)
            qz=torch.stack([x[None].expand(b,n),l[:,None].expand(b,n),k.log()[:,None].expand(b,n)],-1)
            q=self.query(qz);key=self.key(z);val=self.value(z)
            score=torch.einsum('bnr,bmr->bnm',q,key)/32**.5
            score=score.masked_fill(~active[:,None,:],-1e9)
            weights=torch.softmax(F.pad(score,(0,1),value=0.),-1)[...,:m]
            pooled=torch.einsum('bnm,bmr->bnr',weights,val)
            full=self.decode(torch.cat([pooled,qz],-1))[...,0]
            empty=self.decode(torch.cat([torch.zeros_like(pooled),qz],-1))[...,0]
            single_weight=torch.sigmoid(score)
            single_pool=single_weight[:,:,:,None]*val[:,None,:,:]
            single=self.decode(torch.cat([single_pool,qz[:,:,None,:].expand(b,n,m,3)],-1))[...,0]
            correction=full-empty-((single-empty[:,:,None])*active[:,None,:]).sum(-1)
            return lam[:,None]*x*(1-x)*correction,torch.ones_like(s)
        h2=(s.square()-r.square().sum(1))/2
        if self.kind.startswith('reaction'):
            h=lam[:,None]*base.exp()*(torch.expm1(s)-torch.expm1(r).sum(1))
            h=h.clamp_min(0.)  # this experiment restricts increments to nonnegative sources
            if self.kind=='reaction_gate':
                tau=F.softplus(self.threshold[0]+self.threshold[1]*l)[:,None]
                signal=F.relu(h-tau);activation=(h>tau).to(h.dtype)
            else:signal=h;activation=(h>0).to(h.dtype)
            kernel=F.softplus(torch.einsum('bc,cnk->bnk',context,self.kernel[0]))
            return x*(1-x)*torch.einsum('bnk,bk->bn',kernel,signal),activation
        signals=[lam[:,None]*base.exp()*h2]
        if self.kind=='volterra3':signals.append(lam[:,None]*base.exp()*(s**3-(r**3).sum(1))/6)
        result=torch.zeros_like(s)
        for index,signal in enumerate(signals):
            kernel=torch.einsum('bc,cnk->bnk',context,self.kernel[index])
            result=result+torch.einsum('bnk,bk->bn',kernel,signal)
        return x*(1-x)*result,torch.ones_like(s)
