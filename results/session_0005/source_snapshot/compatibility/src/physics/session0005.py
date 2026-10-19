"""Transparent 1D operators for compatibility diagnostics, not new solvers."""
import numpy as np


def joint_operators(k,boundary=(0.,0.),velocity=0.):
    n=len(k)-1;h=1/(n+1)
    grad=np.zeros((n+1,n));grad[0,0]=1/h;grad[-1,-1]=-1/h
    for e in range(1,n):grad[e,e]=1/h;grad[e,e-1]=-1/h
    lift=np.zeros(n+1);lift[0]=-boundary[0]/h;lift[-1]=boundary[1]/h
    c=-np.asarray(k)[:,None]*grad;cg=-np.asarray(k)*lift
    vp=max(velocity,0.);vm=min(velocity,0.)
    for e in range(n+1):
        if e>0:c[e,e-1]+=vp
        else:cg[e]+=vp*boundary[0]
        if e<n:c[e,e]+=vm
        else:cg[e]+=vm*boundary[1]
    div=np.zeros((n,n+1))
    for i in range(n):div[i,i]=-1/h;div[i,i+1]=1/h
    return div,c,cg


def defects(u,j,k,q,boundary=(0.,0.),velocity=0.,lam=0.):
    d,c,cg=joint_operators(k,boundary,velocity)
    return d@j-q-lam*np.exp(u),j-c@u-cg


def defect_gradient(u,j,k,q,boundary=(0.,0.),velocity=0.,lam=0.):
    d,c,_=joint_operators(k,boundary,velocity);b,r=defects(u,j,k,q,boundary,velocity,lam)
    return -lam*np.exp(u)*b-c.T@r,d.T@b+r
