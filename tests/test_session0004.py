import numpy as np
import pytest
import torch

from src.physics.session0004 import (fold,bratu_profile,bratu_lambda,bratu_branches,
    steady_heat,stability,diagnostics)
from src.operators.session0004 import BranchModel,InterventionModel,CoGenerationModel


def test_bratu_analytic_pde_and_fold():
    a,lam=fold();x=np.linspace(0,1,33);t=bratu_profile(a,x)
    second=-2*a*a/np.cosh(a*(x-.5))**2
    np.testing.assert_allclose(-second,lam*np.exp(t),rtol=1e-13)
    assert abs(a*np.tanh(a/2)-2)<1e-13
    assert abs(t[0])+abs(t[-1])<1e-14
    eps=1e-4
    assert bratu_lambda(a)>bratu_lambda(a-eps) and bratu_lambda(a)>bratu_lambda(a+eps)


def test_branch_stability_and_discrete_convergence():
    errors=[]
    for n in [15,31,63]:
        x=np.arange(1,n+1)/(n+1);truth,_=bratu_branches(2.,x);current=[]
        for branch in range(2):
            t,_=steady_heat(np.ones(n+1),np.zeros(n),lam=2.,initial=truth[branch])
            assert (stability(np.ones(n+1),2.,t)>0)==(branch==1)
            assert diagnostics(np.ones(n+1),np.zeros(n),(0,0),2.,t)['pde_residual_inf']<3e-10
            current.append(np.linalg.norm(t-truth[branch])/np.linalg.norm(truth[branch]))
        errors.append(current)
    assert np.all(np.array(errors[1:])<np.array(errors[:-1])*.3)
    with pytest.raises(ValueError):bratu_branches(4.,x)


def test_linear_heat_limit_and_nonzero_boundary_scaling():
    n=15;x=np.arange(1,n+1)/(n+1)
    t,_=steady_heat(np.ones(n+1),np.ones(n),(1.,2.))
    np.testing.assert_allclose(t,1+x+x*(1-x)/2,atol=1e-14)
    t2,_=steady_heat(2*np.ones(n+1),np.ones(n),(1.,2.))
    np.testing.assert_allclose(t2-(1+x),(t-(1+x))/2,atol=1e-14)


def test_finite_intervention_symmetry_and_zero_anchor():
    torch.manual_seed(7);model=InterventionModel().double()
    x=torch.linspace(0,1,17,dtype=torch.float64)
    z=torch.tensor([[.2,.7,.3,.6],[.2,0.,.3,.6]],dtype=torch.float64)
    with torch.no_grad():
        p=model(z,x);s=model(z[:,[1,0,3,2]],x)
    torch.testing.assert_close(p,s,atol=1e-14,rtol=1e-14)
    assert torch.count_nonzero(p[:,[0,-1]])==0
    # The rank channel vanishes identically when one edit is zero.
    model.mode='additive'
    with torch.no_grad():torch.testing.assert_close(p[1],model(z,x)[1])


@pytest.mark.parametrize('lam',[0.,1.5])
def test_cogeneration_balance_during_state_formation(lam):
    torch.manual_seed(19);model=CoGenerationModel().double()
    raw=torch.zeros((3,34),dtype=torch.float64)
    raw[:,16:31]=torch.linspace(-1,2,15);raw[:,31:33]=torch.tensor([.1,.2]);raw[:,-1]=lam
    t,j=model(raw)
    effective=raw[:,16:31]+lam*t.exp()
    torch.testing.assert_close(torch.diff(j,dim=1)*16,effective,atol=1e-13,rtol=1e-13)
    # Balance does not imply constitutive accuracy for random untrained weights.
    d=diagnostics(np.ones(16),raw[0,16:31].numpy(),(.1,.2),lam,t[0].detach().numpy(),j[0].detach().numpy())
    assert d['constitutive_mismatch_inf']>1e-3
    (t.square().mean()+j.square().mean()).backward()
    assert model.reaction.weight.grad is not None


def test_paired_branch_ordering_boundary_and_fold_gate():
    model=BranchModel().double();x=torch.linspace(0,1,17,dtype=torch.float64)
    out=model(torch.tensor([1.,2.],dtype=torch.float64),x)
    assert torch.all(out[:,1]>=out[:,0])
    assert torch.count_nonzero(out[:,:,[0,-1]])==0


def test_directed_transport_reference_and_cogeneration():
    from experiments.session0004_transport import transport_reference,TransportCoGeneration
    n=15;x=np.arange(1,n+1)/(n+1)
    for v in [-1.,1.]:
        u,j=transport_reference(np.full(16,.1),np.full(15,v),(0.,1.),v)
        np.testing.assert_allclose(u,x,atol=1e-14)
        np.testing.assert_allclose(np.diff(j)*16,np.full(15,v),atol=1e-14)
    torch.manual_seed(4);m=TransportCoGeneration().double();raw=torch.zeros((2,34),dtype=torch.float64)
    raw[:,:16]=np.log(.1);raw[:,16:31]=1.;raw[:,-1]=torch.tensor([-1.,1.])
    u,j=m(raw)
    torch.testing.assert_close(torch.diff(j,dim=1)*16,torch.ones((2,15),dtype=torch.float64),rtol=1e-13,atol=1e-13)
