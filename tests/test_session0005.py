import numpy as np
import pytest
import torch
from scipy.linalg import expm
from src.physics.session0005 import joint_operators,defects,defect_gradient
from src.physics.session0004 import steady_heat
from src.operators.session0005_controls import GControl,physical_flux,LearnedJacobi
from src.operators.session0005_interaction import Interaction,SingletonEncoder


def test_balance_constitutive_and_pde_are_distinct():
    n=15;x=np.arange(1,n+1)/16;k=np.ones(16);q=np.ones(15)
    u,j=__import__('experiments.session0004_transport',fromlist=['transport_reference']).transport_reference(k,q,(0,0),0.)
    d,c,cg=joint_operators(k);pert=u+1e-4*np.sin(15*np.pi*x)
    b,r=defects(pert,j,k,q)
    np.testing.assert_allclose(d@(c@pert+cg)-q,b-d@r,atol=1e-13)
    assert max(abs(b))<1e-12 and max(abs(b-d@r))>.1
    assert np.linalg.norm(pert-u)/np.linalg.norm(u)<.001


def test_finite_nonlinear_compatible_increment_not_tangent():
    rng=np.random.default_rng(23);u=rng.uniform(0,.4,7);j=rng.normal(size=8);du=rng.normal(scale=.2,size=7);dj=rng.normal(size=8)
    k=np.exp(rng.normal(size=8));q=np.ones(7);d,c,_=joint_operators(k)
    b0,c0=defects(u,j,k,q,lam=1.5);b1,c1=defects(u+du,j+dj,k,q,lam=1.5)
    np.testing.assert_allclose(b1-b0,d@dj-1.5*np.exp(u)*np.expm1(du),atol=1e-12)
    np.testing.assert_allclose(c1-c0,dj-c@du,atol=1e-12)
    assert max(abs((b1-b0)-(d@dj-1.5*np.exp(u)*du)))>1e-3


@pytest.mark.parametrize('family',['G','H'])
def test_g_control_matched_features_and_exact_generated_balance(family):
    raw=torch.zeros((4,34),dtype=torch.float64);raw[:,16:31]=1.;raw[:,-1]=1.5 if family=='G' else -1.
    for kind in ['original','physics_plus_feature','learned_feature','static_physics']:
        torch.manual_seed(4);m=GControl(kind,family).double();u,j=m(raw)
        source=raw[:,16:31]+(raw[:,-1:]*u.exp() if family=='G' else 0.)
        torch.testing.assert_close(torch.diff(j,dim=1)*16,source,rtol=1e-12,atol=1e-12)
    torch.manual_seed(4);a=GControl('physics_plus_feature',family).double()
    torch.manual_seed(4);b=GControl('learned_feature',family).double()
    assert sum(p.numel() for p in a.parameters())==sum(p.numel() for p in b.parameters())==9472
    for (na,pa),(nb,pb) in zip(a.named_parameters(),b.named_parameters()):
        assert na==nb;torch.testing.assert_close(pa,pb)


def test_jacobi_conditionally_preserves_positive_source_state():
    raw=torch.zeros((3,34),dtype=torch.float64);raw[:,:16]=np.log(.1);raw[:,16:31]=1.;raw[:,-1]=torch.tensor([-1.,0.,1.])
    for family in ['G','H']:
        z=raw.clone()
        if family=='G':z[:,-1]=0.
        m=LearnedJacobi(family,32).double();u,j=m(z)
        assert torch.min(u)>=0.
        torch.testing.assert_close(j,physical_flux(z,u,family))


@pytest.mark.parametrize('kind',['reaction_gate','reaction_no_gate','bilinear','volterra3','attention'])
def test_interaction_singleton_zero_and_set_permutation(kind):
    torch.manual_seed(8);m=Interaction(kind).double();x=torch.arange(1,16,dtype=torch.float64)/16
    k=torch.ones(2,dtype=torch.float64);lam=torch.full((2,),1.5,dtype=torch.float64);base=torch.ones((2,15),dtype=torch.float64)*.1
    r=torch.rand((2,4,15),dtype=torch.float64)*.02;s=torch.ones((2,4,3),dtype=torch.float64)*.2
    s[0,1:,0]=0.;r[0,1:]=0.
    c,_=m(base,r,s,k,lam,x);rev,_=m(base,r.flip(1),s.flip(1),k,lam,x)
    torch.testing.assert_close(c,rev,rtol=1e-10,atol=1e-12)
    assert max(abs(c[0])).item()<1e-12
    if kind.startswith('reaction'):assert c.min()>=0
    zero,_=m(base,r,s,k,lam*0,x);assert torch.count_nonzero(zero)==0


def test_cattaneo_two_snapshots_aliasing_is_real_information_loss():
    omega=3.14;tau=.2;gen=np.array([[0.,-omega],[omega/tau,-1/tau]])
    p=expm(.05*gen);coef=[np.trace(p),-np.linalg.det(p)]
    state=np.array([.2,.4]);a1=(p@state)[0];a2=(p@p@state)[0]
    np.testing.assert_allclose(a2,coef[0]*a1+coef[1]*state[0],atol=1e-15)
    wd=np.sqrt(omega**2/tau-1/(4*tau*tau));pa=expm(np.pi/wd*gen)
    assert abs(pa[0,1])<1e-13
    assert abs((p@pa)[0,1])>.01


def test_identical_mixed_reaction_is_not_a_sufficient_closure_state():
    n=15;x=np.arange(1,n+1)/16;k=np.ones(16);q0=np.zeros(n);lam=1.5
    base=steady_heat(k,q0,lam=lam)[0];d,c,_=joint_operators(k);a=d@c
    defects_and_corrections=[]
    for coefficients in [[.3,.3],[.1,.9]]:
        responses=np.log1p(np.array(coefficients)[:,None]*4*x*(1-x))
        sources=responses@a.T-lam*np.exp(base)*np.expm1(responses)
        assert sources.min()>0
        s=responses.sum(0)
        h=lam*np.exp(base)*(np.expm1(s)-np.expm1(responses).sum(0))
        true=steady_heat(k,sources.sum(0),lam=lam,initial=base+s)[0]
        delta=true-base-s
        np.testing.assert_allclose(a@delta-lam*np.exp(base+s)*np.expm1(delta)-h,0.,atol=1e-9)
        defects_and_corrections.append((h,delta))
    (h1,delta1),(h2,delta2)=defects_and_corrections
    np.testing.assert_allclose(h1,h2,atol=1e-14)
    assert np.linalg.norm(delta1-delta2)>.006
