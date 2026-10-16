import numpy as np
import pytest
import torch

from src.physics.heat import heat_1d, heat_2d, neumann_1d
from src.operators.primitives import (anchored_response, resistance_response,
    exchange_response, action_response, multiplicative_response,
    material_event_paths, LifetimeResponse, spectral_response)
from src.operators.primitives import excitation_contrast


@pytest.mark.parametrize('n,k,boundary',[(7,1.,(0.,0.)),(15,.2,(1.,2.)),(31,8.,(-1.,.5))])
def test_heat_1d_reference_matches_analytic_and_balances_flux(n,k,boundary):
    p=heat_1d(n,conductivity=k,boundary=boundary)
    expected=p.x*(1-p.x)/(2*k)+boundary[0]*(1-p.x)+boundary[1]*p.x
    np.testing.assert_allclose(p.reference(),expected,atol=1e-12)
    assert p.diagnostics(p.reference())['net_energy_balance_abs'] < 1e-10


def test_pure_neumann_compatibility_gauge_and_convergence():
    errors=[]
    for n in [16,32,64]:
        x,a,b,t=neumann_1d(n)
        np.testing.assert_allclose(a@np.ones(n),0,atol=1e-12)
        np.testing.assert_allclose(a@t,b,atol=1e-10)
        assert abs(t.mean())<1e-12
        errors.append(np.linalg.norm(t-np.cos(2*np.pi*x))/np.sqrt(n))
    assert errors[1]<errors[0]/3.8
    assert errors[2]<errors[1]/3.8
    with pytest.raises(ValueError,match='incompatible'):
        neumann_1d(16,source=1.)


@pytest.mark.parametrize('hole',[False,True])
def test_fem_spd_boundary_flux_balance_and_connected_hole(hole):
    p=heat_2d(8,tensor=np.array([[3.,1.],[1.,1.]]),hole=hole)
    np.testing.assert_allclose(p.matrix,p.matrix.T,atol=1e-12)
    assert np.linalg.eigvalsh(p.matrix).min()>0
    np.testing.assert_allclose(p.stiffness.sum(axis=1),0,atol=1e-12)
    metrics=p.diagnostics(p.reference())
    assert metrics['pde_residual_inf'] < 1e-12
    assert metrics['net_energy_balance_abs'] < 1e-12
    assert metrics['boundary_error_abs']==0


def test_fem_rotated_anisotropic_manufactured_convergence():
    errors=[]
    for m in [8,16]:
        p=heat_2d(m,tensor=np.array([[3.,1.],[1.,1.]]),
            source=lambda x,y:6*y*(1-y)+2*x*(1-x)-2*(1-2*x)*(1-2*y))
        x,y=p.points.T
        expected=x*(1-x)*y*(1-y)
        errors.append(np.linalg.norm(p.reference()-expected)/np.linalg.norm(expected))
    assert errors[1]<errors[0]/3


def test_resistance_green_control_matches_independent_direct_solve():
    p=heat_1d(19,conductivity=lambda x:np.where(x<.35,.3,7.),
               source=lambda x:np.sin(3*np.pi*x)+2,boundary=(-1.,2.))
    pred,g=resistance_response(p)
    np.testing.assert_allclose(pred,p.reference(),atol=1e-12)
    np.testing.assert_allclose(g,g.T,atol=1e-14)
    assert np.linalg.eigvalsh(g).min()>0


def test_anchored_candidate_passes_uniform_but_fails_reciprocity_and_local_source():
    p=heat_1d()
    pred,g=anchored_response(p)
    np.testing.assert_allclose(pred,p.reference(),atol=1e-12)
    assert np.linalg.norm(g-g.T)/np.linalg.norm(g)>.01
    local=heat_1d(source=lambda x:np.exp(-((x-.2)/.05)**2))
    estimate,_=anchored_response(local)
    assert np.linalg.norm(estimate-local.reference())/np.linalg.norm(local.reference())>.05


def test_exchange_unstable_step_and_stable_but_slow_revision():
    p=heat_1d()
    initial_error=np.linalg.norm(p.reference())
    divergent=exchange_response(p.matrix,p.rhs,steps=8)
    assert np.linalg.norm(divergent-p.reference())>100*initial_error
    largest=np.linalg.eigvalsh(p.matrix)[-1]
    stable=exchange_response(p.matrix,p.rhs,steps=32,step=1/largest)
    assert 0<np.linalg.norm(stable-p.reference())<initial_error


def test_fixed_action_span_cannot_recover_orthogonal_source():
    p=heat_1d(source=lambda x:np.sin(8*np.pi*x))
    basis=np.sin(np.pi*p.x[:,None]*np.arange(1,5)[None,:])
    pred=action_response(p.matrix,p.rhs,basis)
    assert np.linalg.norm(pred)<1e-12
    assert np.linalg.norm(p.reference())>1e-4


def test_multiplicative_packet_cannot_represent_signed_cooling():
    p=heat_1d(source=-1.)
    pred=multiplicative_response(p.matrix,p.rhs)
    assert np.all(pred>0) and np.all(p.reference()<0)
    assert np.linalg.norm(pred-p.reference())/np.linalg.norm(p.reference())>1


def test_material_event_endpoint_and_loop_defect():
    first,second,loop,start=material_event_paths(heat_1d())
    assert np.linalg.norm(first-second)>.001
    assert np.linalg.norm(loop-start)>.001


def test_lifetime_filter_positive_reciprocal_but_bad_extreme_spectral_tail():
    model=LifetimeResponse(seed=824)
    a=heat_1d().matrix
    identity=np.eye(len(a))
    g=spectral_response(a,identity,model)
    np.testing.assert_allclose(g,g.T,atol=1e-14)
    assert np.linalg.eigvalsh(g).min()>0
    assert model.numpy_filter(np.array([1e9]))[0] == 0
    # The continuous inverse diverges at lambda=0; finite positive lifetime sum does not.
    assert torch.isfinite(model(torch.tensor([0.],dtype=torch.float64))).all()


def test_source_contrast_gauge_is_not_a_temperature_gauge():
    p=heat_1d()
    pred=excitation_contrast(p)
    np.testing.assert_allclose(pred,0,atol=1e-14)
    assert np.linalg.norm(p.reference())>0.1
    local=heat_1d(source=lambda x:np.exp(-((x-.25)/.05)**2))
    assert excitation_contrast(local).min() < -0.01
    assert local.reference().min() > 0


def test_spd_fem_does_not_imply_nodal_positivity():
    from scipy.linalg import expm
    p=heat_2d(8,tensor=np.array([[1.,-.9],[-.9,1.]]))
    assert np.linalg.eigvalsh(p.matrix).min()>0
    assert expm(-1e-6*p.matrix).min() < -1e-5
