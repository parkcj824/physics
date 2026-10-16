"""Physics and information counterexamples, not checks of recorded score constants."""

import numpy as np
import pytest
from scipy.linalg import expm, solve

from src.physics.heat import heat_1d
from src.operators.session0003 import (
    incidence, material_response, oriented_inverse_update, balance_probes,
    witness_collision, change_compose, group_encode, compose_responses,
    segment_compose, segment_response, segment_features, periodic_transport,
)


@pytest.mark.parametrize("n",[7,15])
def test_finite_packets_match_direct_heterogeneous_heat_with_boundary(n):
    p=heat_1d(n,conductivity=lambda x:.1+2*x,source=lambda x:np.sin(2*np.pi*x),boundary=(-.3,1.))
    b=incidence(n)
    np.testing.assert_allclose(b.T@np.diag(p.edges_k/p.h**2)@b,p.matrix,atol=1e-12)
    base=solve(heat_1d(n).matrix,np.eye(n),assume_a="pos")
    prediction=material_response(base,np.ones(n+1),p.edges_k,p.rhs,p.h)
    np.testing.assert_allclose(prediction,p.reference(),atol=1e-11)
    assert p.diagnostics(prediction)["net_energy_balance_abs"]<1e-10


def test_oriented_packets_need_left_and_right_responses():
    n=10;dt=.03
    base=np.eye(n)+dt*periodic_transport(n,0.,.01)
    target=np.eye(n)+dt*periodic_transport(n,1.,.01)
    inverse=solve(base,np.eye(n))
    correct=oriented_inverse_update(inverse,target-base)
    wrong=oriented_inverse_update(inverse,target-base,reciprocal=True)
    np.testing.assert_allclose(target@correct,np.eye(n),atol=1e-12)
    assert np.linalg.norm(target@wrong-np.eye(n))>.01


def test_coarse_witness_and_source_energy_have_a_positive_source_collision():
    p=heat_1d();r=balance_probes(p.x);delta=witness_collision(r)
    plus,minus=1+.8*delta,1-.8*delta
    assert plus.min()>.1 and minus.min()>.1
    np.testing.assert_allclose(r@plus,r@minus,atol=1e-14)
    np.testing.assert_allclose(plus@plus,minus@minus,atol=1e-13)
    a,b=heat_1d(source=plus).reference(),heat_1d(source=minus).reference()
    assert np.linalg.norm(a-b)>1e-4


def test_group_law_is_associative_but_not_physical_source_addition():
    a,b,c=np.array([.2,-.3]),np.array([.7,.4]),np.array([-.5,.1])
    lhs=change_compose(change_compose(a,b),c)
    rhs=change_compose(a,change_compose(b,c))
    np.testing.assert_allclose(lhs,rhs,atol=1e-13)
    np.testing.assert_allclose(group_encode(change_compose(a,b)),group_encode(a)+group_encode(b),atol=1e-14)
    assert np.linalg.norm(change_compose(a,b)-(a+b))>.1


def test_conjugated_group_response_is_exactly_a_reparameterized_linear_sum():
    contributions=np.array([[.1,.2,-.05],[.3,-.2,.4]])
    np.testing.assert_allclose(compose_responses(contributions,conjugate=True),contributions.sum(1),atol=1e-14)
    assert np.linalg.norm(compose_responses(contributions)-contributions.sum(1))>.01


def test_ordered_segment_law_equals_transfer_matrix_and_is_associative():
    def matrix(z):
        r,s,c=z
        return np.array([[1.,-r,-c],[0.,1.,s],[0.,0.,1.]])
    a,b,c=np.array([.7,.2,.1]),np.array([1.3,-.4,.3]),np.array([.2,.3,-.1])
    np.testing.assert_allclose(matrix(segment_compose(a,b)),matrix(b)@matrix(a),atol=1e-14)
    np.testing.assert_allclose(segment_compose(segment_compose(a,b),c),segment_compose(a,segment_compose(b,c)),atol=1e-14)
    assert abs(segment_compose(a,b)[2]-segment_compose(b,a)[2])>.1


@pytest.mark.parametrize("n",[7,15,31])
def test_ordered_moments_match_heat_flux_for_new_materials_sources_and_bc(n):
    p=heat_1d(n,conductivity=lambda x:np.where(x<.3,.02,4.),
              source=lambda x:3*np.exp(-((x-.77)/.08)**2),boundary=(1.,-.5))
    prediction=segment_response(p.edges_k,p.source,p.boundary)
    np.testing.assert_allclose(prediction,p.reference(),atol=1e-11)
    assert p.diagnostics(prediction)["relative_residual"]<1e-11


def test_transfer_coefficient_sensitivity_is_explicit_and_unit_value_is_physical():
    p=heat_1d()
    constant,sensitivity=segment_features(p.edges_k,p.source,p.boundary)
    np.testing.assert_allclose(constant+sensitivity,p.reference(),atol=1e-12)
    assert np.linalg.norm(constant-p.reference())>.1


def test_reverse_advection_changes_phase_not_mass_and_energy():
    n=24;x=np.arange(n)/n;initial=1+.3*np.cos(2*np.pi*x)
    a=expm(-.2*periodic_transport(n,1.,.01))@initial
    b=expm(-.2*periodic_transport(n,-1.,.01))@initial
    np.testing.assert_allclose(a.mean(),b.mean(),atol=1e-13)
    np.testing.assert_allclose(np.mean(a*a),np.mean(b*b),atol=1e-13)
    assert np.linalg.norm(a-b)>.1


def test_temperature_scaling_must_preserve_nonzero_harmonic_lift():
    shape=lambda x:np.where(x<.5,1.,4.)
    p=heat_1d(conductivity=shape,boundary=(.5,1.5))
    changed=heat_1d(conductivity=lambda x:.02*shape(x),boundary=p.boundary)
    lift=heat_1d(conductivity=shape,source=0.,boundary=p.boundary).reference()
    np.testing.assert_allclose(lift+(p.reference()-lift)/.02,changed.reference(),atol=1e-11)
    assert np.linalg.norm(p.reference()/.02-changed.reference())>1
