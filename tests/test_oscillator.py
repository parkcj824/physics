import numpy as np
from scipy.integrate import solve_ivp
import torch

from experiments.oscillator import QuadraticHamiltonian, integrate_baselines, train


def test_scipy_matches_analytic_oscillator():
    t = np.linspace(0, 20, 401)
    solution = solve_ivp(lambda _, y: [y[1], -y[0]], (0, 20), [1, 0],
                         t_eval=t, rtol=1e-10, atol=1e-12)
    assert solution.success
    np.testing.assert_allclose(solution.y.T, np.column_stack((np.cos(t), -np.sin(t))), atol=1e-8)


def test_hamiltonian_vector_field_matches_closed_form_and_conserves_energy():
    model = QuadraticHamiltonian()
    state = torch.tensor([[0.3, -0.8], [1.2, 0.4]], dtype=torch.float64, requires_grad=True)
    field = model.vector_field(state)
    a, b, c = model.coefficients
    q, p = state.unbind(-1)
    expected = torch.stack((b*p + c*q, -a*q - c*p), dim=-1)
    torch.testing.assert_close(field, expected)
    gradient = torch.autograd.grad(model(state).sum(), state)[0]
    torch.testing.assert_close((gradient * field).sum(-1), torch.zeros(2, dtype=torch.float64), atol=1e-14, rtol=0)


def test_training_recovers_oscillator_and_unseen_derivatives():
    model, metrics = train()
    assert metrics['final_loss'] < 1e-12
    assert metrics['final_loss'] < metrics['initial_loss'] * 1e-8
    torch.testing.assert_close(model.coefficients, torch.tensor([1., 1., 0.], dtype=torch.float64), atol=1e-6, rtol=0)
    unseen = torch.tensor([[2., -3.], [-4., 2.]], dtype=torch.float64, requires_grad=True)
    torch.testing.assert_close(model.vector_field(unseen), torch.stack((unseen[:, 1], -unseen[:, 0]), -1), atol=1e-6, rtol=0)
    assert model.coefficients.device.type == 'cpu'


def test_euler_negative_control_has_predicted_energy_growth():
    _, euler, _ = integrate_baselines()
    ratio = np.sum(euler[-1]**2)
    np.testing.assert_allclose(ratio, (1 + 0.05**2)**400, rtol=1e-12)
    assert ratio - 1 > 1.0


def test_leapfrog_keeps_energy_error_bounded():
    _, _, leapfrog = integrate_baselines()
    relative_error = np.abs(np.sum(leapfrog**2, axis=1) - 1)
    assert relative_error.max() < 0.001
