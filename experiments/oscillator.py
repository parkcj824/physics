"""Learn a quadratic Hamiltonian and measure integration error on a unit oscillator."""

import argparse
import json
import platform
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.integrate import solve_ivp
import torch
from torch import nn


class QuadraticHamiltonian(nn.Module):
    """H(q,p) = (a q² + b p²)/2 + c qp; a baseline, not a new architecture."""

    def __init__(self):
        super().__init__()
        self.coefficients = nn.Parameter(torch.tensor([0.7, 1.3, 0.2], dtype=torch.float64))

    def forward(self, state):
        q, p = state.unbind(-1)
        a, b, c = self.coefficients
        return 0.5 * (a * q.square() + b * p.square()) + c * q * p

    def vector_field(self, state):
        grad = torch.autograd.grad(self(state).sum(), state, create_graph=True)[0]
        return torch.stack((grad[..., 1], -grad[..., 0]), dim=-1)


def train(seed=824, steps=400):
    """Fit derivative observations from H=(q²+p²)/2 on CPU in float64."""
    torch.set_num_threads(1)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    state = (2 * torch.rand(128, 2, generator=generator, dtype=torch.float64) - 1).requires_grad_()
    target = torch.stack((state[:, 1], -state[:, 0]), dim=-1).detach()
    model = QuadraticHamiltonian()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    initial_loss = None
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        state.grad = None
        loss = (model.vector_field(state) - target).square().mean()
        if initial_loss is None:
            initial_loss = loss.item()
        loss.backward()
        optimizer.step()
    final_loss = (model.vector_field(state) - target).square().mean().item()
    return model, {"seed": seed, "steps": steps, "initial_loss": initial_loss, "final_loss": final_loss}


def integrate_baselines(dt=0.05, steps=400):
    """Euler is an intentional negative control; leapfrog is the comparator."""
    euler = np.empty((steps + 1, 2))
    leapfrog = np.empty_like(euler)
    euler[0] = leapfrog[0] = [1.0, 0.0]
    for i in range(steps):
        q, p = euler[i]
        euler[i + 1] = [q + dt * p, p - dt * q]
        q, p = leapfrog[i]
        p_half = p - 0.5 * dt * q
        q_next = q + dt * p_half
        leapfrog[i + 1] = [q_next, p_half - 0.5 * dt * q_next]
    return np.arange(steps + 1) * dt, euler, leapfrog


def run(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    model, metrics = train()
    t, euler, leapfrog = integrate_baselines()
    reference = solve_ivp(lambda _, y: [y[1], -y[0]], (0, t[-1]), [1, 0],
                          t_eval=t, rtol=1e-10, atol=1e-12)
    a, b, c = model.coefficients.detach().numpy()
    learned = solve_ivp(lambda _, y: [b*y[1] + c*y[0], -a*y[0] - c*y[1]],
                        (0, t[-1]), [1, 0], t_eval=t, rtol=1e-10, atol=1e-12)
    if not reference.success or not learned.success:
        raise RuntimeError("SciPy integration failed")
    analytic = np.column_stack((np.cos(t), -np.sin(t)))
    energy = lambda y: 0.5 * np.sum(y*y, axis=-1)
    metrics.update({
        "coefficients": [float(a), float(b), float(c)],
        "device": str(model.coefficients.device), "dtype": str(model.coefficients.dtype),
        "dt": 0.05, "integration_steps": 400,
        "reference_max_abs_error": float(np.max(np.abs(reference.y.T - analytic))),
        "learned_rollout_max_abs_error": float(np.max(np.abs(learned.y.T - analytic))),
        "euler_final_relative_energy_error": float(energy(euler)[-1] / 0.5 - 1),
        "leapfrog_max_relative_energy_error": float(np.max(np.abs(energy(leapfrog) / 0.5 - 1))),
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "scipy": scipy.__version__, "matplotlib": matplotlib.__version__, "torch": torch.__version__},
        "novelty": "unverified; baseline only; no literature review performed",
    })
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    np.savez(output / "trajectories.npz", time=t, analytic=analytic,
             reference=reference.y.T, learned=learned.y.T, euler=euler, leapfrog=leapfrog)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(t, analytic[:, 0], label="analytic q")
    axes[0].plot(t, learned.y[0], "--", label="learned q")
    axes[0].set(xlabel="time", ylabel="position")
    for name, values in [("Euler (negative control)", euler), ("leapfrog", leapfrog)]:
        axes[1].plot(t, energy(values) / 0.5 - 1, label=name)
    axes[1].set(xlabel="time", ylabel="relative energy error")
    for ax in axes:
        ax.legend()
    fig.tight_layout()
    fig.savefig(output / "oscillator.png", dpi=150)
    plt.close(fig)
    print(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="artifacts/oscillator")
    run(parser.parse_args().output)
