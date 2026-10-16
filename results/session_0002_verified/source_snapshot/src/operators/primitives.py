"""Six falsifiable hypotheses. Mathematical overlap with classical methods is explicit."""

import numpy as np
from scipy.linalg import expm, solve
import torch
from torch import nn


def anchored_response(problem, length=0.2):
    """C001: boundary-anchored normalized source response (not reciprocal)."""
    x = problem.x
    distances = abs(x[:,None]-x[None,:])
    weights = np.exp(-distances/length)
    weights /= weights.sum(axis=1, keepdims=True)
    effective_k = 1/np.mean(1/problem.edges_k)
    green = x[:,None]*(1-x[:,None])/(2*effective_k)*weights
    lift = problem.boundary[0]*(1-x) + problem.boundary[1]*x
    return lift + green @ problem.source, green


def resistance_response(problem):
    """C001R: exact 1D resistance-coordinate Green function; CLASSICAL control."""
    resistance = np.r_[0., np.cumsum(problem.h/problem.edges_k)]
    r, total = resistance[1:-1], resistance[-1]
    green = problem.h*np.minimum(r[:,None],r[None,:])*(total-np.maximum(r[:,None],r[None,:]))/total
    lift = problem.boundary[0]*(1-r/total) + problem.boundary[1]*r/total
    return lift + green @ problem.source, green


def exchange_response(matrix, rhs, steps=32, step=0.1):
    """C002: explicit conservative relaxation; algebraically Richardson iteration."""
    state = np.zeros_like(rhs)
    for _ in range(steps):
        state = state + step*(rhs-matrix @ state)
    return state


def action_response(matrix, rhs, basis):
    """C003: stationarity in a reduced action span; exactly Ritz/Galerkin."""
    return basis @ solve(basis.T @ matrix @ basis, basis.T @ rhs, assume_a="pos")


def multiplicative_response(matrix, rhs, steps=128, step=0.001):
    """C004: positive response packets; cannot represent signed temperature deviations."""
    state = np.full_like(rhs, 0.1)
    for _ in range(steps):
        state = state*np.exp(step*(rhs-matrix @ state))
    return state


def material_event_paths(problem, amount=np.log(2)):
    """C005: noncommuting material events; compare identical physical endpoints."""
    x = problem.x
    a, b = np.sin(np.pi*x), np.sin(np.pi*x) + 0.8*np.sin(2*np.pi*x)
    a, b = a/np.linalg.norm(a), b/np.linalg.norm(b)
    first, second = -np.outer(a,a), -np.outer(b,b)
    start = problem.reference()
    e1, e2 = expm(amount*first), expm(amount*second)
    path12, path21 = e2 @ e1 @ start, e1 @ e2 @ start
    loop = expm(-amount*second) @ expm(-amount*first) @ e2 @ e1 @ start
    return path12, path21, loop, start


class LifetimeResponse(nn.Module):
    """C006: learned finite positive relaxation-age measure.

    g(lambda)=sum_j mass_j exp(-age_j lambda), approximating integral_0^inf exp(-s lambda) ds.
    An exponential-sum inverse approximation, not an established novel operation.
    """

    def __init__(self, packets=6, seed=0):
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        log_age = torch.linspace(-8., -0.5, packets, dtype=torch.float64)
        log_age += 0.05*torch.randn(packets, generator=generator, dtype=torch.float64)
        self.log_age = nn.Parameter(log_age)
        self.log_mass = nn.Parameter(log_age.clone())

    def forward(self, eigenvalues):
        return torch.exp(-eigenvalues[...,None]*self.log_age.exp()) @ self.log_mass.exp()

    def numpy_filter(self, eigenvalues):
        with torch.no_grad():
            return self(torch.as_tensor(eigenvalues, dtype=torch.float64)).numpy()


def spectral_response(matrix, rhs, model, null_tolerance=1e-10):
    eigenvalues, vectors = np.linalg.eigh(matrix)
    mask = eigenvalues > null_tolerance
    multiplier = np.zeros_like(eigenvalues)
    multiplier[mask] = model.numpy_filter(eigenvalues[mask])
    coordinates=vectors.T @ rhs
    filtered=multiplier*coordinates if coordinates.ndim==1 else multiplier[:,None]*coordinates
    return vectors @ filtered


def resolvent_response(matrix, rhs, lifetime=1e6):
    """C006R: stable high-frequency repair, exactly a shifted linear solve (REJECT novelty)."""
    return solve(matrix + np.eye(len(rhs))/lifetime, rhs, assume_a="pos")


def excitation_contrast(problem, length=0.2):
    """C007: source-dipole response. Symmetric PSD but has the WRONG inverse sign structure.

    Proposed as a contrast/gauge hypothesis, then falsified on constant and local sources.
    Potential gauge symmetry does not imply invariance under a constant source shift.
    """
    x=problem.x
    weights=np.exp(-abs(x[:,None]-x[None,:])/length)
    np.fill_diagonal(weights,0.)
    contrast=np.diag(weights.sum(axis=1))-weights
    effective_k=1/np.mean(1/problem.edges_k)
    return (length**2/effective_k)*contrast@problem.source
