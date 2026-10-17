"""Classical reference physics for Session 0004; none is an invented primitive."""

import numpy as np
from scipy.linalg import eigvalsh_tridiagonal, solve_banded
from scipy.optimize import brentq


def bratu_lambda(a):
    return 2 * a**2 / np.cosh(a / 2)**2


def fold():
    a = brentq(lambda z: z * np.tanh(z / 2) - 2, 1, 4, xtol=1e-14)
    return a, bratu_lambda(a)


def bratu_profile(a, x):
    # logaddexp form also avoids unnecessarily large cosh intermediates.
    logcosh = lambda z: np.logaddexp(z, -z) - np.log(2)
    return 2 * (logcosh(a / 2) - logcosh(a * (np.asarray(x) - .5)))


def bratu_branches(lam, x):
    astar, critical = fold()
    if not 0 < lam < critical:
        raise ValueError("two-branch reference requires 0 < lambda < fold")
    low = brentq(lambda a: bratu_lambda(a) - lam, 1e-10, astar, xtol=1e-13)
    high = brentq(lambda a: bratu_lambda(a) - lam, astar, 80, xtol=1e-13)
    return np.array([bratu_profile(a, x) for a in [low, high]]), np.array([low, high])


def diffusion_bands(k, n):
    h = 1 / (n + 1)
    k = np.broadcast_to(k, (n + 1,))
    if np.any(k <= 0):
        raise ValueError("positive conductivity required")
    bands = np.zeros((3, n))
    bands[1] = (k[:-1] + k[1:]) / h**2
    bands[0, 1:] = -k[1:-1] / h**2
    bands[2, :-1] = -k[1:-1] / h**2
    return bands


def apply_bands(bands, t):
    v = bands[1] * t
    v[:-1] += bands[0, 1:] * t[1:]
    v[1:] += bands[2, :-1] * t[:-1]
    return v


def source_rhs(k, q, boundary):
    n = len(q); h = 1 / (n + 1)
    k = np.broadcast_to(k, (n + 1,))
    b = np.array(q, dtype=float, copy=True)
    b[0] += k[0] * boundary[0] / h**2
    b[-1] += k[-1] * boundary[1] / h**2
    return b


def steady_heat(k, q, boundary=(0., 0.), lam=0., initial=None, tolerance=2e-10):
    """Damped Newton with an explicit failure, O(N) banded linear steps.

    A converged root need not be the only root. No claim of global convergence.
    """
    n = len(q); bands = diffusion_bands(k, n); b = source_rhs(k, q, boundary)
    t = solve_banded((1, 1), bands, b) if initial is None else np.array(initial, dtype=float, copy=True)
    for step in range(81):
        residual = apply_bands(bands, t) - b - lam * np.exp(t)
        if np.max(np.abs(residual)) < tolerance:
            return t, step
        if step == 80:
            break
        jac = bands.copy(); jac[1] -= lam * np.exp(t)
        delta = solve_banded((1, 1), jac, -residual)
        alpha = 1.
        while alpha > 2**-20:
            trial = t + alpha * delta
            if np.max(trial) < 50:
                r = apply_bands(bands, trial) - b - lam * np.exp(trial)
                if np.linalg.norm(r) < np.linalg.norm(residual):
                    break
            alpha *= .5
        else:
            break
        t = trial
    raise RuntimeError(f"Newton failed: lambda={lam}, max residual={np.max(abs(residual))}")


def flux(k, t, boundary=(0., 0.)):
    return -np.asarray(k) * np.diff(np.r_[boundary[0], t, boundary[1]]) * (len(t) + 1)


def diagnostics(k, q, boundary, lam, t, generated_flux=None):
    j = flux(k, t, boundary)
    source = np.asarray(q) + lam * np.exp(t)
    h = 1 / (len(t) + 1)
    residual = np.diff(j) / h - source
    result = {"pde_residual_inf": float(max(abs(residual))),
              "relative_pde_residual": float(np.linalg.norm(residual) / max(np.linalg.norm(source), 1e-14)),
              "constitutive_energy_balance_abs": float(abs(j[-1] - j[0] - h * sum(source))),
              "boundary_error_abs": 0.}
    if generated_flux is not None:
        result["generated_balance_inf"] = float(max(abs(np.diff(generated_flux) / h - source)))
        result["generated_energy_balance_abs"] = float(abs(generated_flux[-1] - generated_flux[0] - h * sum(source)))
        result["constitutive_mismatch_inf"] = float(max(abs(generated_flux - j)))
    return result


def stability(k, lam, t):
    """Largest eigenvalue of du/dt = -A u + lambda exp(T) u."""
    bands = diffusion_bands(k, len(t))
    return float(eigvalsh_tridiagonal(-bands[1] + lam * np.exp(t), -bands[0, 1:],
                                    select='i', select_range=(len(t)-1, len(t)-1))[0])
