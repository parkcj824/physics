"""Session 0003 hypotheses and explicit classical-equivalence controls.

No operation in this module is claimed to be a novel architecture.
All arrays are dimensionless; external data and previous results are read-only.
"""

import numpy as np


def incidence(n):
    """Grounded 1D edge differences, A=B.T diag(k/h²) B."""
    b = np.zeros((n + 1, n))
    b[0, 0] = 1
    for i in range(1, n):
        b[i, i - 1:i + 1] = [-1, 1]
    b[-1, -1] = -1
    return b


def material_response(inverse_base, base_k, new_k, rhs, h, mode="finite"):
    """A: finite response packet; the complete update is Sherman–Morrison.

    'frozen' leaves cross-susceptibilities stale; 'tangent' omits finite denominator.
    The supplied base inverse is an explicit strong prior, not a secretly learned model.
    """
    base = np.array(inverse_base, copy=True)
    current = base.copy()
    state = base @ rhs
    for d, delta in zip(incidence(len(rhs)), (new_k - base_k) / h**2):
        if abs(delta) < 1e-15:
            continue
        packet = (base if mode == "frozen" else current) @ d
        susceptibility = d @ packet
        denominator = 1 + delta * susceptibility
        if mode != "tangent" and denominator <= 0:
            raise ValueError("updated packet violates positive compliance condition")
        factor = delta if mode == "tangent" else delta / denominator
        state = state - factor * packet * (d @ state)
        if mode == "finite":
            current -= factor * np.outer(packet, packet)
    return state


def oriented_inverse_update(inverse_base, difference, reciprocal=False):
    """General finite left/right packets; exactly nonsymmetric Sherman–Morrison."""
    inverse = np.array(inverse_base, copy=True)
    n = len(inverse)
    for j in range(n):
        left = inverse @ difference[:, j]
        row = inverse[j, :]
        denominator = 1 + left[j]
        if abs(denominator) < 1e-12:
            raise ValueError("singular intermediate update")
        if reciprocal:
            # A deliberately invalid reciprocal shortcut on nonselfadjoint operators.
            inverse -= np.outer(left, inverse[:, j]) / denominator
        else:
            inverse -= np.outer(left, row) / denominator
    return inverse


def balance_probes(x, rank=4):
    """Coarse weak balances, orthonormalized for conditioning (not learned)."""
    raw = np.column_stack([np.ones(len(x)), x, np.cos(np.pi*x), np.sin(2*np.pi*x)])
    q, _ = np.linalg.qr(raw[:, :rank])
    return q.T


def witness_decode(matrix, rhs, probes):
    """B: minimum-norm closure of coarse witnesses, explicitly a pseudoinverse."""
    constraints = probes @ matrix
    return np.linalg.lstsq(constraints, probes @ rhs, rcond=None)[0]


def witness_collision(probes):
    """A nonzero source contrast invisible to every supplied coarse witness."""
    _, singular, vh = np.linalg.svd(probes, full_matrices=True)
    if len(singular) == probes.shape[1]:
        raise ValueError("full rank witness has no null source")
    contrast = vh[-1]
    return contrast / np.max(np.abs(contrast))


def group_encode(value, gamma=4.0):
    return np.arcsinh(gamma*np.asarray(value))/gamma


def group_decode(value, gamma=4.0):
    return np.sinh(gamma*np.asarray(value))/gamma


def change_compose(a, b, gamma=4.0):
    """C: nonlinear associative change law, conjugate to ordinary addition."""
    return a*np.sqrt(1+(gamma*b)**2) + b*np.sqrt(1+(gamma*a)**2)


def compose_responses(contributions, gamma=4.0, conjugate=False):
    """Each column is one response. Conjugate=True exposes pure reparameterization."""
    state = np.zeros(contributions.shape[0])
    for response in contributions.T:
        value = group_decode(response, gamma) if conjugate else response
        state = change_compose(state, value, gamma)
    return group_encode(state, gamma) if conjugate else state
