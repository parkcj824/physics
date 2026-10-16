# C004 — Multiplicative response mass

**Origin/object:** a positive response mass changed by irreversible driving.

## Inputs and rule

Dimensionless state `u_i>0`, SPD physical operator `A`, load `b`, positive potentially learned rate `eta`. Initialize `u_i=0.1` and compute

\[
r=b-Au,\qquad u_i^+=u_i\exp(\eta r_i),\quad\eta=10^{-3}.
\]

This resembles mirror/entropy-gradient motion for a positive state. Tests run 128 steps. No learned parameters are fitted before testing whether the representation can contain the true solution.

## Guarantees and failure

For finite exponents, positivity is exact. That is the only unconditional guarantee here. Finite `eta` does not guarantee energy decrease, stiffness robustness, conservation or convergence; exponent overflow is possible. A signed cooling source with zero boundary values has a strictly negative temperature *deviation*. The candidate cannot represent it for any finite parameters. Absolute Kelvin positivity is a different modeling choice and cannot remove this counterexample by silently changing variables. Mean-zero nontrivial Neumann states also require signs. Temperature-origin translation equivariance fails.

Duplicating positive packets as `u=u_plus-u_minus` was considered analytically, not executed. It changes the object, introduces nonidentifiability, and has no established improvement; record it as unrun rather than a repaired result.

## Difference, complexity, extension

The multiplicative rule differs from additive attention or convolution but overlaps mirror descent/exponentiated gradient. Sparse cost per step is `O(N k_nb d)` and memory `O(N k_nb+Nd)`; the dense prototype is `O(N²)` per step. It can be meaningful for concentrations or densities, not a generic signed-field primitive. Geometry and resolution robustness are unproved.

**Verdict:** REJECT for the current generic thermal-state representation. This does not prove all entropy-based architectures invalid. Cycle 4 tests both positive and negative sources.
