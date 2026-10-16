# C001 — Boundary-anchored response average

**Origin/object:** equilibrium response anchored by distance to the boundary. This is a source-response object, not a vocabulary token.

## Inputs and implementable rule

Dimensionless interior coordinates `x_i∈(0,1)`, source values `q_i∈R`, positive edge conductivities `k_e`, Dirichlet values `a,b∈R`; tunable length `ell>0` (fixed at 0.2 in these falsification tests). Define

\[
k_{eff}=\left({1\over N+1}\sum_e k_e^{-1}\right)^{-1},\quad
w_{ij}={e^{-|x_i-x_j|/\ell}\over\sum_l e^{-|x_i-x_l|/\ell}},\quad
T_i=a(1-x_i)+b x_i+{x_i(1-x_i)\over 2k_{eff}}\sum_j w_{ij}q_j.
\]

Without nondimensionalization the anchor has units `L²/k`. Length is a learnable parameter in principle; no training is needed to disprove the current structural assumptions. Uniform source and material reproduce the analytic parabola.

## Guarantees and counterexamples

The homogeneous response is positive for nonnegative sources. Boundary values are imposed externally. `G=diag(x(1-x)/(2k_eff)) W` is generally not symmetric, so reciprocity is not guaranteed. A global effective conductivity cannot locate a material interface. Linear boundary lifting is not harmonic when `k` varies. Uniform-source success is insufficient: local sources and heterogeneous boundary tests falsify it. There is no general convergence theorem as resolution changes.

## Existing-operation comparison and complexity

It is a fixed positive kernel with row normalization, mathematically close to normalized kernel attention. There are no learned query/key vectors, but that alone is not novelty. Dense cost/time and memory are `O(N²)` per scalar channel; no GPU speed claim is made.

## Revision C001R (Cycle 6)

Let `R_i=∑_(edges before i) h/k_e`, `R_*=∑_e h/k_e`. Replace averaging by

\[
G_{ij}=h\,{\min(R_i,R_j)[R_*-\max(R_i,R_j)]\over R_*},\quad
T_i=a(1-R_i/R_*)+bR_i/R_*+\sum_j G_{ij}q_j.
\]

This changes both the material representation and coupling normalization. It is symmetric and exactly equals the inverse of this 1D tridiagonal discretization. The derivation is classical resistance-coordinate Green response, not an architecture invention. Dense implementation is `O(N²)`; prefix/suffix evaluation can be `O(N)` and is likewise classical. Generalization to holes or higher dimensions is not established by this formula.

**Verdict:** C001 REJECT for physical defects; C001R REJECT as an independent invention, retain as an exact classical control. Executed evidence: Cycles 1 and 6 in `reports/cycles/` and `results/session_0002_verified/metrics/all_cases.json`.
