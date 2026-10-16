# C006 — Positive relaxation-age response measure

**Origin/object:** an age packet describing how long an excitation has relaxed; not a spatial token.

## Inputs, trainable quantities, rule

Dimensionless self-adjoint positive operator `A=V diag(lambda) Vᵀ`, lifted/mass-scaled load `b∈R^N`, packet count `m=6`. Learn 12 real scalars `alpha_j,beta_j`; set `w_j=exp(alpha_j)>0`, `tau_j=exp(beta_j)>0` and compute

\[
g_\theta(\lambda)=\sum_{j=1}^m w_j e^{-\tau_j\lambda},\qquad
T=V\operatorname{diag}(g_\theta(\lambda))V^T b.
\]

The implementation is fully expanded in `LifetimeResponse` and `spectral_response`. Before nondimensionalization `tau` has units of inverse operator eigenvalue and `w` has inverse-operator units. The model is a finite learned quadrature for `A^-1=integral exp(-sA) ds`, not a claim to invent that identity. Boundary/constraint restriction is fixed, not learned.

## Data flow and learning

Physical problem → classical reference assembly and mass coordinates → admissible positive-mode eigensystem → transform source → six positive age responses → sum → inverse transform → prescribed boundary lift. For 1D Neumann, a compatible source and explicit mean-zero projection are necessary; only the separate classical gauge control was executed, so learned Neumann performance is unrun.

Train on 128 small 1D heat problems (`N=15`, scalar uniform `k∈[0.5,2]`), exact discrete labels, normalized per-sample solution MSE, Adam 1,000 steps, seeds 824/825/826. Validation and test samples are separate. The seed824 model is fixed in advance for expanded physics tests; it is not cherry-picked as the best seed.

## Guarantees, conditional claims, expressivity

- `g(lambda)>0` on finite positive eigenvalues, so reciprocal SPD response and nonnegative quadratic work are exact up to roundoff in these coordinates.
- Permutation/orthogonal basis covariance holds when the physical operator, source and boundary restriction transform together. Rotational Euclidean equivariance of arbitrary learned geometry encoders is not established.
- Pointwise positivity follows from the semigroup only when the discrete heat generator is positivity preserving. SPD alone is insufficient; the anisotropic nonmonotone-FEM counterexample is executed in Cycle 8.
- A finite positive age sum has finite `g(0)` and exponentially decaying high-frequency tail; it cannot uniformly approximate `1/lambda` on `(0,infinity)`. Both low-material and high-frequency OOD tests are mandatory.
- It is linear in `b`; it cannot by itself represent a nonlinear Navier–Stokes solution operator. Projecting velocity enforces divergence, not momentum.
- Changing resolution/geometry changes the supplied eigensystem. Reusing the filter across these inputs is not evidence of a mesh-free learned geometry representation.
- It has no direct temporal semigroup guarantee. A self-adjoint restriction loses advective direction; Cycle 9 tests that separately.

## Complexity and comparison

Dense setup eigendecomposition costs `O(N³)` time and `O(N²)` memory. Per source, cached transforms cost `O(N²d)` and scalar filtering `O(Nm)`; storage includes `O(N²+Nm+Nd)`. Analytic Fourier bases on periodic rectangles can reduce transforms to `O(N log N)`, but this is geometry-specific. The prototype does not implement a fast general matrix-function approximation or demonstrate a CAE speedup.

FNO learns Fourier-mode/channel multipliers; here the eigenbasis comes from the actual physical operator and the scalar multiplier is constrained to a positive exponential sum. That is a concrete difference in restrictions/inputs, not enough to establish novelty relative to spectral operator learning, matrix-function approximation or learned preconditioners.

## Revision C006R

To repair the high-frequency tail, test `g_R(lambda)=tau/(1+tau lambda)=1/(lambda+1/tau)`, `tau=10^6`:

\[
T_R=(A+\tau^{-1}I)^{-1}b.
\]

This changes exponential decay to the correct `1/lambda` high-frequency tail. The response error is `1/(1+tau lambda)` relative to the exact inverse eigenmode. It is exactly a shifted solve/resolvent, with large `tau` approaching the original solver. Dense cost remains `O(N³)` per uncached factorization. Transient `(I+dt L)^-1` extension is backward Euler. Neither is a new neural primitive.

**Verdict:** C006 REVISE as a numerical research probe, **REJECT as a substantiated independent invention**; C006R REJECT for novelty and solver cost. A broad positive spectral parameterization may remain useful, but no existing-art overlap search is complete and no final architecture is selected. Cycles 7–11 supply accuracy, failure, timing, learning and strong-baseline evidence.
