# Physical principles and falsification contract

This session is an invention search, not a claim that an invention has been found. The existing `0001-*` oscillator records are preserved. The project root is the existing `/workspace/physics` checkout; a second nested repository is unnecessary.

## A. Problem information

Domain, geometry, boundary/initial conditions, material tensors, sources, time and units define the physical problem. They are not automatically tokens. All current experiments use dimensionless variables: `x/L`, `k/k0`, `T/T0`, and `Q L²/(k0 T0)`. A response multiplying a source must have inverse-operator units before this scaling. Flow viscosity and forcing are separately nondimensionalized; numerical similarity of heat and Stokes operators does not equate their physical units.

## Principles tested

1. **Response/reciprocity.** A self-adjoint elliptic operator has a symmetric Green response in the correct quadrature/mass inner product. A row-normalized averaging rule need not have it.
2. **Conservation/exchange.** Opposite internal transfers cancel, but this does not make a finite iteration a correct steady solution. Boundary exchange and sources remain essential.
3. **Stationary action.** Minimizing `E(T)=TᵀAT/2-bᵀT` gives `AT=b`. Restricting the variation to a span gives Galerkin, not a new neural primitive.
4. **Irreversibility/positive state.** Positive absolute concentrations are useful objects. Signed temperature deviations and gauge-fixed Neumann temperatures need signed representations. Entropic/multiplicative updates cannot ignore this distinction.
5. **Events/endpoint consistency.** Steady material response depends on the final material field, not on the ordering of artificial parameter edits. Noncommuting event updates can create fictitious history dependence.
6. **Relaxation/response age.** For positive self-adjoint `A`, `A⁻¹=∫₀∞ exp(-sA) ds`. A learned finite positive measure gives a small parameterization but inherits finite spectral support and classical matrix-function ancestry.
7. **Gauge/source contrasts.** A potential's additive gauge does not make an additive source shift a gauge. A PSD source-contrast matrix may have precisely the wrong sign pattern for an inverse diffusion response.

## Governing equations actually used

- A/B: `-div(K grad T)=Q`, including 1D scalar `k`; Dirichlet, and a separate compatible mean-zero 1D Neumann control.
- C: `∂t u + v ∂x u = k ∂xx u + S`; periodic and inflow Dirichlet. The user prompt omitted its equality sign; this standard form is used explicitly. The nonselfadjoint part carries direction.
- D: `rho (u·grad)u = -grad p + mu Δu + f`, `div u=0`. Poiseuille, Couette, periodic manufactured Stokes, and a negative test with nonlinear convection are executed. No RANS claim is made.

## Mathematical claims versus measurements

- SPD of `g(A)` gives positive quadratic work and reciprocity, not nodal positivity or small PDE residual.
- If `-A` is Metzler (equivalently off-diagonal `A_ij≤0`), positive exponential mixtures preserve pointwise positivity. Arbitrary SPD anisotropic FEM matrices need not satisfy this.
- A divergence-free projected velocity can still violate momentum balance severely.
- Imposed boundary values are not learned BC generalization. Zero recorded BC error means prescribed lifting worked.
- Exact agreement with a discrete solver is different from convergence to the continuum equation. Analytic 1D and manufactured 2D convergence checks are separate.
- A classical solver written inside a neural model remains a solver. Setup/eigendecomposition and per-source inference costs are reported separately where possible.

## Stop/restart rule

The session has a maximum of 12 scientific cycles. Replays to verify measurements are labeled pilot/revision/final, not counted as independent discoveries. After three cycles without meaningful novelty progress, reconsider representation; Cycle 12 changes from relaxation-age packets to source contrasts. No candidate is promoted by renaming it. Continue from `next_steps.md` after preserving current results.
