# Cycle 10 — Heat-to-flow challenge

## 1. Physical principle
constraint versus governing dynamics

## 2. Minimal computational object
response in a divergence-free subspace

## 3. Implemented formula
\[
\hat u=g_\theta(\mu|\xi|^2)(I-\xi\xi^T/|\xi|^2)\hat f
\]

The complete inputs, trainable/fixed quantities, guarantees and complexity are in `../../architectures/candidate_*.md`.

## 4. Difference from existing operations
fixed Helmholtz projection plus the heat-trained scalar response is a classical structure-restricted reuse test

## 5. Governing equation, conditions and reference
Poiseuille, Couette, periodic manufactured Stokes and nonzero-inertia manufactured Navier-Stokes
References are independent direct discrete solves, analytic solutions, augmented gauge solves or matrix exponentials as identified in each saved record. The anisotropic positivity counterexample intentionally uses the exact discrete semigroup to expose a discretization defect.

## 6. Executed results
Final verified run: `results/session_0002_final`; 22 records in this cycle. Complete inputs, predictions and references are saved in the NPZ paths in the metrics JSON.

| Candidate | Case | Relative L2 | Max absolute error | Median seconds | Traced peak bytes |
|---|---|---:|---:|---:|---:|
| C006 | poiseuille_n15_mu0.1 | 0.470048 | 0.606728 | 5.5567e-05 | 3544 |
| C006R | poiseuille_n15_mu0.1 | 1.01575e-06 | 1.30615e-06 | 2.3994e-05 | 7360 |
| C006 | poiseuille_n15_mu1.0 | 0.00257208 | 0.000338838 | 5.2709e-05 | 3544 |
| C006R | poiseuille_n15_mu1.0 | 1.01575e-07 | 1.30615e-08 | 2.3962e-05 | 7360 |
| C006 | poiseuille_n15_mu10.0 | 0.00590138 | 7.91528e-05 | 7.7109e-05 | 3544 |
| C006R | poiseuille_n15_mu10.0 | 1.01575e-08 | 1.30615e-10 | 2.5521e-05 | 7360 |
| C006 | poiseuille_n31_mu0.1 | 0.469151 | 0.605596 | 9.2021e-05 | 9832 |
| C006R | poiseuille_n31_mu0.1 | 1.0133e-06 | 1.3031e-06 | 2.9887e-05 | 17068 |
| C006 | poiseuille_n31_mu1.0 | 0.00257069 | 0.000325776 | 0.000106042 | 9832 |
| C006R | poiseuille_n31_mu1.0 | 1.0133e-07 | 1.3031e-08 | 3.1071e-05 | 17068 |
| C006 | poiseuille_n31_mu10.0 | 0.00613672 | 8.32622e-05 | 0.000130657 | 9832 |
| C006R | poiseuille_n31_mu10.0 | 1.0133e-08 | 1.3031e-10 | 3.1457e-05 | 17068 |
| lifting_control | couette | 0 | 0 | 3.95e-07 | 216 |
| C006 | periodic_mu1.0_inertia0.0 | 0.00651245 | 0.04319 | 0.000108416 | 29393 |
| C006R | periodic_mu1.0_inertia0.0 | 1.2172e-08 | 9.23099e-08 | 4.5132e-05 | 29328 |
| spectral_stokes_control | periodic_mu1.0_inertia0.0 | 2.02189e-16 | 2.66454e-15 | 5.0595e-05 | 29328 |
| C006 | periodic_mu0.1_inertia0.0 | 0.000996362 | 0.00823679 | 7.7641e-05 | 29393 |
| C006R | periodic_mu0.1_inertia0.0 | 1.2172e-07 | 9.23099e-07 | 3.8185e-05 | 29328 |
| spectral_stokes_control | periodic_mu0.1_inertia0.0 | 2.05345e-16 | 2.66454e-15 | 4.093e-05 | 29328 |
| C006 | periodic_mu0.1_inertia1.0 | 0.272424 | 2.38209 | 9.4289e-05 | 29393 |
| C006R | periodic_mu0.1_inertia1.0 | 0.272217 | 2.37348 | 3.7661e-05 | 29328 |
| spectral_stokes_control | periodic_mu0.1_inertia1.0 | 0.272217 | 2.37348 | 5.0454e-05 | 29328 |

Array/batch relative L2 and mean per-sample relative L2 are distinct. Per-sample means/std, heat flux errors, physical balances, BC handling, momentum/divergence residuals and timing/memory scope are in the JSON. Tracemalloc does not measure all native/BLAS memory; RSS is cumulative process high-water. Timing is a small CPU microbenchmark, not GPU performance.

## 7. Self-refutation in six roles
- A — Inventor: The same heat-trained filter reuses linear response mathematics in projected Stokes.
- B — Theoretical physicist: Zero divergence does not enforce nonlinear momentum; small viscosity also leaves the trained spectrum.
- C — Numerical analyst: The exact Stokes control fails the nonlinear test too, identifying a model omission rather than a training bug.
- D — Architecture researcher: Projection plus spectral response is not a new primitive simply because two domains use it.
- E — Patent-review perspective: No generic CAE or turbulence claim follows from these manufactured linear successes. This is not a patentability finding.
- F — Alternative-hypothesis researcher: Strengthen baseline conditioning before inferring any architecture advantage.

## 8. Prior work and verification limits
See `../../research/novelty_review.md` for checked official source descriptions/code and the full coverage ledger. Original paper sites returned proxy 403; no unread full text is presented as reviewed. Exact classical algebraic equivalences are sufficient to withhold an invention claim. Architectural novelty and patent coverage are unverified.

## 9. Verdict
**REVISE**. This is a cycle-level decision; final candidate-level dispositions are in `../../research/candidate_registry.md`. No final invention is selected.

## 10. Next cycle hypothesis / stopping state
Strengthen baseline conditioning before inferring any architecture advantage.
