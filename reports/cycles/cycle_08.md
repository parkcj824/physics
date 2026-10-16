# Cycle 8 — Spectral repair and geometry

## 1. Physical principle
relaxation, reciprocity and positivity

## 2. Minimal computational object
age measure or shifted-resolvent response

## 3. Implemented formula
\[
g_R(\lambda)=1/(\lambda+10^{-6});\quad T_R=(A+10^{-6}I)^{-1}b
\]

The complete inputs, trainable/fixed quantities, guarantees and complexity are in `../../architectures/candidate_*.md`.

## 4. Difference from existing operations
tail repair is exactly a shifted solve; geometry enters through classical FEM assembly

## 5. Governing equation, conditions and reference
1D N=7/15/31/63, low/high k; 2D rectangle/hole/aspect/material/anisotropy/BC/source; manufactured convergence
References are independent direct discrete solves, analytic solutions, augmented gauge solves or matrix exponentials as identified in each saved record. The anisotropic positivity counterexample intentionally uses the exact discrete semigroup to expose a discretization defect.

## 6. Executed results
Final verified run: `results/session_0002_verified`; 61 records in this cycle. Complete inputs, predictions and references are saved in the NPZ paths in the metrics JSON.

| Candidate | Case | Relative L2 | Max absolute error | Median seconds | Traced peak bytes |
|---|---|---:|---:|---:|---:|
| C006 | n7_k0.02 | 0.83873 | 0.423509 | 0.000214407 | 1936 |
| C006R | n7_k0.02 | 4.8803e-06 | 2.32732e-06 | 2.0581e-05 | 5952 |
| direct_solver | n7_k0.02 | 0 | 0 | 1.3779e-05 | 1124 |
| C006 | n7_k1.0 | 0.00341879 | 4.79538e-05 | 0.000120224 | 1936 |
| C006R | n7_k1.0 | 9.76065e-08 | 9.30935e-10 | 6.5469e-05 | 5952 |
| direct_solver | n7_k1.0 | 0 | 0 | 1.7811e-05 | 1124 |
| C006 | n7_k20.0 | 0.0329088 | 2.84361e-05 | 4.4093e-05 | 1936 |
| C006R | n7_k20.0 | 4.88033e-09 | 2.32734e-12 | 2.5946e-05 | 5952 |
| direct_solver | n7_k20.0 | 0 | 0 | 1.3951e-05 | 1124 |
| C006 | n15_k0.02 | 0.84199 | 0.662036 | 5.3943e-05 | 3479 |
| C006R | n15_k0.02 | 4.8787e-06 | 3.70812e-06 | 2.1248e-05 | 7360 |
| direct_solver | n15_k0.02 | 0 | 0 | 1.6455e-05 | 2820 |
| C006 | n15_k1.0 | 0.00331363 | 8.45835e-05 | 5.7797e-05 | 3544 |
| C006R | n15_k1.0 | 9.75745e-08 | 1.48325e-09 | 2.0261e-05 | 7360 |
| direct_solver | n15_k1.0 | 0 | 0 | 1.6005e-05 | 2820 |
| C006 | n15_k20.0 | 0.0355747 | 6.15152e-05 | 0.000108133 | 3479 |
| C006R | n15_k20.0 | 4.87873e-09 | 3.70814e-12 | 2.2126e-05 | 7360 |
| direct_solver | n15_k20.0 | 0 | 0 | 1.5935e-05 | 2820 |
| C006 | n31_k0.02 | 0.841986 | 0.658706 | 0.000107688 | 10072 |
| C006R | n31_k0.02 | 4.87015e-06 | 3.68155e-06 | 2.8428e-05 | 17068 |
| direct_solver | n31_k0.02 | 0 | 0 | 2.214e-05 | 9284 |
| C006 | n31_k1.0 | 0.0032857 | 8.44786e-05 | 0.000173063 | 10072 |
| C006R | n31_k1.0 | 9.74034e-08 | 1.47263e-09 | 3.0464e-05 | 17068 |
| direct_solver | n31_k1.0 | 0 | 0 | 2.143e-05 | 9284 |
| C006 | n31_k20.0 | 0.0344932 | 5.61722e-05 | 0.000119404 | 10072 |
| C006R | n31_k20.0 | 4.87017e-09 | 3.68157e-12 | 2.6624e-05 | 17068 |
| direct_solver | n31_k20.0 | 0 | 0 | 3.1283e-05 | 9284 |
| C006 | n63_k0.02 | 0.841992 | 0.658408 | 0.000334871 | 35383 |
| C006R | n63_k0.02 | 4.86811e-06 | 3.67986e-06 | 6.1768e-05 | 66348 |
| direct_solver | n63_k0.02 | 0 | 0 | 4.9365e-05 | 34500 |
| C006 | n63_k1.0 | 0.003281 | 8.5939e-05 | 0.000302861 | 35448 |
| C006R | n63_k1.0 | 9.73627e-08 | 1.47195e-09 | 6.1824e-05 | 66348 |
| direct_solver | n63_k1.0 | 0 | 0 | 5.4244e-05 | 34500 |
| C006 | n63_k20.0 | 0.0343062 | 5.73574e-05 | 0.000375535 | 35448 |
| C006R | n63_k20.0 | 4.86814e-09 | 3.67989e-12 | 6.7188e-05 | 66348 |
| direct_solver | n63_k20.0 | 0 | 0 | 5.6881e-05 | 34500 |
| C006 | rectangle8 | 0.00100961 | 7.29112e-05 | 0.000319493 | 22330 |
| C006R | rectangle8 | 5.10958e-08 | 4.05476e-09 | 4.7958e-05 | 40756 |
| direct_solver | rectangle8 | 0 | 0 | 0.000111697 | 21452 |
| C006 | rectangle16 | 0.000955001 | 0.00010063 | 0.00580565 | 415338 |
| C006R | rectangle16 | 5.05919e-08 | 4.06053e-09 | 0.000683 | 818676 |
| direct_solver | rectangle16 | 0 | 0 | 0.000979585 | 413580 |
| C006 | aspect_changed | 0.00180451 | 0.000187608 | 0.0026142 | 123202 |
| C006R | aspect_changed | 7.01388e-08 | 7.71759e-09 | 0.000228782 | 239188 |
| direct_solver | aspect_changed | 0 | 0 | 0.00013567 | 121964 |
| C006 | square_hole | 0.00987391 | 0.00023023 | 0.00469593 | 329313 |
| C006R | square_hole | 1.65854e-08 | 3.71374e-10 | 0.000456202 | 647776 |
| direct_solver | square_hole | 0 | 0 | 0.000391489 | 327680 |
| C006 | heterogeneous | 0.00424052 | 0.000169511 | 0.00606648 | 415338 |
| C006R | heterogeneous | 2.28706e-08 | 9.1482e-10 | 0.000854669 | 818676 |
| direct_solver | heterogeneous | 0 | 0 | 0.000582698 | 413580 |
| C006 | rotated_anisotropy | 0.00771967 | 0.000398725 | 0.00483744 | 415338 |
| C006R | rotated_anisotropy | 2.66756e-08 | 1.12052e-09 | 0.00156062 | 818676 |
| direct_solver | rotated_anisotropy | 0 | 0 | 0.000661264 | 413580 |
| C006 | changed_boundary | 0.00386391 | 0.00345964 | 0.00103312 | 123202 |
| C006R | changed_boundary | 3.94333e-08 | 1.52639e-08 | 0.00022602 | 239188 |
| direct_solver | changed_boundary | 0 | 0 | 0.000151198 | 121964 |
| C006 | local_source | 0.00274657 | 0.00209615 | 0.00435097 | 415338 |
| C006R | local_source | 4.53175e-08 | 2.61854e-08 | 0.000647094 | 818676 |
| direct_solver | local_source | 0 | 0 | 0.000998256 | 413580 |
| discretization_counterexample | anisotropic_nodal_positivity | 0 | 0 | 8.5414e-05 | 155040 |

Array/batch relative L2 and mean per-sample relative L2 are distinct. Per-sample means/std, heat flux errors, physical balances, BC handling, momentum/divergence residuals and timing/memory scope are in the JSON. Tracemalloc does not measure all native/BLAS memory; RSS is cumulative process high-water. Timing is a small CPU microbenchmark, not GPU performance.

## 7. Self-refutation in six roles
- A — Inventor: The original filter can reuse coefficients on new supplied physical operators.
- B — Theoretical physicist: Reciprocal SPD response is weaker than pointwise positivity or correct flux.
- C — Numerical analyst: Non-M-matrix anisotropic FEM has a negative short-time heat response despite SPD; full eigensetup is expensive.
- D — Architecture researcher: The resolvent becomes the original solver as lifetime increases; not a new fast network.
- E — Patent-review perspective: Neither correct geometry assembly nor a shifted solve establishes a distinct invention. This is not a patentability finding.
- F — Alternative-hypothesis researcher: Challenge the self-adjoint assumption with directed advection.

## 8. Prior work and verification limits
See `../../research/novelty_review.md` for checked official source descriptions/code and the full coverage ledger. Original paper sites returned proxy 403; no unread full text is presented as reviewed. Exact classical algebraic equivalences are sufficient to withhold an invention claim. Architectural novelty and patent coverage are unverified.

## 9. Verdict
**REVISE**. This is a cycle-level decision; final candidate-level dispositions are in `../../research/candidate_registry.md`. No final invention is selected.

## 10. Next cycle hypothesis / stopping state
Challenge the self-adjoint assumption with directed advection.
