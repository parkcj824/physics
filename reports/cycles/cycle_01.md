# Cycle 1 — Boundary-anchored response

## 1. Physical principle
equilibrium and source response

## 2. Minimal computational object
boundary-anchored source response

## 3. Implemented formula
\[
T_i=a(1-x_i)+bx_i+\frac{x_i(1-x_i)}{2k_{eff}}\sum_j\frac{e^{-|x_i-x_j|/\ell}}{\sum_l e^{-|x_i-x_l|/\ell}}q_j
\]

The complete inputs, trainable/fixed quantities, guarantees and complexity are in `../../architectures/candidate_*.md`.

## 4. Difference from existing operations
physical anchor and effective material replace arbitrary features, but normalization is kernel/attention-like

## 5. Governing equation, conditions and reference
-(kT_x)_x=Q; N=15; uniform/local source, heterogeneous k and changed Dirichlet data
References are independent direct discrete solves, analytic solutions, augmented gauge solves or matrix exponentials as identified in each saved record. The anisotropic positivity counterexample intentionally uses the exact discrete semigroup to expose a discretization defect.

## 6. Executed results
Final verified run: `results/session_0002_verified`; 4 records in this cycle. Complete inputs, predictions and references are saved in the NPZ paths in the metrics JSON.

| Candidate | Case | Relative L2 | Max absolute error | Median seconds | Traced peak bytes |
|---|---|---:|---:|---:|---:|
| C001 | uniform | 1.22367e-15 | 1.94289e-16 | 2.4921e-05 | 8712 |
| C001 | localized | 0.337892 | 0.154994 | 1.9251e-05 | 8712 |
| C001 | material | 0.915499 | 0.0431993 | 2.0066e-05 | 8712 |
| C001 | boundary | 0.304488 | 0.388889 | 2.1075e-05 | 8712 |

Array/batch relative L2 and mean per-sample relative L2 are distinct. Per-sample means/std, heat flux errors, physical balances, BC handling, momentum/divergence residuals and timing/memory scope are in the JSON. Tracemalloc does not measure all native/BLAS memory; RSS is cumulative process high-water. Timing is a small CPU microbenchmark, not GPU performance.

## 7. Self-refutation in six roles
- A — Inventor: The uniform-source parabola is exact without solving a system.
- B — Theoretical physicist: Local sources and spatial material distributions require reciprocal nonlocal Green response.
- C — Numerical analyst: A special analytic fit does not control residual or heat balance.
- D — Architecture researcher: Normalized positive kernels are not a new primitive by name.
- E — Patent-review perspective: No specific prior-art distinction or robust technical effect is established. This is not a patentability finding.
- F — Alternative-hypothesis researcher: Use conserved exchanges, then revisit the response with resistance coordinates.

## 8. Prior work and verification limits
See `../../research/novelty_review.md` for checked official source descriptions/code and the full coverage ledger. Original paper sites returned proxy 403; no unread full text is presented as reviewed. Exact classical algebraic equivalences are sufficient to withhold an invention claim. Architectural novelty and patent coverage are unverified.

## 9. Verdict
**REVISE**. This is a cycle-level decision; final candidate-level dispositions are in `../../research/candidate_registry.md`. No final invention is selected.

## 10. Next cycle hypothesis / stopping state
Use conserved exchanges, then revisit the response with resistance coordinates.
