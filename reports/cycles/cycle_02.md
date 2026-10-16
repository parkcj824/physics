# Cycle 2 — Exchange budgets

## 1. Physical principle
conservation

## 2. Minimal computational object
opposite exchange budget

## 3. Implemented formula
\[
T^+=T+\tau(b-AT);\quad\tau_R=1/\lambda_{max}(A)
\]

The complete inputs, trainable/fixed quantities, guarantees and complexity are in `../../architectures/candidate_*.md`.

## 4. Difference from existing operations
antisymmetric exchange explicitly cancels internal transfers; expanded rule is Richardson relaxation

## 5. Governing equation, conditions and reference
-T_xx=1; N=7,15,31; zero Dirichlet; 8 unstable or 32 stable steps
References are independent direct discrete solves, analytic solutions, augmented gauge solves or matrix exponentials as identified in each saved record. The anisotropic positivity counterexample intentionally uses the exact discrete semigroup to expose a discretization defect.

## 6. Executed results
Final verified run: `results/session_0002_verified`; 6 records in this cycle. Complete inputs, predictions and references are saved in the NPZ paths in the metrics JSON.

| Candidate | Case | Relative L2 | Max absolute error | Median seconds | Traced peak bytes |
|---|---|---:|---:|---:|---:|
| C002 | unstable_n7 | 1.58152e+08 | 1.78773e+07 | 2.0169e-05 | 720 |
| C002R | stable_n7 | 0.274571 | 0.0354426 | 0.000104561 | 744 |
| C002 | unstable_n15 | 2.20299e+12 | 3.76459e+11 | 2.0147e-05 | 848 |
| C002R | stable_n15 | 0.731506 | 0.0941586 | 8.5387e-05 | 872 |
| C002 | unstable_n31 | 2.72637e+16 | 6.6023e+15 | 2.1074e-05 | 1176 |
| C002R | stable_n31 | 0.925105 | 0.117169 | 7.7953e-05 | 1200 |

Array/batch relative L2 and mean per-sample relative L2 are distinct. Per-sample means/std, heat flux errors, physical balances, BC handling, momentum/divergence residuals and timing/memory scope are in the JSON. Tracemalloc does not measure all native/BLAS memory; RSS is cumulative process high-water. Timing is a small CPU microbenchmark, not GPU performance.

## 7. Self-refutation in six roles
- A — Inventor: Opposite internal transfers cancel exactly in the incidence formulation.
- B — Theoretical physicist: Internal cancellation does not remove sources or boundary power.
- C — Numerical analyst: Fixed tau violates the spectral stability limit; stable steps remain slow under refinement.
- D — Architecture researcher: This is a classical learned-step iterative solver family.
- E — Patent-review perspective: Stability correction alone is not a differentiated invention. This is not a patentability finding.
- F — Alternative-hypothesis researcher: Try admissible whole-field variations instead of repeated local exchange.

## 8. Prior work and verification limits
See `../../research/novelty_review.md` for checked official source descriptions/code and the full coverage ledger. Original paper sites returned proxy 403; no unread full text is presented as reviewed. Exact classical algebraic equivalences are sufficient to withhold an invention claim. Architectural novelty and patent coverage are unverified.

## 9. Verdict
**REJECT**. This is a cycle-level decision; final candidate-level dispositions are in `../../research/candidate_registry.md`. No final invention is selected.

## 10. Next cycle hypothesis / stopping state
Try admissible whole-field variations instead of repeated local exchange.
