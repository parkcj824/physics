# Cycle 12 — Independent source contrast

## 1. Physical principle
gauge and neutral excitation

## 2. Minimal computational object
signed source pair

## 3. Implemented formula
\[
T_i=\ell^2/k_{eff}\sum_{j\ne i} e^{-|x_i-x_j|/\ell}(q_i-q_j)
\]

The complete inputs, trainable/fixed quantities, guarantees and complexity are in `../../architectures/candidate_*.md`.

## 4. Difference from existing operations
the rule is a kernel Laplacian on sources, not an inverse response

## 5. Governing equation, conditions and reference
1D uniform, localized and heterogeneous heat cases
References are independent direct discrete solves, analytic solutions, augmented gauge solves or matrix exponentials as identified in each saved record. The anisotropic positivity counterexample intentionally uses the exact discrete semigroup to expose a discretization defect.

## 6. Executed results
Final verified run: `results/session_0002_verified`; 3 records in this cycle. Complete inputs, predictions and references are saved in the NPZ paths in the metrics JSON.

| Candidate | Case | Relative L2 | Max absolute error | Median seconds | Traced peak bytes |
|---|---|---:|---:|---:|---:|
| C007 | uniform | 1 | 0.125 | 1.8463e-05 | 9368 |
| C007 | localized | 4.38616 | 0.14368 | 1.9754e-05 | 9368 |
| C007 | material | 1 | 0.0484637 | 1.5719e-05 | 9368 |

Array/batch relative L2 and mean per-sample relative L2 are distinct. Per-sample means/std, heat flux errors, physical balances, BC handling, momentum/divergence residuals and timing/memory scope are in the JSON. Tracemalloc does not measure all native/BLAS memory; RSS is cumulative process high-water. Timing is a small CPU microbenchmark, not GPU performance.

## 7. Self-refutation in six roles
- A — Inventor: The response is symmetric PSD and explicitly cancels uniform source offsets.
- B — Theoretical physicist: Uniform source offsets are physical under Dirichlet data; a source is not a potential gauge.
- C — Numerical analyst: Localized positive forcing produces negative remote temperatures, violating the tested maximum principle.
- D — Architecture researcher: Pair differences are standard graph/kernel Laplacian aggregation.
- E — Patent-review perspective: The mathematical defect precedes any patent or novelty investigation. This is not a patentability finding.
- F — Alternative-hypothesis researcher: Stop at 12 cycles; next session must change objects and complete source verification before further tuning.

## 8. Prior work and verification limits
See `../../research/novelty_review.md` for checked official source descriptions/code and the full coverage ledger. Original paper sites returned proxy 403; no unread full text is presented as reviewed. Exact classical algebraic equivalences are sufficient to withhold an invention claim. Architectural novelty and patent coverage are unverified.

## 9. Verdict
**REJECT**. This is a cycle-level decision; final candidate-level dispositions are in `../../research/candidate_registry.md`. No final invention is selected.

## 10. Next cycle hypothesis / stopping state
Stop at 12 cycles; next session must change objects and complete source verification before further tuning.
