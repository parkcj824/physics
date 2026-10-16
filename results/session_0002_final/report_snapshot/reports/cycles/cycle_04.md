# Cycle 4 — Multiplicative mass

## 1. Physical principle
irreversibility and positivity

## 2. Minimal computational object
positive response mass

## 3. Implemented formula
\[
u_i^+=u_i\exp[\eta(b-Au)_i]
\]

The complete inputs, trainable/fixed quantities, guarantees and complexity are in `../../architectures/candidate_*.md`.

## 4. Difference from existing operations
multiplicative state updates overlap entropy/mirror descent rather than attention

## 5. Governing equation, conditions and reference
-T_xx=+1 or -1; zero Dirichlet; 128 steps eta=0.001
References are independent direct discrete solves, analytic solutions, augmented gauge solves or matrix exponentials as identified in each saved record. The anisotropic positivity counterexample intentionally uses the exact discrete semigroup to expose a discretization defect.

## 6. Executed results
Final verified run: `results/session_0002_final`; 2 records in this cycle. Complete inputs, predictions and references are saved in the NPZ paths in the metrics JSON.

| Candidate | Case | Relative L2 | Max absolute error | Median seconds | Traced peak bytes |
|---|---|---:|---:|---:|---:|
| C004 | positive_source | 0.118644 | 0.0168346 | 0.000333584 | 848 |
| C004 | cooling_deviation | 1.79336 | 0.212695 | 0.000331749 | 848 |

Array/batch relative L2 and mean per-sample relative L2 are distinct. Per-sample means/std, heat flux errors, physical balances, BC handling, momentum/divergence residuals and timing/memory scope are in the JSON. Tracemalloc does not measure all native/BLAS memory; RSS is cumulative process high-water. Timing is a small CPU microbenchmark, not GPU performance.

## 7. Self-refutation in six roles
- A — Inventor: Finite exponential updates keep every positive packet positive.
- B — Theoretical physicist: Cooling temperature deviations are negative; absolute temperature is a different variable.
- C — Numerical analyst: Positivity alone does not guarantee stiffness robustness, energy descent or finite exponentials.
- D — Architecture researcher: Exponentiated-gradient ancestry must not be renamed as novelty.
- E — Patent-review perspective: A generic signed thermal operator is not implemented by this positive cone. This is not a patentability finding.
- F — Alternative-hypothesis researcher: Test events in material-parameter space, an independent representation.

## 8. Prior work and verification limits
See `../../research/novelty_review.md` for checked official source descriptions/code and the full coverage ledger. Original paper sites returned proxy 403; no unread full text is presented as reviewed. Exact classical algebraic equivalences are sufficient to withhold an invention claim. Architectural novelty and patent coverage are unverified.

## 9. Verdict
**REJECT**. This is a cycle-level decision; final candidate-level dispositions are in `../../research/candidate_registry.md`. No final invention is selected.

## 10. Next cycle hypothesis / stopping state
Test events in material-parameter space, an independent representation.
