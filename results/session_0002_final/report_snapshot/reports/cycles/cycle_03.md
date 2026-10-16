# Cycle 3 — Action span

## 1. Physical principle
stationarity of energy

## 2. Minimal computational object
admissible field variation

## 3. Implemented formula
\[
T=\Phi(\Phi^TA\Phi)^{-1}\Phi^Tb
\]

The complete inputs, trainable/fixed quantities, guarantees and complexity are in `../../architectures/candidate_*.md`.

## 4. Difference from existing operations
global stationary response, exactly Ritz/Galerkin instead of learned pair weights

## 5. Governing equation, conditions and reference
-(kT_x)_x=Q; rank4 sine span; includes mode8 forcing
References are independent direct discrete solves, analytic solutions, augmented gauge solves or matrix exponentials as identified in each saved record. The anisotropic positivity counterexample intentionally uses the exact discrete semigroup to expose a discretization defect.

## 6. Executed results
Final verified run: `results/session_0002_final`; 4 records in this cycle. Complete inputs, predictions and references are saved in the NPZ paths in the metrics JSON.

| Candidate | Case | Relative L2 | Max absolute error | Median seconds | Traced peak bytes |
|---|---|---:|---:|---:|---:|
| C003 | uniform | 0.00858991 | 0.00147603 | 2.9963e-05 | 1394 |
| C003 | localized | 0.0515848 | 0.0282086 | 1.9765e-05 | 1394 |
| C003 | material | 0.0892111 | 0.00658941 | 3.3252e-05 | 1394 |
| C003 | high_frequency | 1 | 0.00195313 | 1.941e-05 | 1394 |

Array/batch relative L2 and mean per-sample relative L2 are distinct. Per-sample means/std, heat flux errors, physical balances, BC handling, momentum/divergence residuals and timing/memory scope are in the JSON. Tracemalloc does not measure all native/BLAS memory; RSS is cumulative process high-water. Timing is a small CPU microbenchmark, not GPU performance.

## 7. Self-refutation in six roles
- A — Inventor: The energy is minimized within a compact admissible span.
- B — Theoretical physicist: Missing source directions have nonzero physical solutions but zero predicted response.
- C — Numerical analyst: Projected residual zero does not mean local residual or flux balance zero.
- D — Architecture researcher: Exact reduced-basis equivalence excludes an independent primitive claim.
- E — Patent-review perspective: A learned basis would require evidence beyond existing model reduction. This is not a patentability finding.
- F — Alternative-hypothesis researcher: Test positive-state response masses from irreversibility.

## 8. Prior work and verification limits
See `../../research/novelty_review.md` for checked official source descriptions/code and the full coverage ledger. Original paper sites returned proxy 403; no unread full text is presented as reviewed. Exact classical algebraic equivalences are sufficient to withhold an invention claim. Architectural novelty and patent coverage are unverified.

## 9. Verdict
**REJECT**. This is a cycle-level decision; final candidate-level dispositions are in `../../research/candidate_registry.md`. No final invention is selected.

## 10. Next cycle hypothesis / stopping state
Test positive-state response masses from irreversibility.
