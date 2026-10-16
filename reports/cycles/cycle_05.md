# Cycle 5 — Material events

## 1. Physical principle
state change and endpoint consistency

## 2. Minimal computational object
material-change event

## 3. Implemented formula
\[
T^+=e^{\delta c_jG_j}T;\quad e^{c_2G_2}e^{c_1G_1}\ne e^{c_1G_1}e^{c_2G_2}
\]

The complete inputs, trainable/fixed quantities, guarantees and complexity are in `../../architectures/candidate_*.md`.

## 4. Difference from existing operations
event composition acts in physical parameter space instead of source aggregation

## 5. Governing equation, conditions and reference
steady 1D heat; two half-material edits end at uniform k=2; closed loop returns to k=1
References are independent direct discrete solves, analytic solutions, augmented gauge solves or matrix exponentials as identified in each saved record. The anisotropic positivity counterexample intentionally uses the exact discrete semigroup to expose a discretization defect.

## 6. Executed results
Final verified run: `results/session_0002_verified`; 2 records in this cycle. Complete inputs, predictions and references are saved in the NPZ paths in the metrics JSON.

| Candidate | Case | Relative L2 | Max absolute error | Median seconds | Traced peak bytes |
|---|---|---:|---:|---:|---:|
| C005 | material_path12 | 0.391997 | 0.0308071 | 0.000113541 | 26008 |
| C005 | closed_material_loop | 0.179546 | 0.0272984 | 0.000113541 | 26008 |

Array/batch relative L2 and mean per-sample relative L2 are distinct. Per-sample means/std, heat flux errors, physical balances, BC handling, momentum/divergence residuals and timing/memory scope are in the JSON. Tracemalloc does not measure all native/BLAS memory; RSS is cumulative process high-water. Timing is a small CPU microbenchmark, not GPU performance.

## 7. Self-refutation in six roles
- A — Inventor: Low-rank generators represent compact responses to parameter changes.
- B — Theoretical physicist: The steady endpoint is independent of edit order, unlike the candidate.
- C — Numerical analyst: A closed material loop produces a measurable spurious change.
- D — Architecture researcher: Matrix-exponential flows/Lie products are existing mathematics.
- E — Patent-review perspective: No benefit survives the simplest endpoint test; no patent assertion is warranted. This is not a patentability finding.
- F — Alternative-hypothesis researcher: Repair the original response analytically to identify what information was missing.

## 8. Prior work and verification limits
See `../../research/novelty_review.md` for checked official source descriptions/code and the full coverage ledger. Original paper sites returned proxy 403; no unread full text is presented as reviewed. Exact classical algebraic equivalences are sufficient to withhold an invention claim. Architectural novelty and patent coverage are unverified.

## 9. Verdict
**REJECT**. This is a cycle-level decision; final candidate-level dispositions are in `../../research/candidate_registry.md`. No final invention is selected.

## 10. Next cycle hypothesis / stopping state
Repair the original response analytically to identify what information was missing.
