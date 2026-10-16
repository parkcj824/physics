# Cycle 6 — Resistance repair and gauge

## 1. Physical principle
reciprocal response and compatibility

## 2. Minimal computational object
resistance-coordinate source response

## 3. Implemented formula
\[
G_{ij}=h\min(R_i,R_j)[R_*-\max(R_i,R_j)]/R_*;\quad T=Gq+a(1-R/R_*)+bR/R_*
\]

The complete inputs, trainable/fixed quantities, guarantees and complexity are in `../../architectures/candidate_*.md`.

## 4. Difference from existing operations
normalization replaced by an exact resistance Green function; this is a known classical solution

## 5. Governing equation, conditions and reference
heterogeneous 1D heat, N=15/31; separate compatible pure-Neumann N=16/32/64 with mean zero
References are independent direct discrete solves, analytic solutions, augmented gauge solves or matrix exponentials as identified in each saved record. The anisotropic positivity counterexample intentionally uses the exact discrete semigroup to expose a discretization defect.

## 6. Executed results
Final verified run: `results/session_0002_verified`; 8 records in this cycle. Complete inputs, predictions and references are saved in the NPZ paths in the metrics JSON.

| Candidate | Case | Relative L2 | Max absolute error | Median seconds | Traced peak bytes |
|---|---|---:|---:|---:|---:|
| C001R | uniform | 1.2064e-15 | 1.94289e-16 | 1.9661e-05 | 9091 |
| C001R | localized | 5.6239e-16 | 2.22045e-16 | 2.6584e-05 | 9091 |
| C001R | material | 2.15802e-16 | 1.38778e-17 | 3.1059e-05 | 9091 |
| C001R | boundary | 2.11737e-16 | 3.33067e-16 | 2.1389e-05 | 9091 |
| C001R | resolution31 | 5.68616e-16 | 3.46945e-17 | 2.2788e-05 | 32771 |
| gauge_control | neumann_n16 | 6.57604e-15 | 7.43849e-15 | 3.5988e-05 | 9944 |
| gauge_control | neumann_n32 | 1.37648e-14 | 2.10942e-14 | 6.4349e-05 | 28392 |
| gauge_control | neumann_n64 | 2.88881e-14 | 4.56302e-14 | 0.000357913 | 102152 |

Array/batch relative L2 and mean per-sample relative L2 are distinct. Per-sample means/std, heat flux errors, physical balances, BC handling, momentum/divergence residuals and timing/memory scope are in the JSON. Tracemalloc does not measure all native/BLAS memory; RSS is cumulative process high-water. Timing is a small CPU microbenchmark, not GPU performance.

## 7. Self-refutation in six roles
- A — Inventor: The corrected kernel gives exact discrete response and reciprocity.
- B — Theoretical physicist: A Neumann source must balance boundary flux and the additive constant must be fixed.
- C — Numerical analyst: Check continuum convergence separately from exact agreement with the tridiagonal matrix.
- D — Architecture researcher: This is the classical Green/resistance formula, not a neural invention.
- E — Patent-review perspective: Equation identity disqualifies renaming this correction as a novel primitive. This is not a patentability finding.
- F — Alternative-hypothesis researcher: Explore relaxation-age objects from irreversible response instead of a spatial kernel.

## 8. Prior work and verification limits
See `../../research/novelty_review.md` for checked official source descriptions/code and the full coverage ledger. Original paper sites returned proxy 403; no unread full text is presented as reviewed. Exact classical algebraic equivalences are sufficient to withhold an invention claim. Architectural novelty and patent coverage are unverified.

## 9. Verdict
**REJECT**. This is a cycle-level decision; final candidate-level dispositions are in `../../research/candidate_registry.md`. No final invention is selected.

## 10. Next cycle hypothesis / stopping state
Explore relaxation-age objects from irreversible response instead of a spatial kernel.
