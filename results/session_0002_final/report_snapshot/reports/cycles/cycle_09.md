# Cycle 9 — Directed transport

## 1. Physical principle
causality and irreversible direction

## 2. Minimal computational object
time-step response with an antisymmetric component

## 3. Implemented formula
\[
u_{n+1}=(I+\Delta tL)^{-1}(u_n+\Delta tb);\quad L\ne L^T
\]

The complete inputs, trainable/fixed quantities, guarantees and complexity are in `../../architectures/candidate_*.md`.

## 4. Difference from existing operations
the extension is backward Euler; deleting the skew part removes physical transport direction

## 5. Governing equation, conditions and reference
u_t+v u_x=k u_xx; Pe_cell=0.1,1,10; v=+-1; periodic/inflow; time refinement and t=10
References are independent direct discrete solves, analytic solutions, augmented gauge solves or matrix exponentials as identified in each saved record. The anisotropic positivity counterexample intentionally uses the exact discrete semigroup to expose a discretization defect.

## 6. Executed results
Final verified run: `results/session_0002_final`; 18 records in this cycle. Complete inputs, predictions and references are saved in the NPZ paths in the metrics JSON.

| Candidate | Case | Relative L2 | Max absolute error | Median seconds | Traced peak bytes |
|---|---|---:|---:|---:|---:|
| selfadjoint_transport_failure | periodic_Pe0.1_v-1.0 | 0.0357606 | 0.0504949 | 6.7795e-05 | 75008 |
| backward_euler_control | periodic_Pe0.1_v-1.0 | 0.00568525 | 0.00802881 | 0.000415117 | 19168 |
| selfadjoint_transport_failure | periodic_Pe0.1_v1.0 | 0.0357606 | 0.0504949 | 6.4425e-05 | 75008 |
| backward_euler_control | periodic_Pe0.1_v1.0 | 0.00568525 | 0.00802881 | 0.000454657 | 19168 |
| selfadjoint_transport_failure | periodic_Pe1.0_v-1.0 | 0.106697 | 0.152725 | 6.886e-05 | 75008 |
| backward_euler_control | periodic_Pe1.0_v-1.0 | 0.00360144 | 0.00516381 | 0.00070395 | 19168 |
| selfadjoint_transport_failure | periodic_Pe1.0_v1.0 | 0.106697 | 0.152725 | 7.482e-05 | 75008 |
| backward_euler_control | periodic_Pe1.0_v1.0 | 0.00360144 | 0.00516381 | 0.000428449 | 19168 |
| selfadjoint_transport_failure | periodic_Pe10.0_v-1.0 | 0.118741 | 0.170599 | 6.8053e-05 | 66816 |
| backward_euler_control | periodic_Pe10.0_v-1.0 | 0.00376155 | 0.00540302 | 0.000363213 | 19168 |
| selfadjoint_transport_failure | periodic_Pe10.0_v1.0 | 0.118741 | 0.170599 | 6.9765e-05 | 66816 |
| backward_euler_control | periodic_Pe10.0_v1.0 | 0.00376155 | 0.00540302 | 0.000484842 | 19168 |
| backward_euler_control | inflow_dirichlet | 0.0430538 | 0.0340816 | 0.000407722 | 23936 |
| backward_euler_control | inflow_dt0.02 | 0.0793993 | 0.0620315 | 0.000209872 | 23936 |
| backward_euler_control | inflow_dt0.005 | 0.0225455 | 0.0179763 | 0.000714091 | 23936 |
| backward_euler_control | inflow_dt0.0025 | 0.0115574 | 0.0092423 | 0.00227254 | 23936 |
| backward_euler_control | periodic_longtime1.0 | 0.0180584 | 0.0256669 | 0.00370115 | 19192 |
| backward_euler_control | periodic_longtime10.0 | 0.000115731 | 0.000163166 | 0.0385865 | 19224 |

Array/batch relative L2 and mean per-sample relative L2 are distinct. Per-sample means/std, heat flux errors, physical balances, BC handling, momentum/divergence residuals and timing/memory scope are in the JSON. Tracemalloc does not measure all native/BLAS memory; RSS is cumulative process high-water. Timing is a small CPU microbenchmark, not GPU performance.

## 7. Self-refutation in six roles
- A — Inventor: A positivity-preserving nonsymmetric resolvent handles directed transport stably.
- B — Theoretical physicist: Symmetrizing the operator cannot represent reversed advection.
- C — Numerical analyst: Long-time boundedness and mass conservation do not imply accurate phase or transient response.
- D — Architecture researcher: Backward Euler is a classical control, not an invented temporal neural operation.
- E — Patent-review perspective: Known time integration cannot supply the required architectural distinction. This is not a patentability finding.
- F — Alternative-hypothesis researcher: Check incompressibility and momentum separately in fluid examples.

## 8. Prior work and verification limits
See `../../research/novelty_review.md` for checked official source descriptions/code and the full coverage ledger. Original paper sites returned proxy 403; no unread full text is presented as reviewed. Exact classical algebraic equivalences are sufficient to withhold an invention claim. Architectural novelty and patent coverage are unverified.

## 9. Verdict
**REJECT**. This is a cycle-level decision; final candidate-level dispositions are in `../../research/candidate_registry.md`. No final invention is selected.

## 10. Next cycle hypothesis / stopping state
Check incompressibility and momentum separately in fluid examples.
