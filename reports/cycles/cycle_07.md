# Cycle 7 — Relaxation-age learning

## 1. Physical principle
relaxation and response

## 2. Minimal computational object
positive age packet

## 3. Implemented formula
\[
T=V\operatorname{diag}(\sum_{j=1}^6e^{\alpha_j}e^{-e^{\beta_j}\lambda})V^Tb
\]

The complete inputs, trainable/fixed quantities, guarantees and complexity are in `../../architectures/candidate_*.md`.

## 4. Difference from existing operations
learned positive age measure in the actual physical eigenbasis, not unconstrained Fourier channel weights

## 5. Governing equation, conditions and reference
-kT_xx=Q; 128 training cases, 32 validation, 32 per test condition; three seeds
References are independent direct discrete solves, analytic solutions, augmented gauge solves or matrix exponentials as identified in each saved record. The anisotropic positivity counterexample intentionally uses the exact discrete semigroup to expose a discretization defect.

## 6. Executed results
Final verified run: `results/session_0002_verified`; 36 records in this cycle. Complete inputs, predictions and references are saved in the NPZ paths in the metrics JSON.

| Candidate | Case | Relative L2 | Max absolute error | Median seconds | Traced peak bytes |
|---|---|---:|---:|---:|---:|
| C006 | validation_seed824 | 0.00192935 | 0.00172006 | 3.8231e-05 | 12808 |
| C006 | iid_seed824 | 0.00188609 | 0.0016331 | 7.4991e-05 | 12808 |
| C006 | low_conductivity_ood_seed824 | 0.906872 | 44.97 | 3.7193e-05 | 12808 |
| C006 | high_conductivity_ood_seed824 | 0.0190332 | 0.000679614 | 8.8313e-05 | 12808 |
| C006 | localized_source_ood_seed824 | 0.00219771 | 0.000637467 | 3.7218e-05 | 12808 |
| C006 | high_frequency_source_ood_seed824 | 0.0524547 | 0.000165633 | 8.8104e-05 | 12808 |
| MLP | validation_seed824 | 0.460164 | 0.394332 | 5.6147e-05 | 5040 |
| MLP | iid_seed824 | 0.518262 | 0.723995 | 4.2673e-05 | 5040 |
| MLP | low_conductivity_ood_seed824 | 0.995437 | 49.1968 | 5.9235e-05 | 5040 |
| MLP | high_conductivity_ood_seed824 | 8.13392 | 0.334164 | 3.9252e-05 | 5040 |
| MLP | localized_source_ood_seed824 | 0.460306 | 0.268078 | 5.2661e-05 | 5040 |
| MLP | high_frequency_source_ood_seed824 | 87.629 | 0.366903 | 3.7442e-05 | 5040 |
| C006 | validation_seed825 | 0.00206439 | 0.00182596 | 4.0754e-05 | 12808 |
| C006 | iid_seed825 | 0.00182893 | 0.00185658 | 5.1998e-05 | 12808 |
| C006 | low_conductivity_ood_seed825 | 0.904587 | 44.8653 | 5.4904e-05 | 12808 |
| C006 | high_conductivity_ood_seed825 | 0.0263422 | 0.00091169 | 7.0417e-05 | 12808 |
| C006 | localized_source_ood_seed825 | 0.00235237 | 0.000719552 | 5.1091e-05 | 12808 |
| C006 | high_frequency_source_ood_seed825 | 0.0782423 | 0.000284389 | 7.0493e-05 | 12808 |
| MLP | validation_seed825 | 0.502941 | 0.391186 | 3.3548e-05 | 5040 |
| MLP | iid_seed825 | 0.554585 | 0.753027 | 3.8194e-05 | 5040 |
| MLP | low_conductivity_ood_seed825 | 0.991936 | 49.2953 | 3.5331e-05 | 5040 |
| MLP | high_conductivity_ood_seed825 | 9.39534 | 0.315573 | 0.000131812 | 5040 |
| MLP | localized_source_ood_seed825 | 0.582978 | 0.325052 | 4.7831e-05 | 5040 |
| MLP | high_frequency_source_ood_seed825 | 39.9478 | 0.135901 | 3.5641e-05 | 5040 |
| C006 | validation_seed826 | 0.0019159 | 0.00159591 | 3.5728e-05 | 12808 |
| C006 | iid_seed826 | 0.00204654 | 0.00241329 | 7.1973e-05 | 12808 |
| C006 | low_conductivity_ood_seed826 | 0.908063 | 45.0247 | 5.3157e-05 | 12808 |
| C006 | high_conductivity_ood_seed826 | 0.0210774 | 0.000686896 | 9.7626e-05 | 12808 |
| C006 | localized_source_ood_seed826 | 0.00214219 | 0.000703978 | 4.7782e-05 | 12808 |
| C006 | high_frequency_source_ood_seed826 | 0.0681226 | 0.000201873 | 0.000108144 | 12808 |
| MLP | validation_seed826 | 0.485517 | 0.419819 | 3.6668e-05 | 5040 |
| MLP | iid_seed826 | 0.502791 | 0.673933 | 3.963e-05 | 5040 |
| MLP | low_conductivity_ood_seed826 | 0.990492 | 49.2911 | 3.466e-05 | 5040 |
| MLP | high_conductivity_ood_seed826 | 10.5117 | 0.401977 | 3.5982e-05 | 5040 |
| MLP | localized_source_ood_seed826 | 0.478818 | 0.310883 | 3.4867e-05 | 5040 |
| MLP | high_frequency_source_ood_seed826 | 68.5851 | 0.252852 | 4.1798e-05 | 5040 |

Array/batch relative L2 and mean per-sample relative L2 are distinct. Per-sample means/std, heat flux errors, physical balances, BC handling, momentum/divergence residuals and timing/memory scope are in the JSON. Tracemalloc does not measure all native/BLAS memory; RSS is cumulative process high-water. Timing is a small CPU microbenchmark, not GPU performance.

## 7. Self-refutation in six roles
- A — Inventor: Twelve scalars give accurate in-range linear elliptic response.
- B — Theoretical physicist: Low conductivity exposes the missing long-age response; finite sums cannot diverge like 1/lambda at zero.
- C — Numerical analyst: Large-lambda inverse tails decay exponentially and source smoothness can hide this defect.
- D — Architecture researcher: It is an exponential-sum inverse approximation with a full operator prior; learned quadrature novelty is unverified.
- E — Patent-review perspective: A new term for a known inverse-integral approximation is not a technical distinction. This is not a patentability finding.
- F — Alternative-hypothesis researcher: Repair the spectral tail, then check whether that merely recreates a solver.

## 8. Prior work and verification limits
See `../../research/novelty_review.md` for checked official source descriptions/code and the full coverage ledger. Original paper sites returned proxy 403; no unread full text is presented as reviewed. Exact classical algebraic equivalences are sufficient to withhold an invention claim. Architectural novelty and patent coverage are unverified.

## 9. Verdict
**REVISE**. This is a cycle-level decision; final candidate-level dispositions are in `../../research/candidate_registry.md`. No final invention is selected.

## 10. Next cycle hypothesis / stopping state
Repair the spectral tail, then check whether that merely recreates a solver.
