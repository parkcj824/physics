# Cycle 11 — Baseline self-refutation

## 1. Physical principle
linearity and dimensional scaling

## 2. Minimal computational object
learned fixed-mesh compliance

## 3. Implemented formula
\[
\widehat C=\arg\min_C\|XC-Y\|_F^2,\quad X=Q/k;\quad T=(Q/k)\widehat C
\]

The complete inputs, trainable/fixed quantities, guarantees and complexity are in `../../architectures/candidate_*.md`.

## 4. Difference from existing operations
this is a strong classical data-fitted linear control, not a new architecture

## 5. Governing equation, conditions and reference
same homogeneous scalar-k 1D task; 32/128 labels; normalized MLP with separate validation and 3000-step budget
References are independent direct discrete solves, analytic solutions, augmented gauge solves or matrix exponentials as identified in each saved record. The anisotropic positivity counterexample intentionally uses the exact discrete semigroup to expose a discretization defect.

## 6. Executed results
Final verified run: `results/session_0002_verified`; 32 records in this cycle. Complete inputs, predictions and references are saved in the NPZ paths in the metrics JSON.

| Candidate | Case | Relative L2 | Max absolute error | Median seconds | Traced peak bytes |
|---|---|---:|---:|---:|---:|
| linear_compliance | iid_samples32 | 9.85888e-16 | 1.19349e-15 | 3.658e-06 | 8832 |
| linear_compliance | low_conductivity_ood_samples32 | 1.25116e-15 | 8.70415e-14 | 6.467e-06 | 8832 |
| linear_compliance | high_conductivity_ood_samples32 | 1.47051e-15 | 8.67362e-17 | 7.704e-06 | 8832 |
| linear_compliance | high_frequency_source_ood_samples32 | 4.42235e-13 | 3.76911e-15 | 6.585e-06 | 8832 |
| MLP_normalized | iid_n32_seed824 | 0.588492 | 0.810131 | 4.4072e-05 | 13632 |
| MLP_normalized | low_conductivity_ood_n32_seed824 | 0.995002 | 49.3523 | 5.8009e-05 | 13632 |
| MLP_normalized | high_conductivity_ood_n32_seed824 | 6.4639 | 0.222397 | 5.2504e-05 | 13632 |
| MLP_normalized | high_frequency_source_ood_n32_seed824 | 14.0642 | 0.0782987 | 4.626e-05 | 13632 |
| MLP_normalized | iid_n32_seed825 | 0.543785 | 0.761818 | 6.6515e-05 | 13632 |
| MLP_normalized | low_conductivity_ood_n32_seed825 | 0.996784 | 49.3684 | 0.000334562 | 13632 |
| MLP_normalized | high_conductivity_ood_n32_seed825 | 4.9378 | 0.152966 | 6.4395e-05 | 13632 |
| MLP_normalized | high_frequency_source_ood_n32_seed825 | 18.4015 | 0.0901912 | 6.5159e-05 | 13632 |
| MLP_normalized | iid_n32_seed826 | 0.56363 | 0.774952 | 4.6926e-05 | 13632 |
| MLP_normalized | low_conductivity_ood_n32_seed826 | 0.995975 | 49.3636 | 0.000111705 | 13632 |
| MLP_normalized | high_conductivity_ood_n32_seed826 | 5.96586 | 0.195287 | 4.5048e-05 | 13632 |
| MLP_normalized | high_frequency_source_ood_n32_seed826 | 31.9141 | 0.140422 | 4.3643e-05 | 13632 |
| linear_compliance | iid_samples128 | 5.01274e-16 | 1.11022e-15 | 3.943e-06 | 8832 |
| linear_compliance | low_conductivity_ood_samples128 | 4.55984e-16 | 3.01981e-14 | 5.466e-06 | 8832 |
| linear_compliance | high_conductivity_ood_samples128 | 5.45808e-16 | 3.81639e-17 | 4.934e-06 | 8832 |
| linear_compliance | high_frequency_source_ood_samples128 | 1.40821e-13 | 7.92146e-16 | 3.213e-06 | 8832 |
| MLP_normalized | iid_n128_seed824 | 0.210793 | 0.363592 | 4.6538e-05 | 13632 |
| MLP_normalized | low_conductivity_ood_n128_seed824 | 0.998061 | 49.5131 | 5.1761e-05 | 13632 |
| MLP_normalized | high_conductivity_ood_n128_seed824 | 2.03823 | 0.0585174 | 4.8699e-05 | 13632 |
| MLP_normalized | high_frequency_source_ood_n128_seed824 | 14.8735 | 0.0740818 | 4.4835e-05 | 13632 |
| MLP_normalized | iid_n128_seed825 | 0.210036 | 0.359512 | 6.0399e-05 | 13632 |
| MLP_normalized | low_conductivity_ood_n128_seed825 | 0.998731 | 49.495 | 5.0062e-05 | 13632 |
| MLP_normalized | high_conductivity_ood_n128_seed825 | 2.26065 | 0.0879398 | 0.000167039 | 13632 |
| MLP_normalized | high_frequency_source_ood_n128_seed825 | 12.2958 | 0.0850308 | 6.5019e-05 | 13632 |
| MLP_normalized | iid_n128_seed826 | 0.204483 | 0.341868 | 6.1945e-05 | 13632 |
| MLP_normalized | low_conductivity_ood_n128_seed826 | 0.998387 | 49.4752 | 5.9558e-05 | 13632 |
| MLP_normalized | high_conductivity_ood_n128_seed826 | 1.75194 | 0.0667782 | 4.7215e-05 | 13632 |
| MLP_normalized | high_frequency_source_ood_n128_seed826 | 16.5803 | 0.0765224 | 4.8152e-05 | 13632 |

Array/batch relative L2 and mean per-sample relative L2 are distinct. Per-sample means/std, heat flux errors, physical balances, BC handling, momentum/divergence residuals and timing/memory scope are in the JSON. Tracemalloc does not measure all native/BLAS memory; RSS is cumulative process high-water. Timing is a small CPU microbenchmark, not GPU performance.

## 7. Self-refutation in six roles
- A — Inventor: A full-rank fixed-mesh linear map exploits the exact physical 1/k scaling.
- B — Theoretical physicist: An expressive neural model is unnecessary for this particular linear operator family.
- C — Numerical analyst: Initial MLP conditioning was poor; normalized controls materially change the comparison.
- D — Architecture researcher: More parameters, different priors and different compute budgets prevent sweeping model rankings.
- E — Patent-review perspective: No demonstrated technical advantage supports an invention claim here. This is not a patentability finding.
- F — Alternative-hypothesis researcher: After stalled novelty, replace the age object with an independent source-contrast hypothesis.

## 8. Prior work and verification limits
See `../../research/novelty_review.md` for checked official source descriptions/code and the full coverage ledger. Original paper sites returned proxy 403; no unread full text is presented as reviewed. Exact classical algebraic equivalences are sufficient to withhold an invention claim. Architectural novelty and patent coverage are unverified.

## 9. Verdict
**REJECT**. This is a cycle-level decision; final candidate-level dispositions are in `../../research/candidate_registry.md`. No final invention is selected.

## 10. Next cycle hypothesis / stopping state
After stalled novelty, replace the age object with an independent source-contrast hypothesis.
