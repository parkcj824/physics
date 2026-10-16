# Preserved failures and causal diagnoses

The primary evidence is `../results/session_0002_verified/metrics/all_cases.json` with matching input/prediction/reference arrays. A failing scientific hypothesis is distinct from a failing software test. The tests assert valid physics/reference properties and correctly expose several candidate failures.

| Failure | Executed evidence | Diagnosis | Classification / do not repeat |
|---|---|---|---|
| C001 special-case success | uniform relative error ~1e-15; localized 0.33789, heterogeneous 0.91550 | normalized source average lacks reciprocal resistance information | structural; do not evaluate only a constant source |
| C002 fixed exchange step | 8-step relative errors grow from ~1.58e8 at N7 to ~2.73e16 at N31 | `rho(I-tau A)>1` | numerical stability failure; not a missing package or insufficient training |
| C002R stable revision | 32-step error 0.27457 at N7, 0.92511 at N31 | low modes converge slowly with stable CFL-scaled step | computational failure as a finite-depth steady operator; classical relaxation |
| C003 rank bottleneck | mode8 source has relative error 1 | `Phiᵀb=0` | exact expressivity obstruction; more training of fixed span cannot help |
| C004 positive cone | cooling deviation relative error ~1.7934; all predicted states positive | true temperature deviation is negative | representation failure; distinguish Kelvin temperature from signed deviation |
| C005 event order | endpoint path discrepancy ~0.244 relative to the true endpoint; closed loop response change ~0.17955 | noncommuting material edit generators introduce fictitious history | physical consistency failure for memoryless steady PDEs |
| C001R accurate but known | corrected heterogeneous/local/BC cases match direct solver ~1e-15 | exact classical Green response | novelty failure; retain as control, not a newly named neural operation |
| C006 spectral OOD | low-k test mean error ~0.80424 versus IID ~0.0026435 | finite maximum age makes g(0) finite; source-independent structural limit | range/generalization failure; cannot fix uniformly with finite packet tuning |
| C006 high-frequency/source OOD | mean error ~0.09862 for mode12 source | exponential spectral tail versus inverse algebraic tail | separate asymptotic defect, not only material-data coverage |
| C006R tail repair | accurate shifted solves, but full dense factorization | correction converges to the original solver | computation/novelty failure for the target fast neural primitive |
| Self-adjoint transport | ignoring skew part yields ~0.11874 relative error at cell Pe10 | direction and phase removed | domain-extension failure; positivity and diffusion do not carry advection |
| Divergence-only flow validation | manufactured nonlinear flow error ~0.27242 despite divergence <1e-12 | omitted projected convection; exact Stokes baseline fails similarly | model-class omission, not learned weights or hidden instability |
| Weak MLP baseline | raw IID mean ~0.45762; normalized/validation-selected ~0.05628 with more steps | poor output/input conditioning and different training budgets | measurement-design failure; preserve pilot and report stronger controls |
| Data-fitted linear control defeats apparent benefit | 32 or128 labels recover fixed-mesh response to ~1e-15 with known 1/k scaling | scalar homogeneous-k family is a very simple linear map | no evidence of neural superiority or need for an invented primitive on that family |
| SPD does not imply nodal positivity | exact short-time anisotropic FEM semigroup min value -5.75638e-5 | positive stiffness off-diagonals, non-M-matrix mesh/operator | **reference discretization defect** for a continuous positivity claim; not a scipy.expm error |
| C007 wrong gauge | uniform source relative error 1; local source ~4.38616 and negative remote response | applies potential gauge invariance to the source and wrong inverse sign pattern | structural; learning a length cannot restore a missing constant-source response |

An integrity review also found a matrix-right-hand-side broadcasting bug in `spectral_response`: a vector of eigenvalue multipliers was applied to columns instead of rows for a matrix source. The actual physics experiments had used single-vector sources or a separate correct learning path, so their results were unaffected. The function was corrected, multi-source/individual-source equivalence assertions were added, tests reran, and the entire study was replayed as `session_0002_verified` with matching corrected source snapshots. This was an implementation bug, distinct from the scientific candidate failures.

No existing result was deleted. `session_0002` is the pilot (some diagnostic timings were unmeasured there); `session_0002_revision` adds baseline refinement and measurement scope; `session_0002_final` is the first complete run; `session_0002_verified` is the corrected-code replay with all 198 rows timed. Earlier cycle reports are archived under `session_0002_final/report_snapshot`. These are measurement replays and additions within the same 12 scientific cycles, not independent evidence counts. Source snapshots preserve the executable versions.

Unrun ideas remain unrun: signed paired multiplicative packets, commuting/integrable material generators, nonlinear implicit learned response, arbitrary CAD meshes, learned pure-Neumann generalization, RANS, GPU scaling, a broad architecture baseline sweep, and patent search.
