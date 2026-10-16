# Minimal computational representations

Physical input information (A), computational objects (B), and rules (C) are separate. A temperature tensor stores continuous numbers, not vocabulary IDs. No tokenizer, attention block, embedding, encoder, decoder, or residual connection is a prerequisite here.

| B: object | A: required physical information | C: candidate rule | Strength | Structural limitation |
|---|---|---|---|---|
| Boundary-anchored source response | coordinates, source, boundary values, effective material | C001 weighted global response | one-pass, handles a uniform analytic case | averaging weights can break reciprocity and spatial material dependence |
| Exchange budget between freedoms | incidence, conductance, source/storage scale | C002 opposite transfers and source injection | internal cancellation is explicit | time-step restriction and slow global propagation; same as classical relaxation |
| An admissible variation of a whole field | boundary lift, positive energy form, trial functions | C003 stationary action on a span | global coupling and reduced work | a missing source direction cannot be recovered; classical reduced basis |
| Positive response mass | positive initial packet, dimensionless driving gradient | C004 multiplicative change | positivity by construction for finite exponentials | excludes signed thermal deviations; does not fix stiffness or BC compatibility |
| Material-change event | final material descriptors and ordered perturbations | C005 products of event generators | compact parameter-space effects | noncommuting updates introduce spurious path dependence in steady problems |
| Relaxation-age packet | admissible self-adjoint physical operator and source | C006 positive learned age measure | 12 parameters reused across discretizations/domains | spectral range limits; prototype needs full eigensystem; exponential inverse approximation is known |
| Neutral source pair | source differences and pair geometry | C007 symmetric contrast response | symmetric PSD and zero-sum pair changes | source-shift invariance is physically wrong for Dirichlet heat; inverse sign pattern is wrong |

At least the response, variation, event, age, and contrast representations are distinct hypotheses; they are not five attention variants. Some rules prove equivalent to existing numerical methods and are explicitly excluded as independent inventions.

## Common types and dimensions

`N` is the number of free scalar freedoms, `d` is the number of channels, `k_nb` the number of neighbors, `r` a trial-space rank, and `m` an age-packet count. Current prototypes have `d=1` for heat and share scalar response filters across two projected velocity channels. `b∈R^N` is the source/load after boundary lifting and mass scaling. `A∈R^(N×N)` is a classical reference operator, not learned ground truth hidden in a dataset label. Any candidate requiring `A` is marked as receiving this strong physical prior.

In 2D FEM the mass-coordinate state is `z=M^(1/2)T_free`, `A=M^(-1/2)K_ff M^(-1/2)`, and `b=M^(-1/2)(f-K lift)_free`. Symmetry claims refer to `z`, not an unweighted temperature kernel on arbitrary meshes. Neumann problems additionally require a compatible source and an explicit constant-mode gauge.
