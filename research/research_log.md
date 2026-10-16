# Research log

## Preserved setup stage

`0001-oscillator-baseline.md` and its metrics remain unchanged. Python 3.12 / NumPy / SciPy / Matplotlib / pytest / CPU PyTorch were already validated. This session checked the actual machine again: 5 reported logical CPUs, a cgroup quota of 4 CPU equivalents, 32 GiB cgroup memory limit, and no CUDA device. Runs use one Torch/BLAS thread and small synthetic problems.

## Session 0002 — 2026-10-17, Asia/Seoul

The source machine records UTC timestamps in run metadata; human-facing session date above uses the user's timezone. Existing untracked project files were inspected and preserved. The requested persistent research rules were appended to AGENTS.md. The existing checkout root is used rather than creating a second repository/worktree.

1. Defined at least five independent physical hypotheses before selecting any candidate. Response, exchange, admissible variation, positive mass and material events were evaluated in Cycles 1–5.
2. Executed small Dirichlet heat problems, inspected counterexamples, then repaired the response using resistance coordinates in Cycle 6. Recognized the exact classical Green identity and rejected an invention claim. Verified compatible mean-zero Neumann controls and rejected an incompatible source.
3. Branched to relaxation-age packets (C006), trained three seeds on 128 synthetic problems, and compared held-out conditions with MLP. Severe low-material and spectral-tail limits were measured, not hidden behind IID accuracy.
4. Cycle 8 tested a rational tail repair, resolution changes, 2D rectangles/holes, heterogeneous and rotated anisotropic materials, BC/source changes, analytic manufactured convergence, and a nonmonotone-FEM positivity counterexample. The repair was recognized as a shifted solve.
5. Cycle 9 falsified a self-adjoint extension to directed advection and ran classical implicit controls with changed BCs, cell Péclet numbers, reversed velocities, time refinement and long-time stability.
6. Cycle 10 reused the seed824 heat filter in channel/periodic Stokes and a nonlinear manufactured-flow negative test. Correct divergence did not establish momentum correctness.
7. Cycle 11 corrected a weak-baseline comparison by adding input/output normalization, validation-only checkpoint selection, 32/128-label experiments, and a full-rank learned linear compliance with known 1/k scaling. The stronger classical control removed any observed need for a new neural primitive on this small homogeneous family.
8. After stalled novelty progress, Cycle 12 changed the object to excitation contrasts, then rejected its source-gauge confusion on uniform/local heat forcing. The session stopped at its explicit 12-cycle budget, not because an invention was found.

## Execution and provenance

The authoritative final run is `results/session_0002_verified` (198 result rows). Some rows aggregate 32 cases; 198 is a record count, not 198 independent discoveries or a uniform sample count. Computational section timing is recorded in that run's `run.json`; it excludes researcher analysis, implementation, source retrieval, report writing and test startup. Session duration is not represented by that compute-only measurement.

The initial run and intermediate revisions were preserved. The complete run fixes missing pilot measurement scope and adds stronger controls, long-time transport, and a reference-positivity counterexample. A later matrix-source broadcasting bug was identified and fixed; the physics experiments' vector-source results were unaffected. Multi-source assertions and a full corrected-code replay provide the final verified record. Earlier reports are archived rather than deleted. These replays are not new scientific cycles. Each scientific cycle has a separate ten-part report under `reports/cycles/`.

Final test execution: `21 passed`, no failures/skips, recorded in `artifacts/validation/research_pytest.xml`. Result integrity verification recomputes errors from saved arrays, validates source hashes and PNG files, and records `results/session_0002_verified/integrity.json`. This integrity report is separate from physics or novelty proof.

## Literature status

Official public source retrieval succeeded for ten implementation/library descriptions plus selected core files; seven of those are direct model author/official simulator descriptions. Original arXiv/PMLR/Crossref access returned proxy 403. Source-level verification and the pending literature ledger are in `novelty_review.md`. No original full paper or patent was falsely represented as reviewed.

## External actions

No Git commit/push, branch reset, deletion, paid resource, GPU training or research-data upload was performed. All synthesis and labels are local. GitHub publication remains subject to the user's prior-approval rule. Saved environment drafts, if updated, are configuration only; publication and new-task restoration are not claimed.
