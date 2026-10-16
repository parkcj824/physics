# Exact resumption state — session 0002

## Stop state

The session reached its **12 scientific-cycle budget**. Seven independent hypotheses, explicit revisions and **198 result records** are saved in `results/session_0002_verified`. **21 tests passed**, none failed/skipped. No novel architecture is selected. Earlier measurement replays are preserved, not counted as independent discoveries.

1. **Current best invention:** none. Best learned numerical control is the fixed-mesh classical compliance map `T=(Q/k) C_hat` with known exact scalar-material scaling;32/128 labels recover the full-rank map to floating-point accuracy.
2. **Compact probe:** C006, `g(lambda)=sum_j exp(alpha_j-exp(beta_j)*lambda)`, `T=V diag(g)V^T b`. Twelve parameters. It needs a complete physical eigensystem; positive exponential inverse quadrature is not established as new.
3. **Successes:** IID heat mean relative error0.0026435 across3seeds; supplied-operator 2D heat reuse; some projected Stokes cases; exact classical Green/gauge/reference controls. See `reports/current_best.md`.
4. **Failures:** low-k OOD0.80424, mode12 OOD0.09862, nonlinear flow0.27242 despite divergence~1e-13. Source gauge, event order, positive-only state and exchange stability failures are in `failures.md`.
5. **Unresolved:** full-paper/patent review, learned Neumann generalization, arbitrary CAD, monotone anisotropic reference, nonlinear momentum architecture, GPU/CAE scaling. RANS and other domains are unrun.
6. **Next independent ideas:** oriented perturbation-response objects and constraint-witness objects; these are *unrun questions without a completed primitive equation*, not newly named inventions. They must pass prior-art and equation-level scrutiny before implementation.
7. **Commands:** below.
8. **Exact order:** inspect evidence → verify environment/results → restore original-source access → complete nearest-prior-art checks → define two independent new objects/rules → fastest heat counterexample → only surviving candidates receive learning and flow tests.

## 1. Inspect and verify before changing code

Read `AGENTS.md`, `reports/current_best.md`, `candidate_registry.md`, `failures.md`, `novelty_review.md`, then the relevant ten-part reports in `reports/cycles/`.

```bash
cd /workspace/physics
source /workspace/.physics-venv/bin/activate
export XDG_CACHE_HOME=/workspace/.cache
export MPLCONFIGDIR=/workspace/.cache/matplotlib
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
python -m pytest -q
python scripts/verify_results.py results/session_0002_verified
```

Inspect that run's `run.json`, `integrity.json`, `source_snapshot/sha256.json`, `metrics/all_cases.json` and `comparisons/normalized_mlp.json`. The verifier compares current source with snapshot hashes: intentional later code changes require inspecting the provenance difference, **not** rewriting historical snapshots.

## 2. Original-source access

arXiv/PMLR/Crossref requests returned proxy403. Official source READMEs/code were read; full papers were not. The environment draft save succeeded and requires publication. It preserves the installation script, updates `start_skill`, and adds four custom domains while preserving package-manager presets: `arxiv.org`, `proceedings.mlr.press`, `api.crossref.org`, `jmlr.org`. Saving is not runtime application or publication.

The user can review/save the settings and Publish the prepared environment. After a real runtime change, retry a read-only public request:

```bash
python - <<'PY'
import urllib.request
with urllib.request.urlopen('https://arxiv.org/abs/2010.08895', timeout=20) as response:
    print(response.status, response.headers.get('content-type'))
PY
```

Do not loop identical403 requests or request a GitHub token; the observed prerequisite is public-paper egress. No secret is required for current numerical work. Verify settings propagation through the actual request before claiming access.

## 3. Literature before more model tuning

Start with learned Green kernels, exponential/rational inverse approximations and spectral response learning; fill the GENERIC/metriplectic, port-Hamiltonian, symplectic, domain-decomposition, multiscale, learned-iteration, finite-volume and homogenization gaps in `novelty_review.md`. Record exact metadata, actual equation, reading depth and formula difference. Patent search is unperformed.

Already identified forms must not be renamed: resistance Green response, Richardson, Galerkin, positive exponentiated-gradient motion, generic exponential flows, Laplace inverse quadrature and shifted resolvents. Adding an encoder to them is insufficient novelty.

## 4. New-session exploration/exploitation

Begin Cycle13 onward with a fresh maximum of12cycles/90minutes. No multi-agent delegation is authorized by the current AGENTS.md.

- Independent branch: an **oriented perturbation-response relation including constraint reaction** as the object. Derive the distinction between reciprocal elliptic susceptibility and directed transport. First tests must reverse velocity, superpose sources and check constraints. A simple symmetric/skew split is existing mathematics and must not be claimed as an invention.
- Independent alternative: a **constraint witness** (admissible bounds plus dual balance certificate) rather than all numerical solver states. Ask whether a concrete witness composition yields a direct response with a provable bound. No complete equation or experiment exists yet; projection/bound-propagation prior art may disqualify it.
- Limited exploitation: use C006 only to understand spectral-range limits. Do not spend a session tuning finite positive ages to approximate1/lambda on the whole positive axis, or hiding eigen/solve costs. C007 is structurally rejected; do not tune/rename its source Laplacian.

For each actual new equation, define inputs/types/units/trainable variables and exact guarantees, then attempt a small heat counterexample before training. Maintain a control and independent challenger. An empty current-best-invention slot is acceptable.

## 5. Only surviving candidates receive broader benchmarks

In order: heterogeneous1D/local source/changing BC → learned compatible Neumann/gauge →2D new materials/geometry/resolution with a correctly qualified positivity reference → matched-prior/data/compute controls → directed advection phase and nonlinear manufactured momentum. Include nested32/128-label comparisons; C006's own32-label learning curve is currently unrun. RANS, large datasets and other physics domains are not the next step.

Replay only to resolve a reproduction concern, using a new path:

```bash
python -m experiments.run_research --output results/session_0003_replay
```

Existing output paths are refused. `--phase initial` followed by `--phase continue` supports reviewing the first six cycles. Replays are verification, not new independent discoveries. The report writer is session-specific; do not blindly overwrite old reports when adapting it.

Local coding/testing is authorized. Ask before research-data uploads/Git pushes, deletion, paid resources, major repository changes or large GPU training. GitHub publication, environment publication and fresh-task restoration remain separate actions; none has been performed here.
