# Experiment 0001: CPU oscillator baseline

## Purpose and hypothesis

Validate the research environment before inventing architectures. A quadratic scalar Hamiltonian `Hθ(q,p)=(a q²+b p²)/2+c qp`, differentiated as `(dq/dt,dp/dt)=(∂H/∂p,-∂H/∂q)`, should recover the unit harmonic oscillator from exact derivative observations. This is a baseline with **unverified novelty**; no paper search or source verification has been performed and no architectural novelty is claimed.

## Protocol

- CPU only, float64; one Torch thread; seed 824.
- 128 synthetic states sampled uniformly from `[-1,1]²`, exact targets `(p,-q)`.
- Initial parameters `(0.7,1.3,0.2)`; Adam learning rate 0.05; 400 steps.
- Additional derivative checks use held-out states `(2,-3)` and `(-4,2)`.
- Rollout from `(q,p)=(1,0)` over time `[0,20]`; 401 samples.
- SciPy `solve_ivp` with `rtol=1e-10`, `atol=1e-12` for the reference and learned systems.
- Euler and leapfrog use step size 0.05 for 400 steps.
- Versions and machine-readable measurements: `0001-baseline-metrics.json`; full transitive dependency pins and hashes: `../requirements-cpu.lock`.

## Failed conservation formula: deliberate negative control

Forward Euler uses `q_next=q+h*p`, `p_next=p-h*q`. It fails exact energy conservation: `E_next=(1+h²)*E`, so after 400 steps the relative energy error must be `(1+0.05²)^400-1`. This is an intentionally unsuitable integration baseline, not an unexpected software failure or a newly discovered result. It demonstrates that correct continuous Hamiltonian equations do not guarantee energy conservation under an arbitrary discrete integrator. Preserve this negative result when evaluating later architectures.

Leapfrog uses `p_half=p-h*q/2`, `q_next=q+h*p_half`, `p_next=p_half-h*q_next/2`; it is the comparison for bounded energy error in this experiment.

## Observed results

- Training derivative MSE: `0.03997153115202883` → `2.745474058498152e-21`.
- Learned `(a,b,c)`: `(0.9999999999459299, 0.9999999998992237, 4.47737602504792e-11)`.
- SciPy reference maximum absolute state error: `2.2833668289479192e-10`.
- Learned rollout maximum absolute state error: `1.4688154303943435e-09`.
- Euler final relative energy error: `1.7148917443812932` (about 171.49%).
- Leapfrog maximum absolute relative energy error: `0.0006249937444130493` (about 0.0625%).
- pytest: **5 passed, 0 failed, 0 skipped**, final run without warnings.
- Matplotlib PNG decoded successfully and rollout NPZ contains 401 finite two-dimensional states.

Commands, from `/workspace/physics`:

```bash
bash scripts/setup.sh
export XDG_CACHE_HOME=/workspace/.cache
export MPLCONFIGDIR=/workspace/.cache/matplotlib
export OMP_NUM_THREADS=1
/workspace/.physics-venv/bin/python -m pytest -q --junitxml=artifacts/validation/pytest.xml
/workspace/.physics-venv/bin/python -m experiments.oscillator --output artifacts/oscillator
```

Installation was repeated successfully with the hashed lock and no dependency changes on the second run. Initial setup exposed an unwritable default cache directory; writable cache paths above resolved it. Matplotlib 3.10.3 with pyparsing 3.3 emitted deprecation warnings; pinning pyparsing 3.2.3 resolved them without suppressing warnings.

## Limits and next research work

This tests an exactly representable, noiseless, linear system with one seed and one rollout initial condition. It does not establish performance on nonlinear systems, noisy observations, difficult generalization, or advantages over other architectures. The Hamiltonian coefficients are not constrained to be positive definite during optimization. Continuous energy conservation from the skew-gradient structure does not prove discrete rollout conservation. Literature review and broader baselines are required before any novelty or comparative-performance claim.

No GPU training, paid resource, external dataset, or research upload was used. Records are local and awaiting approval before GitHub publication; the remote repository has not been changed.
