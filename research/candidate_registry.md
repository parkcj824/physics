# Candidate registry — session 0002

No architecture is selected as a substantiated invention. A method can be a useful control and still be rejected as an invention. Names are identifiers, not novelty evidence.

| ID | Independent starting principle / object | Cycles | Final research verdict | Disqualifying evidence / next use |
|---|---|---|---|---|
| C001 | equilibrium source-response anchored by boundaries | 1,6 | REJECT | localized/heterogeneous response and reciprocity fail; corrected C001R is exactly classical 1D Green response |
| C002 | conservation / opposite exchange budgets | 2 | REJECT | fixed step unstable; corrected step is slow Richardson relaxation |
| C003 | stationarity / admissible variations | 3 | REJECT as invention | exact Galerkin/Ritz form; rank-4 span misses mode-8 forcing completely |
| C004 | irreversible positive response mass | 4 | REJECT for generic signed thermal state | negative temperature deviations cannot be represented; positive-state subproblems remain possible |
| C005 | material-parameter events | 5 | REJECT for steady heat | endpoint/closed-loop path defects; no actual physical hysteresis in benchmark |
| C006 | relaxation age / finite positive response measure | 7–11 | REVISE as numerical probe; REJECT as established invention | good limited-range accuracy but severe spectral OOD failure, full eigensystem cost, classical exponential inverse ancestry; novelty review incomplete |
| C006R | explicit revision of C006, not independent hypothesis | 8–10 | REJECT as invention | shifted exact solve/backward Euler; restores accuracy by reverting toward the solver |
| C007 | source gauge / neutral excitation contrasts | 12 | REJECT | annihilates a physical uniform source and predicts negative remote response; equivalent to a source Laplacian |

## Eight-axis assessment (qualitative analysis, not measured scores)

| Candidate | Difference from standard neural rules | Physical validity | Expressivity | Training | Resolution / geometry | Domain extension | Implementation / cost | Potential novelty |
|---|---|---|---|---|---|---|---|---|
| C001 | normalized physical kernel, close to attention normalization | only special uniform case | linear, limited material information | length could train; not a cure for missing structure | normalized sums alone do not guarantee consistency | fails thermal prerequisite | easy; dense N² | weak; correction is classical |
| C002 | explicit antisymmetric exchange | internal conservation conditional on BC/storage | finite-depth iteration limited by spectrum | scalar/rates learnable, not trained here | step/iteration count depends on h | diffusion algebra only | easy; sN k_nb sparse | no independent novelty |
| C003 | stationarity in global span | energy optimal within span | rank-limited | basis could train; fixed span tested | continuous basis helps, geometry support untested | self-adjoint heat/Stokes | small reduced solve plus assembly | Galerkin identity excludes claim |
| C004 | multiplicative instead of additive | signed-state mismatch | positive cone only | rate learnable; overflow/stiffness risks | no resolution robustness proof | concentrations possible, signed fields fail | easy; repeated sparse products | overlaps mirror descent |
| C005 | time-ordered material events | inconsistent steady endpoint | noncommuting history effects | generators learnable; not trained | input geometry not specified by event rule | hysteresis might differ; unrun | rank-1 easy, dense exponential expensive | standard exponential flows; no validated distinction |
| C006 | positive age measure in physical eigenbasis | reciprocal work-positive under assumptions | finite spectral window, source-linear | 12 trained scalars; three seeds | reuse demonstrated only with full supplied operator | heat and projected Stokes reuse; advection/nonlinearity fail | dense setup N³, cached transforms N² | specific learned quadrature novelty unverified; insufficient evidence |
| C007 | source differences rather than potential differences | wrong gauge and inverse sign | constant source removed | learning length cannot repair nullspace | any mesh shares same flaw | heat failure blocks promotion | easy; dense N² | graph Laplacian equivalence |

The initial hypothesis set spans response, conservation, action, irreversibility and events. C006 is a new relaxation-object branch after those failures; C007 is an independent challenger after the spectral branch stalls. No rejected candidate has been renamed as a success.

Authoritative measurements: `../results/session_0002_verified/metrics/all_cases.json`; pilot and revised measurements are preserved separately. Full formulas and complexity assumptions: `../architectures/candidate_*.md`. Cycle decisions: `../reports/cycles/`.
