"""Render cycle reports from executed measurements; never invent numerical results."""

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


ROOT=Path(__file__).resolve().parents[1]
RESULT=ROOT/'results/session_0002_verified'


def write_new(path,text):
    if path.exists():
        old=path.read_text()
        if not old.startswith('# Cycle ') or 'results/session_0002_final' not in old:
            raise FileExistsError(f'Preserve existing report: {path}')
        archive=ROOT/'results/session_0002_final/report_snapshot'/path.relative_to(ROOT)
        archive.parent.mkdir(parents=True,exist_ok=True)
        if archive.exists():
            raise FileExistsError(f'Preserve existing archive: {archive}')
        archive.write_text(old)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(text)


def main():
    rows=json.loads((RESULT/'metrics/all_cases.json').read_text())
    # principle, object, explicit rule, distinction, physics, inventor, physicist,
    # numerical analyst, architecture reviewer, patent-view check, challenger, verdict
    cycles=[
      ('Boundary-anchored response','equilibrium and source response','boundary-anchored source response',
       r'T_i=a(1-x_i)+bx_i+\frac{x_i(1-x_i)}{2k_{eff}}\sum_j\frac{e^{-|x_i-x_j|/\ell}}{\sum_l e^{-|x_i-x_l|/\ell}}q_j',
       'physical anchor and effective material replace arbitrary features, but normalization is kernel/attention-like',
       '-(kT_x)_x=Q; N=15; uniform/local source, heterogeneous k and changed Dirichlet data',
       'The uniform-source parabola is exact without solving a system.',
       'Local sources and spatial material distributions require reciprocal nonlocal Green response.',
       'A special analytic fit does not control residual or heat balance.',
       'Normalized positive kernels are not a new primitive by name.',
       'No specific prior-art distinction or robust technical effect is established.',
       'Use conserved exchanges, then revisit the response with resistance coordinates.', 'REVISE'),
      ('Exchange budgets','conservation','opposite exchange budget',r'T^+=T+\tau(b-AT);\quad\tau_R=1/\lambda_{max}(A)',
       'antisymmetric exchange explicitly cancels internal transfers; expanded rule is Richardson relaxation',
       '-T_xx=1; N=7,15,31; zero Dirichlet; 8 unstable or 32 stable steps',
       'Opposite internal transfers cancel exactly in the incidence formulation.',
       'Internal cancellation does not remove sources or boundary power.',
       'Fixed tau violates the spectral stability limit; stable steps remain slow under refinement.',
       'This is a classical learned-step iterative solver family.',
       'Stability correction alone is not a differentiated invention.',
       'Try admissible whole-field variations instead of repeated local exchange.', 'REJECT'),
      ('Action span','stationarity of energy','admissible field variation',r'T=\Phi(\Phi^TA\Phi)^{-1}\Phi^Tb',
       'global stationary response, exactly Ritz/Galerkin instead of learned pair weights',
       '-(kT_x)_x=Q; rank4 sine span; includes mode8 forcing',
       'The energy is minimized within a compact admissible span.',
       'Missing source directions have nonzero physical solutions but zero predicted response.',
       'Projected residual zero does not mean local residual or flux balance zero.',
       'Exact reduced-basis equivalence excludes an independent primitive claim.',
       'A learned basis would require evidence beyond existing model reduction.',
       'Test positive-state response masses from irreversibility.', 'REJECT'),
      ('Multiplicative mass','irreversibility and positivity','positive response mass',r'u_i^+=u_i\exp[\eta(b-Au)_i]',
       'multiplicative state updates overlap entropy/mirror descent rather than attention',
       '-T_xx=+1 or -1; zero Dirichlet; 128 steps eta=0.001',
       'Finite exponential updates keep every positive packet positive.',
       'Cooling temperature deviations are negative; absolute temperature is a different variable.',
       'Positivity alone does not guarantee stiffness robustness, energy descent or finite exponentials.',
       'Exponentiated-gradient ancestry must not be renamed as novelty.',
       'A generic signed thermal operator is not implemented by this positive cone.',
       'Test events in material-parameter space, an independent representation.', 'REJECT'),
      ('Material events','state change and endpoint consistency','material-change event',r'T^+=e^{\delta c_jG_j}T;\quad e^{c_2G_2}e^{c_1G_1}\ne e^{c_1G_1}e^{c_2G_2}',
       'event composition acts in physical parameter space instead of source aggregation',
       'steady 1D heat; two half-material edits end at uniform k=2; closed loop returns to k=1',
       'Low-rank generators represent compact responses to parameter changes.',
       'The steady endpoint is independent of edit order, unlike the candidate.',
       'A closed material loop produces a measurable spurious change.',
       'Matrix-exponential flows/Lie products are existing mathematics.',
       'No benefit survives the simplest endpoint test; no patent assertion is warranted.',
       'Repair the original response analytically to identify what information was missing.', 'REJECT'),
      ('Resistance repair and gauge','reciprocal response and compatibility','resistance-coordinate source response',
       r'G_{ij}=h\min(R_i,R_j)[R_*-\max(R_i,R_j)]/R_*;\quad T=Gq+a(1-R/R_*)+bR/R_*',
       'normalization replaced by an exact resistance Green function; this is a known classical solution',
       'heterogeneous 1D heat, N=15/31; separate compatible pure-Neumann N=16/32/64 with mean zero',
       'The corrected kernel gives exact discrete response and reciprocity.',
       'A Neumann source must balance boundary flux and the additive constant must be fixed.',
       'Check continuum convergence separately from exact agreement with the tridiagonal matrix.',
       'This is the classical Green/resistance formula, not a neural invention.',
       'Equation identity disqualifies renaming this correction as a novel primitive.',
       'Explore relaxation-age objects from irreversible response instead of a spatial kernel.', 'REJECT'),
      ('Relaxation-age learning','relaxation and response','positive age packet',
       r'T=V\operatorname{diag}(\sum_{j=1}^6e^{\alpha_j}e^{-e^{\beta_j}\lambda})V^Tb',
       'learned positive age measure in the actual physical eigenbasis, not unconstrained Fourier channel weights',
       '-kT_xx=Q; 128 training cases, 32 validation, 32 per test condition; three seeds',
       'Twelve scalars give accurate in-range linear elliptic response.',
       'Low conductivity exposes the missing long-age response; finite sums cannot diverge like 1/lambda at zero.',
       'Large-lambda inverse tails decay exponentially and source smoothness can hide this defect.',
       'It is an exponential-sum inverse approximation with a full operator prior; learned quadrature novelty is unverified.',
       'A new term for a known inverse-integral approximation is not a technical distinction.',
       'Repair the spectral tail, then check whether that merely recreates a solver.', 'REVISE'),
      ('Spectral repair and geometry','relaxation, reciprocity and positivity','age measure or shifted-resolvent response',
       r'g_R(\lambda)=1/(\lambda+10^{-6});\quad T_R=(A+10^{-6}I)^{-1}b',
       'tail repair is exactly a shifted solve; geometry enters through classical FEM assembly',
       '1D N=7/15/31/63, low/high k; 2D rectangle/hole/aspect/material/anisotropy/BC/source; manufactured convergence',
       'The original filter can reuse coefficients on new supplied physical operators.',
       'Reciprocal SPD response is weaker than pointwise positivity or correct flux.',
       'Non-M-matrix anisotropic FEM has a negative short-time heat response despite SPD; full eigensetup is expensive.',
       'The resolvent becomes the original solver as lifetime increases; not a new fast network.',
       'Neither correct geometry assembly nor a shifted solve establishes a distinct invention.',
       'Challenge the self-adjoint assumption with directed advection.', 'REVISE'),
      ('Directed transport','causality and irreversible direction','time-step response with an antisymmetric component',
       r'u_{n+1}=(I+\Delta tL)^{-1}(u_n+\Delta tb);\quad L\ne L^T',
       'the extension is backward Euler; deleting the skew part removes physical transport direction',
       'u_t+v u_x=k u_xx; Pe_cell=0.1,1,10; v=+-1; periodic/inflow; time refinement and t=10',
       'A positivity-preserving nonsymmetric resolvent handles directed transport stably.',
       'Symmetrizing the operator cannot represent reversed advection.',
       'Long-time boundedness and mass conservation do not imply accurate phase or transient response.',
       'Backward Euler is a classical control, not an invented temporal neural operation.',
       'Known time integration cannot supply the required architectural distinction.',
       'Check incompressibility and momentum separately in fluid examples.', 'REJECT'),
      ('Heat-to-flow challenge','constraint versus governing dynamics','response in a divergence-free subspace',
       r'\hat u=g_\theta(\mu|\xi|^2)(I-\xi\xi^T/|\xi|^2)\hat f',
       'fixed Helmholtz projection plus the heat-trained scalar response is a classical structure-restricted reuse test',
       'Poiseuille, Couette, periodic manufactured Stokes and nonzero-inertia manufactured Navier-Stokes',
       'The same heat-trained filter reuses linear response mathematics in projected Stokes.',
       'Zero divergence does not enforce nonlinear momentum; small viscosity also leaves the trained spectrum.',
       'The exact Stokes control fails the nonlinear test too, identifying a model omission rather than a training bug.',
       'Projection plus spectral response is not a new primitive simply because two domains use it.',
       'No generic CAE or turbulence claim follows from these manufactured linear successes.',
       'Strengthen baseline conditioning before inferring any architecture advantage.', 'REVISE'),
      ('Baseline self-refutation','linearity and dimensional scaling','learned fixed-mesh compliance',
       r'\widehat C=\arg\min_C\|XC-Y\|_F^2,\quad X=Q/k;\quad T=(Q/k)\widehat C',
       'this is a strong classical data-fitted linear control, not a new architecture',
       'same homogeneous scalar-k 1D task; 32/128 labels; normalized MLP with separate validation and 3000-step budget',
       'A full-rank fixed-mesh linear map exploits the exact physical 1/k scaling.',
       'An expressive neural model is unnecessary for this particular linear operator family.',
       'Initial MLP conditioning was poor; normalized controls materially change the comparison.',
       'More parameters, different priors and different compute budgets prevent sweeping model rankings.',
       'No demonstrated technical advantage supports an invention claim here.',
       'After stalled novelty, replace the age object with an independent source-contrast hypothesis.', 'REJECT'),
      ('Independent source contrast','gauge and neutral excitation','signed source pair',
       r'T_i=\ell^2/k_{eff}\sum_{j\ne i} e^{-|x_i-x_j|/\ell}(q_i-q_j)',
       'the rule is a kernel Laplacian on sources, not an inverse response',
       '1D uniform, localized and heterogeneous heat cases',
       'The response is symmetric PSD and explicitly cancels uniform source offsets.',
       'Uniform source offsets are physical under Dirichlet data; a source is not a potential gauge.',
       'Localized positive forcing produces negative remote temperatures, violating the tested maximum principle.',
       'Pair differences are standard graph/kernel Laplacian aggregation.',
       'The mathematical defect precedes any patent or novelty investigation.',
       'Stop at 12 cycles; next session must change objects and complete source verification before further tuning.', 'REJECT'),
    ]
    for index,entry in enumerate(cycles,1):
        name,principle,obj,formula,difference,physics,a,b,c,d,e,f,verdict=entry
        selected=[r for r in rows if r['cycle']==index]
        lines=[f'# Cycle {index} — {name}', '',
          '## 1. Physical principle', principle, '', '## 2. Minimal computational object', obj, '',
          '## 3. Implemented formula', '\\[',formula,'\\]', '',
          'The complete inputs, trainable/fixed quantities, guarantees and complexity are in `../../architectures/candidate_*.md`.', '',
          '## 4. Difference from existing operations',difference,'',
          '## 5. Governing equation, conditions and reference',physics,
          'References are independent direct discrete solves, analytic solutions, augmented gauge solves or matrix exponentials as identified in each saved record. The anisotropic positivity counterexample intentionally uses the exact discrete semigroup to expose a discretization defect.', '',
          '## 6. Executed results',
          f'Final verified run: `{RESULT.relative_to(ROOT)}`; {len(selected)} records in this cycle. Complete inputs, predictions and references are saved in the NPZ paths in the metrics JSON.', '',
          '| Candidate | Case | Relative L2 | Max absolute error | Median seconds | Traced peak bytes |',
          '|---|---|---:|---:|---:|---:|']
        for r in selected:
            lines.append(f"| {r['candidate']} | {r['case']} | {r['relative_l2']:.6g} | {r['max_absolute_error']:.6g} | {r['latency_median_seconds']:.6g} | {r['traced_peak_bytes']} |")
        lines += ['', 'Array/batch relative L2 and mean per-sample relative L2 are distinct. Per-sample means/std, heat flux errors, physical balances, BC handling, momentum/divergence residuals and timing/memory scope are in the JSON. Tracemalloc does not measure all native/BLAS memory; RSS is cumulative process high-water. Timing is a small CPU microbenchmark, not GPU performance.', '',
          '## 7. Self-refutation in six roles',
          f'- A — Inventor: {a}',f'- B — Theoretical physicist: {b}',
          f'- C — Numerical analyst: {c}',f'- D — Architecture researcher: {d}',
          f'- E — Patent-review perspective: {e} This is not a patentability finding.',
          f'- F — Alternative-hypothesis researcher: {f}', '',
          '## 8. Prior work and verification limits',
          'See `../../research/novelty_review.md` for checked official source descriptions/code and the full coverage ledger. Original paper sites returned proxy 403; no unread full text is presented as reviewed. Exact classical algebraic equivalences are sufficient to withhold an invention claim. Architectural novelty and patent coverage are unverified.', '',
          '## 9. Verdict',f'**{verdict}**. This is a cycle-level decision; final candidate-level dispositions are in `../../research/candidate_registry.md`. No final invention is selected.', '',
          '## 10. Next cycle hypothesis / stopping state',f,'']
        write_new(ROOT/f'reports/cycles/cycle_{index:02d}.md','\n'.join(lines))
    # A final figure includes the stronger controls rather than hiding them behind the weak pilot MLP.
    aggregate=json.loads((RESULT/'comparisons/aggregate.json').read_text())
    labels=['iid','low_conductivity_ood','high_conductivity_ood','high_frequency_source_ood']
    fig,ax=plt.subplots(figsize=(10,4.5))
    positions=np.arange(len(labels))
    models=['C006','MLP_normalized','linear_compliance']
    for j,name in enumerate(models):
        means=[];std=[]
        for condition in labels:
            if name=='C006':
                stats=aggregate[name][condition]; means.append(stats['mean_over_seeds']);std.append(stats['std_over_seeds'])
            elif name=='MLP_normalized':
                values=[r['mean_sample_relative_l2'] for r in rows if r['candidate']==name and r['training_samples']==128 and r['case'].startswith(condition+'_')]
                means.append(np.mean(values));std.append(np.std(values,ddof=1))
            else:
                r=next(r for r in rows if r['candidate']==name and r['training_samples']==128 and r['case'].startswith(condition+'_'))
                with np.load(RESULT/r['predictions_file']) as data:
                    values=np.linalg.norm(data['prediction']-data['reference'],axis=1)/np.linalg.norm(data['reference'],axis=1)
                means.append(values.mean());std.append(0.)
        ax.bar(positions+(j-1)*.25,means,.25,yerr=std,label=name)
    ax.set_xticks(positions,['IID','low k OOD','high k OOD','high frequency OOD'])
    ax.set_yscale('log');ax.set_ylabel('mean sample relative L2 (log scale)')
    ax.set_title('Controls include known 1/k scaling; budgets/priors differ')
    ax.legend();fig.tight_layout();fig.savefig(RESULT/'plots/strong_control_comparison.png',dpi=160);plt.close(fig)
    print('Wrote 12 cycle reports and a comparison figure including stronger controls.')


if __name__=='__main__': main()
