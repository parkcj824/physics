# C002 — Opposite exchange budgets

**Origin/object:** conservation of internal exchange between physical freedoms.

## Inputs and rule

`T∈R^N`, source `b∈R^N`, fixed reference `A=BᵀCB` including grounded boundary edges, incidence `B`, positive conductance `C`; a potentially learned positive scalar `tau`. In dimensionless storage units,

\[
f=C BT,\qquad T^+=T+\tau(b-B^T f)=T+\tau(b-AT).
\]

Opposite signs in `Bᵀ` cancel transfers on internal edges. Boundary and source terms do not cancel. The experiment begins at zero and evaluates a finite number of steps as an attempted steady operator.

## Mathematics

The error obeys `e^+=(I-tau A)e`. For SPD `A`, convergence requires `0<tau<2/lambda_max`; positivity requires the stronger rowwise condition for a grounded M-matrix (`tau≤1/max A_ii` suffices). On closed compatible Neumann systems, the gauge mode must be removed; this update does not determine it. Internal conservation does not imply steady energy balance after 32 updates.

The initially fixed `tau=0.1` fails under mesh refinement. Revision C002R uses `tau=1/lambda_max`, ensuring contraction on positive modes but slowing low-frequency convergence. This is a materially changed step constraint, not a new candidate name hiding the failure.

## Prior art and cost

Expanding the rule yields Richardson relaxation (and diffusion-like message passing), with no independent neural calculation principle. Sparse work per step is `O(N k_nb d)` and storage `O(N k_nb+Nd)`; `s` steps cost `O(s N k_nb d)`. The pilot dense implementation costs `O(sN²)`. Stable work to reduce the slowest grid-scale error grows like the condition number, generally `O(h^-2)` for fixed domain Poisson. Computing a full eigenvalue bound densely is `O(N³)` here; practical row bounds would avoid it.

It extends to diffusion-type operators but does not automatically handle directed advection, constraints, or momentum coupling. Tuning one scalar does not make this architecture novel.

**Verdict:** REJECT as an independent invention; retain unstable/stable controls for detecting the conservation-versus-convergence confusion. Cycle 2 executes `N=7,15,31`.
