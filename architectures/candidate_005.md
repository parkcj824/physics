# C005 — Material-change event generators

**Origin/object:** an event in material-parameter space, with state response carried between events.

## Inputs and rule

Dimensionless event amounts `delta c_1,delta c_2∈R`, response `T∈R^N`, and symmetric potentially learned generators `G_1,G_2∈R^(N×N)`. Tests use fixed normalized vectors `a∝sin(pi x)`, `b∝sin(pi x)+0.8 sin(2pi x)` with `G_1=-aaᵀ`, `G_2=-bbᵀ`. An event applies

\[
E_j(\delta c_j)T=\exp(\delta c_j G_j)T.
\]

Both ordered sequences aim at the same final material, `c_1=c_2=log 2`, whose physical interpretation is doubling conductivity in each of two halves. Starting from the unit-conductivity heat solution, compare `E_2 E_1 T` with `E_1 E_2 T` and the exact uniform-`k=2` solution. A closed loop tests `E_2^-1 E_1^-1 E_2 E_1 T` against the starting state.

## Mathematics

Each positive event is contractive for these negative-semidefinite generators. Negative edits are expansive. The commutator `[G_1,G_2]` is generally nonzero, producing order dependence at second order. Individual symmetry does not make products symmetric. Endpoint consistency is required for this memoryless, single-valued steady linear heat problem; the candidate violates it. This criticism does not apply to physically hysteretic materials or true time-ordered dynamics.

Commuting generators would remove this one defect but would not establish correct material response, reciprocity or expressive heterogeneity. No such repaired architecture was trained. Flatness/integrability of a parameter-space connection is an open object-level question, not an invention result here.

## Difference, cost, novelty

These are parameter events rather than source averaging or local messages. They overlap matrix exponential flows/Lie-product representations. Dense exponentials cost `O(N³)` and memory `O(N²)`; the rank-one special case can apply each event in `O(N)`. A future low-rank `r` generator would cost at least `O(Nr)` per application after setup. No general geometry compatibility is established.

**Verdict:** REJECT for steady elliptic solution operators because the direct endpoint/loop counterexamples are structural. Cycle 5 saves both paths and the loop.
