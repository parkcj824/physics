# C007 — Source-contrast response

**Origin/object:** neutral excitation pairs, explored after response-age methods failed generality/novelty tests. It is not a renamed C006.

## Inputs and implemented operation

Dimensionless `x_i`, signed source `q_i`, positive conductivity effective value `k_eff`, potentially learnable positive length `ell` (test value 0.2). Define symmetric off-diagonal weights and source-pair transfer

\[
w_{ij}=e^{-|x_i-x_j|/\ell},\quad w_{ii}=0,\qquad
T_i={\ell^2\over k_{eff}}\sum_j w_{ij}(q_i-q_j).
\]

With dimensional coordinates, `ell²/k_eff` restores source-to-temperature units. This hypothesis asks whether invariance of potential to an additive constant can be expressed as excitation-contrast computation.

## Analysis and counterexample

The resulting source-space Laplacian is symmetric PSD and annihilates constant sources. But a nonzero uniform source is physically meaningful under Dirichlet conditions: its exact response is a nonzero parabola. A localized nonnegative source also produces negative predicted values at other nodes because the off-diagonal response coefficients have the wrong sign for the grounded diffusion inverse. This violates the maximum principle for the tested 1D discretization.

The mistake is assigning *potential* gauge symmetry to the *source*. Even in a compatible Neumann problem, neutral sources require an inverse response, not a Laplacian applied directly to sources. Adding a diagonal ground term does not repair negative off-diagonal inverse coefficients.

## Prior art, complexity and domains

The rule is a graph/kernel Laplacian on sources; it is not distinct from established pairwise difference aggregation. Dense time and memory `O(N²)`; local truncation `O(N k_nb)` but changes the tested response. The source contrast object is independently motivated, yet its present rule neither supplies a general solution operator nor novel math. Geometric and resolution generalization are unproven, and learning the length cannot remove the constant-source nullspace.

**Verdict:** REJECT structurally; preserve uniform, localized-source and material counterexamples. Cycle 12 exhausts the session's 12-cycle budget. Do not rename or continue tuning this same source-Laplacian form.
