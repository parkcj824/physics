# C003 — Admissible action span

**Origin/object:** a global admissible field variation, not a node or pair message.

## Inputs and rule

After boundary lifting, take SPD `A∈R^(N×N)`, load `b∈R^N`, and full-column-rank trial matrix `Phi∈R^(N×r)`. The matrix could be trained; falsification uses fixed `Phi_ij=sin(pi*j*x_i)`, `j=1..4`. Solve the explicit small computation

\[
H=\Phi^T A\Phi,\quad z=H^{-1}\Phi^T b,\quad T=\Phi z.
\]

It is the stationary point of `E(Phi z)=zᵀHz/2-zᵀPhiᵀb`. Energy dimensions are consistent under the common dimensionless scaling.

## Mathematics and limitations

`H` is SPD under the stated rank condition, yielding stability in the energy norm and Galerkin orthogonality `Phiᵀ(AT-b)=0`. It does not give full local conservation, positivity, or strong residual control. If `Phiᵀb=0`, prediction is zero even if the true solution is nonzero. A mode-8 source directly tests this counterexample. Changing meshes requires evaluating a continuous basis and consistent quadrature; this is possible but no theorem covers learned arbitrary geometries here.

The response is symmetric PSD; rank is at most `r`, so full-rank inverse expressivity is impossible for `r<N`. Near-rank-deficient learned bases create conditioning problems.

## Existing methods and cost

This is exactly Ritz/Galerkin or a reduced-basis model. Learning `Phi` would not by itself create a distinct primitive. It differs from attention by solving a variational stationarity system, but overlaps established model reduction and operator learning.

Dense assembly `O(N²r+Nr²)`, reduced solve `O(r³)`, per-source cached response `O(Nr+r²)`; stored basis `O(Nr)`. With sparse `A`, assembly can be `O(N k_nb r+Nr²)`. These setup costs are not free. Heat and projected Stokes share the algebra without implying the same weights work everywhere.

**Verdict:** REJECT as an independent invention; useful reduced-order control. Cycle 3 measures a complete failure on an orthogonal source and partial error for localized/heterogeneous cases.
