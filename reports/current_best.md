# Current best — no invention selected

There is **no current architecture eligible for an invention claim**. Seven distinct primitive hypotheses and explicit revisions were examined over twelve cycles. Useful mathematical structure and successful numerical controls do not establish originality or generic CAE superiority.

The most accurate *learned control* in this small fixed-mesh uniform-material family is the classical fitted compliance map `T=(Q/k) C_hat`. With 32 or128 labels it reconstructs the full-rank linear response to approximately floating-point precision. It requires the known exact `1/k` scaling, is mesh-specific, and is not a proposed generic architecture.

The most compact *trained structural probe* is C006:

\[
T=V\operatorname{diag}\left(\sum_{j=1}^{6} e^{\alpha_j}\exp[-e^{\beta_j}\lambda]\right)V^Tb.
\]

It has twelve trainable scalars, verified reciprocal positive-work structure, and mean IID relative error `0.0026435` across three seeds. Its low-conductivity error is `0.80424`; high-frequency error is `0.09862`. It receives a full physical eigensystem and is slower than a direct dense solve in the tested uncached 2D examples. The expression is a learned positive exponential-sum inverse approximation; it is not established as new. It remains a numerical probe only.

The exact C001R resistance kernel, C003 Galerkin model, C006R shifted solve, backward Euler transport and Fourier Stokes projection are explicitly classical controls. Do not present their successful cases as architecture invention.

Independent challenger C007 was rejected, not silently dropped. The next challenger must change representation and pass source/BC/gauge checks before training. See `../research/next_steps.md` for the exact resume order, limits and unrun hypothesis directions.
