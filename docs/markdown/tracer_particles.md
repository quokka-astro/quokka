# Monte Carlo tracer particles

Quokka’s default tracer mode follows the *Monte Carlo tracer* scheme introduced by Nelson, Vogelsberger, Genel, Sijacki, Springel & Hernquist (2013, arXiv:1305.2913; documented locally as `stt1383.md`). Instead of pushing tracer particles with the interpolated velocity field, each tracer is attached to the cell that currently owns it. Whenever the hydro solver exchanges mass across a face, we draw a Bernoulli trial that decides whether each tracer leaves with that flux. Over many tracers, this samples the Eulerian mass fluxes and reproduces the cell-by-cell mass distribution by construction.

## Algorithm overview

1. During the RK stage we already have face-centered conservative fluxes. For every cell we keep a “reduced mass” bookkeeping variable—initially the cell mass—and visit each outgoing face in the same order used for the solver.
2. From the face flux we derive the mass that actually leaves during the time step and divide it by the current reduced mass. That ratio is the probability `p_{i,j}^{flux}` given in the paper.
3. Each tracer that belongs to the cell draws a uniform random number. If `x < p_{i,j}^{flux}` the tracer is moved to the neighbouring cell. The reduced mass is then decreased by the amount that just left so the next face sees the updated denominator.
4. After the Monte Carlo decisions, the tracer container redistributes so particles migrate to their new AMR boxes and any invalid tracers (boundary losses) are removed.

This matches the published scheme:

> $$p_{i,j}^{\text{flux}} = \frac{\Delta M_{i,j}}{\widetilde{M}_i}, \qquad \widetilde{M}_i = \widetilde{M}_i - \Delta M_{i,j}$$

### Properties

- **Mass unbiased:** Because the probabilities are derived from the conservative fluxes, the ensemble of tracers reproduces the Eulerian mass distribution.
- **Diffusive noise:** Every tracer performs a random walk relative to the bulk flow (variance ≈ mean number of exchanges). Using more tracers per cell reduces sampling noise but not this intrinsic diffusion.
- **Low coupling cost:** The algorithm only needs face fluxes and cell masses, so it integrates cleanly with Quokka’s existing hyperbolic update.
- **Host-side particle advection:** The actual particle advection (movement) is performed on the host (CPU). While the probabilities are computed on the device (GPU), the individual stochastic decisions for each particle and their position updates occur on the CPU. This can be a performance consideration for very large numbers of tracer particles in GPU-accelerated simulations.

## Caveats and current limitations

- **Mesh refinement/derefinement:** The open issue is specifically grid creation/destruction. During *normal* coarse–fine exchanges the tracer probabilities already line up with the conservative flux registers, so coarse/fine interfaces behave the same as uniform-grid faces. However, when the AMR hierarchy is rebuilt (refine or derefine), we currently let AMReX move tracers solely by spatial position via `TracerPC->Redistribute`. We do **not** yet apply the probabilistic rules from Nelson et al. that apportion tracers to new daughter cells (or drain tracers from a cell that will be removed) in proportion to the associated mass fluxes. Consequently, tracer counts can momentarily decorrelate from mass **only** at regrid events. (Planned for a future PR.)
- **Other particle sinks/sources:** The Monte Carlo transfers described here only cover hydro fluxes. Coupling to star formation, sink particles, etc., still uses the traditional particle register paths.

If you need exact mass-tracking across AMR transitions today, you’ll have to treat those transitions manually (e.g., via custom hooks) until the probabilistic refinement logic is implemented.
