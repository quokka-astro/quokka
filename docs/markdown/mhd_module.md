# Magnetohydrodynamics Module

The magnetohydrodynamics (MHD) module augments Quokka's Euler solver with
magnetic fields while keeping the discrete divergence of $\vec{B}$ exactly
zero. It is built on the same method-of-lines framework described in
[Equations](equations.md), but introduces face- and edge-centred fields to
support constrained transport.

## Finite-volume discretization

- Cell-centred conserved variables ($\rho$, $\rho\vec{v}$, $E_{\text{gas}}$ and
  optional scalars) live in the same `MultiFab` as the hydrodynamics module.
  Magnetic fluxes $B_1$, $B_2$, $B_3$ are stored on face-centred `MultiFab`s so
  that each component is collocated with its normal face area.
- Edge-centred electromotive forces (EMFs) are defined on a third staggered grid
  and assembled on demand. This staggered arrangement lets the induction update
  follow Stokes' theorem exactly, ensuring $\nabla \cdot \vec{B}=0$ to machine
  precision on uniform grids.
- Source terms (cooling, gravity, etc.) act on the cell-centred state. Since only
  ideal MHD is supported, there are no source terms in the magnetic induction
  equation.

## Reconstruction and Riemann solves

Each sweep through `computeHydroFluxes` performs the following steps for every
coordinate direction:

1. Convert conserved variables (plus face-centred magnetic fluxes) to primitive
   variables with `HydroSystem::ConservedToPrimitive`.
2. Build shock-flattening coefficients that detect strong compression and
   locally limit the reconstruction stencil to maintain monotonicity.
3. Reconstruct left/right interface states using the requested order:
   donor-cell (1), piecewise linear with a minmod limiter (2), PPM (3) or
   extrema-preserving PPM (5, the default). The transverse magnetic components
   are reconstructed alongside the fluid primitives so that the Riemann solver
   sees a consistent state.
4. Solve the one-dimensional Riemann problem with the HLLD solver. The normal
   magnetic field is taken directly from the face-centred data so that the
   solution respects the jump condition $B_n^- = B_n^+$. The solver returns both
   fluxes for the conserved state and a face-centred velocity that is reused by
   particle advection and, optionally, by the EMF calculation.

If the high-order solve produces negative densities, the code marks the
affected cells and later recomputes their update with a first-order upwind flux
(see *Fallbacks* below). Negative pressures are corrected separately by the
dual-energy synchronisation step.

## Edge-centred electromotive forces

`MHDSystem::ComputeEMF` constructs either cell-centered or edge-centred EMFs $\mathcal{E} = -\vec{v}\times\vec{B}$ by
combining face-centred magnetic fields with one of the three schemes selected by
`emf_computinging_scheme`: 

- `FelkerStone2017` – reconstructed velocities from cell-centered states to edges.
- `Balsara2025` – average of face-centered magnetic fields with cell-centered velocities, with subsequent EMF calculation at cell-center then reconstructed EMF to edges
- `Quokka2026` - face-centered velocity returned by the Riemann solver reconstructed to edges 

The stencil used in this reconstruction is controlled by `emf_reconstruction_order`
with the same order options as the flux reconstruction. After reconstruction,
Quokka averages the four quadrant-centred EMFs surrounding each edge using one of
two formulas selected by `emf_averaging_scheme`:

- `LondrilloDelZanna2004` – the Londrillo & Del Zanna (2004) upwind constrained-transport
  formula, which weights the quadrants using characteristic MHD signal speeds.
During the LondrilloDelZanna2004 average, the code leverages the fast magnetosonic speeds
computed at interfaces during the Riemann solve so that the EMF averaging is properly upwinded
without requiring a full state reconstruction and eigenvalue computation.

## Constrained-transport update

Once EMFs are available, `MHDSystem::SolveInductionEqn` updates the face-centred
magnetic fluxes using the area-integrated form of Faraday's law. For a face with
normal direction $w_0$ and transverse directions $w_1$, $w_2$ the update reads

$$
B_{w_0}^{n+1} = B_{w_0}^n + \frac{\Delta t}{\Delta A_{w_0}}
\left[ \mathcal{E}_{w_2}(i+\tfrac{1}{2},j+1,k) - \mathcal{E}_{w_2}(i+\tfrac{1}{2},j,k)
     - \mathcal{E}_{w_1}(i+1,j+\tfrac{1}{2},k) + \mathcal{E}_{w_1}(i,j+\tfrac{1}{2},k) \right].
$$

Because the EMFs on opposite edges are equal and opposite, the discrete
divergence, computed as a finite difference of the face-centred fluxes, remains
unchanged by the update.

## Time integration

The MHD module uses the same second-order strong stability preserving Runge–Kutta
(SSPRK2) integrator that advances the hydrodynamics:

1. **Predictor stage:** Evaluate numerical fluxes (and EMFs) from the state at
   $t^n$ and form a provisional update using `HydroSystem::PredictStep`.
2. **Corrector stage:** Recompute fluxes/EMFs from the predictor state, average
   them with the first-stage fluxes, and apply `HydroSystem::PredictStep` a
   second time to obtain $U^{n+1}$.

Strang-split source terms (chemistry, cooling, gravity) advance by half steps
before and after the hydrodynamic RK stages. Radiation transport advances in a
separate, fully operator-split solve after the hydrodynamic update.

## Fallbacks and positivity control

- Cells flagged during reconstruction because the high-order state produced a
  negative density are re-updated with a first-order local Lax–Friedrichs flux
  and EMFs reconstructed with donor-cell stencils. This "first-order flux
  correction" preserves robustness while keeping the divergence constraint. (The
  dual-energy sync step is responsible for repairing negative pressures.)
- Density and temperature floors are enforced after every RK stage.
- When the dual-energy formalism is enabled, the auxiliary internal energy is
  synchronised with the magnetized total energy to avoid large truncation errors
  in the internal energy in high Mach number regions.

## Runtime controls

The following input parameters tune the MHD discretization and are documented in
more detail in [Runtime parameters](parameters.md):

- `reconstruction_order` – spatial order for hydro flux reconstruction (default
  3 = PPM).
- `emf_reconstruction_order` – spatial order for the EMF reconstruction
  (default 5 = extrema-preserving PPM).
- `emf_compute_scheme` – choose `FelkerStone2017`, `Balsara2025`, or `Quokka2026` for computing the emf either at cell-center or at the edge (default `Balsara2025`).
- `emf_averaging_scheme` – currently only `LondrilloDelZanna2004` for edge averaging
  (default `LondrilloDelZanna2004`).
- `artificial_viscosity_k` – optional scalar viscosity coefficient that adds a
  diffusive flux to the momentum equations and can damp post-shock oscillations.
- `quokka.bc` – (required) choose `periodic` or `reflecting` for the boundary conditions. For reflecting boundaries, we use `amrex::BCType::reflect_even` for all magnetic field components. Support for properly reflecting magnetic field boundaries will be added in the future.
