# ADR-0003: Comoving Cosmology and Where the Scale Factor Enters
Date: 2026-10-27 • Status: Proposed

Quokka evolves cosmological runs in comoving variables. The scale factor $`a(t)`$ is a function of time, never mutable state, and every factor of $`a`$ lives in a small, fixed set of hooks. No physics kernel contains a $`1/a`$.

## Context
The `mfulghieri/test-cosmology` branch ([#2340](https://github.com/quokka-astro/quokka/pull/2340)) adds comoving hydro, self-gravity and CIC dark matter. It works on a single level and has two test problems (`CosmologicalExpansion`, `CosmoSphereDM`). Its structure blocks the next steps:

- **Mutable scale factor.** `a_now_` is a member advanced inside the per-level Strang source operator. Its meaning changes mid-step ($`a(t)`$, then $`a(t+\Delta t/2)`$, then $`a(t+\Delta t)`$). This forces `amr.max_level = 0`, forbids restart, needs save/restore on hydro retries, and needs five extra particle hooks on `AMRSimulation`.
- **Scattered factors of $`a`$.** $`1/a`$ is inserted separately in both flux functions, the signal speed, the CFL check, the $`P\,dV`$ term, tracer advection, the gravity source, and the particle kick, drift and CFL. Each new operator must find its own places.
- **Ideal gas only.** The expansion operator hard-codes $`e \propto a^{-3(\gamma-1)}`$, and hydro EOS calls receive comoving density.
- **No MHD.** Magnetic energy is dropped when total energy is recomputed, and the induction equation has no comoving form.

Touches:
- **ParmParse:** new `cosmology.*` runtime parameters; cosmological numbers move out of `Physics_Traits`.
- **Output format:** no change. Cosmological state is added to `metadata.yaml` as diagnostics.
- **Problem-file interface:** `Physics_Traits<problem_t>::is_cosmology_enabled`; initial conditions are specified in comoving variables (including $`B_c`$, below).

Out of scope: radiation. Comoving radiation transport ($`c/a`$, photon redshift) is not covered here and stays blocked by the existing `static_assert`.

## Options
- **Option A: $`1/a`$ at each call site (current branch).** Works on one level, but every new flux path or operator must find its own factors, and the mutable `a_now_` blocks AMR, restart and retries.
- **Option B: Supercomoving coordinates** (Martel & Shapiro 1998). These remove $`a`$ from the equations only for $`\gamma = 5/3`$. Quokka supports a general EOS.
- **Option C: Comoving variables with a fixed set of hooks** (this ADR), storing the magnetic field as $`\tilde B = a^{3/2} B_\mathrm{phys}`$ with a $`-\tfrac12 H \tilde B`$ expansion term. This is the Li, Li & Cen (2008) form. It breaks $`\nabla\cdot B = 0`$ under AMR subcycling (see [Why a factor on B breaks ∇·B](#why-a-factor-on-b-breaks-b-at-sync)).
- **Option D: As C, but storing the flux-frozen field $`B_c = a^2 B_\mathrm{phys}`$** (Collins et al. 2010) and supplying $`\tilde B`$ through a view to every consumer that expects magnetic pressure in comoving energy units.

## Decision
Choose Option D.

### Principle
Use comoving coordinates $`x`$, peculiar velocity $`v`$, and comoving conserved variables. Every hook follows from one principle:

> **At fixed $`a`$, the comoving equations are the physical equations in different units.** Densities and energy densities scale by $`a^3`$, cell size by $`a`$ ($`dx_\mathrm{phys} = a\,dx`$), and per-unit-mass quantities ($`v`$, $`e`$, $`T`$, $`c_s`$, abundances) are unchanged. The only terms that are not a unit change are the $`\dot a/a`$ expansion terms.

Consequences of that principle:

1. **$`a(t)`$ is a function.** The cosmology module provides `scaleFactorAt(t)` and `dtau(t, dt)` $`= \int dt'/a(t')`$, evaluated from a table built from the cosmological parameters alone, or directly as $`a_\mathrm{init}`$ and $`dt/a_\mathrm{init}`$ when $`H_0 = 0`$ (see [Restart and reproducibility](#restart-and-reproducibility)). Every caller asks for $`a`$ at the time it needs. AMR subcycling, retries and restart need no special handling (`tNew_` is already checkpointed).
2. **Transport runs on the clock $`\tau`$.** Every term that moves material through comoving space carries the same $`1/a`$, so it is advanced with $`d\tau`$ instead of $`dt`$.
3. **Expansion is one Strang-split operator** on the physical $`dt`$, and the only code that knows about $`\dot a/a`$.
4. **Local and diffusive physics run unmodified** inside a physical-frame wrapper. Diffusive terms that live inside the transport step (viscosity, resistivity) use physical-frame gradients and coefficients, with their returned fluxes converted to the transport frame.

### The hooks
These seven hooks are the complete list of places where $`a`$ appears outside `src/cosmology/`.

| Hook | Location | Covers |
|---|---|---|
| `dtau` | `advanceHydroAtLevel` (PredictStep, AddFluxesRK2, `SolveInductionEqn`, flux and EMF registers, tracers, magnetic view and EMF conversion, `PhysicalDx` and coefficient/flux conversions for viscous and resistive terms); `applyPoissonGravityAtLevel`; particle kick and drift | Flux divergence, $`P\,dV`$, induction, viscosity, resistivity, gravity on gas and particles, particle drift, tracers |
| Timestep conversion: `dt_cfl = a * dtau_cfl` for hyperbolic transport; physical diffusive limits | `computeTimestepAtLevel`; `computeMaxSignalLocal` | Hydro, MHD and particle CFL; viscous and resistive stability |
| Expansion operator | Strang source chain; particle velocity drag; expansion-frame adapter for ghost filling | All $`\dot a/a`$ terms, including general EOS; coarse and boundary states mapped to the consuming stage's expansion frame ($`B_c`$ has no expansion term) |
| `inPhysicalFrame(op)`, plus $`dx \cdot a`$ for diffusion and conversion of returned integrated fluxes | Strang source chain and its flux-register contributions | Cooling, chemistry, photoelectric heating, electron conduction (state and reflux), future Strang-split local or diffusive physics |
| $`G/a`$ and $`\bar\rho`$ | `fillPoissonRhsAtLevel`; particle `depositMass`; $`\bar\rho`$ from the level-0 composite field after average-down | Comoving Poisson equation |
| EOS view (`a3`) | EOS calls in hydro | Non-ideal EOS |
| Magnetic view ($`B_c/\sqrt a`$) | Transport, CFL estimation, initialization, state fixups and diagnostics that consume magnetic energy | Consistent magnetic pressure, energy and wavespeeds at each consumer's evaluation time |

In `advanceHydroAtLevel`, `dtau_lev` is used only for the update itself. Time labels (`fillBoundaryConditions(time + dt_lev)`, the Strang source calls, `tNew_`) stay on the physical `dt_lev`. A physical time label does not identify the expansion frame of a split-stage state: ghost filling must also receive the explicit stage frame described below. Any hyperbolic term later added to the hydro RHS inherits the correct factor automatically; diffusive terms need the treatment below.

Hydro and particles use the same $`d\tau`$ for each step, which keeps gas and dark matter moving together (the property `CosmoSphereDM` tests). The particle drift uses $`d\tau`$ over the whole step; each half-kick uses $`d\tau`$ over its own half-interval, $`\tau(t+\Delta t/2) - \tau(t)`$ and $`\tau(t+\Delta t) - \tau(t+\Delta t/2)`$. The midpoint value $`\Delta t/a(t+\Delta t/2)`$ is second-order; the exact integral is preferred.

### Expansion-aware ghost filling
Both RK transport stages use the frozen expansion frame $`a_\star = a(t_\mathrm{lev}+\Delta t_\mathrm{lev}/2)`$, even though their physical time labels differ. The first stage's interior has already undergone the first expansion half-step. `fillBoundaryConditions` currently obtains coarse endpoints through `GetData` at the requested physical time; copying those endpoints unchanged would leave a uniform gas's ghosts warmer than its interior. `PreInterpState`/`PostInterpState` only change the energy representation for interpolation and do not apply expansion.

The expansion hook must provide a ghost-fill adapter carrying the requested physical time and the consuming stage's expansion frame separately:

- Copy each coarse endpoint into a temporary and map it from its own expansion frame to $`a_\star`$ **before time interpolation**. Use the expansion operator (or its inverse when mapping backward): keep comoving density and $`B_c`$ fixed, scale momentum, evolve internal energy with the EOS ODE, and reconstruct total energy with the magnetic view at $`a_\star`$. Then interpolate in physical time and prolong spatially with all source data in that common frame. For ideal-gas homogeneous expansion, the mapped endpoints are identical, so interpolation introduces no pressure gradient. Mapping an already time-interpolated raw state does not have this property.
- Fine stage data and same-level/periodic copies already carry the target frame; do not expand them again. Preserve stored coarse endpoints and face fields. Energy interpolation and fixups must use the target-frame EOS and magnetic views.
- Evaluate prescribed physical-boundary data at the requested physical time, convert physical units to comoving units at that time if necessary, then map the result into the consuming stage's expansion frame. Extrapolating and reflecting boundaries act directly on stage-frame interior data. Carry this distinction through every boundary callback, including corners, so no boundary fill overwrites converted ghosts with endpoint-frame values.
- A physical-frame source such as conduction uses the same adapter with its source-stage expansion frame, then converts all valid and ghost data to physical units at that frame. The stage descriptor must follow the actual source ordering; the source call's physical time argument alone is insufficient.

This adapter belongs to the expansion hook, so it does not add scale-factor logic to interpolation kernels. Keep the AMR guard until a refined homogeneous-expansion test verifies stage ghost/interior pressure agreement and absence of interface-generated velocity, including physical boundaries and subcycling.

### Strang-split conduction fluxes
The physical-frame wrapper must convert returned fluxes as well as state and spacing. `addStrangSplitSourcesWithBuiltin` currently adds each physical `heat_flux` to `recal_fluxes` with weight `0.5`; `advanceHydroAtLevel` combines this with hydro fluxes and deposits using the full step. Replacing that last multiplier with `dtau_lev` alone gives the wrong conduction reflux.

For a conduction half-step of physical duration $`\Delta t_s`$ evaluated at frozen source-frame scale factor $`a_s`$, the integrated comoving energy flux required by the register is

```math
I_s = a_s^2\,q_{\mathrm{phys},s}\,\Delta t_s
    = a_s^3\,q_{\mathrm{phys},s}\,\Delta\tau_s
\quad\text{at fixed }a_s,\qquad \Delta\tau_s=\Delta t_s/a_s.
```

Here $`q_{\mathrm{phys},s}`$ is the returned `heat_flux`; the $`a_s^2`$ follows from converting the energy update by $`a_s^3`$ and using physical spacing $`a_s dx`$. In an expanding run, retain the quadrature used by the physical-time source update: accumulate $`a_s^2 q_{\mathrm{phys},s}\Delta t_s`$ for each source evaluation, rather than substituting the hydro step's $`d\tau`$.

Before combining with hydro register contributions, multiply each $`I_s`$ by its own remaining expansion weight $`W_s`$ from the point where that source increment enters the split state to $`t_\mathrm{sync}`$. Determine $`W_s`$ from the actual forward/reverse source ordering and all later fine substeps; do not assign both half-steps the hydro midpoint weight. For ideal-gas thermal increments, $`W_s`$ is the product of subsequent internal-energy expansion factors, $`\prod (a_\mathrm{before}/a_\mathrm{after})^{3(\gamma-1)}`$, and applies to the conduction contribution in both internal and total energy. A general EOS requires the response of the remaining expansion ODE to that thermal increment; a single exponent is not exact. Any approximate energy weighting must be documented and validated under the energy-reflux accuracy limitation below.

Keep the two half-step contributions separate until unit conversion and expansion weighting are complete. Prefer an integrated-flux accumulator $`\sum_s W_s I_s`$, with no further time multiplier; if retaining the full-step `dtau_lev` deposit API, add $`\sum_s W_s I_s/dtau_lev`$ to its flux array and apply no second expansion weight. At frozen $`a`$, this reduces to the existing `0.5` per half-step times $`a^3 q_\mathrm{phys}`$. Commit these contributions only after a successful hydro/source step, as with the current retry handling.

### Diffusive terms inside the transport step
Physical viscosity is computed inside `HydroSystem::ComputeFluxes`, and Ohmic resistivity inside the EMF computation and `MHDSystem::AddResistiveEnergyFlux`. Only electron conduction is Strang-split. A diffusive term needs $`1/a^2`$, but running on $`d\tau`$ supplies only one factor, so the transport hook also:

- computes gradients inside diffusive fluxes with $`dx_\mathrm{phys} = a(t+\Delta t/2)\,dx`$, while the flux divergence keeps the comoving $`dx`$;
- evaluates coefficients from physical state: dynamic shear and bulk viscosities have $`\mu_c = a^3\mu_\mathrm{phys}`$; resistivity $`\eta`$ is a diffusivity and is unchanged;
- converts the physical viscous momentum and energy fluxes back by $`a^3`$ before adding them to the comoving fluxes in `ComputeFluxes`. Equivalently, supply $`a^3\mu_\mathrm{phys}`$ (for both shear and bulk viscosity) to `ComputeViscousFlux` with physical velocity gradients. Apply exactly one of these conversions: $`d\tau`$ supplies the coordinate/time factor, not the flux-unit conversion;
- evaluates resistive EMFs and energy fluxes using $`\tilde B`$, physical spacing and unchanged $`\eta`$. The energy flux is then already in comoving units because it is quadratic in $`\tilde B`$; multiply the EMF by $`\sqrt a`$ when returning to the stored $`B_c`$ update.

`ComovingDx` and `PhysicalDx` are distinct types, so passing the comoving spacing to a diffusive gradient does not compile.

### Timestep limits
The hyperbolic conversion $`dt_\mathrm{cfl} = a\,d\tau_\mathrm{cfl}`$ assumes physical characteristic speeds, obtained with the magnetic and EOS views at the state evaluation time. Diffusive limits must also use physical spacing and physical state/coefficient values. For constant coefficients, the existing limits become

```math
\Delta t_\eta \le C\,\frac{(a\,dx)^2}{2\eta}, \qquad
\Delta t_\mu \le C\,\frac{(a\,dx)^2\rho_\mathrm{phys}}{2\left(\tfrac43\mu_\mathrm{phys}+\zeta_\mathrm{phys}\right)}, \qquad
\rho_\mathrm{phys}=\rho_c/a^3,
```

where $`\mu`$ and $`\zeta`$ are shear and bulk dynamic viscosity and $`C`$ retains the existing CFL and dimensional restrictions. Take the minimum of the hyperbolic and diffusive limits in physical time. If `computeMaxSignalLocal` continues to combine them as effective speeds, use $`2\eta/(a\,dx)`$ and $`2(\tfrac43\mu_\mathrm{phys}+\zeta_\mathrm{phys})/(\rho_\mathrm{phys}\,a\,dx)`$ before the final multiplication by $`a`$. Do not multiply a timestep computed with the old $`2\eta/dx`$ by $`a`$: that gives $`a\,dx^2/(2\eta)`$, ten times too large at $`a=0.1`$. Apply the same physical-frame rule to any additional explicit diffusion limit; reevaluate stability for the state and scale factor used by the transport update.

### Scaling exponents
A single per-component table in `src/cosmology/` gives each conserved component two exponents:

- the **frame exponent** $`k`$, with $`q_\mathrm{comoving} = a^k q_\mathrm{phys}`$, used by the frame wrapper and the equivalence tests;
- the **dilution exponent** $`m`$, the power in the expansion factor $`(a_\mathrm{old}/a_\mathrm{new})^m`$, used by the expansion operator and the reflux weights.

A new field needs one entry with both.

| Component | $`k`$ (frame) | $`m`$ (dilution) | Expansion step |
|---|---|---|---|
| Density, partial densities, passive scalars | 3 | 0 | none ($`\rho_c`$ constant) |
| Momentum density | 3 | 1 | $`\times\, a_\mathrm{old}/a_\mathrm{new}`$ |
| Internal energy density | 3 | $`3(\gamma-1)`$ | EOS ODE (below); the power law for an ideal gas |
| Magnetic field $`B_c`$ (face-centred, stored) | 2 | 0 | none (pure curl evolution) |
| Solver-frame field $`\tilde B = B_c/\sqrt a`$ (not stored) | 3/2 | — | not stored |
| Total energy | 3 | ≈ 2 (reflux only) | recomputed: $`e_\mathrm{int} + \mathrm{KE} + \tfrac12 B_c^2/a`$ |
| Per-unit-mass quantities | 0 | 0 | none |

### Expansion operator
The expansion operator is Strang-split around the transport step and uses the physical $`dt`$. For the first half step $`a_\mathrm{old} = a(t)`$ and $`a_\mathrm{new} = a(t + \Delta t/2)`$; for the second, $`a_\mathrm{old} = a(t + \Delta t/2)`$ and $`a_\mathrm{new} = a(t + \Delta t)`$:

```math
\begin{aligned}
\text{momentum:}\quad & p \leftarrow p\,\frac{a_\mathrm{old}}{a_\mathrm{new}} && \text{(gas and particles, exact for any EOS)} \\
\text{internal:}\quad & \frac{\partial E_{\mathrm{int},c}}{\partial t} = -3H\,p_c, \qquad p_c = a^3\,p\!\left(\rho_c/a^3,\ e\right) \\
\text{as ODE:}\quad & \frac{de}{d\ln a} = -3\left(\frac{p}{\rho}\right)_\mathrm{phys}\!\left(\rho_c/a^3,\ e\right) && \text{at fixed } \rho_c \\
\text{magnetic:}\quad & B_c \text{ unchanged} \\
\text{total:}\quad & E \leftarrow E_\mathrm{int} + \tfrac12 |p|^2/\rho + \tfrac12 |B_c|^2/a_\mathrm{new}
\end{aligned}
```

For a general EOS the internal-energy ODE is solved with the existing RK integrator. For an ideal gas it reduces to the closed-form power law the branch uses today.

### Magnetic field
**Notation.** Throughout this ADR, $`B`$ is Quokka's code field $`B' = B_\mathrm{Gauss}/\sqrt{4\pi}`$, so magnetic energy is $`\tfrac12 B^2`$ and the Alfvén speed is $`B/\sqrt\rho`$. The factor $`\sqrt{4\pi}`$ is independent of $`a`$, so every exponent and equation here holds unchanged for $`B'`$. Enzo absorbs the same factor (Collins et al. 2010, Eq. 5), so their equations match term for term.

Store the flux-frozen field $`B_c = a^2 B_\mathrm{phys}`$ (Collins et al. 2010, Eq. 11). Its induction equation is a pure curl with no expansion term (their Eq. 4):

```math
\frac{\partial B_c}{\partial t} = \frac{1}{a}\,\nabla_x \times \left(v \times B_c\right)
```

- **Solver frame.** At the transport hook the solver receives $`\tilde B = B_c/\sqrt{a(t+\Delta t/2)} = a^{3/2} B_\mathrm{phys}`$, so magnetic pressure scales like gas pressure. The Alfvén and fast speeds from $`\tilde B`$ and $`\rho_c`$ equal the physical speeds, so the HLLD and wavespeed kernels can be reused with this view. The EMF is scaled back by $`\sqrt a`$ before updating $`B_c`$. Enzo does the same conversion around its solver (Collins et al. 2010, Eqs. 30–31).
- **Other magnetic consumers.** The same view is required everywhere that combines stored faces with comoving cell energies or densities. `computeMaxSignalLocal` uses $`a`$ at the time of the state being estimated. `updateInitialMagneticEnergyFromFaceField` uses $`a_\mathrm{init}`$, so it adds $`B_c^2/(2a_\mathrm{init})`$. `ApplyHydroStateFixup` (including floors and `SyncDualEnergy`) uses the scale factor of the state being repaired: the frozen midpoint frame inside transport, the endpoint after the second expansion half-step, and $`a(t_\mathrm{sync})`$ for `FixupState` after AMR synchronization. Pass that evaluation time/frame explicitly; a level number alone is insufficient. Energy diagnostics require the same view. These routines may retain their existing $`B^2/2`$ expressions only when given $`\tilde B`$; passing stored $`B_c`$ would subtract the wrong magnetic energy and can overwrite internal or total energy. Use a temporary view without rescaling stored faces; AMR face operations continue to act on $`B_c`$.
- **Transport.** The curl update runs on $`d\tau`$, like the hydro fluxes.
- **No expansion factor on $`B`$.** The expansion operator never touches $`B_c`$. A multiplicative factor preserves $`\nabla\cdot B`$ on one level, but under subcycling the two levels apply different factor histories to the field, and sync assembles coarse cells at the interface from both (next section). Collins et al. (2010, §2.4) report exactly this for the source-term form of Li et al. (2008), and "greatly improved divergence" with $`B_c`$.
- **AMR consistency.** Because $`B_c`$ evolves only through the curl, divergence-free prolongation, time interpolation and EMF reflux stay exact. Every grid sharing a face must still compute identical face EMFs (Collins et al. 2010, §2.4).
- **Initial conditions.** Problems specify $`B_c = a_\mathrm{init}^2\,B_\mathrm{Gauss}/\sqrt{4\pi}`$. This is documented next to the comoving-density convention; it is the most likely place for a missing factor, in problem setup files and in the frozen-$`a`$ test inputs.
- **Diagnostics.** Output in Gauss multiplies by $`\sqrt{4\pi}\,a^{-2}`$. This is an output conversion, not a hook, and is not part of the exponent table.

### Why a factor on B breaks ∇·B at sync
The error appears in the uncovered coarse cell $`C`$ next to the coarse-fine interface. At sync its faces come from two levels:

- **Interface face $`F_I`$:** overwritten by average-down from the fine faces in that plane (as `amrex::average_down_faces` does for faces bounding the fine region). It carries the fine history.
- **Faces $`P`$ sharing an edge with $`F_I`$, and $`C`$'s other faces:** coarse values plus the EMF-reflux correction on the shared edges. They carry the coarse history.

```text
          P: coarse + reflux
      +-----------------------●-----------+-----------+
      |                       |           |           |
      |                       |   fine    |   fine    |
      |          C            |           |           |
      |   (uncovered coarse)  F_I --------+-----------+
      |                       | (average- |           |
      |                       |   down)   |   fine    |
      |                       |           |           |
      +-----------------------●-----------+-----------+
          P: coarse + reflux
   ● = shared edges e (2D sketch: edges appear as points)
```

**Without expansion, div(C) is exactly zero.** Average-down gives $`F_I = B_I^0 - \nabla\times\langle E_f\rangle`$: interior fine edges cancel, leaving the coarse edge perimeter with $`\langle E_f\rangle = \sum_j \Delta t_f\,E_{f,j}`$. Reflux puts the same $`\langle E_f\rangle`$ on the shared edges of $`P`$. Every edge EMF enters the two faces of $`C`$ sharing it with equal and opposite sign, and the old-field terms cancel because div(C) was zero.

**A multiplicative factor $`\eta`$ on $`B`$ leaves two terms uncancelled** (schematic):

```math
\begin{aligned}
\text{old field:}\quad & \mathrm{div}(C)\,V \supset \Big(\prod_j \eta_{f,j} - \eta_c\Big)\, B_I^0\, A_I \\
\text{EMF:}\quad & \mathrm{div}(C)\,V \supset \sum_j (w_j - 1)\,\Delta t_f\, E_{f,j} + (\eta_{c,2} - 1)\,\Delta t\, E_c
\end{aligned}
```

1. **Old-field term.** $`F_I`$ is scaled by the product of the fine substep factors, the other faces by the coarse factor. The error is proportional to $`B`$ itself, not to a flux mismatch. Enzo's semi-implicit factor $`(1 - x/2)/(1 + x/2)`$, with $`x = \Delta t\,\dot a/a`$, does not telescope, so this term is $`O(x^3)\,B`$ per step (Collins et al. 2010, §2.4). Exact factors from $`a(t)`$ make it zero in exact arithmetic, leaving roundoff times $`B`$ every coarse step.
2. **EMF term.** In $`F_I`$, the substep-$`j`$ fine EMF is scaled by its later expansion factors $`w_j`$. In $`P`$, the coarse EMF is scaled by the coarse second half-step factor $`\eta_{c,2}`$, but the reflux correction is added unweighted. This is the [reflux commutation problem](#amr-subcycling-and-reflux), appearing as divergence instead of a momentum error.

Constrained transport preserves $`d(\nabla\cdot B)/dt = 0`$ and never removes divergence, so the error accumulates every coarse step and is inherited by later regrids and prolongations from $`C`$.

- **$`B_c = a^2 B_\mathrm{phys}`$ (chosen):** $`\eta = 1`$, so both terms vanish identically.
- **Factor on $`B`$ with exact $`a(t)`$ and weighted EMF registers ($`m = 1/2`$, including the coarse half-step factor):** the EMF term becomes exact and the old-field term drops to roundoff, but that roundoff grows without bound, and every copy of face $`B`$ (average-down, regrid, ghost exchange) would have to apply the factors identically.

Without cosmology, constrained-transport AMR with divergence-free prolongation and EMF reflux keeps $`\nabla\cdot B`$ at machine precision. Time-interpolated coarse data only fills ghost cells, which feed EMFs, so interpolation affects accuracy, not divergence. The error comes from sync assembling one coarse cell's faces from two levels' histories, which is exact only when the field carries no multiplicative factor.

### General EOS
An ideal-gas EOS is invariant under the unit change: $`p = (\gamma-1)\rho e`$ and $`T(e)`$ hold in both frames, so hydro on comoving variables needs no EOS hook. A general EOS needs all inputs in physical units at every EOS call.

Hydro EOS calls go through an EOS view carrying `a3`. Before calling the EOS, it divides every volume-density input by $`a^3`$: density, internal energy density, pressure, and species densities in `massScalars`. In particular, `ComputePressure` and `ComputeTgasFromEint` require both physical density and physical internal energy density, while `ComputeSoundSpeed` requires physical density and pressure. Scaling energy and species densities alongside density preserves the specific energy $`e = E_\mathrm{int}/\rho`$ and species mass fractions; converting density alone would spuriously multiply these ratios by $`a^3`$.

The view multiplies per-volume outputs (pressure, energy density) by $`a^3`$. Specific energy, temperature, sound speed, and mass fractions are unchanged by the unit conversion. Non-cosmological runs use `a3 = 1`, and an `is_scale_invariant` EOS trait removes the view entirely for the ideal gas.

Until the view exists, a `static_assert` requires a scale-invariant EOS whenever cosmology is enabled.

### AMR subcycling and reflux
Reflux does not commute with the expansion operator. In `timeStepWithSubcycling` the coarse level completes its whole step, including its second Strang half-step, before the fine substeps and `Reflux` run.

- The coarse flux is applied at the coarse midpoint, then scaled by $`(a(t+\Delta t/2)/a(t+\Delta t))^m`$.
- Each fine substep's flux is applied at its own midpoint and scaled by the expansion of every later substep.
- Reflux adds $`\sum F_f\,d\tau_f - F_c\,d\tau_c`$ unscaled at $`t_\mathrm{sync}`$.

The corrected coarse cell then matches no consistent evolution. Without gravity, transport plus expansion conserves $`a\,p_c`$, and it drifts by $`O(H\Delta t)`$ times the flux mismatch every coarse step, which creates spurious bulk momentum in a periodic box. The stored magnetic field $`B_c`$ has no expansion factor, so EMF reflux is unaffected.

```text
             w = (a(t+Δt/2)/a_sync)^m
coarse  |=================◆=================|  ┆ Reflux
fine    |=======◆=========|========◆========|  ┆
                w1                 w2           ┆
        t               t+Δt/2          t+Δt = t_sync

  w1 = (a(t+Δt/4)/a_sync)^m     w2 = (a(t+3Δt/4)/a_sync)^m
  ◆ = where a level applies its flux (the Strang midpoint of its step)
```

**Fix:** weight each transport register deposit by the expansion it will undergo before reflux (Strang-split conduction uses its own source-stage weights as specified above):

```math
\text{register} \mathrel{+}= \left(\frac{a(t_\mathrm{dep})}{a(t_\mathrm{sync})}\right)^{m} F\,d\tau, \qquad t_\mathrm{dep} = t_\mathrm{lev} + \tfrac12\,\Delta t_\mathrm{lev}
```

For the coarse level the weight equals its second half-step factor. For fine substep $`j`$ it is the product of all later factors, because $`a(t)`$ is a function and the momentum factors compose exactly. Reflux is then additive and exact for density and momentum. `FluxRegister::CrseInit` and `FineAdd` take one scalar per call, so deposit per dilution-exponent group or pre-scale a temporary copy of the fluxes, using the $`m`$ column of the exponent table. EMF registers need no weight ($`m = 0`$ for $`B_c`$).

Collins et al. (2010, Appendix D) apply the flux correction after the expansion step and describe no weighting, so Enzo appears to carry this momentum error. That is inferred from the step order; the paper does not discuss it.

Total energy has no single exact weight, since kinetic, magnetic and internal parts scale differently. Use $`m = 2`$ (exact for $`\gamma = 5/3`$ without $`B`$) and accept an $`O(H\Delta t)`$ error in the energy correction. Energy is not conserved in an expanding box (Layzer–Irvine), so consistency is the requirement; `FixupState` and dual-energy sync recompute $`E_\mathrm{int}`$.

### Restart and reproducibility
$`a(t)`$ is a pure function of the cosmological parameters and $`t`$. `WriteCheckpointFile` already stores `tNew_` and `dt_` with `precision(17)`, which round-trips doubles exactly, so a restarted run reproduces $`a`$, $`d\tau`$ and $`H`$ bitwise with no cosmology state of its own.

What would break it:
- Advancing $`a`$ step by step, or integrating from the last cached point. The result depends on the step history, and a restart starts from a different point.
- An adaptive integrator started from a moving point. Its step sequence depends on the start, not only on the target $`t`$.
- Restoring $`a`$ from `metadata.yaml`. yaml-cpp does not guarantee round-trip precision, and the stored value carries the history anyway. Metadata $`a`$, $`z`$ and $`H`$ are diagnostics only.
- A running sum of $`d\tau`$ over steps.

**Zero expansion.** For $`H_0 = 0`$, take an explicit constant-scale-factor branch: require $`a_\mathrm{init}>0`$, return `scaleFactorAt(t) = a_init`, `H(t) = 0` and `dtau(t, dt) = dt / a_init` for every interval. Do not construct or invert a table: the integrands below are singular, and a constant $`a`$ cannot parameterize elapsed time. Expansion is an identity operation (skip its ODE and energy reconstruction), reflux weights are one, and any timestep bound proportional to $`1/H`$ is inactive. This branch supports both frozen-$`a`$ and identity tests, including half-kicks, subcycling, retries and restart, without accumulating a transport clock.

For an expanding cosmology with $`H(a)>0`$ throughout the table domain, implementation is based on the two integrals

```math
t(a) = \int \frac{da}{a\,H(a)}, \qquad \tau(a) = \int \frac{da}{a^2 H(a)}
```

1. **Table from parameters only.** At startup, tabulate $`t(a)`$ and $`\tau(a)`$ on a fixed grid uniform in $`\ln a`$, with fixed-order Gauss–Legendre quadrature per interval. Nothing adaptive, nothing history-dependent; the table is rebuilt identically on restart.
2. **$`a(t)`$ by interpolation.** Cubic Hermite on the table using the exact derivative $`da/dt = a\,H(a)`$, optionally followed by a fixed number of Newton iterations on $`t(a) - t`$.
3. **$`d\tau`$ as a difference.** $`d\tau(t_1, t_2) = \tau(a(t_2)) - \tau(a(t_1))`$. Nothing accumulates, and hydro, MHD and particles see the same value.
4. **Host only.** $`a`$, $`d\tau`$ and $`H`$ are computed on the host and captured as scalars, so GPU and CPU builds and all MPI ranks agree without a broadcast.
5. **Parameter check.** The checkpoint stores the cosmological parameters and a hash of the table, or a constant-scale-factor mode marker when $`H_0=0`$ (no table). Restart checks the mode and parameters, including $`a_\mathrm{init}`$, and the table hash when present; it aborts with an error on a mismatch.

Bitwise reproducibility holds for the same binary on the same machine. A different compiler, FMA contraction or libm can change the last bits of $`H(a)`$, as it can for the hydro update. The time sequence also has to match; it does, because restart restores `dt_` and the cosmological timestep limit depends only on $`H(a(t))`$.

### Rules for new physics
1. **Every operator declares its frame.** A transport operator runs inside the $`d\tau`$ step. A Strang-split local or diffusive operator is wrapped in `inPhysicalFrame`, including conversion and source-stage expansion weighting of any returned flux-register contributions. Ghost fills carry the consuming stage's expansion frame separately from physical time. A diffusive term inside the transport step uses `PhysicalDx`, coefficients evaluated in the physical frame and fluxes converted back to transport units; its stability limit follows the same frame convention. Magnetic energy and wavespeed consumers use the magnetic view at their evaluation time. Only the expansion operator acts directly on comoving state. An undeclared operator fails to compile.
2. **No $`a`$ inside kernels.** Outside `src/cosmology/`, the scale factor appears only in the seven hooks. The type system and the AST check enforce this.
3. **New fields get exponents.** Add one entry, with $`k`$ and $`m`$, to the exponent table; the wrapper, expansion operator, reflux weights and tests pick it up.

### Enforcement
Three layers, from hardest to softest: types make misuse a compile error, an AST check catches what types cannot express, and the equivalence tests check the numbers.

**1. Strong types (compile-time gate).** The scale factor is a type whose raw value only the hooks can read, through a passkey:

```cpp
namespace quokka::cosmology {
class ScaleFactor {
      public:
	auto value(HookKey /*key*/) const -> amrex::Real { return a_; } // host-only, needs a key
      private:
	amrex::Real a_;
	explicit ScaleFactor(amrex::Real a) : a_(a) {}
	friend auto scaleFactorAt(amrex::Real t) -> ScaleFactor;
};
class HookKey { // passkey: only the hooks can construct one
	HookKey() = default;
	friend class TransportStep;   // dtau, dt_cfl
	friend class ExpansionOperator;
	friend class PhysicalFrame;   // inPhysicalFrame, dx * a
	friend class ComovingPoisson; // G/a, rho_bar
	friend class EOSView;         // a3, private inside the view
	friend class MagneticView;    // B_c / sqrt(a), at the consumer's evaluation time
};
} // namespace quokka::cosmology
```

- `ScaleFactor` has no implicit conversion to `Real`, so a use outside a hook fails on every build. `value()` is host-only, so a kernel that reaches it also fails under CUDA and HIP.
- Hooks hand out only derived quantities (`dtau`, `inv_a`, `sqrt_a`, `a3`, `PhysicalDx`), never $`a`$ itself.
- `TransportDt` and `PhysicalDt` are distinct types, so `fillBoundaryConditions(time + dtau)` does not compile. Likewise `ComovingDx` and `PhysicalDx` keep diffusive gradients on the physical spacing.
- `callInOrder` accepts only `Transport{op}`, `PhysicalFrame{op}` or `Comoving{op}`. A `static_assert` rejects a bare lambda, and only the expansion operator can construct `Comoving`.
- The `HookKey` friend list is the allowlist. `src/cosmology/` is a CODEOWNERS path, so adding a friend requires review.

**2. AST check (CI gate).** A `clang-query` matcher script runs against `compile_commands.json` in a CI job next to the existing clang-tidy workflow. It exits non-zero with file:line on any of:
1. a reference to `quokka::cosmology::*` (`HubbleFactor`, `scaleFactorAt`, `CosmologyParams` members) outside `src/cosmology/` and the allowlisted hook functions;
2. a hook-output type (`inv_a`, `sqrt_a`, `a3`) used inside a lambda body outside an allowlisted function;
3. `ScaleFactor`, `HookKey` or a hook-output type in a `HydroSystem`, `MHDSystem` or operator signature.

AST matching is deterministic but needs a configured build, so it is too slow for `quokka-pre-commit.sh`. A grep for the same symbols outside allowed paths can run there as an early warning, but it is easy to evade and is not a gate.

**3. Equivalence tests (numerical check).** Linting cannot see a wrong exponent, a hard-coded literal standing in for $`a`$, or a quantity a problem file has already scaled. The frozen-$`a`$ equivalence tests cover those. Full dataflow tracking with the Clang Static Analyzer taint checker is not used: it is heavier and less predictable, and the strong types already track $`a`$ up to the one point where a hook unwraps it.

## Consequences
- **Accuracy.** Expansion source terms are exact for any EOS given $`a(t)`$; transport uses the exact $`\int dt/a`$. Momentum reflux becomes exact under subcycling; total-energy reflux carries an $`O(H\Delta t)`$ error in the correction term.
- **Performance.** A few host-side scalars per level step, one table built at startup for expanding runs, per-group scaling of flux-register deposits, and magnetic views for transport and other consumers. The cost of materializing temporary magnetic views must be measured during implementation.
- **Portability.** $`a`$, $`d\tau`$ and $`H`$ are host-only; kernels receive scalars or converted data. The expansion operator, EOS view and magnetic view must support CPU and GPU execution.
- **Maintenance.** Factors of $`a`$ are confined to `src/cosmology/` and seven hooks, enforced by types and CI. New physics declares its frame instead of locating factors by hand.
- **Problem files.** Initial conditions are comoving, including $`B_c = a_\mathrm{init}^2 B_\mathrm{Gauss}/\sqrt{4\pi}`$. `Physics_Traits` keeps only `is_cosmology_enabled`; cosmological numbers become `cosmology.*` runtime parameters.
- **Output.** No format change. Current $`a`$, $`z`$ and $`H`$ are written to `metadata.yaml` for plotfiles and checkpoints as diagnostics.

## Rollout & Testing
Migration from the current branch:

- [ ] Add the parameter-only $`a(t)`$ and $`\tau(a)`$ table with `scaleFactorAt(t)` and `dtau(t, dt)` to `src/cosmology/`, plus the table-free $`H_0=0`$ branch returning $`a_\mathrm{init}`$ and $`dt/a_\mathrm{init}`$; remove `a_now_`, `a_half_` and the retry save/restore.
- [ ] Replace the five particle cosmology virtuals on `AMRSimulation` with direct calls (or one `scaleFactorAt` virtual).
- [ ] Pass `dtau_lev` through `advanceHydroAtLevel`; remove the $`1/a`$ in both flux functions, `AddInternalEnergyPdV`, tracers, `computeMaxSignalLocal` and `isCflViolated`.
- [ ] Apply `dt_cfl = a * dtau_cfl` to hyperbolic hydro/MHD and particle limits; convert the viscous and resistive limits in `computeMaxSignalLocal` using physical spacing, state and coefficients before combining limits in physical time.
- [ ] Merge the duplicated kernels in `applyPoissonGravityAtLevel` and `fillPoissonRhsAtLevel` into one each, with host-computed scalars; compute $`\bar\rho`$ from the level-0 composite field after average-down at each solve.
- [ ] Rewrite the expansion operator for general EOS, including magnetic energy $`\tfrac12 B_c^2/a`$ in the total.
- [ ] Store $`B_c = a^2 B_\mathrm{phys}`$; supply $`\tilde B`$ through the magnetic view to transport, CFL estimation, initialization, `ApplyHydroStateFixup`/`SyncDualEnergy` (including post-AMR `FixupState`) and energy diagnostics at their evaluation times. Scale the EMF by $`\sqrt a`$ in the transport hook; keep stored faces in $`B_c`$ units.
- [ ] Add the exponent table ($`k`$ and $`m`$) and `inPhysicalFrame`; lift the cooling and chemistry guards by wrapping those operators.
- [ ] Convert each conduction half-step's returned heat flux to an integrated comoving register contribution and apply its remaining source-stage expansion weight before combining with hydro. Validate frozen-$`a`$ AMR equivalence and expanding source-stage weights before enabling conduction with cosmological AMR.
- [ ] Pass `PhysicalDx` and $`a^3\mu_\mathrm{phys}`$ (both shear and bulk) to the viscous terms in `ComputeFluxes`, or multiply their physical fluxes by $`a^3`$. Use `PhysicalDx`, $`\tilde B`$ and unchanged $`\eta`$ for the resistive EMF and `AddResistiveEnergyFlux`; then lift the viscosity guard.
- [ ] Move cosmological numbers out of `Physics_Traits` into runtime parameters; keep only `is_cosmology_enabled`.
- [ ] Write current $`a`$, $`z`$ and $`H`$ into metadata for checkpoints as well as plotfiles.
- [ ] Write the cosmological parameters and table hash (or constant-scale-factor mode marker) to the checkpoint and check them on restart; then lift the restart guard.
- [ ] Add explicit expansion-frame metadata to stage ghost fills; map coarse endpoints before time interpolation and prescribed boundary data before use, including physical-frame source fills. Pass the refined homogeneous-expansion test with subcycling and physical boundaries.
- [ ] Weight flux register deposits and pass the AMR conservation test. Lift the `max_level = 0` guard only after the expansion-aware ghost-fill requirements also pass; conduction with AMR additionally requires the flux-conversion tests above.
- [ ] Add the EOS view; then lift the scale-invariant-EOS guard.
- [ ] Introduce `ScaleFactor`, `HookKey`, `TransportDt`/`PhysicalDt`, `ComovingDx`/`PhysicalDx` and the operator frame wrappers; add `src/cosmology/` to CODEOWNERS.
- [ ] Add the clang-query check as a CI job and the grep warning to `quokka-pre-commit.sh`.
- [ ] Add the verification tests to CTest.

Verification (all in CTest, CPU and at least one GPU backend; the current problems use `ADD_TEST OFF` and should be enabled):

| Test | Setup | Catches |
|---|---|---|
| Frozen-$`a`$ equivalence | Cosmology on, $`H_0 = 0`$, $`a = 0.1`$, compared with a non-cosmological run with box $`\times a`$, densities $`\times a^{-3}`$, $`B \times a^{-2}`$ (code units, $`B_\mathrm{Gauss}/\sqrt{4\pi}`$); one per operator | Any missing or extra factor of $`a`$ in any operator |
| Identity | Cosmology on, $`H_0 = 0`$, $`a = 1`$ | Must match the non-cosmological run bitwise |
| Frozen-$`a`$ diffusion | $`H_0=0`$, $`a=0.1`$, nonzero shear/bulk viscosity and, separately, constant resistivity; choose parameters so diffusion limits the timestep | Match physical-box momentum/energy fluxes and decay rates, and physical timesteps proportional to $`a^2 dx^2`$ at fixed physical diffusivity |
| Frozen-$`a`$ AMR conduction | $`H_0=0`$, $`a=0.1`$, temperature gradient crossing a static coarse-fine interface, subcycling and reflux enabled; compare with the corresponding physical-box run | Both half-step integrated heat fluxes, internal/total energy reflux corrections and final state match after unit conversion; exercises a nonzero flux mismatch |
| Expanding AMR conduction | Ideal gas, temperature gradient crossing a refined interface, subcycling; record each source-stage contribution and refine the timestep | Register contributions match $`\sum_s W_s a_s^2 q_s\Delta t_s`$ for the actual source ordering; energy errors converge under the stated reflux approximation |
| Frozen-$`a`$ magnetic consumers | $`H_0=0`$, $`a=0.1`$, nonzero $`B_c`$; exercise initialization, CFL, floors and both dual-energy branches, including after AMR synchronization | Magnetic energy $`B_c^2/(2a)`$, physical wavespeeds, and internal/total energy agreement with the physical-box run |
| Constant-scale-factor clock | $`H_0=0`$, $`a_\mathrm{init}=0.1`$ and $`1`$; query whole steps and half-intervals, repeat after restart | No table construction or inversion; constant $`a`$, zero $`H`$, and nonzero $`d\tau=dt/a_\mathrm{init}`$ for nonzero $`dt`$ |
| `CosmologicalExpansion` | Uniform box against the Einstein–de Sitter solution; also a non-ideal EOS once the view exists | Expansion operator, $`a(t)`$ table |
| Refined homogeneous expansion | Stationary uniform ideal gas, static refined patch, subcycling on; periodic boundaries and separately prescribed homogeneous-expansion boundaries; also a non-ideal EOS once supported | At every transport/source ghost fill, pressure agrees with the interior to roundoff for the ideal-gas map (ODE tolerance for a general EOS); no interface/boundary-generated velocity; final cooling matches the homogeneous solution |
| Zel'dovich pancake | Hydro, gravity and CIC particles together | Coupling of $`d\tau`$, Poisson RHS and particle kicks |
| Restart | $`N`$ steps, compared with $`N/2`$ steps, restart, $`N/2`$ steps | Bitwise match of $`a`$, $`d\tau`$ and the final state; also covers the existing hydro restart path |
| AMR conservation | Periodic box, static refined patch, bulk flow, subcycling on | $`\sum a\,p_c`$ constant to roundoff (fails on unweighted reflux); $`\nabla\cdot B_c`$ at machine precision across coarse-fine boundaries |

## Links
- Collins, D. C., Xu, H., Norman, M. L., Li, H., & Li, S. 2010, ApJS, 186, 308. *Cosmological Adaptive Mesh Refinement Magnetohydrodynamics with Enzo.* Eqs. 1–11 (comoving MHD and $`B_c`$), §2.4 (divergence under AMR), §2.9 and Appendix D (expansion terms and step order).
- Li, S., Li, H., & Cen, R. 2008, ApJS, 174, 1. *CosmoMHD*: the source-term formulation of magnetic dilution.
- Martel, H., & Shapiro, P. R. 1998, MNRAS, 297, 467. Supercomoving variables.
- [#2340](https://github.com/quokka-astro/quokka/pull/2340): CI test PR for the `mfulghieri/test-cosmology` branch.
- `src/cosmology/Cosmology.hpp`, `src/QuokkaSimulation.hpp`, `src/simulation.hpp`, `src/particles/PhysicsParticles.hpp`, `src/hydro/hydro_system.hpp`
