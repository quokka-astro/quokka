# Plan For Adding Cosmological Terms To Quokka

## Purpose

This note describes a concrete plan for adding cosmological expansion terms to Quokka, using Nyx's comoving hydro treatment as the reference design while adapting it to Quokka's method-of-lines (MOL) integrator.

The main conclusion is that Quokka should not add cosmological expansion as only an operator-split source term. Nyx's implementation is a coordinated change to

- the stored conservative variables,
- the primitive-variable recovery,
- the flux-divergence operator,
- the internal-energy work terms, and
- the final conservative update.

Quokka needs the same kind of coordinated treatment, but expressed in semidiscrete MOL form rather than in a Godunov predictor-corrector formulation.

## Current Quokka Hydro Structure

Quokka currently advances the hydro state in ordinary conserved variables

- `rho`
- `rho v`
- `rho E`
- `rho e` (auxiliary internal energy, when dual energy is enabled)

using an SSP-RK2 method-of-lines update. The main hydro flow is

1. apply Strang-split source terms before the hydro step,
2. reconstruct primitive variables from the stage state,
3. compute interface fluxes,
4. form the stage RHS from flux divergences,
5. add the `-P div(v)` term to the auxiliary internal-energy RHS,
6. update the conserved state with `U_new = U_old + dt * rhs`,
7. apply floors and dual-energy sync,
8. apply Strang-split source terms after the hydro step.

The key code locations are

- `QuokkaSimulation<problem_t>::advanceHydroAtLevel` in [`src/QuokkaSimulation.hpp`](/Users/benwibking/amrex_codes/quokka/src/QuokkaSimulation.hpp)
- `HydroSystem<problem_t>::ConservedToPrimitive` in [`src/hydro/hydro_system.hpp`](/Users/benwibking/amrex_codes/quokka/src/hydro/hydro_system.hpp)
- `HydroSystem<problem_t>::AddInternalEnergyPdV` in [`src/hydro/hydro_system.hpp`](/Users/benwibking/amrex_codes/quokka/src/hydro/hydro_system.hpp)
- `HydroSystem<problem_t>::PredictStep` in [`src/hydro/hydro_system.hpp`](/Users/benwibking/amrex_codes/quokka/src/hydro/hydro_system.hpp)

This means that Quokka has no direct analog of Nyx's primitive-variable predictor source step. In Quokka, the equivalent changes must appear in the semidiscrete stage operator `L(U,t)`.

## Target Formulation

The cleanest path is to adopt Nyx-style comoving gas variables for hydro:

- `rho`
- `m = a rho v`
- `E_c = a^2 rho E`
- `e_c = a^2 rho e`

where `a(t)` is the cosmological scale factor.

In these variables, the continuum system is

```text
∂t rho + (1/a) ∇·(rho v) = S_rho

∂t (a rho v) + ∇·(rho v v + p I) = rho g + a S_m

∂t (a^2 rho E) + a ∇·[(rho E + p) v] = a rho v·g + a^2 S_E

∂t (a^2 rho e) + a ∇·(rho e v) = -a p ∇·v + a^2 S_e
```

For a method-of-lines scheme, Quokka should evaluate this system stage-by-stage:

```text
dU/dt = L_flux(U, a(t)) + S_cosmo(U, a(t), adot(t)) + S_other(U, t)
```

The exact split between `L_flux` and `S_cosmo` depends on how much is absorbed by the variable choice. If Quokka stores Nyx-style comoving variables, then most of the cosmological treatment belongs in variable conversion and flux scaling rather than in a separate explicit source term.

## High-Level Design Decision

### Recommended approach

Implement cosmology as a built-in hydro formulation with comoving conserved variables.

This means:

- hydro stores comoving gas momentum and energies,
- primitive-variable recovery divides by the correct powers of `a`,
- the flux-divergence operator applies component-dependent `a` factors,
- the internal-energy work term is modified consistently,
- gravity and other modules are adapted afterward.

### Not recommended

Do not begin by adding only an operator-split Hubble source inside `addStrangSplitSources()`.

That would be inconsistent with Nyx's treatment because it would leave

- reconstruction,
- Riemann states,
- fluxes,
- total/internal energy consistency,
- and gravity coupling

in physical variables while trying to treat expansion as a separate source. That is likely to be less accurate and harder to maintain.

## Architecture Changes

## 1. Add a cosmology runtime module

Create a small built-in cosmology layer responsible for

- `a(t)`,
- `adot(t)`,
- `H(t) = adot/a`,
- optional conversion between scale factor and redshift,
- parsing cosmology inputs,
- publishing metadata to diagnostics and plotfiles.

This module should be independent of any specific problem.

Suggested inputs:

- `cosmology.enabled = 0/1`
- `cosmology.model = constant_H | tabulated_a_of_t | user`
- `cosmology.a_initial`
- `cosmology.H0`
- `cosmology.Omega_m`
- `cosmology.Omega_lambda`

The first implementation can support only the minimum required model, for example a supplied `a(t)` or constant `H`, and grow later.

## 2. Define the hydro state convention explicitly

When `cosmology.enabled = 1`, Quokka should document that the stored gas variables are

- density in comoving form: `rho`
- momentum in comoving form: `a rho v`
- total gas energy in comoving form: `a^2 rho E`
- auxiliary internal energy in comoving form: `a^2 rho e`

The code should avoid silent mixing of physical and comoving quantities. A helper layer should provide explicit conversions:

- conserved comoving state to physical primitive variables,
- physical primitive variables to conserved comoving state,
- physical source terms to comoving source terms.

This is the most important guard against subtle bugs.

### Required refactor: centralize state interpretation

Adopting comoving variables is not only a physics change. It is also a software-architecture change.

Today, different parts of Quokka can directly interpret conserved variables as if they were always the physical quantities

- `rho`,
- `rho v`,
- `rho E`,
- `rho e`.

That is manageable in the current non-cosmological code path, but it becomes unsafe once the stored state changes to comoving variables. At that point, correctness depends on ensuring that all modules obtain physical density, velocity, pressure, temperature, internal energy, and total energy through a single consistent conversion layer.

In practice, this means:

- `ConservedToPrimitive` must become the canonical path from stored state to physical primitive variables,
- the inverse mapping back to the stored conservative state must be equally explicit,
- helper routines should be added for common physical queries so modules do not manually recompute them from conserved components,
- code that directly divides momentum by density or subtracts kinetic energy from total energy should be audited and, where possible, replaced by central helpers.

This refactor is one of the main costs of the comoving-variable approach, but it is also one of its main benefits: once done correctly, it makes the variable convention explicit and prevents inconsistent physical/comoving interpretations from spreading through the code.

## 3. Thread stage-time cosmology data through the hydro integrator

Nyx computes `a_old` and `a_new` around a hydro step. Quokka needs the stage-time equivalent.

At minimum, each hydro stage should know

- `time_stage`,
- `a_stage = a(time_stage)`,
- `H_stage = H(time_stage)`.

If the update formula uses begin/end values explicitly, it should also have

- `a_old = a(t^n)`,
- `a_new = a(t^{n+1})`,
- `a_half = a(t^n + 0.5 dt)`.

The required changes begin in `advanceHydroAtLevel`, which already owns `time` and `dt_lev`.

## 4. Modify primitive-variable recovery

The current `ConservedToPrimitive` routine assumes physical conserved variables. With comoving storage, primitive recovery must compute

- `v = (a rho v) / (a rho) = m / (a rho)`
- physical internal energy from `a^2 rho e`
- physical total energy from `a^2 rho E`
- gas pressure from the physical internal energy

This is the first place where the new formulation becomes visible to the Riemann solve.

Implementation notes:

- keep the primitive variables themselves in physical form,
- keep reconstruction algorithms unchanged if possible,
- keep the Riemann solver unaware of cosmology if primitive inputs already represent the physical state correctly.

This isolates most cosmology logic from the reconstruction and solver code.

## 5. Modify the stage flux operator

This is the method-of-lines analog of Nyx's component-dependent hydro update.

Quokka currently forms one stage RHS from flux divergences with a uniform conservative interpretation. For the comoving system, the stage operator should instead apply:

- density RHS: `-(1/a) div(F_rho)`
- momentum RHS: `-div(F_m)`
- total-energy RHS: `-a div(F_E)`
- internal-energy RHS: `-a div(F_e)`
- passive scalar RHS: decide case-by-case whether they should be evolved as comoving densities or in another form

This is likely best implemented by extending the stage RHS assembly function to accept cosmology data and multiply each component by the correct factor.

The corresponding Quokka integration points are the stage RHS calls inside `advanceHydroAtLevel`.

## 6. Modify the internal-energy work term

Quokka currently adds

```text
-P div(v)
```

to the auxiliary internal-energy equation.

For the comoving internal-energy equation, the corresponding term is

```text
-a P div(v)
```

so `AddInternalEnergyPdV` must be made cosmology-aware.

In addition, if the chosen variable form leaves any explicit expansion correction not absorbed into the flux prefactor and rescaling, it should be added here or in a dedicated cosmology source routine.

## 7. Replace the plain conservative stage update

The current update in `PredictStep` is a plain

```text
U_new = U_old + dt rhs
```

That is correct only if `U` is already the stage variable being integrated.

If Quokka stores Nyx-style comoving conserved variables, then this update can still remain algebraically simple, but only if the stage RHS is already written in those variables and evaluated with the correct `a(t)` factors.

This leads to two viable implementations:

### Option A: fully semidiscrete comoving operator

Write `rhs` entirely in terms of the comoving conserved variables. Then `PredictStep` can remain unchanged.

This is the preferred option because it keeps the RK framework intact.

### Option B: explicit begin/end rescaling

Use `a_old`, `a_new`, and `a_half` explicitly in the state update, closer to Nyx's final conservative update.

This is less natural in Quokka's MOL structure and should be avoided unless testing shows it is needed.

## 8. Introduce a dedicated cosmology source hook inside hydro

Some terms may not be fully absorbed into variable scaling and flux prefactors. Those should not be implemented as user Strang-split sources.

Instead, add a built-in hydro routine, conceptually

```text
AddCosmologySources(rhs, state, time_stage, a_stage, H_stage)
```

This routine would be called during stage RHS assembly, beside the existing internal-energy work term.

Examples of terms that might belong here:

- explicit Hubble drag, if a non-comoving variable subset is retained,
- adiabatic expansion corrections not already captured by the conservative form,
- any temporary compatibility terms needed during an incremental rollout.

## 9. Decide how to treat passive scalars, dust, MHD, and radiation

These subsystems should not be left ambiguous.

### Passive scalars

If passive scalars represent advected mass densities, they should usually follow the density equation and therefore receive the same `1/a` flux prefactor as `rho`.

This needs to be documented and tested explicitly.

### Dust

Dust momentum and drag source terms depend strongly on the chosen frame and variable convention. Dust should initially be disabled when cosmology is enabled unless a consistent comoving formulation is implemented.

### MHD

MHD introduces additional choices for the magnetic-field scaling in comoving coordinates. This is a separate design problem and should be deferred until pure hydro is working.

### Radiation

Radiation can be formulated in either physical or comoving variables. Since Quokka's radiation system is already nontrivial, cosmological radiation terms should be deferred until the hydro-only implementation is validated.

## 10. Adapt gravity in phase 2

Quokka currently applies self-gravity through operator splitting after the elliptic solve.

Nyx's gravity update is consistent with its comoving variable choice:

- momentum source enters the `a rho v` equation,
- gravitational work enters the `a^2 rho E` equation with one factor of `a`.

Once hydro is converted to comoving storage, Quokka's gravity update must be reformulated to act on those same variables. This should be treated as a separate phase after hydro-only cosmology is stable.

## Implementation Phases

## Phase 0: design and guardrails

Goals:

- add a cosmology parameter namespace,
- add a central cosmology helper class or namespace,
- add assertions and comments documenting physical versus comoving variables,
- disable unsupported subsystems when cosmology is enabled.

Deliverables:

- parser support,
- metadata support,
- no physics changes yet.

## Phase 1: hydro-only comoving gas variables

Goals:

- implement comoving gas variable storage,
- modify primitive recovery,
- modify stage flux prefactors,
- modify the internal-energy work term,
- keep the RK structure intact.

Scope:

- pure hydro only,
- no self-gravity,
- no radiation,
- no dust,
- no MHD.

Deliverables:

- `cosmology.enabled = 1` works for hydro-only problems,
- `cosmology.enabled = 0` reproduces current behavior.

## Phase 2: gravity coupling

Goals:

- reformulate gravity source terms in comoving variables,
- verify momentum and energy consistency,
- ensure source ordering is well defined relative to the hydro step.

Deliverables:

- self-gravitating hydro works with cosmology enabled.

## Phase 3: passive scalars, chemistry, cooling

Goals:

- define which source modules consume physical variables,
- convert states to physical form before source application when needed,
- convert back to comoving form after the source update,
- document unit conventions.

Deliverables:

- chemistry and cooling can run consistently in expanding backgrounds.

## Phase 4: dust, MHD, radiation, particles

These are all separate extensions with their own design questions. They should be implemented only after hydro and gravity are stable.

## Detailed Code Touch Points

The following Quokka locations are expected to change.

### `src/QuokkaSimulation.hpp`

- `advanceHydroAtLevel`
  - compute and thread stage cosmology data,
  - call cosmology-aware RHS assembly,
  - gate unsupported features.
- `addStrangSplitSourcesWithBuiltin`
  - optionally convert to and from physical variables for source modules in later phases,
  - do not place the main cosmological hydro terms here.
- `computeHydroFluxes`
  - continue to reconstruct physical primitive variables, but derived from comoving conserved state.

### `src/hydro/hydro_system.hpp`

- `ConservedToPrimitive`
  - recover physical primitives from comoving conserved variables.
- `ComputeConsVars`
  - if used in reconstruction or tests, add the inverse mapping back to comoving conserved variables.
- `AddInternalEnergyPdV`
  - multiply by the proper `a` factor and add any remaining cosmology-specific energy corrections.
- `PredictStep`
  - ideally unchanged if the stage RHS is fully expressed in comoving variables.
- `isStateValid`, `EnforceLimits`, `SyncDualEnergy`
  - review carefully for hidden assumptions that the stored energies are physical rather than comoving.

### `src/simulation.hpp`

- gravity coupling paths such as `gravAccelAllLevels`
  - phase-2 work only,
  - reformulate sources for comoving momentum and energy.

## Numerical Validation Plan

The implementation should be validated in increasing order of complexity.

## 1. Exact preservation of current behavior when disabled

With `cosmology.enabled = 0`, all existing hydro tests should continue to pass within current tolerances.

This is the most important regression guard.

## 2. Homogeneous adiabatic expansion

Set up a uniform gas with no spatial gradients and an externally specified `a(t)`.

Expected behavior:

- density follows the chosen comoving convention,
- velocity decays with Hubble drag if a peculiar velocity is present,
- for an ideal monatomic gas, temperature scales as `T ∝ a^{-2}`.

This is the cleanest unit test of the expansion terms.

## 3. Uniform moving gas in an expanding background

Start with constant density, pressure, and peculiar velocity.

Expected behavior:

- no spurious spatial structure appears,
- momentum and kinetic energy evolve according to the comoving formulation,
- the solution remains machine-smooth.

## 4. Cosmological sound wave

Add a small-amplitude perturbation in an expanding background.

This checks that reconstruction, fluxing, and source coupling remain second-order and stable.

## 5. Cosmological Sod-like problem with `a = const`

Use the new code path with `a(t) = 1`.

Expected behavior:

- results match the non-cosmological code path to roundoff or near-roundoff,
- this verifies that the cosmology path reduces correctly to the existing equations.

## 6. Self-gravitating collapse in an expanding background

This should be added only after phase 2.

## Risks And Failure Modes

## 1. Silent mixing of physical and comoving variables

This is the largest risk. Pressure, temperature, kinetic energy, total energy, and source terms must all be computed in a clearly defined variable convention.

Mitigation:

- centralize conversions,
- add comments and assertions,
- avoid ad hoc `a` factors scattered throughout the code.
- audit modules for direct interpretation of conserved components and replace those call sites with canonical helper functions.

## 2. Dual-energy inconsistency

Quokka stores both total and auxiliary internal energy. In a comoving formulation both must be scaled consistently, and synchronization logic must be audited carefully.

Mitigation:

- port total and auxiliary energy together,
- add dedicated dual-energy tests under expansion.

## 3. Source modules expecting physical state

Cooling, chemistry, dust, and other modules likely assume physical densities, energies, and temperatures.

Mitigation:

- disable them initially,
- later wrap them in explicit comoving-to-physical conversions.

## 4. Gravity mismatch

If hydro uses comoving variables but gravity still updates physical momentum and energy, the combined method will be inconsistent.

Mitigation:

- defer gravity until the hydro-only path is stable,
- then update gravity in one focused phase.

## 5. MHD scaling ambiguity

The correct comoving scaling for magnetic fields requires a separate derivation and should not be improvised during the hydro port.

Mitigation:

- keep MHD disabled for cosmology until the formulation is written down explicitly.

## Recommended Initial Milestone

The first milestone should be deliberately narrow:

- hydro only,
- no MHD,
- no radiation,
- no dust,
- no chemistry,
- no cooling,
- no self-gravity,
- externally specified `a(t)` or constant `H`.

Success criteria:

- homogeneous adiabatic expansion test passes,
- a moving uniform gas remains uniform,
- existing non-cosmological hydro tests still pass with cosmology disabled,
- the cosmology code path reproduces the non-cosmological path exactly when `a = 1`.

## Summary

The correct Nyx-like port to Quokka is not "add cosmological source terms." It is "add a cosmological hydro formulation."

For Quokka's method-of-lines architecture, the equivalent of Nyx's predictor changes is to modify the stage operator:

- recover physical primitive variables from comoving conserved state,
- apply the correct `a`-dependent scaling to flux divergences,
- modify the internal-energy work term consistently,
- add any remaining expansion terms as built-in stage sources,
- leave operator-split source hooks for user physics, not for the core cosmological hydro terms.

That keeps the RK integrator intact, minimizes changes to reconstruction and Riemann solvers, and provides a clear phased path for extending cosmology support to gravity and other physics modules later.
