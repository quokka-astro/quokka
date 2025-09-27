# Hydro Integrator

The hydro integrator advances the conservative gas (and optionally MHD) variables that live in `state_new_cc_` and the face-centered magnetic fields in `state_new_fc_`. The entry point is `QuokkaSimulation<problem_t>::advanceHydroAtLevelWithRetries`, which is called once per AMR level from `AdvanceTimeStepOnLevel`. The routine is designed to deliver a second-order accurate, Strang-split update while remaining robust in the presence of stiff source terms, sharp shocks, and MHD-specific stability constraints.

## High-level workflow
- **Source splitting:** Each accepted substep begins and ends with `addStrangSplitSourcesWithBuiltin`, applying the problem-specific source terms for half a timestep. This keeps reactions, gravity, and other local physics time-centered relative to the hydrodynamic update.
- **Runge–Kutta stages:** `advanceHydroAtLevel` implements an SSP-RK2 integrator (with an optional first-order Euler fallback) over two stages. Stage 1 advances from the old state to an intermediate state; Stage 2 combines the intermediate fluxes with Stage 1 using RK weights stored in `flux_rk2` and `avgFaceVel`.
- **Flux register coupling:** When refluxing is enabled the combined RK fluxes are accumulated via `incrementFluxRegisters`, guaranteeing flux conservation across AMR coarse/fine interfaces.
- **Tracer and dual-energy support:** Face-averaged velocities produced during the update are reused to advance tracer particles and to synchronize the dual-energy formulation after each stage.
- **CFL validation:** A final `isCflViolated` check compares the accepted timestep against the local maximum signal speed. Any violation forces the retry logic to reduce the timestep, ensuring the user-specified `cflNumber_` is respected.

## Reconstruction and flux evaluation
`advanceHydroAtLevel` delegates the spatial discretisation to `computeHydroFluxes`:
- Primitive variables are reconstructed from the conservative state in cell-centered (`state_old_cc_tmp`) and, when MHD is enabled, face-centered (`state_old_fc_tmp`) data structures.
- Multi-dimensional flattening coefficients (`ComputeFlatteningCoefficients`) limit the slopes near shocks before higher-order reconstruction is applied.
- Directional flux functions call the appropriate Riemann solver: `HLLC` for pure hydro, `HLLD` plus constrained transport (`SolveInductionEqn`) for MHD. The solver also returns face-centered velocities and the fastest MHD wave speeds used by the EMF update.
- A first-order (donor-cell) set of fluxes is computed in parallel through `computeFOHydroFluxes`. These rely on the diffusive `LLF` solvers so that first-order flux correction has the most stable possible fallback without discarding the high-order solution everywhere.

## First-order fallback and stability guards
Throughout each timestep the solver decorates the update with physics-aware stability checks:
- `HydroSystem<problem_t>::AddInternalEnergyPdV` and `PredictStep` compute the stage RHS and provisional states. `redoFlag` flips only when the provisional density in a cell becomes non-positive, so flagged cells have their fluxes replaced with the pre-computed first-order counterparts via `replaceFluxes`/`replaceEMFs`. Because both RK stages share the corrected fluxes, the full timestep in those cells is carried out at first order.
- If any flagged cells remain after the first-order correction, the retry machinery escalates (unless `abortOnFofcFailure_` is set, in which case the attempt aborts immediately).
- Post-update limiters (`EnforceLimits`) and optional dual-energy synchronization ensure the final conservative state obeys positivity constraints prior to CFL evaluation and refluxing.

## Source terms, radiation, and coupling hooks
`advanceHydroAtLevel` only updates the conservative hydro variables. Radiation subcycling (`subcycleRadiationAtLevel`), additional source modules, tracer advection, and diagnostic output are invoked in the surrounding time-stepping loop. The hydro integrator exposes the face-averaged velocities and EMFs needed by those subsystems and writes detailed debug plotfiles when `lowLevelDebuggingOutput_` is enabled, easing diagnosis of instabilities.

## Hydro retries algorithm
The retry loop surrounding the integrator guards against pathological timesteps and first-order fallback failures:

1. **State checkpointing:** Before each attempt the routine snapshots the last accepted solution into `accepted_state_cc` (and `accepted_state_fc` for MHD). Failed attempts always restore these buffers so that no partial update contaminates subsequent retries.
2. **Adaptive substepping:** Each retry increases the number of substeps as `nsubsteps = 2^{retry_count}` (capped by `max_retries = 6`). The remaining timestep `dt_remaining` is divided by this count so the integrator can succeed with a smaller CFL number without modifying the user-requested global step.
3. **Substep execution:** For every substep the integrator reuses the checkpointed state as input, advances by `dt_step`, and, upon success, commits the new state to the accepted buffers. Partial progress updates `completed_time` so the next attempt only covers the unfinished interval.
4. **Partial progress handling:** If an attempt fails after completing one or more substeps, the loop records that partial progress and restarts with at least one additional level of subdivision. This avoids losing successful substeps while still shrinking the timestep for the remaining portion.
5. **Failure diagnostics:** Exceeding the retry budget triggers a fatal diagnostic: the code writes a `debug_hydro_state_fatal` plotfile (or Blueprint output when Ascent is enabled) and aborts, ensuring that difficult-to-integrate states leave actionable breadcrumbs.

Together, the retry logic and the stage-level first-order fallback provide a layered defence against stiff source terms, extreme shocks, and unexpected CFL violations while preserving as much of the high-order solution as possible.
