# Hydro Integrator

The hydro integrator advances the conservative gas (and optionally MHD) variables that live in `state_new_cc_` and the face-centered magnetic fields in `state_new_fc_`. The entry point is `QuokkaSimulation<problem_t>::advanceHydroAtLevelWithRetries`, which is called once per AMR level from `AdvanceTimeStepOnLevel`. The routine is designed to deliver a second-order accurate, Strang-split update while remaining robust in the presence of stiff source terms and strong shocks and rarefactions.

## High-level workflow
- **Source splitting:** Each accepted substep begins and ends with `addStrangSplitSourcesWithBuiltin`, applying the problem-specific source terms for half a timestep.
- **Runge–Kutta stages:** `advanceHydroAtLevel` implements an SSP-RK2 integrator (with an optional first-order Euler fallback) over two stages. Stage 1 advances from the old state to an intermediate state; Stage 2 starts from the intermediate state and computes the second stage fluxes. The final update is done from the initial state using the average from the Stage 1 and Stage 2 fluxes.
- **Flux register coupling:** When refluxing is enabled the combined RK fluxes are accumulated via `incrementFluxRegisters`, guaranteeing flux conservation across AMR coarse/fine interfaces.
- **Tracer and dual-energy support:** At each stage, face-centered velocities are used to update the auxiliary internal energy and perform a dual energy sync. Face-centered velocities produced during the update are averaged over the timestep and reused to advance tracer particles at the end of the step.
- **CFL validation:** A final `isCflViolated` check compares the accepted timestep against the maximum signal speed of the provisional updated state. Any violation forces the retry logic to reduce the timestep and redo the hydro update, ensuring the user-specified `cflNumber_` is respected.

## Reconstruction and flux evaluation
`advanceHydroAtLevel` delegates the spatial discretisation to `computeHydroFluxes`:
- Primitive variables are reconstructed from the conservative state stored in the cell-centered (`state_old_cc_tmp`) and, when MHD is enabled, face-centered (`state_old_fc_tmp`) data structures.
- Multi-dimensional flattening coefficients (`ComputeFlatteningCoefficients`) limit the slopes near shocks before higher-order reconstruction is applied.
- Directional flux functions call the appropriate Riemann solver: `HLLC` for pure hydro, `HLLD` plus constrained transport (`SolveInductionEqn`) for MHD. The solver also returns face-centered velocities and the fastest MHD wave speeds used by the EMF update.
- Before the second-order update, a first-order (donor-cell) set of fluxes is computed through `computeFOHydroFluxes`. These rely on the diffusive `LLF` solvers so that first-order flux correction has the most stable possible fallback without discarding the high-order solution everywhere.

## First-order fallback and stability guards
Throughout each timestep the solver decorates the update with physics-aware stability checks:
- `HydroSystem<problem_t>::AddInternalEnergyPdV` and `PredictStep` compute the stage RHS and provisional states. `redoFlag` flips only when the provisional density in a cell becomes non-positive, so flagged cells have their fluxes replaced with the pre-computed first-order counterparts via `replaceFluxes`/`replaceEMFs`. The full timestep in those cells is therefore carried out at first order in both space and time.
- If any flagged cells remain after the first-order correction, the retry machinery escalates (unless `abortOnFofcFailure_` is set, in which case the attempt aborts immediately).
- Post-update limiters (`EnforceLimits`) and optional dual-energy synchronization ensure the final conservative state obeys positivity constraints prior to CFL evaluation and refluxing.

## Source terms, radiation, and coupling hooks
`advanceHydroAtLevel` only updates the conservative hydro variables. Radiation subcycling (`subcycleRadiationAtLevel`), additional operator-split source modules, and diagnostic output are invoked in the surrounding time-stepping loop.

## Hydro retries algorithm
The retry loop surrounding the integrator helps prevent crashes in the presence of strong nonlinearities in the solution:

1. **State checkpointing:** Before each attempt the routine snapshots the last accepted solution into `accepted_state_cc` (and `accepted_state_fc` for MHD). Failed attempts always restore these buffers so that no partial update contaminates subsequent retries.
2. **Adaptive substepping:** Each retry chooses `nsubsteps = 2^{retry_count}` (with `retry_count ≤ max_retries = 6`). Progress is recorded as an integer number of base units, where one unit represents `dt_lev / (2^{max_retries})`. This guarantees that any retry can reinterpret previously accepted work exactly on its canonical grid.
3. **Substep execution:** For the current retry level the code computes `units_per_substep = 2^{max_retries - retry_count}` and evaluates `completed_substeps = completed_units / units_per_substep`. Only the remaining substeps are executed. The floating-point time passed to `advanceHydroAtLevel` is reconstructed on demand from the integer unit count: `t_sub = t0 + completed_units * dt_lev / 2^{max_retries}`.
4. **Partial progress handling:** Accepting a substep immediately increments the integer progress counter. If a failure occurs after some substeps succeed, the retry level is bumped (capped by `max_retries`) so the next attempt uses smaller substeps; a fully successful pass exits the outer loop immediately.
5. **Failure diagnostics:** Exceeding the retry budget triggers a fatal diagnostic: the code writes a `debug_hydro_state_fatal` plotfile (or Blueprint output when Ascent is enabled) and aborts, ensuring that difficult-to-integrate states can be debugged.

### Pseudocode outline

```text
t0 = currentLevelTime
maxUnits = 1 << maxRetries
acceptedState = snapshotCellCenteredState()
acceptedFaceState = snapshotFaceCenteredState()

completedUnits = 0
curRetryLevel = 0
while completedUnits < maxUnits and curRetryLevel <= maxRetries:
    totalSubsteps = 1 << curRetryLevel
    unitsPerSubstep = maxUnits // totalSubsteps
    startSubstep = completedUnits // unitsPerSubstep
    dtSubstep = dtTotal / totalSubsteps

    for substepIndex in range(startSubstep, totalSubsteps):
        substepTime = t0 + substepIndex * dtSubstep
        status = advanceHydroAtLevel(substepTime, dtSubstep)
        if status.failed():
            restoreStateFrom(acceptedState, acceptedFaceState)
            curRetryLevel += 1
            break
        else:
            completedUnits += unitsPerSubstep
            commitAcceptedState()

if completedUnits < maxUnits:
    writeFatalDiagnostics()
    abortHydroStep()
```
