# Radiation Integrator

The radiation integrator advances the coupled radiation–matter system using the IMEX PD-ARS scheme. It is called once per AMR level from `AdvanceTimeStepOnLevel`, after the hydro advance, via `QuokkaSimulation<problem_t>::subcycleRadiationAtLevel`. Because the radiation CFL number is typically much tighter than the hydro CFL number (the reduced speed of light `c_hat` can still exceed the hydrodynamic wave speed), the radiation step is subdivided into multiple substeps within a single hydro timestep.

## High-level workflow

- **Substep count:** `computeNumberOfRadiationSubsteps` determines the integer number of radiation substeps needed to cover the hydro timestep at the radiation CFL number. When hydro is disabled or a constant timestep is in use, a single substep is taken.
- **State management:** At the start of each substep after the first, `swapRadiationState` copies the radiation hyperbolic variables from `state_new_cc_` back into `state_old_cc_`, so that the integrator always has a clean "old" radiation state to advance from while the hydro variables remain in `state_new_cc_`.
- **IMEX stages per substep:** Each substep applies the 3-stage IMEX PD-ARS scheme — one explicit Forward Euler stage and one explicit RK2 corrector stage, each followed by an implicit Newton–Raphson solve for the stiff matter–radiation coupling.
- **Particle source injection:** In 3D, stellar particles deposit their luminosity into `radEnergySource` before each implicit solve, giving a cell-centred luminosity density (erg s⁻¹ cm⁻³).
- **Flux register coupling:** Radiation fluxes are accumulated into `FluxRegister`s for later refluxing across AMR coarse/fine interfaces.

## IMEX PD-ARS scheme

The coupled system has the form

```
∂U/∂t = s(U) + g(U)
```

where `s` is the stiff-explicit part (radiation transport: flux divergence) and `g` is the stiff-implicit part (matter–radiation exchange: emission, absorption, momentum coupling). The IMEX PD-ARS scheme integrates this with a 3-stage, 2nd-order accurate, L-stable method.

### Butcher tableaux

The explicit (Aex) and implicit (Aim) tableaux are:

```
Explicit Aex:            Implicit Aim:
c | A                    c | A
--+-----                 --+-----
0 | 0    0    0          0 | 0    0    0
1 | 1    0    0          1 | 0    1    0
1 | 1/2  1/2  0          1 | 0   1/2  1/2
  |-----------             |-----------
  | 1/2  1/2  0            | 0   1/2  1/2
```

The scheme is **stiffly accurate**: the quadrature weights b equal the last row of A in both tableaux, so the solution at the end of each step equals the last stage value. In code the entries are defined as `static constexpr` members of `QuokkaSimulation`:

| Constant | Value | Meaning |
|---|---|---|
| `IMEX_Aex_21` | 1.0 | Explicit flux weight, stage 2 ← stage 1 |
| `IMEX_Aex_31` | 0.5 | Explicit flux weight, stage 3 ← stage 1 |
| `IMEX_Aex_32` | 0.5 | Explicit flux weight, stage 3 ← stage 2 |
| `IMEX_Aim_22` | 1.0 | Implicit solve timestep fraction, stage 2 |
| `IMEX_Aim_32` | 0.5 | Off-diagonal implicit weight, stage 3 ← stage 2 |
| `IMEX_Aim_33` | 0.5 | Implicit solve timestep fraction, stage 3 |
| `IMEX_alpha`  | 0.5 | Derived: Aim_32 / Aim_22 (Shu-Osher weight) |

### Stage equations

Let U^n denote the state at the beginning of a radiation substep. Stage 1 is trivial (U^(1) = U^n). The two active stages are:

**Stage 2 — predictor:**

```
U^(2) = U^n + dt * Aex_21 * s(U^(1)) + dt * Aim_22 * g(U^(2))
```

The explicit part is a Forward Euler step with coefficient `Aex_21 = 1`; the implicit part is a backward Euler solve with `dt_implicit = Aim_22 * dt = dt`.

**Stage 3 — corrector:**

```
U^(3) = U^n + dt * [Aex_31 * s(U^(1)) + Aex_32 * s(U^(2))]
             + dt * [Aim_32 * g(U^(2)) + Aim_33 * g(U^(3))]
```

The explicit part is the standard SSP-RK2 combination of stage-1 and stage-2 fluxes; the implicit part involves the off-diagonal contribution `Aim_32 * g(U^(2))` carried from stage 2, plus a new diagonal implicit solve with `dt_implicit = Aim_33 * dt = dt/2`.

### Shu-Osher form for stage 3

The off-diagonal implicit contribution `Aim_32 * g(U^(2))` does not need to be stored separately. Using the stage-2 equation to eliminate `dt * g(U^(2))`:

```
dt * Aim_22 * g(U^(2)) = U^(2) - U^n - dt * Aex_21 * s(U^(1))
```

Substituting into the stage-3 equation and collecting terms gives the **Shu-Osher form**:

```
let alpha = Aim_32 / Aim_22 = 0.5

U^(3)* = (1 - alpha) * U^n + alpha * U^(2)
        + dt * (Aex_31 - alpha * Aex_21) * s(U^(1))
        + dt * Aex_32 * s(U^(2))
```

For PD-ARS the coefficient `Aex_31 - alpha * Aex_21 = 0.5 - 0.5 * 1 = 0`, so no stage-1 flux divergence term appears:

```
U^(3)* = 0.5 * U^n + 0.5 * U^(2) + 0.5 * dt * s(U^(2))
```

This is exactly the SSP-RK2 corrector formula, extended to also carry forward the implicit contribution from stage 2 through the `alpha * U^(2)` weight.

## Implementation

### Stage 2

```
// 1. Copy hydro-updated state into temporary (preserves gas variables)
state_tmp1_cc = state_new_cc_[lev]

// 2. Forward Euler overwrites radiation vars in state_tmp1_cc from state_old_cc_
advanceRadiationForwardEuler(..., state_tmp1_cc)
//    → state_tmp1_rad = state_old_rad + dt * Aex_21 * s(state_old_rad)
//      state_tmp1_gas = gas_n (unchanged by PredictStep)

// 3. Implicit solve: full Newton-Raphson with dt_implicit = Aim_22 * dt
AddSourceTerms(state_tmp1_cc, dt_implicit = Aim_22 * dt, gas_update_factor = 1.0)
//    → state_tmp1 = U^(2) (radiation + gas fully updated)
```

### Stage 3

```
// 4. Explicit corrector (radiation vars only — Shu-Osher form)
advanceRadiationMidpointRK2(..., state_inter = state_tmp1_cc)
//    calls AddFluxesRK2(alpha=0.5, Aex_s1_coeff=0, Aex_s2_coeff=0.5)
//    → state_new_rad = 0.5*U^n_rad + 0.5*U^(2)_rad + 0.5*dt*s(U^(2)_rad)

// 5. Shu-Osher combination for gas variables
//    AddFluxesRK2 only touches radiation hyperbolic indices; gas must be combined
//    explicitly via a ParallelFor kernel on components [0, nstartHyperbolic_)
//    and [nstartHyperbolic_ + ncompHyperbolic_, nComp):
//    state_new_gas = (1 - alpha) * state_new_gas + alpha * state_tmp1_gas
//                 = 0.5 * gas_n + 0.5 * (gas_n + Δg_stage2)
//                 = gas_n + 0.5 * Δg_stage2

// 6. Implicit solve: full Newton-Raphson with dt_implicit = Aim_33 * dt = 0.5 * dt
AddSourceTerms(state_new_cc_, dt_implicit = Aim_33 * dt, gas_update_factor = 1.0)
//    → state_new = U^(3)
```

### Key functions

| Function | Role |
|---|---|
| `subcycleRadiationAtLevel` | Outer loop: substep count, state swap, per-substep IMEX stages |
| `advanceRadiationForwardEuler` | Stage 2 explicit: `PredictStep` with `dt * Aex_21` into `state_out` |
| `advanceRadiationMidpointRK2` | Stage 3 explicit: `AddFluxesRK2` with Shu-Osher coefficients, reading `state_inter` |
| `RadSystem::AddFluxesRK2` | GPU kernel: `(1-alpha)*U0 + alpha*U1 + Aex_s1_coeff*F0 + Aex_s2_coeff*F1` for radiation |
| `RadSystem::AddSourceTermsSingleGroup` | GPU kernel: Newton-Raphson implicit solve for single-group coupling |
| `RadSystem::AddSourceTermsMultiGroup` | GPU kernel: Newton-Raphson implicit solve for multi-group coupling |
| `swapRadiationState` | Copies radiation hyperbolic vars from `state_new` → `state_old` for next substep |

### Source term interface

Both `AddSourceTermsSingleGroup` and `AddSourceTermsMultiGroup` accept explicit `(dt_implicit, gas_update_factor)` parameters rather than a stage integer. The caller computes:

- `dt_implicit = Aim_ii * dt_radiation` — the effective implicit timestep for the diagonal solve
- `gas_update_factor = 1.0` — full update to all variables (no partial-update approximation)

This makes the mapping from Butcher tableau entries to solver calls transparent.

### Temporary state storage

`state_tmp1_cc` is allocated once per call to `subcycleRadiationAtLevel` (before the substep loop) and reused across substeps. It holds the complete U^(2) state after stage 2, enabling the Shu-Osher combination for gas variables in step 5 above without needing to store `g(U^(2))` separately.

## Equivalence with the previous implementation (single-group)

Before this refactor the code used a `gas_update_factor = IMEX_a32 = 0.5` trick: stage 2 applied only half the gas update, avoiding the need to store U^(2). The new implementation applies the full gas update at stage 2, then applies the Shu-Osher combination `0.5*gas_n + 0.5*gas_stage2` before stage 3's implicit solve. The starting point for the stage 3 Newton-Raphson is identical in both cases, so for single-group radiation the numerical results are algebraically equivalent. For multi-group radiation the old `gas_update_factor` also entered the work-term iteration inside `UpdateFlux`; the new implementation with `gas_update_factor = 1.0` is the mathematically correct IMEX formulation.
