# Radiation Integrator

The radiation integrator advances the coupled radiation–matter system using the IMEX PD-ARS scheme. It is called once per AMR level from `AdvanceTimeStepOnLevel`, after the hydro advance, via `QuokkaSimulation<problem_t>::subcycleRadiationAtLevel`. Because the radiation timestep is typically much smaller than the hydro timestep (the reduced speed of light `c_hat` can still exceed the hydrodynamic wave speed), the radiation step is subdivided into multiple substeps within a single hydro timestep.

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


| Constant      | Value | Meaning                                         |
| ------------- | ----- | ----------------------------------------------- |
| `IMEX_Aex_21` | 1.0   | Explicit flux weight, stage 2 ← stage 1         |
| `IMEX_Aex_31` | 0.5   | Explicit flux weight, stage 3 ← stage 1         |
| `IMEX_Aex_32` | 0.5   | Explicit flux weight, stage 3 ← stage 2         |
| `IMEX_Aim_22` | 1.0   | Implicit solve timestep fraction, stage 2       |
| `IMEX_Aim_32` | 0.5   | Off-diagonal implicit weight, stage 3 ← stage 2 |
| `IMEX_Aim_33` | 0.5   | Implicit solve timestep fraction, stage 3       |
| `IMEX_alpha`  | 0.5   | Derived: Aim_32 / Aim_22 (Shu-Osher weight)     |


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

The rest of stage 3 is a single backward Euler step: `U^(3) = U^(3)* + dt Aim_33 * g(U^(3))` 

## Implementation

Define `state_xxx_gas` and `state_xxx_rad` as the gas and radiation components of `state_xxx_cc`, respectively, where `xxx` can be `new` or `tmp1`.

### Stage 1

Trivial `state_new_cc_ = state_new_cc_`, skipped.

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
//  → state_new_rad = (1 - alpha) * state_new_rad + alpha * state_tmp1_rad
//                    + dt * (Aex_31 - alpha * Aex_21) * s(U^(1))
//                    + dt * Aex_32 * s(state_tmp1_rad)
//                  = 0.5 * state_new_rad + 0.5 * state_tmp1_rad + dt * 0.5 * s(state_tmp1_rad)

// 5. Shu-Osher combination for gas variables
//    AddFluxesRK2 only touches radiation hyperbolic indices; gas must be combined
//    explicitly via a ParallelFor kernel on components [0, nstartHyperbolic_)
//    and [nstartHyperbolic_ + ncompHyperbolic_, nComp):
//    state_new_gas = (1 - alpha) * state_new_gas + alpha * state_tmp1_gas
//                  = 0.5 * gas_n + 0.5 * state_tmp1_gas

// 6. Implicit solve of `U^(3) = U^(3)* + dt Aim_33 * g(U^(3))`: Newton-Raphson with dt_implicit = Aim_33 * dt = 0.5 * dt
AddSourceTerms(state_new_cc_, dt_implicit = Aim_33 * dt, gas_update_factor = 1.0)
//    → state_new = U^(3)
```

### Key functions


| Function                               | Role                                                                                    |
| -------------------------------------- | --------------------------------------------------------------------------------------- |
| `subcycleRadiationAtLevel`             | Outer loop: substep count, state swap, per-substep IMEX stages                          |
| `advanceRadiationForwardEuler`         | Stage 2 explicit: `PredictStep` with `dt * Aex_21` into `state_out`                     |
| `advanceRadiationMidpointRK2`          | Stage 3 explicit: `AddFluxesRK2` with Shu-Osher coefficients, reading `state_inter`     |
| `RadSystem::AddFluxesRK2`              | GPU kernel: `(1-alpha)*U0 + alpha*U1 + Aex_s1_coeff*F0 + Aex_s2_coeff*F1` for radiation |
| `RadSystem::AddSourceTermsSingleGroup` | GPU kernel: Newton-Raphson implicit solve for single-group coupling                     |
| `RadSystem::AddSourceTermsMultiGroup`  | GPU kernel: Newton-Raphson implicit solve for multi-group coupling                      |
| `swapRadiationState`                   | Copies radiation hyperbolic vars from `state_new` → `state_old` for next substep        |


### Source term interface

Both `AddSourceTermsSingleGroup` and `AddSourceTermsMultiGroup` accept explicit `(dt_implicit, gas_update_factor)` parameters rather than a stage integer. The caller computes:

- `dt_implicit = Aim_ii * dt_radiation` — the effective implicit timestep for the diagonal solve
- `gas_update_factor = 1.0` — full update to all variables (no partial-update approximation)

This makes the mapping from Butcher tableau entries to solver calls transparent.

### Temporary state storage

`state_tmp1_cc` is allocated once per call to `subcycleRadiationAtLevel` (before the substep loop) and reused across substeps. It holds the complete U^(2) state after stage 2, enabling the Shu-Osher combination for gas variables in step 5 above without needing to store `g(U^(2))` separately.

## Matter–radiation Newton solve

Each implicit stage solves, cell by cell, a coupled system whose unknowns are one scalar for the matter (the gas internal energy, or the dust temperature when gas and dust are thermally decoupled) plus one per radiation group. The group residual is conservation,

<script type="math/tex; mode=display">
F_g = E_g - E_{0,g} - R_g - \mathrm{Src}_g \, ,
</script>

and the group unknown is tied to \\(E\_g\\) by the definition of the exchange term,

<script type="math/tex; mode=display">
R_g = \left( \frac{4 \pi B_g}{c} - \frac{E_g}{(\kappa_P/\kappa_E)_g} \right) \tau_g + w_g \, ,
</script>

with \\(\tau\_g = \Delta t\, \rho\, \kappa\_{P,g}\, \hat c\\) the optical depth across the step and \\(w\_g\\) the work term. Two properties of this system are easy to get wrong and are worth stating.

### Group Planck temperature derivative

The Jacobian entry \\(\partial F\_g / \partial T\\) requires the temperature derivative of the group-integrated Planck function \\(a T^4 f\_g(T)\\), and **both** terms of the product rule matter:

<script type="math/tex; mode=display">
\frac{\mathrm{d}}{\mathrm{d}T} \left[ a T^4 f_g(T) \right] = 4 a T^3 f_g + a T^4 \frac{\mathrm{d} f_g}{\mathrm{d}T} \, .
</script>

The second term is negligible for the group that carries the spectral peak, where \\(f\_g\\) is close to constant, but it dominates for a group on the Wien tail, where \\(f\_g\\) can rise as steeply as \\(T^{20}\\). Dropping it does not change the converged answer, but it makes the Newton step too long by the same factor, and the iteration then overshoots and diverges.

`ComputeThermalRadiationTempDerivativeMultiGroup` therefore evaluates the exact derivative. Integrating the kernel by parts gives a cumulative form built from the same normalized Planck integral \\(P\\) used for the energy fractions,

<script type="math/tex; mode=display">
D(x) \equiv \frac{15}{\pi^4} \int_0^x \frac{s^4 e^s}{(e^s - 1)^2} \, \mathrm{d}s = 4 P(x) - \frac{15}{\pi^4} \frac{x^4}{e^x - 1} \, ,
</script>

so that \\(\mathrm{d}(4 \pi B\_g / c) / \mathrm{d}T = a T^3 \left[ D(x\_{g+1}) - D(x\_g) \right]\\) with \\(x = h \nu / (k T)\\). The cost is one extra term per group boundary, and \\(D(\infty) = 4\\) recovers \\(\mathrm{d}(a T^4)/\mathrm{d}T\\).

### Choice of per-group unknown

\\(E\_g\\) and \\(R\_g\\) are affinely related, so either may serve as the unknown and Newton takes the same steps in exact arithmetic. Round-off is not invariant, however: whichever quantity is carried between iterations keeps full relative precision, and the other inherits the error of the map. The two directions fail in opposite regimes.

- With \\(R\_g\\) as the unknown, \\(E\_g = (\kappa\_P/\kappa\_E)\_g \left( 4 \pi B\_g / c - (R\_g - w\_g)/\tau\_g \right)\\). For \\(\tau\_g \ll 1\\) the two terms agree to within a factor \\(\tau\_g\\), so an **optically thin** group loses its leading digits.
- With \\(E\_g\\) as the unknown, \\(R\_g\\) comes from the definition above. That difference cancels when \\(E\_g / (\kappa\_P/\kappa\_E)\_g\\) approaches \\(4 \pi B\_g / c\\), which is the **optically thick**, near-equilibrium case.

The solver therefore switches per group on \\(\tau\_g\\), at `newton_erad_base_tau_threshold` (currently 1): thin groups are based on \\(E\_g\\), thick groups on \\(R\_g\\). Rebasing a group scales its Jacobian column by \\(\mathrm{d}R\_g/\mathrm{d}E\_g = -\tau\_g / (\kappa\_P/\kappa\_E)\_g\\), changes \\(\partial F\_g / \partial x\\) (the partial derivative at fixed \\(E\_g\\) is not the one at fixed \\(R\_g\\)), and adds a term to \\(\partial F\_0 / \partial x\\), since \\(R\_g\\) now varies with the matter variable. This is done by `RebaseThinGroupsOntoErad`.

Rebasing is applied in `SolveGasRadiationEnergyExchange` and in the decoupled branch of `SolveGasDustRadiationEnergyExchange`. It is **not** applied when gas and dust are thermally coupled, because there \\(T\_d\\) is itself a function of \\(\sum\_g R\_g\\) and `ComputeJacobianForGasAndDust` has already eliminated that coupling in the \\(R\_g\\) unknowns, so the columns are no longer a plain change of variable away from the \\(E\_g\\) ones.

Groups that are still based on \\(R\_g\\) keep a round-off floor on the convergence test: their residual cannot be resolved below \\(\varepsilon \times 4 \pi B\_g / c\\), so a purely relative tolerance would be unreachable. Rebased groups use the round-off of the residual's own terms instead.

## Equivalence with the previous implementation (single-group)

Before this refactor the code used a `gas_update_factor = IMEX_a32 = 0.5` trick: stage 2 applied only half the gas update, avoiding the need to store U^(2). The new implementation applies the full gas update at stage 2, then applies the Shu-Osher combination `0.5*gas_n + 0.5*gas_stage2` before stage 3's implicit solve. The starting point for the stage 3 Newton-Raphson is identical in both cases, so for single-group radiation the numerical results are algebraically equivalent. For multi-group radiation the old `gas_update_factor` also entered the work-term iteration inside `UpdateFlux`; the new implementation with `gas_update_factor = 1.0` is the mathematically correct IMEX formulation.