# Radiation Integrator

The radiation integrator advances the coupled radiation–matter system using the IMEX PD-ARS scheme. It is called once per AMR level from `AdvanceTimeStepOnLevel`, after the hydro advance, via `QuokkaSimulation<problem_t>::subcycleRadiationAtLevel`. Because the radiation timestep is typically much smaller than the hydro timestep (the reduced speed of light `c_hat` can still exceed the hydrodynamic wave speed), the radiation step is subdivided into multiple substeps within a single hydro timestep.

## High-level workflow

- **Substep count:** `computeNumberOfRadiationSubsteps` determines the integer number of radiation substeps needed to cover the hydro timestep at the radiation CFL number. When hydro is disabled or a constant timestep is in use, a single substep is taken.
- **State management:** At the start of each substep after the first, `swapRadiationState` copies the radiation hyperbolic variables from `state_new_cc_` back into `state_old_cc_`, so that the integrator always has a clean "old" radiation state to advance from while the hydro variables remain in `state_new_cc_`.
- **IMEX stages per substep:** Each substep applies the 3-stage IMEX PD-ARS scheme — one explicit Forward Euler stage and one explicit RK2 corrector stage, each followed by an implicit solve for the stiff matter–radiation coupling.
- **Particle source injection:** In 3D, stellar particles deposit their luminosity into `radEnergySource` before each implicit solve, giving a cell-centred luminosity density (erg s⁻¹ cm⁻³).
- **User source injection:** `RadSystem<problem_t>::AddRadSource` is called before each implicit solve and lets a problem add its own radiation source. It writes two scratch buffers of its own, zeroed beforehand: `radEnergySource`, a luminosity volume density per group, and `reducedFluxSource`, the *reduced* flux \\(f = F/(cE)\\) of the injected radiation in component \\(3g + n\\) for group \\(g\\) along direction \\(n\\). `MergeUserRadSource` then converts the pair into a real flux source and adds both to the solver's buffers. Asking for a reduced flux rather than a flux makes \\(|F| > cE\\) unrepresentable: a reduced flux of unit magnitude injects fully beamed, free-streaming radiation for either kind of band, and leaving it at zero injects isotropically. The energy source is scaled internally by \\(\hat c / c\\) for a thermal group and not at all for a chemistry band, and the derived flux source inherits the same scaling.
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

// 3. Implicit solve: the bracketed coupling solve with dt_implicit = Aim_22 * dt
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

// 6. Implicit solve of `U^(3) = U^(3)* + dt Aim_33 * g(U^(3))`: the bracketed coupling solve with dt_implicit = Aim_33 * dt = 0.5 * dt
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
| `RadSystem::AddSourceTerms`            | GPU kernel: implicit coupling solve and flux update, single-group and multigroup        |
| `swapRadiationState`                   | Copies radiation hyperbolic vars from `state_new` → `state_old` for next substep        |


### Source term interface

`AddSourceTerms` accepts explicit `(dt_implicit, gas_update_factor)` parameters rather than a stage integer. The caller computes:

- `dt_implicit = Aim_ii * dt_radiation` — the effective implicit timestep for the diagonal solve
- `gas_update_factor = 1.0` — full update to all variables (no partial-update approximation)

This makes the mapping from Butcher tableau entries to solver calls transparent.

### Temporary state storage

`state_tmp1_cc` is allocated once per call to `subcycleRadiationAtLevel` (before the substep loop) and reused across substeps. It holds the complete U^(2) state after stage 2, enabling the Shu-Osher combination for gas variables in step 5 above without needing to store `g(U^(2))` separately.

## Matter-radiation coupling solve

Each implicit stage solves, cell by cell, one backward-Euler step of the energy exchange between the gas (or the dust) and the \\(N\_g\\) radiation groups. The unknowns are one scalar for the matter and one energy per group, but the group equations are linear in the group energies at a fixed matter temperature, so they are solved in closed form and the whole step reduces to **one scalar equation**. The method follows the hydro3d.jl reference implementation; the code is `radiation_coupling.hpp`.

### The closed-form group block

With \\(\mathrm{rad0}\_g = E\_g^0 + S\_g + W\_g\\) (the group's starting energy plus its external source and lagged work term), \\(\tau\_s = \Delta t \\, \hat c\\) (times the Lorentz factor on the single-group `beta_order >= 2` path), and the emission and absorption coefficients \\(\varepsilon\_g = \rho \kappa\_{P,g} \\, 4\pi B\_g / c\\) and \\(\alpha\_g = \rho \kappa\_{E,g}\\) evaluated at the matter temperature \\(T\_m\\),

<script type="math/tex; mode=display">
E_g = \frac{\mathrm{rad0}_g + \tau_s \, \varepsilon_g(T_m)}{1 + \tau_s \, \alpha_g(T_m)} \, ,
</script>

a convex combination of where the group started and the Planck value at the matter temperature, weighted by the optical depth of the step. All of the stiffness is in that one weight and is handled exactly; a transparent group keeps \\(\mathrm{rad0}\_g\\), source included.

### One equation

**Without dust** the matter temperature is the gas temperature, and what is left is energy conservation,

<script type="math/tex; mode=display">
G(E_{\rm gas}) = E_{\rm gas} + \frac{c}{\hat c} \sum_g E_g\!\left(T(E_{\rm gas})\right) - \mathcal{E} = 0 \, , \qquad \mathcal{E} = E_{\rm gas}^0 + \frac{c}{\hat c} \sum_g \left( E_g^0 + S_g \right) \, ,
</script>

the new total energy minus the old.

**With dust** (`ISM_Traits::enable_dust_gas_thermal_coupling_model`) the radiation couples to the dust at \\(T\_d\\), the dust holds no energy, and gas and dust exchange energy at the rate \\(K T^{1/2} (T - T\_d)\\) with \\(K\\) the coefficient `radiation.dust_gas_interaction_coeff` times \\(n\_{\rm H}^2\\). The unknown is \\(T\_d\\): the group block is evaluated at \\(T\_d\\), the gas energy follows from conservation, and what is left is the gas equation,

<script type="math/tex; mode=display">
H(T_d) = \left( E_{\rm gas}^0 - \frac{c}{\hat c} \sum_g W_g - E_{\rm gas} \right) - \Delta t \, K \, T^{1/2} \left( T - T_d \right) = 0 \, .
</script>

Total energy is conserved to round-off at every trial \\(T\_d\\), not only at the root. One equation covers every coupling strength: at \\(K = 0\\) the gas is untouched and \\(H = 0\\) is radiative equilibrium of the dust; as \\(K \to \infty\\), \\(T\_d \to T\\) and the dust-free step is recovered. There is no longer a well-coupled and a decoupled regime.

### Bracket, root finder, tolerance

Both equations increase with their unknown except where a steep opacity law makes them non-monotone, in which case the step can have several roots. The bracket is therefore built by **marching outward from the old state** (\\(E\_{\rm gas}^0\\), or the start-of-step gas temperature for \\(T\_d\\)) by factors of two in the direction the sign of the residual indicates, which isolates the root continuously connected to where the cell started. The march in \\(E\_{\rm gas}\\) never goes below \\(E\_{\rm min} = E\_{\rm int}(\rho, T\_{\rm floor})\\): if it reaches \\(E\_{\rm min}\\) without a sign change, the root lies below the admissible range, the gas is clamped to the floor and the groups take the closed form at \\(T\_{\rm floor}\\), the cell counts as converged, and the energy \\(G(E\_{\rm min}) > 0\\) is created by the temperature floor, as any floor does, with no threshold on that amount. With a zero temperature floor (every problem in `UnitSystem::CONSTANTS`) \\(E\_{\rm min} = 0\\), and the march instead floors at round-off of the initial gas energy, \\(16 \\, \epsilon \\, E\_{\rm gas}^0\\) with \\(\epsilon\\) the machine epsilon, so that it still ends; probing exactly \\(E = 0\\) would evaluate the Planck function at \\(T = 0\\). This is routine, not pathological: a transparent, radiation-dominated cell whose gas sits at the floor hits this every step. The march in \\(T\_d\\) has no floor: \\(T\_{\rm floor}\\) is a floor on the gas, and dust in a weak field is colder than it, so as \\(T\_d \to 0\\) the emission vanishes and \\(H\\) always finds its sign change. Only an upward march that exhausts its 200 doublings, or a march that meets a non-finite residual, is reported unconverged, and the run aborts.

The root is found with `quokka::math::brent_solve` (Brent's method with a minimum step; see `bracketing_root_finding.hpp`), which stops when the bracket is narrow relative to the unknown: \\(|hi - lo| \le \mathrm{tol} \\, \min(|lo|, |hi|)\\) with `tol` the input `radiation.iteration_tolerance`. Its two bisection tests on the last steps compare with twice Brent's minimum step \\(\delta = 2\varepsilon|b|\\) rather than with machine epsilon, so that a residual flat at round-off next to the root cannot make the minimum-step safeguard creep for the whole budget; `MathUnitTests` pins this. For an ideal gas the unknown \\(E\_{\rm gas} = c\_V T\\) makes this a relative tolerance on the gas temperature. With dust the promise is on the gas energy across Brent's final bracket in \\(T\_d\\), \\(|E\_{\rm gas}(T\_d^-) - E\_{\rm gas}(T\_d^+)| \le \max(\mathrm{tol}\\,|E\_{\rm gas}|,\\ 4\varepsilon\\,|\mathcal{E}|)\\), because at small \\(K\\) the gas energy follows from conservation and its error is the \\(T\_d\\) error times \\((c/\hat c)\sum\_g E\_g / E\_{\rm gas}\\) (about \\(10^5\\) in `DTypeFront1D`); if the test fails, Brent is repeated on its own final bracket with the \\(T\_d\\) tolerance set from the measured slope, at most three times, and a bracket at the floating-point resolution of \\(T\_d\\) counts as converged. The state is evaluated at the midpoint of the final bracket. Convergence is judged on the unknown, never on the residual: \\(H\\) in particular multiplies the round-off in \\(T - T\_d\\) by \\(\Delta t K T^{1/2}\\) and cannot be tested directly. The conservation error of a step is then about \\(\mathrm{d}G/\mathrm{d}E\_{\rm gas}\\) times the tolerance times the gas energy. Over hydro3d's 240-cell sweep spanning three opacity laws and sixteen decades of optical depth the solve needs about 10 evaluations of the equation per cell without dust and about 14 with dust, and fails on none; `RadCouplingUnitTests` reproduces that sweep and the dust sweep.

Each evaluation costs one pass over the groups (one Planck integral per group) and no derivative, no matrix and no linear solve. `radiation.print_iteration_counts` reports the mean and maximum number of evaluations per solve.

## Equivalence with the previous implementation (single-group)

Before this refactor the code used a `gas_update_factor = IMEX_a32 = 0.5` trick: stage 2 applied only half the gas update, avoiding the need to store U^(2). The new implementation applies the full gas update at stage 2, then applies the Shu-Osher combination `0.5*gas_n + 0.5*gas_stage2` before stage 3's implicit solve. The starting point for the stage 3 implicit solve is identical in both cases, so for single-group radiation the numerical results are algebraically equivalent. For multi-group radiation the old `gas_update_factor` also entered the work-term iteration inside `UpdateFlux`; the new implementation with `gas_update_factor = 1.0` is the mathematically correct IMEX formulation.