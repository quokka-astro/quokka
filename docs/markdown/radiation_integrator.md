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

Each implicit stage solves, cell by cell, one backward-Euler step of the energy exchange between the matter and the \\(N\_g\\) radiation groups. The group equations are solved directly and the step reduces to **one scalar equation**, which is bracketed and solved with Brent's method. The method follows the hydro3d.jl reference implementation; the code is `radiation_coupling.hpp`.

### Reduction to one equation

The step, of length \\(\Delta t\\) = `dt_implicit`, covers the [energy exchange with the matter](radiation_hydrodynamics.md#energy-exchange-with-the-matter), for the two dust models defined there. The external radiation source \\(S\_g\\) and the \\(v/c\\) work term \\(W\_g\\), which is lagged across the outer iteration, are held fixed over the step and folded into the old state,

<script type="math/tex; mode=display">
\mathrm{rad0}_g = E_g^n + S_g + W_g \, , \qquad \mathrm{gas0} = E_{\rm gas}^n - \frac{c}{\hat c} \sum_g W_g \, ,
</script>

and the net emission is written as \\(Q\_g = c \\, (\varepsilon\_g - \alpha\_g E\_g)\\), with the emission and absorption coefficients \\(\varepsilon\_g = 4 \pi \chi\_{0B,g} B\_g / c\\) and \\(\alpha\_g = \chi\_{0E,g}\\) (\\(\chi\_{0P}\\) and \\(\chi\_{0E}\\) for one group; \\(\chi = \rho \kappa\\)) evaluated at the matter temperature. On the single-group `beta_order >= 2` path, \\(\Delta t \\, \hat c\\) is multiplied by the Lorentz factor.

**One-temperature model.** The unknowns are \\(E\_{\rm gas}\\) and the \\(N\_g\\) group energies; the gas temperature \\(T\\) follows from \\(E\_{\rm gas}\\) through the equation of state. The backward-Euler step is

<script type="math/tex; mode=display">
\begin{aligned}
E_g - \mathrm{rad0}_g &= \Delta t \, \hat c \left[ \varepsilon_g(T) - \alpha_g(T) \, E_g \right] , \qquad g = 1 \ldots N_g \, , \\[4pt]
E_{\rm gas} - \mathrm{gas0} &= - \frac{c}{\hat c} \sum_g \left( E_g - \mathrm{rad0}_g \right) .
\end{aligned}
</script>

No group appears in the equation of another group: the groups meet only through \\(T\\). (A term that moved energy between groups directly would break this and need a different solver.) At a fixed \\(T\\), each group equation is therefore linear in its own \\(E\_g\\) and is solved directly,

<script type="math/tex; mode=display">
E_g(T) = \frac{\mathrm{rad0}_g + \Delta t \, \hat c \, \varepsilon_g(T)}{1 + \Delta t \, \hat c \, \alpha_g(T)} \, ,
</script>

a weighted mean of the energy the group starts from and its equilibrium value \\(\varepsilon\_g / \alpha\_g\\), with weights \\(1\\) and \\(\Delta t \\, \hat c \\, \alpha\_g\\), the optical depth of the step. The stiffness of an optically thick group, \\(\Delta t \\, \hat c \\, \alpha\_g \gg 1\\), is handled exactly by this formula, and a transparent group keeps \\(\mathrm{rad0}\_g\\). Substituting \\(E\_g(T(E\_{\rm gas}))\\) into the gas equation leaves one equation in one unknown,

<script type="math/tex; mode=display">
G(E_{\rm gas}) \equiv E_{\rm gas} - \mathrm{gas0} + \frac{c}{\hat c} \sum_g \left[ E_g\big(T(E_{\rm gas})\big) - \mathrm{rad0}_g \right] = 0 \, ,
</script>

which states that the total energy \\(E\_{\rm gas} + (c / \hat c) \sum\_g E\_g\\) is the same after the step as before it.

**Two-temperature model.** The unknowns are \\(E\_{\rm gas}\\), the group energies, and \\(T\_d\\). With the collisional rate \\(\Lambda\_{\rm gd}(T, T\_d) = K \\, T^{1/2} (T - T\_d)\\), \\(K = k\_{\rm gd} \\, n\_{\rm H}^2\\) and \\(k\_{\rm gd}\\) the input `radiation.dust_gas_interaction_coeff`, the backward-Euler step consists of the group equations, the dust energy balance, and the gas equation,

<script type="math/tex; mode=display">
\begin{aligned}
E_g - \mathrm{rad0}_g &= \Delta t \, \hat c \left[ \varepsilon_g(T_d) - \alpha_g(T_d) \, E_g \right] , \qquad g = 1 \ldots N_g \, , \\[4pt]
\frac{c}{\hat c} \sum_g \left( E_g - \mathrm{rad0}_g \right) &= \Delta t \, \Lambda_{\rm gd}(T, T_d) \, , \\[4pt]
E_{\rm gas} - \mathrm{gas0} &= - \Delta t \, \Lambda_{\rm gd}(T, T_d) \, .
\end{aligned}
</script>

Now the groups meet only through \\(T\_d\\), so at a fixed \\(T\_d\\) the group equations are again linear and give \\(E\_g(T\_d)\\) by the formula above with \\(T\\) replaced by \\(T\_d\\). Adding the last two equations gives the gas energy directly as well, from energy conservation,

<script type="math/tex; mode=display">
E_{\rm gas}(T_d) = \mathrm{gas0} - \frac{c}{\hat c} \sum_g \left[ E_g(T_d) - \mathrm{rad0}_g \right] ,
</script>

and with it the gas temperature \\(T(T\_d)\\). What is left is the dust energy balance, one equation in one unknown,

<script type="math/tex; mode=display">
H(T_d) \equiv \frac{c}{\hat c} \sum_g \left[ E_g(T_d) - \mathrm{rad0}_g \right] - \Delta t \, \Lambda_{\rm gd}\big(T(T_d), T_d\big) = 0 \, .
</script>

Total energy is conserved at every trial \\(T\_d\\), not only at the root. At \\(K = 0\\) the gas energy stays at \\(\mathrm{gas0}\\) and \\(H = 0\\) is the radiative equilibrium of the dust; as \\(K \to \infty\\), \\(T\_d \to T\\) and \\(H = 0\\) becomes \\(G = 0\\) of the one-temperature model.

### Comparison with the Newton-Raphson iteration

The inner solve differs from the Newton-Raphson iteration of [@Howell_2003], used in [@Wibking_2022], [@He_2024], and [@He_2024b], which iterates on all \\(1 + N\_g\\) energy variables and tests convergence on the residuals of their equations:

- **The group energies are eliminated exactly** rather than iterated on. A Newton step linearises them about the current iterate, and in an optically thick cell far from equilibrium the first step, taken about the starting group energies, can point away from the root.
- **The root is bracketed before it is refined**, so the solve cannot diverge. Marching from the old state also selects the root connected to it when a steep opacity law gives the step more than one.
- **Convergence is judged on the unknown, not on a residual.** A group residual of the form \\(\tau\_g (4 \pi B\_g / c - E\_g)\\), with \\(\tau\_g\\) the optical depth of the step, carries a round-off error of about \\(\epsilon \\, \tau\_g\\) times the cell's energy (\\(\epsilon\\) is the machine epsilon), so a residual test at \\(10^{-11}\\) cannot be met once \\(\tau\_g \gtrsim 10^5\\). A relative tolerance on the gas energy means the same at every optical depth.
- **No Jacobian and no linear solve.** Each evaluation of the scalar equation costs one Planck integral per group.

### Bracket, root finder, tolerance

Both equations increase with their unknown except where a steep opacity law makes them non-monotone, in which case the step can have several roots. The bracket is therefore built by **marching outward from the old state** (\\(E\_{\rm gas}^0\\), or the start-of-step gas temperature for \\(T\_d\\)) by factors of two in the direction the sign of the residual indicates, which isolates the root continuously connected to where the cell started. The march in \\(E\_{\rm gas}\\) never goes below \\(E\_{\rm min} = E\_{\rm int}(\rho, T\_{\rm floor})\\): if it reaches \\(E\_{\rm min}\\) without a sign change, the root lies below the admissible range, the gas is clamped to the floor and the groups take the closed form at \\(T\_{\rm floor}\\), the cell counts as converged, and the energy \\(G(E\_{\rm min}) > 0\\) is created by the temperature floor, as any floor does, with no threshold on that amount. With a zero temperature floor (every problem in `UnitSystem::CONSTANTS`) \\(E\_{\rm min} = 0\\), and the march instead floors at round-off of the initial gas energy, \\(16 \\, \epsilon \\, E\_{\rm gas}^0\\) with \\(\epsilon\\) the machine epsilon, so that it still ends; probing exactly \\(E = 0\\) would evaluate the Planck function at \\(T = 0\\). This is routine, not pathological: a transparent, radiation-dominated cell whose gas sits at the floor hits this every step. The march in \\(T\_d\\) has no floor: \\(T\_{\rm floor}\\) is a floor on the gas, and dust in a weak field is colder than it, so as \\(T\_d \to 0\\) the emission vanishes and \\(H\\) always finds its sign change. Only an upward march that exhausts its 200 doublings, or a march that meets a non-finite residual, is reported unconverged, and the run aborts.

The root is found with `quokka::math::brent_solve` (Brent's method with a minimum step; see `bracketing_root_finding.hpp`), which stops when the bracket is narrow relative to the unknown: \\(|hi - lo| \le \mathrm{tol} \\, \min(|lo|, |hi|)\\) with `tol` the input `radiation.iteration_tolerance`. For an ideal gas the unknown \\(E\_{\rm gas} = c\_V T\\) makes this a relative tolerance on the gas temperature. With dust the promise is on the gas energy across Brent's final bracket in \\(T\_d\\), \\(|E\_{\rm gas}(T\_d^-) - E\_{\rm gas}(T\_d^+)| \le \max(\mathrm{tol}\\,|E\_{\rm gas}|,\\ 4\varepsilon\\, e\_{\rm r})\\), with \\(e\_{\rm r} = |E\_{\rm gas}^0 - (c/\hat c)\sum\_g W\_g| + (c/\hat c)\sum\_g |E\_g - \mathrm{rad0}\_g|\\) the scale of the round-off in the gas energy, because at small \\(K\\) the gas energy follows from conservation and its error is the \\(T\_d\\) error times \\((c/\hat c)\sum\_g E\_g / E\_{\rm gas}\\) (about \\(10^5\\) in `DTypeFront1D`); if the test fails, Brent is repeated on its own final bracket with the \\(T\_d\\) tolerance set from the measured slope, at most three times, and a bracket at the floating-point resolution of \\(T\_d\\) counts as converged. The state is evaluated where the chord through the ends of Brent's final bracket crosses zero (`secant_point`), not at an end or the midpoint. Where Brent stops inside its tolerance window depends on the path of the iteration, which a round-off change of the inputs can alter, so a state taken at an end or the midpoint would jump by up to the tolerance in response to round-off: physically identical cells, such as the two mirror halves of a symmetric problem, would end the step different at the level of the tolerance. The chord crossing is within second order in the bracket width of the root, so the state changes only by round-off when the inputs do. With dust, the gas energy at that point is then taken either from conservation or from the gas equation, \\(E\_{\rm gas}^0 - (c/\hat c)\sum\_g W\_g - \Delta t \\, K T^{1/2} (T - T\_d)\\), whichever has the smaller estimated round-off; at \\(K = 0\\) this leaves the gas energy exactly unchanged. Convergence is judged on the unknown, never on the residual: \\(H\\) in particular multiplies the round-off in \\(T - T\_d\\) by \\(\Delta t K T^{1/2}\\) and cannot be tested directly. The conservation error of a step is then at most about \\(\mathrm{d}G/\mathrm{d}E\_{\rm gas}\\) times the tolerance times the gas energy. Over a 240-cell sweep taken from hydro3d.jl, spanning three opacity laws and sixteen decades of optical depth the solve needs about 10 evaluations of the equation per cell without dust and about 14 with dust, and fails on none; `RadCouplingUnitTests` reproduces that sweep and the dust sweep. `radiation.print_iteration_counts` reports the mean and maximum number of evaluations per solve.

## Equivalence with the previous implementation (single-group)

Before this refactor the code used a `gas_update_factor = IMEX_a32 = 0.5` trick: stage 2 applied only half the gas update, avoiding the need to store U^(2). The new implementation applies the full gas update at stage 2, then applies the Shu-Osher combination `0.5*gas_n + 0.5*gas_stage2` before stage 3's implicit solve. The starting point for the stage 3 implicit solve is identical in both cases, so for single-group radiation the numerical results are algebraically equivalent. For multi-group radiation the old `gas_update_factor` also entered the work-term iteration inside `UpdateFlux`; the new implementation with `gas_update_factor = 1.0` is the mathematically correct IMEX formulation.