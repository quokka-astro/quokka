## Physics & Algorithms

This section is a target specification for the planned reimplementation of Quokka's radiation-matter coupling module. It documents the local source-term model and the intended nonlinear solve structure, not the full transport integrator. The goal is to define a clean physics interface that can be modularized without changing the underlying coupling model.

## Model scope and assumptions

- Quokka is a finite-volume, grid-based code.
- Hydrodynamic and radiation variables are cell-centered. MHD uses a staggered representation and is out of scope here.
- The radiation subsystem evolves group-wise radiation energy density `Erad[n]` and radiation flux `Frad[n]`. This section focuses on local matter coupling; the hyperbolic M1 transport update is intentionally not re-derived here.
- The gas is compressible and described by density `rho`, momentum `rho v`, total energy `E`, and internal energy `Egas`. Gas thermodynamics are closed by the ideal-gas EOS with mean molecular weight `mu` and adiabatic index `gamma`.
- Dust is not an explicit conserved fluid in this module. Instead, dust enters only through an auxiliary dust temperature `Td`, its thermal emission, and dust-mediated opacities.
- The separate Eulerian dust dynamics module is not part of this design.
- Opacities are frequency-averaged and group-dependent: `kappaP[n]`, `kappaE[n]`, and `kappaF[n]`. In code-native notation we also use

```
xidB[n] = rho * kappaP[n]
xidE[n] = rho * kappaE[n]
xidF[n] = rho * kappaF[n]
```

- The reduced speed of light `chat` appears in the radiation update, while `c` is the physical speed of light used in the gas-radiation energy bookkeeping.
- Additive non-core source terms are part of the coupling interface: stellar radiation injection through `Src[n]`, gas cooling `Lambda(n_H, T)`, and gas heating `Gamma(n_H, T)`.

## Continuous local coupling model

For each radiation group `n`, define the group-integrated thermal source term

```
c G[n] = c * (xidE[n] * Erad[n] - xidB[n] * planck[n])
```

where

```
planck[n] = (4 pi / c) * B_n(Td)
```

is the Planck source integrated over group `n` and evaluated at the dust temperature `Td`.

The local thermal exchange model is

```
d/dt Egas    =  sum_n c G[n]
d/dt Erad[n] = -chat * G[n]
```

so that gas internal energy and radiation energy exchange consistently in the reduced-speed-of-light formulation.

This specification treats the thermal part of the coupling as the canonical nonlinear solve. Radiation flux `Frad[n]` and gas momentum exchange are updated by a separate but coupled submodule that uses the same opacities and the same converged temperatures; in practice this submodule also provides the lagged work contribution that appears in the radiation energy residual.

## Dust closure

Dust is assumed to be in instantaneous thermal equilibrium with the local gas-radiation exchange model. Define the radiative cooling of dust in group `n` as

```
Lambda_gd[n] = -c G[n]
             = -c * (xidE[n] * Erad[n] - xidB[n] * planck[n])
```

The net gas-dust thermal exchange is modeled as

```
Lambda_gd = Theta_gd * n_H^2 * sqrt(T) * (T - Td)
```

with `T` the gas temperature and `n_H` the hydrogen number density. The dust temperature is defined by the balance

```
sum_n Lambda_gd[n] = Lambda_gd
```

This makes `Td` an auxiliary thermodynamic variable: it is part of the physics model, but not necessarily a primary unknown of the nonlinear solve.

### Gas-only limit

The gas-only case is treated as the constrained specialization

```
Td = T
```

evaluated before computing `planck[n]`, `kappaP[n]`, `kappaE[n]`, and `kappaF[n]`.

This is not the same as taking the imperfect-coupling closure above and then setting `T = Td` afterward. In the gas-only model there is no separate dust balance equation; rather, the auxiliary dust temperature is identified with the gas temperature by construction.

## Backward-Euler thermal update

Over one implicit source step of duration `dt`, define

```
R[n] = -chat * G[n] * dt
     = (chat / c) * Lambda_gd[n] * dt
```

Then the Backward-Euler update for the thermal subsystem is

```
Egas    - Egas0    + (c / chat) * sum_n R[n] = 0
Erad[n] - Erad0[n] - R[n]                    = 0
```

where `Egas0` and `Erad0[n]` denote the state at the beginning of the source step.

The practical source update includes additional additive terms:

```
extra_src[n] = work[n] + Src[n]
```

where

- `Src[n]` is the cell-centered stellar radiation source passed in through `radEnergySource`, and
- `work[n]` is the lagged velocity-dependent work contribution. In the algorithm this term is held fixed during the inner Newton iteration and updated by an outer iteration loop.

With these additive terms included, the residuals used by the source update are

```
Egas    - Egas0    + (c / chat) * sum_n R[n] + dt * Lambda(n_H, T) - dt * Gamma(n_H, T) = 0
Erad[n] - Erad0[n] - (R[n] + extra_src[n])                                  = 0
```

The gas cooling/heating terms above represent the non-core thermochemical hooks admitted by the coupling interface. They are evaluated in the gas energy equation, not in the definition of `R[n]`.

## Reduced-basis formulation

The reimplementation should retain a reduced-basis nonlinear solve with primary unknowns

```
(Egas, R[0], R[1], ..., R[N_groups-1])
```

rather than solving directly in `(Egas, Erad[n])`. The main reasons are:

- `Td` is recovered algebraically once `Egas` and `R[n]` are known.
- The Jacobian has a simple "first row + first column + diagonal" structure that admits a small specialized linear solve.
- The same formulation works for both the canonical gas-dust model and the constrained gas-only model.

For convenience define

```
Nd    = (chat / c) * Theta_gd * n_H^2 * dt
tau[n] = chat * dt * xidB[n]
X[n]   = xidB[n] / xidE[n] = kappaP[n] / kappaE[n]
```

In the canonical gas-dust model the dust balance implies

```
sum_n R[n] = Nd * sqrt(T) * (T - Td)
```

so the dust temperature can be reconstructed as

```
Td = T - sum_n R[n] / (Nd * sqrt(T))
```

Once `Td` is known, the radiation energy in each group can be recovered from

```
Erad[n] = X[n] * (planck[n] - R[n] / tau[n])
```

This is the clean thermal form of the reduced-basis update. The reimplementation should store `R[n]` directly and should not retain the older scaled variable `D[n] = R[n] / tau0[n]`.

## Newton solve for the thermal subsystem

The nonlinear solve is a Backward-Euler Newton-Raphson iteration on `(Egas, R[n])`.

Define the residuals

```
F[0] = Egas - Egas0 + (c / chat) * sum_n R[n] + dt * Lambda(n_H, T) - dt * Gamma(n_H, T)
F[n] = Erad[n] - Erad0[n] - (R[n] + extra_src[n])
```

with `extra_src[n] = work[n] + Src[n]`, and with `Erad[n]` understood as the derived quantity above. Let

```
Cv = dEgas / dT
```

at fixed density and composition.

For the canonical gas-dust model, a convenient approximate Jacobian is

```
J[0][0] = 1
J[0][n] = c / chat
J[n][0] = (1 / Cv) * X[n] * (d planck[n] / dTd) * (dTd / dT)
J[n][n] = -X[n] / tau[n] + X[n] * (d planck[n] / dTd) * (dTd / dR[n]) - 1
```

with

```
dTd / dT    = 3/2 - Td / (2 T)
dTd / dR[n] = -1 / (Nd * sqrt(T))
```

The Jacobian linearization assumes that temperature derivatives of the opacities are neglected inside the Newton matrix, i.e. `d/dT kappaP[n] = 0` and `d/dT kappaE[n] = 0` for the purpose of linearization. Equivalently, the temperature derivative of `kappaP[n] / kappaE[n]` is ignored. This should be stated explicitly in inline implementation comments because it affects convergence behavior, but not the converged fixed point when the nonlinear solve succeeds.

### Gas-only Jacobian

In the constrained gas-only specialization,

```
Td = T
dTd / dT = 1
dTd / dR[n] = 0
```

so the Jacobian simplifies to

```
J[0][0] = 1
J[0][n] = c / chat
J[n][0] = (1 / Cv) * X[n] * (d planck[n] / dTd)
J[n][n] = -X[n] / tau[n] - 1
```

## Radiation flux and momentum update

The new module should preserve the current optimized treatment of radiation flux relaxation and gas momentum exchange, but isolate it behind a separate interface. This part of the algorithm is not just a post-processing step: it participates in the outer fixed-point iteration through the lagged work term that enters `extra_src[n]`.

The intended sequencing is

1. Form the old-state quantities needed by the source step: `Egas0`, `Erad0[n]`, gas momentum, `Frad0[n]`, and any injected stellar source `Src[n]`.
2. Start an outer iteration loop. The purpose of this loop is to lag the work term, solve the thermal subsystem with that work term held fixed, update `Frad[n]` and gas momentum, then re-evaluate whether the lagged work term is self-consistent.
3. Inside each outer iteration, run an inner Newton-Raphson solve on the reduced basis `(Egas, R[n])`.
4. After the inner solve converges, update `Frad[n]` and gas momentum using the preserved optimized flux-relaxation formulas and the converged thermodynamic state.
5. Recompute the work term and check outer-loop convergence. If the work term has not converged, repeat from step 3 with the updated lagged work.
6. Once the outer loop converges, commit `Erad[n]`, `Frad[n]`, gas momentum, gas internal energy, and gas total energy back to the cell state.

### Inner Newton iteration

The inner iteration solves the thermal residual system with `extra_src[n]` treated as fixed. In particular:

- `Src[n]` is constant throughout the source step.
- `work[n]` is also held fixed during one inner Newton solve.
- `Td`, `planck[n]`, `kappaP[n]`, and `kappaE[n]` are updated every Newton iteration from the current iterate.
- `kappaF[n]` is evaluated when needed for the flux/momentum update and for the work term.

This separation is essential for robustness: the thermal solve sees a fixed additive source term, while the outer loop absorbs the nonlinearity associated with the work correction.

### Outer lagged-work iteration

The outer iteration preserves the current implementation strategy:

- On the first outer iteration, compute `work[n]` from the old-state gas momentum and radiation flux.
- Pass that lagged `work[n]` into the inner thermal solve through `extra_src[n]`.
- Use the converged thermal state to update `Frad[n]` and gas momentum.
- Recompute the work term implied by that updated state.
- Compare the new work term against the previous lagged value. If the difference exceeds the lagged-work tolerance, continue outer iteration; otherwise accept the source update.

This separation is important: the thermal solve defines the matter-radiation energy exchange model, while the `Frad` update and work-lag iteration are algorithmic submodules that should be preserved and modularized, not re-derived from scratch in this document.

### Weak-coupling fallback

The reimplementation should preserve the current decoupled-dust fallback path for very small `Theta_gd`. In that regime the fully coupled gas-dust thermal solve becomes numerically pathological because the dust balance can become effectively singular relative to the gas energy scale. The current threshold-based fallback should be kept as a deliberate part of the design rather than treated as a temporary workaround.

### Floors and clipping policy

The source update should enforce positivity floors during iteration, not only after convergence:

- Radiation energy floor: `RadSystem_Traits::Erad_floor`
- Gas internal-energy floor: `Cv * tempFloor_`
- Dust temperature floor: `dustTempFloor_`

`dustTempFloor_` should be added to `AMRSimulation` and should default to `tempFloor_`.

Allowing clipping during iteration is intentional. Negative or zero `Erad`, `Egas`, or `Td` leads to undefined or nonphysical opacity and emission evaluations, so waiting until after convergence is not acceptable for a robust implementation.

## Module boundaries for the reimplementation

The reimplemented coupling module should be split into the following responsibilities:

- `OpacityEvaluator`: evaluate `kappaP[n]`, `kappaE[n]`, and `kappaF[n]` at a supplied `Td`.
- `PlanckIntegrator`: compute `planck[n]` and its temperature derivative group-by-group.
- `DustClosure`: recover `Td(T, R[n])` for the canonical gas-dust model, or enforce `Td = T` for the gas-only specialization.
- `ThermalCouplingSolve`: assemble residuals/Jacobian in the reduced basis and carry out the Newton solve.
- `FluxMomentumUpdate`: apply the preserved `Frad` relaxation and gas momentum update using the converged thermodynamic state.
- `CouplingDriver`: orchestrate the inner thermal solve, outer lagged-work iteration, floors/clipping policy, and final per-cell state update.

## Implementation decisions now fixed

- `R[n]` remains the pure thermal exchange variable. The work term is included separately through `extra_src[n] = work[n] + Src[n]`.
- The reduced-basis implementation stores `R[n]` directly; `D[n]` is dropped and `use_D_as_base = false` is assumed.
- The weak-coupling decoupled-dust fallback is preserved.
- The Newton Jacobian continues to neglect temperature derivatives of `kappaP[n]` and `kappaE[n]`, and this should be documented explicitly in inline comments.
- The coupling interface admits additive non-core source terms: stellar radiation `Src[n]`, gas cooling `Lambda(n_H, T)`, and gas heating `Gamma(n_H, T)`.
- Floors are enforced during iteration: `Erad_floor`, `Cv * tempFloor_`, and `dustTempFloor_`, with `dustTempFloor_` defaulting to `tempFloor_`.

