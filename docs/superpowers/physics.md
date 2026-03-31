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

This specification treats the thermal part of the coupling as the canonical nonlinear solve. Radiation flux `Frad[n]` and gas momentum exchange are updated by a separate submodule that uses the same opacities and the same converged temperatures, but is not part of the core thermal residual definition below.

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

If an external cell-centered radiation energy source `Src[n]` is injected during the same implicit step, the radiation residual becomes

```
Erad[n] - Erad0[n] - (R[n] + Src[n]) = 0
```

but `Src[n]` is conceptually an external hook, not part of the core matter-coupling physics.

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

This is the clean thermal form of the reduced-basis update. Any retained velocity-dependent work correction should be treated as a separate algorithmic layer on top of this definition, not folded into the physics meaning of `R[n]`.

## Newton solve for the thermal subsystem

The nonlinear solve is a Backward-Euler Newton-Raphson iteration on `(Egas, R[n])`.

Define the residuals

```
F[0] = Egas - Egas0 + (c / chat) * sum_n R[n]
F[n] = Erad[n] - Erad0[n] - R[n]
```

with `Erad[n]` understood as the derived quantity above. Let

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

The Jacobian linearization assumes that `d/dT (kappaE[n]` and `d/dT (kappaP[n] / kappaE[n])` are neglected inside the Newton matrix. This should be stated explicitly in the implementation because it affects convergence behavior, but not the converged fixed point when the nonlinear solve succeeds.

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

The new module should preserve the current optimized treatment of radiation flux relaxation and gas momentum exchange, but isolate it behind a separate interface.

The intended sequencing is

1. Solve the thermal subsystem for `Egas`, `Erad[n]`, `T`, and `Td`.
2. Re-evaluate the opacities needed by the flux update, especially `kappaF[n]`, at the converged `Td`.
3. Apply the retained `Frad[n]` update and gas momentum exchange using the existing optimized formulas.
4. Reconstruct gas total energy after momentum has been updated.

This separation is important: the thermal solve defines the matter-radiation energy exchange model, while the `Frad` update is an algorithmic submodule that should be preserved and modularized, not re-derived in this document.

## Module boundaries for the reimplementation

The reimplemented coupling module should be split into the following responsibilities:

- `OpacityEvaluator`: evaluate `kappaP[n]`, `kappaE[n]`, and `kappaF[n]` at a supplied `Td`.
- `PlanckIntegrator`: compute `planck[n]` and its temperature derivative group-by-group.
- `DustClosure`: recover `Td(T, R[n])` for the canonical gas-dust model, or enforce `Td = T` for the gas-only specialization.
- `ThermalCouplingSolve`: assemble residuals/Jacobian in the reduced basis and carry out the Newton solve.
- `FluxMomentumUpdate`: apply the preserved `Frad` relaxation and gas momentum update using the converged thermodynamic state.
- `CouplingDriver`: orchestrate the two stages above and expose a single per-cell source-update interface.

## Ambiguities and decisions to resolve before implementation

- Meaning of `R[n]`: keep it as the pure thermal exchange variable defined above, or continue to fold lagged work corrections into the same reduced variable. For modularity, the recommended choice is to keep `R[n]` purely thermal.
- Reduced basis exposed to code: store `R[n]` directly, or use the scaled variable `D[n] = R[n] / tau0[n]` internally for conditioning while keeping `R[n]` as the documented physics variable.
- Weak gas-dust coupling: decide whether to preserve the current decoupled-dust fallback path for very small `Theta_gd`, or require the canonical coupled model everywhere and handle stiffness purely through solver robustness.
- Opacity derivatives in the Jacobian: decide whether the new implementation should continue neglecting `d/dT (kappaP / kappaE)` in the Newton matrix, or optionally include it for difficult opacity laws.
- External hooks: decide exactly which non-core source terms are admitted through the coupling driver interface. The recommended core set is radiation absorption/emission, radiation momentum exchange, and dust-gas thermal exchange only.
- Floors and failure policy: define where positivity floors on `Erad`, `Egas`, and `Td` are enforced, and whether the thermal solver may clip during iteration or only after convergence checks.

