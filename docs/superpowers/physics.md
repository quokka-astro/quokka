## Physics & Algorithms

### Scope

This section describes the local radiation-gas-dust source update used by Quokka's radiation module. It is intended for planning and implementation work, not as a line-by-line code reference.

Relevant context:

- Quokka is a finite-volume, grid-based AMR code.
- Hydrodynamic and radiation state variables are cell-centered.
- MHD variables are staggered and are not part of the coupling update described here.
- Radiation is evolved with an M1 two-moment scheme, with group-wise radiation energy density `E_r,g` and radiation flux `F_r,g`.
- Gas is evolved as compressible hydrodynamics with density `rho`, momentum `rho v`, total energy `E`, and internal energy `e`.
- The gas equation of state is ideal, with mean molecular weight `mu` and adiabatic index `gamma`.
- In the radiation module, dust is not evolved as a separate fluid or particle population. Instead, dust only enters through its temperature `T_d`, which sets thermal emission and opacities.
- Quokka also contains a separate dust dynamics module, but that module is not coupled to radiation and is out of scope here.

### State Variables and Thermodynamics

For the thermal coupling step, the primary unknowns are:

- gas internal energy density `e`
- multigroup radiation energy densities `E_r,g`
- optionally, dust temperature `T_d` as an algebraic auxiliary quantity

Gas temperature `T_g` is obtained from the EOS, schematically

```text
T_g = T_g(rho, e, mu, gamma)
```

and the heat capacity entering the Newton solve is

```text
C_v = d e / d T_g .
```

The radiation module uses group-averaged opacities:

- `kappa_P,g`: Planck mean opacity
- `kappa_E,g`: energy mean opacity
- `kappa_F,g`: flux mean opacity

These are evaluated group by group and may depend on density, temperature, and the chosen opacity model.

### Governing Source Terms

The full RHD system contains transport, pressure, and work terms. The stiff local source update described here focuses on thermal exchange between gas and radiation, plus the associated damping of radiation flux.

For each radiation group `g`, the thermal source term is written as

```text
c G^0_g = rho c (kappa_E,g E_r,g - kappa_P,g B_g(T_d)) ,
```

where `B_g(T_d)` is the Planck spectrum integrated over radiation group `g` and written in energy-density units:

```text
B_g(T_d) = integral_{nu_{g-1}}^{nu_g} (4 pi / c) B_nu(T_d) dnu .
```

With this sign convention:

- `c G^0_g > 0` means radiation energy is absorbed by matter
- `-c G^0_g > 0` means dust emits net energy into radiation group `g`

Ignoring transport terms, the thermal exchange equations are

```text
d e / dt = sum_g c G^0_g
d E_r,g / dt = - c_hat G^0_g ,
```

where `c_hat <= c` is the reduced speed of light used in the radiation update.

In the full Quokka source step, the radiation energy equation may also include lagged or external terms such as:

- radiation energy source terms
- line cooling / heating contributions
- optional `v . F` work terms at `O(v/c)`
- optional photoelectric heating terms

Those terms matter for the implementation, but they are best treated as additive extensions to the core thermal coupling described here.

### Dust Model Used by the Radiation Module

Dust is not advanced with its own conservation law. Instead, it is represented by a temperature `T_d` that mediates emission and absorption.

For the gas-dust thermal exchange model, Quokka assumes that the net dust radiative loss is balanced by gas-dust heat exchange:

```text
sum_g (-c G^0_g) = Lambda_gd .
```

The gas-dust exchange law currently used is

```text
Lambda_gd = Theta_gd n_H^2 sqrt(T_g) (T_g - T_d) ,
```

where:

- `n_H` is the hydrogen number density
- `Theta_gd` is the gas-dust coupling coefficient

This relation implies:

- if `T_g > T_d`, gas heats dust and dust reradiates that energy
- if `T_d > T_g`, dust heats the gas

### Implicit Backward-Euler Update

For a timestep `Delta t`, Quokka solves the stiff thermal coupling implicitly. A convenient variable is

```text
R_g = - c_hat G^0_g Delta t ,
```

which is the radiation-energy increment transferred into group `g` during the source update.

The backward-Euler residuals for the simplified thermal system are

```text
F_gas = e^(n+1) - e^n + (c / c_hat) sum_g R_g
F_g   = E_r,g^(n+1) - E_r,g^n - R_g .
```

In the actual implementation, `F_gas` and `F_g` are augmented by optional cooling, heating, work, and source terms, but the Newton solve is built around the same basic variable choice.

### Why Solve in `(e, R_g)` Instead of `(e, E_r,g)`?

The code solves for `(e, R_g)` rather than directly for `(e, E_r,g)`. This has two practical advantages:

1. The gas-radiation energy exchange enters linearly in the conservation equations through `sum_g R_g`.
2. The dust temperature can be reconstructed algebraically from the gas-dust balance relation.

Define

```text
N_d = (c_hat / c) Theta_gd n_H^2 Delta t
tau_g = rho kappa_P,g c_hat Delta t
X_g = kappa_P,g / kappa_E,g .
```

Then the gas-dust balance becomes

```text
sum_g R_g = N_d sqrt(T_g) (T_g - T_d) ,
```

so

```text
T_d = T_g - sum_g R_g / (N_d sqrt(T_g)) .
```

Given `T_d`, the updated radiation energy in each group is

```text
E_r,g = X_g [ B_g(T_d) - R_g / tau_g ] ,
```

for `tau_g > 0`. This is just the implicit source equation rewritten in terms of `R_g`.

### Newton Iteration and Jacobian Structure

The nonlinear solve is performed with Newton-Raphson iteration. For the simplified thermal-only system, the Jacobian has the block structure

```text
J_00   = d F_gas / d e
J_0g   = d F_gas / d R_g
J_g0   = d F_g / d e
J_gg   = d F_g / d R_g .
```

With the `(e, R_g)` variables and the approximation

```text
d/dT_g (kappa_P,g / kappa_E,g) ~= 0 ,
```

the leading terms are

```text
J_00 = 1
J_0g = c / c_hat
J_g0 = (1 / C_v) X_g (d B_g / d T_d) (d T_d / d T_g)
J_gg = - X_g / tau_g + X_g (d B_g / d T_d) (d T_d / d R_g) - 1 .
```

From

```text
sum_g R_g = N_d sqrt(T_g) (T_g - T_d) ,
```

one obtains, at fixed `R_g`,

```text
d T_d / d T_g = 3/2 - T_d / (2 T_g) ,
```

and at fixed `T_g`,

```text
d T_d / d R_g = -1 / (N_d sqrt(T_g)) .
```

These are the derivatives that make the `(e, R_g)` basis attractive for the coupled gas-dust problem.

### Flux and Momentum Update

After the implicit energy solve, Quokka updates radiation fluxes and gas momentum using the flux-mean opacity `kappa_F,g`.

At lowest order, the source term behaves like a stiff damping term:

```text
F_r,g^(n+1) approx F_r,g^n / (1 + rho kappa_F,g c_hat Delta t) .
```

The gas receives the equal-and-opposite momentum impulse. At `O(v/c)`, the implementation also includes additional work and pressure-coupling terms.

This separation is important:

- energy exchange is solved implicitly through the Newton iteration
- momentum / flux coupling is updated afterward using the converged opacities and temperatures

### Important Special Cases

#### No explicit gas-dust thermal separation

If the dust-gas thermal coupling model is disabled, the radiation module effectively uses

```text
T_d = T_g .
```

Then

```text
d T_d / d T_g = 1
d T_d / d R_g = 0 ,
```

and the Jacobian reduces to the gas-radiation form

```text
J_00 = 1
J_0g = c / c_hat
J_g0 = (1 / C_v) X_g (d B_g / d T_g)
J_gg = - X_g / tau_g - 1 .
```

#### Weak gas-dust coupling fallback

The current implementation also contains a second regime for weak gas-dust coupling, where the code treats `T_d` as a separate nonlinear variable instead of enforcing the strong algebraic relation above. That branch is a solver detail of the current implementation and should be called out explicitly if this document is meant to be a full design spec.

### Ambiguities / Decisions Needed

The draft should make the following design choices explicit:

1. Scope of the document. Decide whether this section describes only the core thermal coupling solve, or the full production source step including line cooling, cosmic rays, `v . F` work terms, and photoelectric heating.
2. Dust model scope. Decide whether to document only the strong-coupling algebraic dust model, or also the weak-coupling fallback that exists in the current solver.
3. Variable naming. Use `E` for total gas energy and `e` for gas internal energy consistently throughout; the current draft mixes these.
4. Opacity notation. Use `kappa_P`, `kappa_E`, and `kappa_F` consistently. The earlier `xidB` / `xidE` notation obscures the physical meaning.
5. Reduced-speed-of-light convention. State clearly that gas updates use `c G^0` while radiation energy updates use `c_hat G^0`.
6. Dust interpretation. The phrase "dust is in thermal equilibrium between gas and radiation" is too vague. What is actually imposed is radiative equilibrium of dust plus a gas-dust heat-exchange closure, not a separate evolved dust energy equation.

