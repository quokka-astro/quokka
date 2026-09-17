# Radiation Hydrodynamics

> **Warning: Beta feature**
>
> The radiation module has been verified against the test problems in [@Wibking_2022], [@He_2024], and [@He_2024b], but is still marked **beta** for science-use maturity. Please cite the relevant methods paper and record the exact commit hash used.
>

Quokka solves the equations of radiation hydrodynamics (RHD) with a two-moment (M1) method in the mixed-frame formulation, accurate to first order in \\(v/c\\). Radiation may be grey (a single frequency-integrated group) or multigroup, and the solver is *asymptotic-preserving*: it recovers the correct diffusion solution even when the photon mean free path is far smaller than a cell. This page summarises what is solved, how, and how to set it up. The stage-by-stage structure of the time integrator is documented separately in the [Radiation Integrator](radiation_integrator.md) page, and the full conservation-law system including MHD, dust, and gravity in [Equations](equations.md).

## Equations solved

### The RHD system

The radiation quantities are the lab-frame group-integrated moments of the specific intensity,

<script type="math/tex; mode=display">
E_g = \int_{\nu_{g-}}^{\nu_{g+}} \frac{1}{c} \oint I(\boldsymbol{n}, \nu) \, d\Omega \, d\nu \, , \quad
F_g^i = \int_{\nu_{g-}}^{\nu_{g+}} \oint n^i I(\boldsymbol{n}, \nu) \, d\Omega \, d\nu \, , \quad
P_g^{ij} = \int_{\nu_{g-}}^{\nu_{g+}} \frac{1}{c} \oint n^i n^j I(\boldsymbol{n}, \nu) \, d\Omega \, d\nu \, ,
</script>

for \\(g = 1 \ldots N_g\\), where \\(\nu\_{g-}\\) and \\(\nu\_{g+}\\) are the lower and upper frequency edges of group \\(g\\). Together with the gas variables they obey the conservative system

<script type="math/tex; mode=display">
\frac{\partial}{\partial t}
\left[\begin{array}{c} \rho \\ \rho \boldsymbol{v} \\ E_{\rm gas} \\ E_g \\ c^{-2} \boldsymbol{F}_g \end{array}\right]
+ \nabla \cdot
\left[\begin{array}{c} \rho \boldsymbol{v} \\ \rho \boldsymbol{v} \otimes \boldsymbol{v} + p \\ (E_{\rm gas} + p) \boldsymbol{v} \\ \boldsymbol{F}_g \\ \mathsf{P}_g \end{array}\right]
=
\left[\begin{array}{c} 0 \\ \sum_g \boldsymbol{G}_g \\ c \sum_g G^0_g \\ - c\, G^0_g \\ - \boldsymbol{G}_g \end{array}\right] ,
</script>

where \\((c G^0\_g, \boldsymbol{G}\_g)\\) is the radiation four-force of group \\(g\\). Writing the equations in the lab frame makes them manifestly conservative, so total energy and momentum are conserved to machine precision.

The system is closed with the [@Levermore_1984] M1 closure, which expresses \\(\mathsf{P}\_g\\) in terms of \\(E\_g\\) and \\(\boldsymbol{F}\_g\\) through the reduced flux \\(f = |\boldsymbol{F}\_g| / (c E\_g)\\). In multigroup runs the closure is applied group by group.

### Matter-radiation coupling

The four-force is written in the *mixed-frame* form: the opacities and emissivities are evaluated in the comoving frame, where they are isotropic and simple, while all radiation moments stay in the lab frame. Assuming the gas is in local thermodynamic equilibrium and neglecting scattering, the frequency-integrated (grey) four-force to order \\(v/c\\) is

<script type="math/tex; mode=display">
\begin{aligned}
- c G^0 &= 4 \pi \chi_{0P} B - c \chi_{0E} E + c^{-1} (2 \chi_{0E} - \chi_{0F}) v^i F^i \, , \\
- G^i &= - c^{-1} \chi_{0F} F^i + 4 \pi c^{-2} v^i \chi_{0P} B + c^{-1} \chi_{0F} v^j P^{ji} + c^{-1} (\chi_{0F} - \chi_{0E}) v^i E \, ,
\end{aligned}
</script>

where \\(\chi\_{0P}\\), \\(\chi\_{0E}\\), and \\(\chi\_{0F}\\) are the comoving-frame Planck-, energy-, and flux-mean absorption coefficients and \\(B\\) is the Planck function at the gas temperature. Quokka works with *mass* opacities \\(\kappa = \chi / \rho\\) in \\(\mathrm{cm^2\\,g^{-1}}\\), which is what a problem generator supplies. The leading terms are the familiar emission, absorption, and radiation force; the terms in \\(v/c\\) carry the work done by the radiation force on the gas and the frame-transformation ("frame-dragging") effects that become order unity in the dynamic diffusion regime. Terms of order \\(v^2/c^2\\) and higher are not documented here; the single-group solver can optionally include them (see `beta_order` below).

For multigroup the same expressions are integrated over each group. The result contains one term with no grey counterpart,

<script type="math/tex; mode=display">
- G_g^i \supset - \frac{4 \pi}{3 c^{2}} v^i \, \Delta_g (\nu \chi_0 B_{\nu}) \, , \qquad \Delta_g(Q) \equiv Q(\nu_{g+}) - Q(\nu_{g-}) \, ,
</script>

which redistributes photons between groups. Physically, a moving emitter Doppler-shifts its own thermal emission, and the shift changes the distribution of momentum over frequency differently from the distribution of energy. These terms cancel when summed over all groups, provided the frequency grid is wide enough that \\(\nu B\_\nu\\) is negligible at both edges, so the grey limit is recovered exactly.

### Reduced speed of light

To relax the radiation timestep, the radiation subsystem may be solved with a reduced speed of light \\(\hat{c} < c\\) (the RSLA), set through `c_hat_over_c`. This scales the transport term by \\(\hat{c}/c\\) and leaves the equations exact when \\(\hat{c} = c\\) (the default). \\(\hat{c}\\) must remain much larger than every hydrodynamic speed in the problem. Energy and momentum are conserved to machine precision only for \\(\hat{c} = c\\).

## Numerical method

A full timestep is operator-split into two parts.

1. **Hydrodynamic transport**, advanced explicitly with the PPM + RK2-SSP Godunov scheme described in [@Wibking_2022].
2. **Radiation transport and matter-radiation coupling**, advanced with an implicit-explicit (IMEX) scheme, subcycled with respect to the hydro step at `radiation.cfl`.

Within the radiation step, the transport terms \\(\nabla \cdot \boldsymbol{F}\_g\\) and \\(\nabla \cdot \mathsf{P}\_g\\) are treated **explicitly** and the four-force terms **implicitly**. This split is what keeps the method cheap on GPUs: the implicit part contains no spatial derivatives, so every cell is solved independently and the radiation update needs no more communication than a pure hydro update.

### IMEX PD-ARS

The two parts are combined with the asymptotic-preserving IMEX PD-ARS integrator, which advances the state \\(\boldsymbol{U}\\) over \\(\Delta t\\) in two stages,

<script type="math/tex; mode=display">
\begin{aligned}
\boldsymbol{U}^{(n+1/2)} &= \boldsymbol{U}^{(n)} + \Delta t \, \mathsf{T}(\boldsymbol{U}^{(n)}) + \Delta t \, \mathsf{S}(\boldsymbol{U}^{(n+1/2)}) \, , \\
\boldsymbol{U}^{(n+1)} &= \boldsymbol{U}^{(n)} + \frac{\Delta t}{2} \left[ \mathsf{T}(\boldsymbol{U}^{(n)}) + \mathsf{T}(\boldsymbol{U}^{(n+1/2)}) \right] + \frac{\Delta t}{2} \left[ \mathsf{S}(\boldsymbol{U}^{(n+1/2)}) + \mathsf{S}(\boldsymbol{U}^{(n+1)}) \right] ,
\end{aligned}
</script>

with \\(\mathsf{T}\\) the transport terms and \\(\mathsf{S}\\) the source terms. Transport and source terms enter symmetrically at both stages, which is what allows the near-exact cancellation between them that the diffusion limit requires. The scheme is second-order accurate and reduces to RK2-SSP in the streaming limit.

One practical consequence: because the scheme is asymptotic-preserving by construction, the *ad hoc* correction to the HLL wavespeeds that earlier explicit radiation schemes needed in order to recover the diffusion limit is no longer required. Quokka uses the uncorrected HLL fluxes with PPM reconstruction by default. The correction survives only as the diagnostic flag `use_wavespeed_correction_`, used by `RadMarshakAsymptotic` to demonstrate what it does.

### The implicit solve

Each implicit stage solves, cell by cell, a system of \\(4 + 4 N\_g\\) equations for the gas energy, the gas momentum, and the energy and flux of every group. It is split into two nested iterations:

- an **inner** Newton-Raphson iteration over the \\(1 + N\_g\\) energy variables (gas energy and the group exchange terms \\(R\_g\\)), with \\(\boldsymbol{v}\\) and \\(\boldsymbol{F}\_g\\) frozen;
- an **outer** iteration that updates \\(\boldsymbol{F}\_g\\) and the gas momentum analytically, then returns to the inner solve if the velocity-dependent terms have changed.

The inner Jacobian is sparse — groups couple to the gas but not directly to each other — so it is inverted by Gauss-Jordan elimination in \\(O(N\_g)\\) operations rather than \\(O(N\_g^3)\\). Outside the dynamic diffusion limit the outer loop almost always converges in one pass. The gas energy is recovered from the converged exchange terms rather than solved for independently, which is what makes the update conservative to machine precision regardless of how tightly the iteration converged. Convergence tolerances are set by `radiation.iteration_tolerance` and `radiation.iteration_tolerance_rel`; the choice of per-group unknown and the round-off floor on the residual are discussed in [Radiation Integrator](radiation_integrator.md).

## Multigroup opacity models

The group-integrated four-force involves integrals of the opacity multiplied by a radiation quantity over each group, so a model for the frequency dependence of \\(\chi\_0\\) *within* a group is needed. Quokka offers two, selected with the `opacity_model` trait.

### Piecewise constant (PC)

The opacity is taken to be constant across each group, \\(\chi\_0(\nu) = \chi\_{0,g}\\). The Planck-, energy-, and flux-mean opacities of a group are then all equal to that one value, and the \\(\partial \chi\_0 / \partial \nu\\) terms vanish. This is the simplest model and the one used by most earlier multigroup codes. It is accurate when the frequency grid is fine enough to resolve the variation of the opacity, and inaccurate when it is not.

### Piecewise power law (PPL)

The opacity is instead assumed to be a power law across each group,

<script type="math/tex; mode=display">
\chi_{0}(\nu) = \chi_{0,g-} \left( \frac{\nu}{\nu_{g-}} \right)^{\alpha_{\chi_0, g}} \, , \qquad \nu_{g-} \le \nu \le \nu_{g+} \, ,
</script>

specified by its value at the lower group edge and a power-law index. This costs very little extra and is much more accurate than PC at low frequency resolution, which is the common case in practice. Power laws are also the natural description of many real opacities — infrared dust opacity and free-free opacity are both close to power laws in frequency.

Evaluating the group means also requires the shape of the radiation spectrum within each group. Assuming both \\(E\_\nu\\) and \\(B\_\nu\\) are power laws of index \\(\alpha\_{Q,g}\\) within the group gives the closed form

<script type="math/tex; mode=display">
\chi_{0Q,g} = \chi_{0,g-} \left[ \frac{ r_g^{\alpha_{Q,g} + 1} - 1}{\alpha_{Q,g} + 1} \right]^{-1} \left[ \frac{r_g^{\alpha_{\chi_0,g} + \alpha_{Q,g} + 1} - 1}{\alpha_{\chi_0,g} + \alpha_{Q,g} + 1} \right] , \qquad r_g \equiv \frac{\nu_{g+}}{\nu_{g-}} \, ,
</script>

with the bracketed factors replaced by \\(\ln r\_g\\) when the corresponding exponent is \\(-1\\). Two ways of choosing \\(\alpha\_{Q,g}\\) are implemented:

- **Fixed slope** (`PPL_opacity_fixed_slope_spectrum`, recommended): assume \\(\nu Q\_\nu\\) is constant across each group, i.e. \\(\alpha\_{Q,g} = -1\\). This is not arbitrary — weighted by the spectrum itself, the mean power-law index of any spectrum that integrates to a finite energy is exactly \\(-1\\). It costs almost nothing and performs well even with a handful of groups.
- **Full spectrum** (`PPL_opacity_full_spectrum`): fit \\(\alpha\_{E,g}\\) and \\(\alpha\_{B,g}\\) to the actual radiation and Planck spectra on the fly, refitting during the first few Newton iterations. This is significantly more expensive and, in the tests of [@He_2024b], no more accurate than the fixed-slope method. It is kept for testing and is not recommended for production.

Note that \\(\alpha\_{Q,g}\\) matters only when \\(\alpha\_{\chi\_0,g} \ne 0\\): the spectrum shape within a group is relevant only if the opacity varies across that group. Setting all exponents to zero reduces PPL to PC.

### Flux-mean opacity

The flux-mean opacity \\(\chi\_{0F,g}\\) is not supplied by the user. It is computed internally from \\(\chi\_{0E,g}\\), \\(\chi\_{0B,g}\\), and the gas temperature so that \\(\boldsymbol{G}\_g \to 0\\) in an optically thick moving medium. Enforcing this relation is what guarantees that the multigroup scheme reaches the correct diffusion limit; in the PC case it reduces to \\(\chi\_{0F,g} = \chi\_{0,g}\\), as expected.

## Setting up a problem

### Compile-time traits

Enable radiation in `Physics_Traits` and set the number of groups:

```c++
template <> struct Physics_Traits<MyProblem> : DefaultPhysicsTraits {
	static constexpr bool is_hydro_enabled = true;
	static constexpr bool is_radiation_enabled = true;
	static constexpr int nGroups = 4; // 1 (default) means grey radiation
};
```

Then specialise `RadSystem_Traits`:

| Trait            | Type                       | Default          | Meaning                                                                                                                        |
| ---------------- | -------------------------- | ---------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| `c_hat_over_c`   | `double`                   | `1.0`            | Reduced speed of light \\(\hat{c}/c\\). Use `1.0` unless the radiation timestep is prohibitive.                                 |
| `Erad_floor`     | `double`                   | `0.`             | Floor on the radiation energy density of each group, in \\(\mathrm{erg\\,cm^{-3}}\\).                                           |
| `energy_unit`    | `double`                   | `C::ev2erg`      | Unit in which `radBoundaries` is expressed. Use `C::hplanck` to give group boundaries in Hz, `C::ev2erg` to give them in eV.    |
| `radBoundaries`  | `GpuArray<double, nGroups+1>` | `{0., inf}`   | Group boundaries, monotonically increasing, in units of `energy_unit`. Must span the full range where \\(\nu B\_\nu\\) matters. |
| `beta_order`     | `int`                      | `1`              | Highest order of \\(v/c\\) retained in the four-force. `0` drops all velocity terms; `1` is the documented scheme. The single-group solver also accepts `2` and `3`. |
| `opacity_model`  | `OpacityModel`             | `single_group`   | `single_group` for grey radiation; one of the multigroup models above when `nGroups > 1`.                                       |

### Grey opacities

For `nGroups = 1`, define the mean opacities as functions of density and gas temperature. Each returns a mass opacity in \\(\mathrm{cm^2\\,g^{-1}}\\). Only `ComputePlanckOpacity` is mandatory; the other two default to it.

```c++
template <> struct RadSystem_Traits<MyProblem> {
	static constexpr double c_hat_over_c = 1.0;
	static constexpr double Erad_floor = 1.0e-20;
	static constexpr int beta_order = 1;
};

template <>
AMREX_GPU_HOST_DEVICE auto RadSystem<MyProblem>::ComputePlanckOpacity(const double /*rho*/, const double /*Tgas*/) -> amrex::Real
{
	return kappa0;
}

template <>
AMREX_GPU_HOST_DEVICE auto RadSystem<MyProblem>::ComputeFluxMeanOpacity(const double rho, const double Tgas) -> amrex::Real
{
	return ComputePlanckOpacity(rho, Tgas);
}
```

`ComputeEnergyMeanOpacity` may be specialised in the same way.

### Multigroup opacities

For `nGroups > 1` the grey hooks are not used. Instead specialise a single function, `DefineOpacityExponentsAndLowerValues`, which returns two arrays of length `nGroups + 1`: the power-law exponents \\(\alpha\_{\chi\_0,g}\\) in element `[0]`, and the opacity at the lower edge of each group, \\(\kappa\_{0,g-}\\), in element `[1]`. Under the PC model the exponents are ignored and `[1][g]` is used directly as the constant opacity of group `g`.

A piecewise constant opacity, uniform across all groups:

```c++
template <> struct RadSystem_Traits<MyProblem> {
	static constexpr double c_hat_over_c = 1.0;
	static constexpr double Erad_floor = 1.0e-20;
	static constexpr double energy_unit = C::hplanck;      // radBoundaries given in Hz
	static constexpr amrex::GpuArray<double, n_groups_ + 1> radBoundaries = rad_boundaries_;
	static constexpr int beta_order = 1;
	static constexpr OpacityModel opacity_model = OpacityModel::piecewise_constant_opacity;
};

template <>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto
RadSystem<MyProblem>::DefineOpacityExponentsAndLowerValues(amrex::GpuArray<double, nGroups_ + 1> /*rad_boundaries*/, const double /*rho*/,
							   const double /*Tgas*/) -> amrex::GpuArray<amrex::GpuArray<double, nGroups_ + 1>, 2>
{
	amrex::GpuArray<amrex::GpuArray<double, nGroups_ + 1>, 2> exponents_and_values{};
	for (int g = 0; g < nGroups_ + 1; ++g) {
		exponents_and_values[0][g] = 0.0;    // ignored by the PC model
		exponents_and_values[1][g] = kappa0; // cm^2 g^-1
	}
	return exponents_and_values;
}
```

A piecewise power-law opacity \\(\kappa \propto \nu^{-2}\\) differs only in the choice of model and the two lines that fill the arrays:

```c++
	static constexpr OpacityModel opacity_model = OpacityModel::PPL_opacity_fixed_slope_spectrum;
```

```c++
	for (int g = 0; g < nGroups_ + 1; ++g) {
		exponents_and_values[0][g] = -2.0;
		exponents_and_values[1][g] = kappa0 * std::pow(rad_boundaries[g] / nu_ref, -2.0);
	}
```

The function is called on the device with the current cell density and gas temperature, so the opacity may depend on both. If the opacity is tabulated rather than analytic, a convenient choice of exponent is the secant slope across the group,

<script type="math/tex; mode=display">
\alpha_{\chi_0,g} = \frac{\ln \left[ \chi_0(\nu_{g+}) / \chi_0(\nu_{g-}) \right]}{\ln (\nu_{g+} / \nu_{g-})} \, .
</script>

### Radiation sources

A problem can inject radiation directly — for example from stellar sources — by specialising `RadSystem<problem_t>::AddRadSource`, which is called before each implicit solve. See the [Radiation Integrator](radiation_integrator.md) page for the buffer conventions.

## Runtime parameters

| Parameter                              | Type          | Default           | Description                                                                                                                  |
| -------------------------------------- | ------------- | ----------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| `radiation.cfl`                        | Float         | `0.3`             | CFL number for the radiation substeps, based on \\(\hat{c}\\). Independent of the hydro CFL number.                           |
| `radiation.reconstruction_order`       | Integer       | `3`               | Spatial reconstruction for the radiation variables: 1 (donor cell), 2 (PLM), 3 (PPM), 5 (extremum-preserving PPM).            |
| `radiation.iteration_tolerance`        | Float         | `1e-11`           | Relative tolerance on the Newton-Raphson residuals of the implicit solve.                                                     |
| `radiation.iteration_tolerance_rel`    | Float         | `-1.0` (disabled) | Optional tolerance on the relative change between consecutive Newton iterations.                                              |
| `radiation.print_iteration_counts`     | Boolean (0/1) | `0`               | Print the number of Newton iterations per step. Useful when diagnosing a stiff or non-converging problem.                     |
| `radiation.dust_gas_interaction_coeff` | Float         | `2.5e-34`         | Coefficient of the dust-gas thermal coupling term, used when `ISM_Traits::enable_dust_gas_thermal_coupling_model` is enabled. |

The number of radiation substeps per hydro step is computed automatically from `radiation.cfl` and \\(\hat{c}\\); it is not set directly.

## Test problems

The following test problems exercise the solver across the streaming, static diffusion, and dynamic diffusion regimes:

- [Radiative shock test](tests/radshock.md) — non-equilibrium radiating shock.
- [Matter-radiation temperature equilibrium test](tests/energy_exchange.md) — the implicit coupling in isolation.
- [Advecting radiation pulse test](tests/radhydro_pulse.md) — static and dynamic diffusion, single-group and multigroup.
- [Uniform advecting radiation in diffusive limit](tests/radhydro_uniform_adv.md) — the \\(v/c\\) terms in the dynamic diffusion limit.
- [1D H II region and dust reprocessing test](tests/DTypeFront1D.md) — multigroup radiation with dust.

## References

The methods on this page are described in full in [@Wibking_2022] (the Godunov radiation solver and the M1 closure), [@He_2024] (the IMEX PD-ARS scheme and its asymptotic-preserving properties), and [@He_2024b] (the multigroup formulation and the opacity models). The mixed-frame formulation follows [@Krumholz2007] and [@MihalasMihalas]; the inner Newton-Raphson scheme follows [@Howell_2003].
