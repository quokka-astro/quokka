# Thermal Conduction

Quokka solves explicit, flux-limited thermal conduction as a Strang-split source term on the gas internal energy. The conductivity model and the geometry (isotropic or field-aligned) are selected at compile time in the problem's `Physics_Traits`. Conduction is then switched on at runtime with `conduction.enabled = 1`.

## Choosing a model and geometry

```cpp
template <> struct Physics_Traits<MyProblem> : DefaultPhysicsTraits {
    static constexpr bool is_hydro_enabled = true;
    static constexpr ConductionModel conduction_model = ConductionModel::spitzer;
    static constexpr ConductionGeometry conduction_geometry = ConductionGeometry::isotropic;
};
```

| `conduction_model` | Conductivity \\(\kappa\\) | Set by |
|--------------------|---------------------------|--------|
| `none` (default) | no conduction | — |
| `constant` | \\(\kappa = \kappa\_0\\) | `conduction.*` prefactors in the input file |
| `spitzer` | \\(\kappa = \kappa\_0 T^{5/2}\\) | `conduction.*` prefactors in the input file |
| `problem_defined` | \\(\kappa(\rho, T)\\) | `computeConductivity` in the problem file |

| `conduction_geometry` | Heat flux | Solver |
|-----------------------|-----------|--------|
| `isotropic` (default) | \\(\mathbf{q} = -\kappa \nabla T\\) | `ElectronConduction` |
| `anisotropic` | \\(\mathbf{q} = -\kappa\_\parallel \hat{\mathbf{b}} (\hat{\mathbf{b}} \cdot \nabla T)\\) | `AnisoConduction` (Sharma & Hammett 2007); requires `is_mhd_enabled = true` |

In both geometries the classical flux is saturated as \\(q / (1 + |q| / q\_{\rm sat})\\), with \\(q\_{\rm sat} = f\_{\rm sat}\,\phi\,\rho c\_s^3\\) (Cowie & McKee 1977), where \\(f\_{\rm sat}\\) is `conduction.saturation_factor` and \\(\phi\\) is `conduction.flux_limiter_phi`.

## Units

Conductivities are always given as a full conductivity \\(\kappa\\), in erg cm\\(^{-1}\\) s\\(^{-1}\\) K\\(^{-1}\\), in both geometries. For `spitzer`, the prefactor \\(\kappa\_0\\) is in erg cm\\(^{-1}\\) s\\(^{-1}\\) K\\(^{-7/2}\\). With a non-CGS `unit_system`, use the corresponding code units.

Internally, both solvers obtain \\(\kappa\\) through `quokka::conduction::EvaluateDiffusivity` (`src/conduction/conductivity.hpp`). It returns the diffusivity \\(\chi = \kappa / (n k\_B)\\), with \\(n = \rho / \mu\\), which is the \\(\chi\\) of Sharma & Hammett (2007), whose flux is \\(-n k\_B \chi \nabla T\\). Conduction therefore requires `EOS_Traits::mean_molecular_weight` to be set; this is checked at compile time.

## Problem-defined conductivity

With `ConductionModel::problem_defined`, specialize `computeConductivity` in the problem file. It returns \\(\{\kappa\_\parallel, \kappa\_\perp\}\\); in the isotropic geometry only the first component is used.

```cpp
template <>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE auto computeConductivity<MyProblem>(amrex::Real rho, amrex::Real Tgas)
    -> quokka::valarray<amrex::Real, 2>
{
    return {kappa0 * std::pow(Tgas, 2.5), 0.0};
}
```

The function runs on the GPU, so it must not use host-only data; use `constexpr` constants for any parameters. Both returned conductivities must be non-negative, because the anisotropic solver's L2 limiter relies on \\(\kappa\_\parallel \hat{b}\_n^2 \ge 0\\). If a problem selects `problem_defined` without specializing `computeConductivity`, it fails to compile.

The solvers evaluate \\(\kappa\\) at face centres (isotropic, from the face-averaged \\(\rho\\) and \\(T\\)) or at cell corners (anisotropic, from the average of the adjacent cells).

## Input parameters

The `constant` and `spitzer` models read their prefactor from the key that matches the geometry:

| Geometry | Keys |
|----------|------|
| `isotropic` | `conduction.conductivity_prefactor` (required when conduction is enabled) |
| `anisotropic` | `conduction.kappaPar` (required when conduction is enabled), `conduction.kappaPerp` (optional, default 0) |

Quokka aborts at startup in these cases:
- a prefactor is set that does not match the geometry, or any prefactor is set with `problem_defined`;
- a prefactor is negative;
- `conduction.enabled = 1` while the problem's `conduction_model` is `none`;
- the input file still sets the removed `conduction.conduction_type` key.

See [Runtime parameters](parameters.md#thermal-conduction) for the full list of `conduction.*` parameters.

## Timestep

The conduction timestep is \\(\Delta t = C\,\Delta x\_{\min}^2 / D\\), minimized over all cells. Here \\(C\\) is `conduction.conduction_cfl`, and the thermal diffusivity is \\(D = \kappa\_{\rm eff} / (\partial e / \partial T)\\), evaluated with the cell's own \\(\rho\\) and \\(T\\). \\(\kappa\_{\rm eff} = \kappa\\) in the isotropic geometry and \\(2(\kappa\_\parallel + \kappa\_\perp)\\) in the anisotropic geometry.

## Limitations

- AMR subcycling is not supported; set `do_subcycle = 0`.
- The perpendicular conductivity \\(\kappa\_\perp\\) is accepted but not yet used by the anisotropic flux.
- The anisotropic solver has been tested in 3D. It compiles in 2D, but the 2D path has not been validated.

## Test problems

| Problem | Model | Geometry |
|---------|-------|----------|
| `ThermalConductionConstantAMR` | `constant` | isotropic |
| `ThermalConductionPattle` | `spitzer` | isotropic |
| `ThermalConductionSpitzerGaussian` | `problem_defined` (Spitzer) | isotropic |
| `ThermalConductionAniso` | `constant` | anisotropic |
