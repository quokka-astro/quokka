# Thermal Conduction

Quokka solves explicit, flux-limited thermal conduction as a Strang-split source term on the gas internal energy. The conductivity model (constant or Spitzer) and the geometry (isotropic or field-aligned) are selected at compile time in the problem's `Physics_Traits`. Conduction is then switched on at runtime with `conduction.enabled = 1`.

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
| `isotropic` (default) | \\(\mathbf{q} = -\kappa \nabla T\\) | `IsoConduction` |
| `anisotropic` | see below | `AnisoConduction` (Sharma & Hammett 2007); requires `is_mhd_enabled = true` |

In the anisotropic geometry, heat flows along the unit magnetic-field vector \\(\hat{\mathbf{b}} = \mathbf{B}/|\mathbf{B}|\\) with conductivity \\(\kappa\_\parallel\\), and across it with \\(\kappa\_\perp\\):

<script type="math/tex; mode=display">
\mathbf{q} = -\kappa_\perp \nabla T - (\kappa_\parallel - \kappa_\perp)\, \hat{\mathbf{b}} \left( \hat{\mathbf{b}} \cdot \nabla T \right).
</script>

Setting \\(\kappa\_\perp = \kappa\_\parallel\\) recovers the isotropic flux. In both geometries, each face-normal component \\(q\\) of the classical flux is then saturated (Cowie & McKee 1977):

<script type="math/tex; mode=display">
q \rightarrow \frac{q}{1 + |q| / q_{\rm sat}}, \qquad q_{\rm sat} = f_{\rm sat}\, \phi\, \rho\, c_s^3,
</script>

where \\(f\_{\rm sat}\\) is `conduction.saturation_factor`, \\(\phi\\) is `conduction.flux_limiter_phi`, and \\(c\_s\\) is the sound speed.

## Units

Conductivities are always given as a full conductivity \\(\kappa\\), in erg cm\\(^{-1}\\) s\\(^{-1}\\) K\\(^{-1}\\), in both geometries. For `spitzer`, the prefactor \\(\kappa\_0\\) is in erg cm\\(^{-1}\\) s\\(^{-1}\\) K\\(^{-7/2}\\). With a non-CGS `unit_system`, use the corresponding code units.

Internally, both solvers obtain \\(\kappa\\) through `quokka::conduction::EvaluateDiffusivity` (`src/conduction/conductivity.hpp`). It returns the diffusivity

<script type="math/tex; mode=display">
\chi = \frac{\kappa}{n k_B}, \qquad n = \frac{\rho}{\mu},
</script>

which is the \\(\chi\\) of Sharma & Hammett (2007), whose flux is \\(\mathbf{q} = -n k\_B \chi \nabla T\\). Conduction therefore requires `EOS_Traits::mean_molecular_weight` to be set; this is checked at compile time.

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

The function runs on the GPU, so it must not use host-only data; use `constexpr` constants for any parameters. Both returned conductivities must be non-negative, and in the anisotropic geometry \\(\kappa\_\perp \le \kappa\_\parallel\\), because the anisotropic solver's L2 limiter relies on \\((\kappa\_\parallel - \kappa\_\perp) \hat{b}\_n^2 \ge 0\\) (checked by an assertion in debug builds). If a problem selects `problem_defined` without specializing `computeConductivity`, it fails to compile.

The solvers evaluate \\(\kappa\\) at face centres (isotropic, from the face-averaged \\(\rho\\) and \\(T\\)) or at cell corners (anisotropic, from the average of the adjacent cells).

## Input parameters

The `constant` and `spitzer` models read their prefactor from the key that matches the geometry:

| Geometry | Keys |
|----------|------|
| `isotropic` | `conduction.conductivity_prefactor` (required when conduction is enabled) |
| `anisotropic` | `conduction.kappaPar` (required when conduction is enabled), `conduction.kappaPerp` (optional, default 0) |

With the anisotropic geometry, `conduction.aniso_flux_limiter` selects the limiter for the transverse (cross) terms of the flux, `mc` (default) or `minmod`, for any `conduction_model`.

Quokka aborts at startup in these cases:
- a prefactor is set that does not match the geometry, or any prefactor is set with `problem_defined`;
- `conduction.aniso_flux_limiter` is set with the isotropic geometry, or is not `mc` or `minmod`;
- a prefactor is negative, or `conduction.kappaPerp` is larger than `conduction.kappaPar`;
- `conduction.enabled = 1` while the problem's `conduction_model` is `none`;
- the input file still sets the removed `conduction.conduction_type` key.

See [Runtime parameters](parameters.md#thermal-conduction) for the full list of `conduction.*` parameters.

## Timestep

The explicit conduction timestep is the minimum over all cells of

<script type="math/tex; mode=display">
\Delta t = C\, \frac{\Delta x_{\min}^2}{D}, \qquad D = \frac{\kappa_{\rm eff}}{\partial e / \partial T}, \qquad
\kappa_{\rm eff} =
\begin{cases}
\kappa, & \text{isotropic,} \\
2 \left( \kappa_\parallel + \kappa_\perp \right), & \text{anisotropic,}
\end{cases}
</script>

where \\(C\\) is `conduction.conduction_cfl`, \\(e\\) is the internal energy per unit volume, and \\(D\\) is evaluated with each cell's own \\(\rho\\) and \\(T\\).

## Test problems

| Problem | Model | Geometry |
|---------|-------|----------|
| `ThermalConductionConstantAMR` | `constant` | isotropic |
| `ThermalConductionPattle` | `spitzer` | isotropic |
| `ThermalConductionSpitzerGaussian` | `problem_defined` (Spitzer) | isotropic |
| `ThermalConductionAniso` | `constant` | anisotropic |
| `ThermalConductionAnisoGaussian` | `constant` (with \\(\kappa\_\perp > 0\\)) | anisotropic |
| `ThermalConductionAnisoPattle` | `spitzer` | anisotropic |
