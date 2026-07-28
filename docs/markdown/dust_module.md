# Dust Module

> **Warning: Beta feature**
>
> The dedicated dust dynamics module has not yet been exercised in a published science application with Quokka and should currently be treated as **beta**.
>

This module implements dust transport and dust-gas source terms. When dust is enabled without MHD, the source term is aerodynamic drag. When both dust and MHD are enabled, Quokka integrates aerodynamic drag together with the charged-dust Lorentz force.

## Equations for Gas-Dust-MHD System

<script type="math/tex; mode=display">
\begin{align}
\frac{\partial \rho_{\mathrm{g}}}{\partial t} 
    + \nabla \cdot (\rho_{\mathrm{g}} \mathbf{v}_{\mathrm{g}}) 
    &= 0, \\
\frac{\partial (\rho_{\mathrm{g}} \mathbf{v}_{\mathrm{g}})}{\partial t}
    + \nabla \cdot (\rho_{\mathrm{g}} \mathbf{v}_{\mathrm{g}} \otimes \mathbf{v}_{\mathrm{g}} 
        + (P_{\mathrm{g}} + \tfrac{1}{2} B^2) \mathbf{I}
        - \mathbf{B} \otimes \mathbf{B})
    &= \sum_{n=1}^{N} \rho_{\mathrm{d},n} 
        \frac{\mathbf{v}_{\mathrm{d},n} - \mathbf{v}_{\mathrm{g}}}{T_{\mathrm{s},n}}
        - \sum_{n=1}^{N} \rho_{\mathrm{d},n} \Omega_{\mathrm{L},n}
        \left(\mathbf{v}_{\mathrm{d},n} - \mathbf{v}_{\mathrm{g}}\right) \times \hat{\mathbf{b}}
        + \rho_{\mathrm{g}} \mathbf{a}_{\mathrm{ext},\mathrm{g}}, \\
\frac{\partial E_{\mathrm{g}}}{\partial t}
    + \nabla \cdot \left[
        (E_{\mathrm{g}} + P_{\mathrm{g}} + \tfrac{1}{2} B^2) \mathbf{v}_{\mathrm{g}}
        - (\mathbf{v}_{\mathrm{g}} \cdot \mathbf{B}) \mathbf{B}\right]
    &= \sum_{n=1}^{N} \rho_{\mathrm{d},n}
        \frac{\mathbf{v}_{\mathrm{d},n} - \mathbf{v}_{\mathrm{g}}}{T_{\mathrm{s},n}}
        \cdot \mathbf{v}_{\mathrm{g}}
        - \sum_{n=1}^{N} \rho_{\mathrm{d},n} \Omega_{\mathrm{L},n}
        \left[\left(\mathbf{v}_{\mathrm{d},n} - \mathbf{v}_{\mathrm{g}}\right) \times \hat{\mathbf{b}}\right]
        \cdot \mathbf{v}_{\mathrm{g}}
        + \rho_{\mathrm{g}} \mathbf{a}_{\mathrm{ext},\mathrm{g}} \cdot \mathbf{v}_{\mathrm{g}}
        + \omega_{\rm drag} \sum_{n=1}^{N} \rho_{\mathrm{d},n}
        \frac{\left|\mathbf{v}_{\mathrm{d},n} - \mathbf{v}_{\mathrm{g}}\right|^{2}}{T_{\mathrm{s},n}}, \\
\frac{\partial \mathbf{B}}{\partial t}
    - \nabla \times (\mathbf{v}_{\mathrm{g}} \times \mathbf{B})
    &= 0, \\
\frac{\partial \rho_{\mathrm{d},n}}{\partial t}
    + \nabla \cdot (\rho_{\mathrm{d},n} \mathbf{v}_{\mathrm{d},n})
    &= 0, \\
\frac{\partial (\rho_{\mathrm{d},n} \mathbf{v}_{\mathrm{d},n})}{\partial t}
    + \nabla \cdot (\rho_{\mathrm{d},n} 
        \mathbf{v}_{\mathrm{d},n} \otimes \mathbf{v}_{\mathrm{d},n})
    &= \rho_{\mathrm{d},n} 
        \frac{\mathbf{v}_{\mathrm{g}} - \mathbf{v}_{\mathrm{d},n}}{T_{\mathrm{s},n}}
        + \rho_{\mathrm{d},n} \Omega_{\mathrm{L},n}
        \left(\mathbf{v}_{\mathrm{d},n} - \mathbf{v}_{\mathrm{g}}\right) \times \hat{\mathbf{b}}
        + \rho_{\mathrm{d},n} \mathbf{a}_{\mathrm{ext},\mathrm{d},n},
\end{align}
</script>

where

-   \\(\rho_{\mathrm{g}}\\) is the gas density,
-   \\(\mathbf{v}_{\mathrm{g}}\\) is the gas velocity,
-   \\(P_{\mathrm{g}}\\) is the gas pressure,
-   \\(\mathbf{I}\\) is the identity tensor,
-   \\(\mathbf{B}\\) is the magnetic field,
-   \\(E_{\mathrm{g}}\\) is the gas total energy density, including magnetic energy when MHD is enabled,
-   \\(\rho_{\mathrm{d},n}\\) is the dust mass density for dust species \\(n\\) (\\(n \in [1, N]\\)),
-   \\(\mathbf{v}_{\mathrm{d},n}\\) is the dust velocity for dust species \\(n\\),
-   \\(T_{\mathrm{s},n}\\) is the aerodynamic stopping time for dust species \\(n\\),
-   \\(\Omega_{\mathrm{L},n}=q_n|\vec{B}|/(m_n c)\\) is the signed angular gyrofrequency for dust species \\(n\\), where \\(q_n\\) is its signed Heaviside–Lorentz charge, \\(m_n\\) is its grain mass, and \\(c\\) is the speed of light,
-   \\(\hat{\mathbf{b}}\\) is the unit vector along the magnetic field,
-   \\(\mathbf{a}_{\mathrm{ext},\mathrm{g}}\\) is the external acceleration applied to the gas,
-   \\(\mathbf{a}_{\mathrm{ext},\mathrm{d},n}\\) is the external acceleration applied to dust species \\(n\\),
-   \\(\omega_{\rm drag}\\) is the fraction of dust-drag dissipation deposited into the gas.

The Lorentz work term in the gas total-energy equation accounts for energy exchanged with the dust. The Lorentz force transfers kinetic energy between gas and dust but does not heat the combined gas-dust system, because the gas-side and dust-side work terms for each species sum to zero:

<script type="math/tex; mode=display">
\begin{aligned}
&- \rho_{\mathrm{d},n} \Omega_{\mathrm{L},n}
   \left[\left(\mathbf{v}_{\mathrm{d},n} - \mathbf{v}_{\mathrm{g}}\right) \times \hat{\mathbf{b}}\right]
   \cdot \mathbf{v}_{\mathrm{g}}
 + \rho_{\mathrm{d},n} \Omega_{\mathrm{L},n}
   \left[\left(\mathbf{v}_{\mathrm{d},n} - \mathbf{v}_{\mathrm{g}}\right) \times \hat{\mathbf{b}}\right]
   \cdot \mathbf{v}_{\mathrm{d},n} \\
&= \rho_{\mathrm{d},n} \Omega_{\mathrm{L},n}
   \left[\left(\mathbf{v}_{\mathrm{d},n} - \mathbf{v}_{\mathrm{g}}\right) \times \hat{\mathbf{b}}\right]
   \cdot \left(\mathbf{v}_{\mathrm{d},n} - \mathbf{v}_{\mathrm{g}}\right)
 = 0 .
\end{aligned}
</script>

In `DustSources::computeDustDragAndLorentz`, Quokka splits the deposited gas-energy increment into a drag-like contribution controlled by `dust.omega_drag_heating` and a gyrofrequency-dependent residual contribution controlled by `dust.omega_gyro_residual`.

### Dimensionless dust charge-to-mass ratio

The equations above use dimensional Heaviside–Lorentz variables. In ideal MHD,

<script type="math/tex; mode=display">
\mathbf{E}=-\frac{\mathbf{v}_{\mathrm{g}}}{c}\times\mathbf{B},
\qquad
\mathbf{a}_{\mathrm{L},n}
=\frac{q_n}{m_n c}
\left(\mathbf{v}_{\mathrm{d},n}-\mathbf{v}_{\mathrm{g}}\right)\times\mathbf{B}.
</script>

For code units defined by \\(L_0\\), \\(M_0\\), and \\(\tau_0\\), let
\\(\rho_0=M_0/L_0^3\\) and
\\(B_0=\sqrt{\rho_0}L_0/\tau_0\\), so that
\\(\widetilde{\mathbf{B}}=\mathbf{B}/B_0\\). The dust source integrator takes

<script type="math/tex; mode=display">
\xi_n=\frac{q_nL_0\sqrt{\rho_0}}{m_n c},
\qquad
\widetilde{\Omega}_{\mathrm{L},n}
=\xi_n|\widetilde{\mathbf{B}}|.
</script>

Here \\(\xi_n\\) is signed and dimensionless. `UnitSystem::CGS` uses
\\(L_0=1\\,\mathrm{cm}\\), \\(M_0=1\\,\mathrm{g}\\), and
\\(\tau_0=1\\,\mathrm{s}\\); `UnitSystem::CUSTOM` uses the base units in
`Physics_Traits`; and `UnitSystem::CONSTANTS` requires the problem to prescribe
\\(\xi_n\\) directly.

## Variable Storage

The dust cell-centred conserved variables (\\(\rho_{\mathrm{d}}\\), \\(\rho_{\mathrm{d}}\mathbf{v}_{\mathrm{d}}\\)) are added to MultiFab.

## Reconstruction and Riemann Solver

Dust reconstruction is performed together with gas using the same method. The Riemann solver used is as follows:

In one dimension along the x-direction, given the left/right states \\(W_{\mathrm{d}}^{\mathrm{L}/\mathrm{R}}\\), one can provide the Riemann flux for conserved variables as follows. The density flux reads (Huang & Bai 2022):

<script type="math/tex; mode=display">
\begin{align*}
F_{x}^{\mathrm{a}}(\rho_{\mathrm{d}}) = 
\begin{cases}
\rho_{\mathrm{d}}^{\mathrm{L}} v_{\mathrm{d},x}^{\mathrm{L}} & \text{if } v_{\mathrm{d},x}^{\mathrm{L}} > 0, \, v_{\mathrm{d},x}^{\mathrm{R}} \ge 0, \\
\rho_{\mathrm{d}}^{\mathrm{R}} v_{\mathrm{d},x}^{\mathrm{R}} & \text{if } v_{\mathrm{d},x}^{\mathrm{L}} \le 0, \, v_{\mathrm{d},x}^{\mathrm{R}} < 0, \\
\rho_{\mathrm{d}}^{\mathrm{L}} v_{\mathrm{d},x}^{\mathrm{L}} + \rho_{\mathrm{d}}^{\mathrm{R}} v_{\mathrm{d},x}^{\mathrm{R}} & \text{if } v_{\mathrm{d},x}^{\mathrm{L}} > 0, \, v_{\mathrm{d},x}^{\mathrm{R}} < 0, \\
0 & \text{else}.
\end{cases}
\end{align*}
</script>

Similar expressions hold for the momentum flux for all directions.

This is implemented in `src/dust/dustRiemannSolver.hpp` and called in `DustSystem::ComputeDustFluxes` to compute the dust advection flux.

## Time Integrator

A Strang-split method is used to integrate the dust-gas source terms together with the explicit transport update:

<script type="math/tex; mode=display">
\mathbf{u}^{n+1} = \mathcal{C}_{\Delta t/2} \mathcal{H}_{\Delta t} \mathcal{C}_{\Delta t/2} \mathbf{u}^n
</script>

where \\(\mathcal{H}\\) is the explicit gas/MHD and dust transport update, and \\(\mathcal{C}\\) denotes the local combined drag-plus-Lorentz update. In non-MHD runs, \\(\mathcal{C}\\) reduces to a drag-only update; in MHD runs, it integrates aerodynamic drag and the charged-dust Lorentz force in the same solve. The \\(\mathcal{C}\\) update is implemented in `src/dust/DustSources.hpp` and called from `QuokkaSimulation::addStrangSplitSourcesWithBuiltin`:

- If `Physics_Traits<problem_t>::is_dust_enabled = true` and MHD is disabled, Quokka calls `DustSources::computeDustDrag`, following Tedeschi-Prades et al. (2025).
- If both `Physics_Traits<problem_t>::is_dust_enabled = true` and `Physics_Traits<problem_t>::is_mhd_enabled = true`, Quokka calls `DustSources::computeDustDragAndLorentz`.

`DustSources::computeDustDragAndLorentz` integrates drag and Lorentz forces in the same source solve; it does not operator-split the Lorentz force from drag. The method uses a two-stage generalized implicit Runge-Kutta (GIRK) update for the local gas and dust momenta. For dust species \\(n\\), the relevant local rates in code units are the drag rate \\(\alpha_n = 1/T_{\mathrm{s},n}\\) and the gyrofrequency \\(\Omega_{\mathrm{L},n} = \xi_n |\mathbf{B}|\\). The implementation selects the resolved or stiff GIRK coefficients from the local timescale, using \\((\alpha_n^2 + \Omega_{\mathrm{L},n}^2)^{-1/2}\\) for the drag-plus-Lorentz system. The resolved coefficients used in `computeDustDragAndLorentz` may be selected at runtime with `dust.resolved_rk_scheme`: `GL4` chooses the current two-stage Gauss-Legendre coefficients, `Midpoint` chooses the implicit midpoint coefficients, and `TP2025` reuses the resolved-branch coefficients from `DustSources::computeDustDrag`.

### Optional Picard iteration for dust–gas source update

Users may optionally enable Picard iteration for the local update represented by \\(\mathcal{C}\\). When the stopping time depends on the gas or dust velocity, enabling iteration is required to maintain an implicit dust source update. This option applies to both `DustSources::computeDustDrag` and `DustSources::computeDustDragAndLorentz`. See [Runtime parameters](parameters.md) for details.

### User-defined dust stopping time and charge

For a given problem, users must define a problem-specific dust stopping time by implementing the `DustSources::ComputeReciprocalStoppingTime` function (note that this function should return the reciprocal of the stopping time). An example can be found in the `src/problems/DustDamping` test.

Users can directly use the dust stopping time calculation helper `DustSources::ComputeReciprocalStoppingTimeKwok` to compute the physical dust stopping time, following Kwok (1975) with an optional supersonic correction. Problem setups that use this helper must provide the dust grain radius \\(a\\) and material density \\(\rho_{\mathrm{gr}}\\) for each dust group. These values can be read from the optional runtime parameters `dust.grain_radius` and `dust.grain_density` by calling `quokka::dust::readDustGrainParams`. The stopping time of dust \\(t_{\mathrm{s}}\\) is given by:

<script type="math/tex; mode=display">
t_{\mathrm{s}} = \frac{\sqrt{\pi \gamma}}{2\sqrt{2}} \frac{a \rho_{\mathrm{gr}}}{\rho_{\mathrm{g}} c_{\mathrm{s}}} \times 
\begin{cases}
\left( 1 + \dfrac{9\pi\gamma}{128} \left| \dfrac{\mathbf{v}_{\mathrm{d}} - \mathbf{v}_{\mathrm{g}}}{c_{\mathrm{s}}} \right|^2 \right)^{-1/2}, & \text{if supersonic correction is enabled,} \\[1.5em]
1, & \text{if supersonic correction is disabled.}
\end{cases}
</script>

When \\(\gamma=1\\), this expression reduces exactly to the isothermal \\(t_{\mathrm{s}}\\). An example of its usage can be found in the `src/problems/DustDampingIteration` test.

For charged dust in MHD, users must also specialize `DustSources::ComputeDustDimensionlessChargeToMassRatio`. This function returns the signed dimensionless \\(\xi_n\\) defined above for each dust group. The default implementation returns zero for all groups, so dust behaves as neutral dust unless a problem overrides it. Examples can be found in `src/problems/DustDampedGyromotion`.

## CFL Condition for Dust

For the dust-gas coupled system with \\(N\\) dust species, we use the following CFL condition:

<script type="math/tex; mode=display">
\Delta t_{\mathrm{CFL}} = C_{\mathrm{CFL}} \cdot \min_{\mathrm{cells}} \left( \frac{\Delta x}{\max\left( |\mathbf{v}_{\mathrm{g}}| + c_{\mathrm{s}}, \max_{n=1}^{N} |\mathbf{v}_{\mathrm{d},n}| + c_{\mathrm{s}} \right)} \right).
</script>

## Runtime Controls

The following input parameters tune the dust module and are documented in more detail in [Runtime parameters](parameters.md):

- `enable_iter_stoptime` – switch of iterative dust stopping time calculation.
- `omega_drag_heating` – controls deposition of the drag-like heating contribution in the dust source update.
- `omega_gyro_residual` – controls deposition of the gyrofrequency-dependent residual contribution in `computeDustDragAndLorentz`.
- `resolved_rk_scheme` – selects the GIRK coefficients in resolved branch used by `DustSources::computeDustDragAndLorentz`. Supported values are `TP2025`, `GL4`, and `Midpoint`.
- `print_iteration_counts` - switch to turn on/off printing of dust source iteration counts for debugging.
- `dust.density_floor` - the minimum dust density value allowed in the simulation.
- `dust.grain_radius` - optional dust grain radius values for problem setups that use the Kwok stopping-time helper.
- `dust.grain_density` - optional dust grain material density values for problem setups that use the Kwok stopping-time helper.
