# Dust Module

> **Warning: Beta feature**
>
> The dedicated dust dynamics module has not yet been exercised in a published science application with Quokka and should currently be treated as **beta**.
>

This module implements dust transport and dust-gas source terms. When dust is enabled without MHD, the source term is aerodynamic drag. When both dust and MHD are enabled, Quokka integrates aerodynamic drag together with the charged-dust Lorentz force.

## Equations for Gas-Dust-MHD System

\\[
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
\\]

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
-   \\(\xi_n\\) is the charge-to-mass ratio for dust species \\(n\\),
-   \\(\Omega_{\mathrm{L},n}= \xi_n |\vec{B}|\\) is the dust gyrofrequency for dust species \\(n\\),
-   \\(\hat{\mathbf{b}}\\) is the unit vector along the magnetic field,
-   \\(\mathbf{a}_{\mathrm{ext},\mathrm{g}}\\) is the external acceleration applied to the gas,
-   \\(\mathbf{a}_{\mathrm{ext},\mathrm{d},n}\\) is the external acceleration applied to dust species \\(n\\),
-   \\(\omega_{\rm drag}\\) is the fraction of physical dust-drag dissipation deposited into the gas.

The Lorentz work term in the gas total-energy equation is the gas-side work from the dust back-reaction. It transfers kinetic energy between gas and dust, but it does not heat the combined gas-dust system. Adding the gas-side and dust-side Lorentz work terms for each dust species gives

\\[
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
\\]

Only aerodynamic drag produces physical gas heating in these equations, through the \\(\omega_{\rm drag}\\) term. The `dust.omega_rk_residual` runtime parameter controls deposition of the discrete RK energy residual from the combined drag-plus-Lorentz update. This residual is not a separate physical heating rate and is not a Lorentz-heating parameter.

## Variable Storage

The dust cell-centred conserved variables (\\(\rho_{\mathrm{d}}\\), \\(\rho_{\mathrm{d}}\mathbf{v}_{\mathrm{d}}\\)) are added to MultiFab.

## Reconstruction and Riemann Solver

Dust reconstruction is performed together with gas using the same method. The Riemann solver used is as follows:

In one dimension along the x-direction, given the left/right states \\(W_{\mathrm{d}}^{\mathrm{L}/\mathrm{R}}\\), one can provide the Riemann flux for conserved variables as follows. The density flux reads (Huang & Bai 2022):

\\[
\begin{align*}
F_{x}^{\mathrm{a}}(\rho_{\mathrm{d}}) = 
\begin{cases}
\rho_{\mathrm{d}}^{\mathrm{L}} v_{\mathrm{d},x}^{\mathrm{L}} & \text{if } v_{\mathrm{d},x}^{\mathrm{L}} > 0, \, v_{\mathrm{d},x}^{\mathrm{R}} \ge 0, \\
\rho_{\mathrm{d}}^{\mathrm{R}} v_{\mathrm{d},x}^{\mathrm{R}} & \text{if } v_{\mathrm{d},x}^{\mathrm{L}} \le 0, \, v_{\mathrm{d},x}^{\mathrm{R}} < 0, \\
\rho_{\mathrm{d}}^{\mathrm{L}} v_{\mathrm{d},x}^{\mathrm{L}} + \rho_{\mathrm{d}}^{\mathrm{R}} v_{\mathrm{d},x}^{\mathrm{R}} & \text{if } v_{\mathrm{d},x}^{\mathrm{L}} > 0, \, v_{\mathrm{d},x}^{\mathrm{R}} < 0, \\
0 & \text{else}.
\end{cases}
\end{align*}
\\]

Similar expressions hold for the momentum flux for all directions.

This is implemented in `src/dust/dustRiemannSolver.hpp` and called in `DustSystem::ComputeDustFluxes` to compute the dust advection flux.

## Time Integrator

A Strang-split method is used to integrate the dust-gas source terms together with the explicit transport update:

\\[
\mathbf{u}^{n+1} = \mathcal{C}_{\Delta t/2} \mathcal{H}_{\Delta t} \mathcal{C}_{\Delta t/2} \mathbf{u}^n
\\]

where \\(\mathcal{H}\\) is the explicit gas/MHD and dust transport update, and \\(\mathcal{C}\\) denotes the local combined drag-plus-Lorentz update. In non-MHD runs, \\(\mathcal{C}\\) reduces to a drag-only update; in MHD runs, it integrates aerodynamic drag and the charged-dust Lorentz force in the same solve. The \\(\mathcal{C}\\) update is implemented in `src/dust/DustSources.hpp` and called from `QuokkaSimulation::addStrangSplitSourcesWithBuiltin`:

- If `Physics_Traits<problem_t>::is_dust_enabled = true` and MHD is disabled, Quokka calls `DustSources::computeDustDrag`, following Tedeschi-Prades et al. (2025).
- If both `Physics_Traits<problem_t>::is_dust_enabled = true` and `Physics_Traits<problem_t>::is_mhd_enabled = true`, Quokka calls `DustSources::computeDustDragAndLorentz`.

`DustSources::computeDustDragAndLorentz` integrates drag and Lorentz forces in the same source solve; it does not operator-split the Lorentz force from drag. The method uses a two-stage generalized implicit Runge-Kutta (GIRK) update for the local gas and dust momenta. For dust species \\(n\\), the relevant local rates are the drag rate \\(\alpha_n = 1/T_{\mathrm{s},n}\\) and the gyrofrequency \\(\Omega_{\mathrm{L},n} = \xi_n |\mathbf{B}|\\). The implementation selects the non-stiff or stiff GIRK coefficients from the local timescale, using \\((\alpha_n^2 + \Omega_{\mathrm{L},n}^2)^{-1/2}\\) for the drag-plus-Lorentz system.

### Optional Picard iteration for dust–gas source update

Users may optionally enable Picard iteration for the local update represented by \\(\mathcal{C}\\). When the stopping time depends on the gas or dust velocity, enabling iteration is required to maintain an implicit dust source update. This option applies to both `DustSources::computeDustDrag` and `DustSources::computeDustDragAndLorentz`. See [Runtime parameters](parameters.md) for details.

### User-defined dust stopping time and charge

For a given problem, users must define a problem-specific dust stopping time by implementing the `DustSources::ComputeReciprocalStoppingTime` function (note that this function should return the reciprocal of the stopping time). An example can be found in the `src/problems/DustDamping` test.

Users can directly use the dust stopping time calculation helper `DustSources::ComputeReciprocalStoppingTimeKwok` to compute the physical dust stopping time, following Kwok (1975) with an optional supersonic correction. Problem setups that use this helper must provide the dust grain radius \\(a\\) and material density \\(\rho_{\mathrm{gr}}\\) for each dust group. These values can be read from the optional runtime parameters `dust.grain_radius` and `dust.grain_density` by calling `quokka::dust::readDustGrainParams`. The stopping time of dust \\(t_{\mathrm{s}}\\) is given by:

\\[
t_{\mathrm{s}} = \frac{\sqrt{\pi \gamma}}{2\sqrt{2}} \frac{a \rho_{\mathrm{gr}}}{\rho_{\mathrm{g}} c_{\mathrm{s}}} \times 
\begin{cases}
\left( 1 + \dfrac{9\pi\gamma}{128} \left| \dfrac{\mathbf{v}_{\mathrm{d}} - \mathbf{v}_{\mathrm{g}}}{c_{\mathrm{s}}} \right|^2 \right)^{-1/2}, & \text{if supersonic correction is enabled,} \\[1.5em]
1, & \text{if supersonic correction is disabled.}
\end{cases}
\\]

When \\(\gamma=1\\), this expression reduces exactly to the isothermal \\(t_{\mathrm{s}}\\). An example of its usage can be found in the `src/problems/DustDampingIteration` test.

For charged dust in MHD, users must also define the problem-specific dust charge-to-mass ratio by specializing `DustSources::ComputeDustChargeToMassRatio`. This function returns \\(\xi_n\\) for each dust group. The default implementation returns zero for all groups, so dust behaves as neutral dust unless a problem overrides it. Examples can be found in `src/problems/DustDampedGyromotion`.

## CFL Condition for Dust

For the dust-gas coupled system with \\(N\\) dust species, we use the following CFL condition:

\\[
\Delta t_{\mathrm{CFL}} = C_{\mathrm{CFL}} \cdot \min_{\mathrm{cells}} \left( \frac{\Delta x}{\max\left( |\mathbf{v}_{\mathrm{g}}| + c_{\mathrm{s}}, \max_{n=1}^{N} |\mathbf{v}_{\mathrm{d},n}| + c_{\mathrm{s}} \right)} \right).
\\]

## Runtime Controls

The following input parameters tune the dust module and are documented in more detail in [Runtime parameters](parameters.md):

- `enable_iter_stoptime` – switch of iterative dust stopping time calculation.
- `omega_drag_heating` – controls deposition of physical dust-drag heating into the gas.
- `omega_rk_residual` – controls deposition of the discrete RK energy residual from the combined dust drag-plus-Lorentz source update. It is only relevant when MHD and dust are both enabled.
- `print_iteration_counts` - switch to turn on/off printing of dust source iteration counts for debugging.
- `dust.density_floor` - the minimum dust density value allowed in the simulation.
- `dust.grain_radius` - optional dust grain radius values for problem setups that use the Kwok stopping-time helper.
- `dust.grain_density` - optional dust grain material density values for problem setups that use the Kwok stopping-time helper.
