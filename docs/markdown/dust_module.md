# Dust Module

> **Warning: Beta feature**
>
> The dedicated dust dynamics module has not yet been exercised in a published science application with Quokka and should currently be treated as **beta**.
>

This module implements dust transport and dust-gas source terms. When dust is enabled without MHD, the source term is aerodynamic drag. When both dust and MHD are enabled, Quokka integrates aerodynamic drag together with the charged-dust Lorentz force.

## Equations for Gas-Dust System

<script type="math/tex; mode=display">
\begin{align}
\frac{\partial \rho_{\mathrm{g}}}{\partial t} 
    + \nabla \cdot (\rho_{\mathrm{g}} \mathbf{v}_{\mathrm{g}}) 
    &= 0, \\
\frac{\partial (\rho_{\mathrm{g}} \mathbf{v}_{\mathrm{g}})}{\partial t}
    + \nabla \cdot (\rho_{\mathrm{g}} \mathbf{v}_{\mathrm{g}} \otimes \mathbf{v}_{\mathrm{g}} 
        + P_{\mathrm{g}} \mathbf{I})
    &= \sum_{n=1}^{N} \rho_{\mathrm{d},n} 
        \frac{\mathbf{v}_{\mathrm{d},n} - \mathbf{v}_{\mathrm{g}}}{T_{\mathrm{s},n}}
        - \sum_{n=1}^{N} \rho_{\mathrm{d},n} \Omega_{\mathrm{L},n}
        \left(\mathbf{v}_{\mathrm{d},n} - \mathbf{v}_{\mathrm{g}}\right) \times \hat{\mathbf{b}}, \\
\frac{\partial E_{\mathrm{g}}}{\partial t}
    + \nabla \cdot \left[(E_{\mathrm{g}} + P_{\mathrm{g}}) \mathbf{v}_{\mathrm{g}}\right]
    &= \sum_{n=1}^{N} \rho_{\mathrm{d},n}
        \frac{\mathbf{v}_{\mathrm{d},n} - \mathbf{v}_{\mathrm{g}}}{T_{\mathrm{s},n}}
        \cdot \mathbf{v}_{\mathrm{g}}
        + \omega_1 \sum_{n=1}^{N} \rho_{\mathrm{d},n}
        \frac{(\mathbf{v}_{\mathrm{d},n} - \mathbf{v}_{\mathrm{g}})^{2}}{T_{\mathrm{s},n}}, \\
\frac{\partial \rho_{\mathrm{d},n}}{\partial t}
    + \nabla \cdot (\rho_{\mathrm{d},n} \mathbf{v}_{\mathrm{d},n})
    &= 0, \\
\frac{\partial (\rho_{\mathrm{d},n} \mathbf{v}_{\mathrm{d},n})}{\partial t}
    + \nabla \cdot (\rho_{\mathrm{d},n} 
        \mathbf{v}_{\mathrm{d},n} \otimes \mathbf{v}_{\mathrm{d},n})
    &= \rho_{\mathrm{d},n} 
        \frac{\mathbf{v}_{\mathrm{g}} - \mathbf{v}_{\mathrm{d},n}}{T_{\mathrm{s},n}}
        + \rho_{\mathrm{d},n} \Omega_{\mathrm{L},n}
        \left(\mathbf{v}_{\mathrm{d},n} - \mathbf{v}_{\mathrm{g}}\right) \times \hat{\mathbf{b}},
\end{align}
</script>

where \\(\omega_1\\) controls the level of physical frictional heating, with \\(\omega_1 = 0\\) turning it off and \\(\omega_1 = 1\\) depositing all drag dissipation into the gas.

## Variable Storage

The dust cell-centred conserved variables (\\(\rho_{\mathrm{d}}\\), \\(\rho_{\mathrm{d}}\vec{v_{\mathrm{d}}}\\)) are added to MultiFab.

## Reconstruction and Riemann Solver

Dust reconstruction is performed together with gas using the same method. The Riemann Solver used is as follows:

In one dimension along the x-direction, given the left/right states \\(W_d^{L/R}\\), one can provide the Riemann flux for conserved variables as follows. The density flux reads (Huang & Bai 2022):

<script type="math/tex; mode=display">
\begin{align*}
F^{\text a}_x(\rho_d) = 
\begin{cases}
\rho_d^L v_{d,x}^L & \text{if } v_{d,x}^L > 0, \, v_{d,x}^R \ge 0, \\
\rho_d^R v_{d,x}^R & \text{if } v_{d,x}^L \le 0, \, v_{d,x}^R < 0, \\
\rho_d^L v_{d,x}^L + \rho_d^R v_{d,x}^R & \text{if } v_{d,x}^L > 0, \, v_{d,x}^R < 0, \\
0 & \text{else}.
\end{cases}
\end{align*}
</script>

Similar expressions hold for the momentum flux for all directions.

This is implemented in `src/dust/dustRiemannSolver.hpp` and called in `DustSystem::ComputeDustFluxes` to compute the dust advection flux.

## Time Integrator

A Strang-split method is used to integrate the dust-gas source terms together with the explicit transport update:

<script type="math/tex; mode=display">
\mathbf{u}^{n+1} = \mathcal{S}_{\Delta t/2} \mathcal{H}_{\Delta t} \mathcal{S}_{\Delta t/2} \mathbf{u}^n
</script>

where \\(\mathcal{H}\\) is the hydrodynamics operator, including both gas and dust transport, and \\(\mathcal{S}\\) is the built-in dust source operator. The hydrodynamics operator \\(\mathcal{H}\\) is handled using the explicit RK2 scheme. The source operator \\(\mathcal{S}\\) is implemented in `src/dust/DustSources.hpp` and called by `QuokkaSimulation::addStrangSplitSourcesWithBuiltin`:

- If `Physics_Traits<problem_t>::is_dust_enabled = true` and MHD is disabled, Quokka calls `DustSources::computeDustDrag`, following Tedeschi-Prades et al. (2025).
- If both `Physics_Traits<problem_t>::is_dust_enabled = true` and `Physics_Traits<problem_t>::is_mhd_enabled = true`, Quokka calls `DustSources::computeDustDragAndLorentz`.

`DustSources::computeDustDragAndLorentz` integrates drag and Lorentz forces in the same source solve; it does not operator-split the Lorentz force from drag. The method uses a two-stage generalized implicit Runge-Kutta (GIRK) update for the local gas and dust momenta. For dust species \\(n\\), the relevant local rates are the drag rate \\(\alpha_n = 1/T_{\mathrm{s},n}\\) and the gyrofrequency \\(\Omega_{\mathrm{L},n} = \xi_n |\mathbf{B}|\\). The implementation selects the non-stiff or stiff GIRK coefficients from the local source timescale, using \\((\alpha_n^2 + \Omega_{\mathrm{L},n}^2)^{-1/2}\\) for the drag-plus-Lorentz system.

### Optional Picard iteration for dust–gas drag

Users may optionally enable Picard iteration for the dust source operator \\(\mathcal{S}\\). When the stopping time depends on the gas or dust velocity, enabling iteration is required to maintain an implicit dust drag update. This option applies to both `DustSources::computeDustDrag` and `DustSources::computeDustDragAndLorentz`. See [Runtime parameters](parameters.md) for details.

### User-defined dust stopping time and charge

For a given problem, users must define a problem-specific dust stopping time by implementing the `DustSources::ComputeReciprocalStoppingTime` function (note that this function should return the reciprocal of the stopping time). An example can be found in the `src/problems/DustDamping` test.

Users can directly use the dust stopping time calculation helper `DustSources::ComputeReciprocalStoppingTimeKwok` to compute the physical dust stopping time, following Kwok (1975) with an optional supersonic correction. Problem setups that use this helper must provide the dust grain radius \\(a\\) and material density \\(\rho_{\mathrm{gr}}\\) for each dust group. The stopping time of dust \\(t_{\mathrm{s}}\\) is given by:

<script type="math/tex; mode=display">
t_{\mathrm{s}} = \frac{\sqrt{\pi \gamma}}{2\sqrt{2}} \frac{a \rho_{\mathrm{gr}}}{\rho_{\mathrm{g}} c_{\mathrm{s}}} \times 
\begin{cases}
\left( 1 + \dfrac{9\pi\gamma}{128} \left| \dfrac{\mathbf{v}_{\mathrm{d}} - \mathbf{v}_{\mathrm{g}}}{c_{\mathrm{s}}} \right|^2 \right)^{-1/2}, & \text{if supersonic correction is enabled,} \\[1.5em]
1, & \text{if supersonic correction is disabled.}
\end{cases}
</script>

When \\(\gamma=1\\), this expression reduces exactly to the isothermal \\(t_s\\). An example of its usage can be found in the `src/problems/DustDampingIteration` test.

For charged dust in MHD, users must also define the problem-specific dust charge-to-mass ratio by specializing `DustSources::ComputeDustChargeToMassRatio`. This function returns \\(\xi_n\\) for each dust group. The default implementation returns zero for all groups, so dust behaves as neutral dust unless a problem overrides it. Examples can be found in `src/problems/DustDampedGyromotion`.

## CFL Condition for Dust

For the dust-gas coupled system with N dust species, we use the following CFL condition:

<script type="math/tex; mode=display">
\Delta t_{\mathrm{CFL}} = C_{\mathrm{CFL}} \cdot \min_{\mathrm{cells}} \left( \frac{\Delta x}{\max\left( |v_{\mathrm{g}}| + c_s, \max_{n=1}^{N} |v_{\mathrm{d},n}|+c_s \right)} \right).
</script>

## Runtime Controls

The following input parameters tune the dust module and are documented in more detail in [Runtime parameters](parameters.md):

- `enable_iter_stoptime` – switch of iterative dust stopping time calculation.
- `omega1` – controls deposition of physical dust-drag heating into the gas.
- `omega2` – controls deposition of the numerical energy correction from the coupled dust drag-plus-Lorentz source update. It is only relevant when MHD and dust are both enabled.
- `print_iteration_counts` - switch to turn on/off printing of dust source iteration counts for debugging.
- `dust.density_floor` - the minimum dust density value allowed in the simulation.
