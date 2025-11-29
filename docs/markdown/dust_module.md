# Dust Module

This module primarily implements two components: the dust transport term and the dust drag source term.

## Equations for Gas-Dust System

$$\begin{align}
\frac{\partial \rho_{\mathrm{g}}}{\partial t} 
    + \nabla \cdot (\rho_{\mathrm{g}} \boldsymbol{v}_{\mathrm{g}}) 
    &= 0, \\
\frac{\partial (\rho_{\mathrm{g}} \boldsymbol{v}_{\mathrm{g}})}{\partial t}
    + \nabla \cdot (\rho_{\mathrm{g}} \boldsymbol{v}_{\mathrm{g}} \otimes \boldsymbol{v}_{\mathrm{g}} 
        + P_{\mathrm{g}} \mathbf{I})
    &= \sum_{n=1}^{N} \rho_{\mathrm{d},n} 
        \frac{\boldsymbol{v}_{\mathrm{d},n} - \boldsymbol{v}_{\mathrm{g}}}{T_{\mathrm{s},n}}, \\
\frac{\partial E_{\mathrm{g}}}{\partial t}
    + \nabla \cdot \left[(E_{\mathrm{g}} + P_{\mathrm{g}}) \boldsymbol{v}_{\mathrm{g}}\right]
    &= \sum_{n=1}^{N} \rho_{\mathrm{d},n} 
        \frac{\boldsymbol{v}_{\mathrm{d},n} - \boldsymbol{v}_{\mathrm{g}}}{T_{\mathrm{s},n}}
        \cdot \boldsymbol{v}_{\mathrm{g}}
        + \omega \sum_{n=1}^{N} \rho_{\mathrm{d},n} 
        \frac{(\boldsymbol{v}_{\mathrm{d},n} - \boldsymbol{v}_{\mathrm{g}})^{2}}{T_{\mathrm{s},n}}, \\
\frac{\partial \rho_{\mathrm{d},n}}{\partial t}
    + \nabla \cdot (\rho_{\mathrm{d},n} \boldsymbol{v}_{\mathrm{d},n})
    &= 0, \\
\frac{\partial (\rho_{\mathrm{d},n} \boldsymbol{v}_{\mathrm{d},n})}{\partial t}
    + \nabla \cdot (\rho_{\mathrm{d},n} 
        \boldsymbol{v}_{\mathrm{d},n} \otimes \boldsymbol{v}_{\mathrm{d},n})
    &= \rho_{\mathrm{d},n} 
        \frac{\boldsymbol{v}_{\mathrm{g}} - \boldsymbol{v}_{\mathrm{d},n}}{T_{\mathrm{s},n}},
\end{align}
$$

where $\omega$ controls the level of frictional heating, with $\omega = 0$ turning it off and $\omega = 1$ depositing all dissipation into the gas.

## Variable Storage

The dust cell-centred conserved variables ($\rho_{\mathrm{d}}$, $\rho_{\mathrm{d}}\vec{v_{\mathrm{d}}}$) are added to MultiFab.

## Reconstruction and Riemann Solver

Dust reconstruction is performed together with gas using the same method. The Riemann Solver used is as follows:

In one dimension along the x-direction, given the left/right states $W_d^{L/R}$, one can provide the Riemann flux for conserved variables as follows. The density flux reads (Huang & Bai 2022):

$$
\begin{align*}
F^{\text a}_x(\rho_d) = 
\begin{cases}
\rho_d^L v_{d,x}^L & \text{if } v_{d,x}^L > 0, \, v_{d,x}^R > 0, \\
\rho_d^R v_{d,x}^R & \text{if } v_{d,x}^L < 0, \, v_{d,x}^R < 0, \\
0 & \text{if } v_{d,x}^L < 0, \, v_{d,x}^R > 0, \\
\rho_d^L v_{d,x}^L + \rho_d^R v_{d,x}^R & \text{if } v_{d,x}^L > 0, \, v_{d,x}^R < 0.
\end{cases}
\end{align*}
$$

Similar expressions hold for the momentum flux for all directions.

This is implemented in `src/dust/dustRiemannSolver.hpp` and called in `HydroSystem::ComputeFluxes` to compute the dust advection flux.

## Time Integrator

A Strang-split method is used to integrate the dust-gas system. The Strang-split update can be expressed as:

$$
\boldsymbol u^{n+1} = \mathcal{D}_{\Delta t/2} \mathcal{H}_{\Delta t} \mathcal{D}_{\Delta t/2} \boldsymbol u^n
$$

where $\mathcal{D}$ is the dust-gas drag operator and $\mathcal{H}$ is the hydrodynamics operator (including both gas and dust transport). The hydrodynamics operator $\mathcal{H}$ is handled using the explicit RK2 scheme. The drag operator $\mathcal{D}$ is implemented in `src/dust/DustDrag.hpp` and called in `QuokkaSimulation::addStrangSplitSourcesWithBuiltin` via `quokka::DustDrag::computeDustDrag`.

## CFL Condition for Dust

For the dust-gas coupled system with N dust species, we use the following CFL condition:

$$
\Delta t_{\mathrm{CFL}} = C_{\mathrm{CFL}} \cdot \min_{\mathrm{cells}} \left( \frac{\Delta x}{\max\left( |v_{\mathrm{g}}| + c_s, \max_{n=1}^{N} |v_{\mathrm{d},n}|+c_s \right)} \right).
$$

## Runtime Controls

The following input parameters tune the dust module and are documented in more detail in [Runtime parameters](parameters.md):

- `alpha` – Inverse of dust stopping time.
- `omega` – Controls the level of frictional heating.