# Equations

## Fluids, radiation, and dust

Assuming the speed of light is not reduced ($\hat{c} = c$), Quokka solves the system of conservation laws:

$$\frac{\partial \vec{U}}{\partial t}+\nabla \cdot \vec{F}(\vec{U}) = \vec{S}(\vec{U}),$$

$$\begin{aligned}
\vec{U} =\left[
\begin{array}{c}
  \rho \\
  \rho \vec{v} \\
  E_{\rm gas} \\
  \rho X_n \\
  E_g \\
  \vec{F}_g \\
  \rho_{\mathrm{d},k} \\
  \rho_{\mathrm{d},k} \vec{v}_{\mathrm{d},k}
\end{array}\right], \;
\vec{F}(U) = \left[
\begin{array}{c}
  \rho \vec{v} \\
  \rho \vec{v} \otimes \vec{v}+p \\
  (E_{\rm gas} + p) \vec{v} \\
  \rho X_n \vec{v} \\
  \vec{F}_g \\
  c^2 \mathsf{P}_g \\
  \rho_{\mathrm{d},k} \vec{v}_{\mathrm{d},k} \\
  \rho_{\mathrm{d},k} \vec{v}_{\mathrm{d},k} \otimes \vec{v}_{\mathrm{d},k}
\end{array}\right], \;
\vec{S}(U)=\left[
\begin{array}{c}
  0 \\
  \sum_g \vec{G}_g + \rho \vec{g} + \sum_{k=1}^{N_{\mathrm{dust}}} \rho_{\mathrm{d},k} \frac{\vec{v}_{\mathrm{d},k} - \vec{v}}{T_{\mathrm{s},k}} \\
  c \sum_g G^0_{g} + \rho \vec{v} \cdot \vec{g} + \mathcal{H} - \mathcal{C} + \sum_{k=1}^{N_{\mathrm{dust}}} \rho_{\mathrm{d},k} \frac{\vec{v}_{\mathrm{d},k} - \vec{v}}{T_{\mathrm{s},k}} \cdot \vec{v} + \omega \sum_{k=1}^{N_{\mathrm{dust}}} \rho_{\mathrm{d},k} \frac{(\vec{v}_{\mathrm{d},k} - \vec{v})^{2}}{T_{\mathrm{s},k}} \\
  \rho \dot{X}_n \\
  - c G^0_{g} \\
  - c^2 \vec{G}_g \\
  0 \\
  \rho_{\mathrm{d},k} \frac{\vec{v} - \vec{v}_{\mathrm{d},k}}{T_{\mathrm{s},k}}
\end{array}\right],
\end{aligned}$$

along with the non-conservative auxiliary internal energy equation:

$$\begin{aligned}
\frac{\partial (\rho e_{\text{aux}})}{\partial t} = - \nabla \cdot (\rho e_{\text{aux}} \vec{v}) - p \nabla \cdot \vec{v} + S_{\text{rad}} + \mathcal{H} - \mathcal{C} + \omega \sum_{k=1}^{N_{\mathrm{dust}}} \rho_{\mathrm{d},k} \frac{(\vec{v}_{\mathrm{d},k} - \vec{v})^{2}}{T_{\mathrm{s},k}}, \\
\Delta S_{\text{rad}} = \int \sum_g c G^0_g \ dt - \frac{1}{2} \Delta \left(\rho v^2 \right),
\end{aligned}$$

and the gravitational Poisson equation:

$$\begin{aligned}
\nabla^2 \phi = -4 \pi G \left( \rho + \sum_i \rho_i \right), \\
\vec{g} \equiv -\nabla \phi,
\end{aligned}$$

where

-   $\rho$ is the gas density,
-   $\vec{v}$ is the gas velocity,
-   $E_{\text{gas}}$ is the total gas energy,
-   $\rho e_{\text{aux}}$ is the auxiliary gas internal energy,
-   $X_n$ is the fractional concentration of species $n$,
-   $\dot{X}_n$ is the chemical reaction term for species $n$,
-   $\mathcal{H}$ is the optically-thin volumetric heating term (radiative and chemical),
-   $\mathcal{C}$ is the optically-thin volumetric cooling term (radiative and chemical),
-   $p(\rho, e)$ is the gas pressure derived from a general convex equation of state,
-   $E_g$ is the radiation energy density for group $g$,
-   $F_g$ is the radiation flux for group $g$,
-   $\mathsf{P}_g$ is the radiation pressure tensor for group $g$,
-   $G_g$ is the radiation four-force $[G^0_g, \vec{G}_g]$ due to group $g$,
-   $\Delta S_{\text{rad}}$ is the change in gas internal energy due to radiation over a timestep,
-   $\phi$ is the Newtonian gravitational potential,
-   $\vec{g}$ is the gravitational acceleration,
-   $\rho_i$ is the mass density due to particle $i$,
-   $\rho_{\mathrm{d},k}$ is the dust mass density for dust species $k$ ($k \in [1, N_{\mathrm{dust}}]$),
-   $\vec{v}_{\mathrm{d},k}$ is the dust velocity for dust species $k$,
-   $T_{\mathrm{s},k}$ is the aerodynamic stopping time for dust species $k$,
-   $\omega$ is the fraction of frictional heating deposited into the gas.

Note that since work done by radiation on the gas is included in the $c \sum_g G^0_g$ term, $S_{\text{rad}}$ is not the same as $c \sum_g G^0_g$.

## Ideal MHD

When the MHD module is enabled, Quokka solves the ideal-MHD system with a face-centred magnetic field evolved by constrained transport. In this case the cell-centred energy is the **magnetized** total energy,

$$
E_{\text{tot}} = \rho e + \frac{1}{2} \rho v^2 + \frac{1}{2} B^2,
$$

and the conserved system is

$$
\frac{\partial \vec{U}_{\text{MHD}}}{\partial t} + \nabla \cdot \vec{F}_{\text{MHD}}(\vec{U}_{\text{MHD}}) = \vec{S}_{\text{MHD}}(\vec{U}_{\text{MHD}}),
$$

with

$$\begin{aligned}
\vec{U}_{\text{MHD}} = \left[
\begin{array}{c}
  \rho \\
  \rho \vec{v} \\
  E_{\text{tot}} \\
  \rho X_n \\
  \vec{B}
\end{array}\right], \;
\vec{F}_{\text{MHD}} = \left[
\begin{array}{c}
  \rho \vec{v} \\
  \rho \vec{v} \otimes \vec{v} + \left(p + \frac{1}{2} B^2\right)\mathsf{I} - \vec{B} \otimes \vec{B} \\
  \left(E_{\text{tot}} + p + \frac{1}{2} B^2\right)\vec{v} - (\vec{v} \cdot \vec{B}) \vec{B} \\
  \rho X_n \vec{v} \\
  \vec{v} \otimes \vec{B} - \vec{B} \otimes \vec{v}
\end{array}\right], \;
\vec{S}_{\text{MHD}} = \left[
\begin{array}{c}
  0 \\
  \rho \vec{g} \\
  \rho \vec{v} \cdot \vec{g} + \mathcal{H} - \mathcal{C} \\
  \rho \dot{X}_n \\
  0
\end{array}\right].
\end{aligned}$$

The magnetic field also satisfies the divergence constraint

$$
\nabla \cdot \vec{B} = 0.
$$

Quokka stores $\vec{B}$ on faces and updates it with a constrained-transport discretization. MHD + radiation is currently untested.


## Collisionless particles

Quokka solves the following equation of motion for collisionless particles:

$$\frac{d^2 \vec{x}_i}{d t^2} = \vec{g} ,$$

where $\vec{x}_i$ is the position vector of particle $i$.
