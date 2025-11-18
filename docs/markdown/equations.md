# Equations

## Fluids and radiation 

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
  \vec{F}_g
\end{array}\right], \;
\vec{F}(U) = \left[
\begin{array}{c}
  \rho \vec{v} \\
  \rho \vec{v} \otimes \vec{v}+p \\
  (E_{\rm gas} + p) \vec{v} \\
  \rho X_n \vec{v} \\
  \vec{F}_g \\
  c^2 \mathsf{P}_g
\end{array}\right], \;
\vec{S}(U)=\left[
\begin{array}{c}
  0 \\
  \sum_g \vec{G}_g + \rho \vec{g} \\
  c \sum_g G^0_{g} + \rho \vec{v} \cdot \vec{g} + \mathcal{H} - \mathcal{C} \\
  \rho \dot{X}_n \\
  - c G^0_{g} \\
  - c^2 \vec{G}_g
\end{array}\right],
\end{aligned}$$

along with the non-conservative auxiliary internal energy equation:

$$\begin{aligned}
\frac{\partial (\rho e_{\text{aux}})}{\partial t} =
- \nabla \cdot (\rho e_{\text{aux}} \vec{v}) - p \nabla \cdot \vec{v}
+ S_{\text{rad}} + \mathcal{H} - \mathcal{C}, \\
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
-   $\rho_i$ is the mass density due to particle $i$.

Note that since work done by radiation on the gas is included in the $c \sum_g G^0_g$ term, $S_{\text{rad}}$ is not the same as $c \sum_g G^0_g$.

## Collisionless particles

Quokka solves the following equation of motion for collisionless particles:

$$\frac{d^2 \vec{x}_i}{d t^2} = \vec{g} ,$$

where $\vec{x}_i$ is the position vector of particle $i$.
