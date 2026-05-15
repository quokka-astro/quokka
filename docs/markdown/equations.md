# Equations

## Fluids, radiation, magnetic fields, and dust

Assuming the speed of light is not reduced (\\(\hat{c} = c\\)), Quokka solves the following conservation laws for the cell-centered variables, as written below in the [Heaviside–Lorentz system of units](https://en.wikipedia.org/wiki/Heaviside%E2%80%93Lorentz_units):

<script type="math/tex; mode=display">
\frac{\partial \vec{U}}{\partial t}+\nabla \cdot \vec{F}(\vec{U}) = \vec{S}(\vec{U}),
</script>

<script type="math/tex; mode=display">
\begin{aligned}
\vec{U} =\left[
\begin{array}{c}
  \rho \\
  \rho \vec{v} \\
  E \\
  \rho X_n \\
  E_g \\
  \vec{F}_g \\
  \rho_{\mathrm{d},k} \\
  \rho_{\mathrm{d},k} \vec{v}_{\mathrm{d},k}
\end{array}\right], \;
\vec{F}(U) = \left[
\begin{array}{c}
  \rho \vec{v} \\
  \rho \vec{v} \otimes \vec{v} + \left(p + \frac{1}{2} B^2\right)\mathsf{I} - \vec{B} \otimes \vec{B} \\
  \left(E + p + \frac{1}{2} B^2\right)\vec{v} - (\vec{v} \cdot \vec{B}) \vec{B} \\
  \rho X_n \vec{v} \\
  \vec{F}_g \\
  c^2 \mathsf{P}_g \\
  \rho_{\mathrm{d},k} \vec{v}_{\mathrm{d},k} \\
  \rho_{\mathrm{d},k} \vec{v}_{\mathrm{d},k} \otimes \vec{v}_{\mathrm{d},k}
\end{array}\right], \;
\vec{S}(U)=\left[
\begin{array}{c}
  0 \\
  \sum_g \vec{G}_g + \rho \vec{g} + \sum_{k=1}^{N_{\mathrm{dust}}} \left[\rho_{\mathrm{d},k} \frac{\vec{v}_{\mathrm{d},k} - \vec{v}}{T_{\mathrm{s},k}} - \rho_{\mathrm{d},k} \Omega_{\mathrm{L},k} \left(\vec{v}_{\mathrm{d},k} - \vec{v}\right) \times \hat{\vec{b}}\right] \\
  c \sum_g G^0_{g} + \rho \vec{v} \cdot \vec{g} + \mathcal{H} - \mathcal{C} + \sum_{k=1}^{N_{\mathrm{dust}}} \rho_{\mathrm{d},k} \frac{\vec{v}_{\mathrm{d},k} - \vec{v}}{T_{\mathrm{s},k}} \cdot \vec{v} - \sum_{k=1}^{N_{\mathrm{dust}}} \rho_{\mathrm{d},k} \Omega_{\mathrm{L},k} \left[\left(\vec{v}_{\mathrm{d},k} - \vec{v}\right) \times \hat{\vec{b}}\right] \cdot \vec{v} + \omega_1 \sum_{k=1}^{N_{\mathrm{dust}}} \rho_{\mathrm{d},k} \frac{(\vec{v}_{\mathrm{d},k} - \vec{v})^{2}}{T_{\mathrm{s},k}} \\
  \rho \dot{X}_n \\
  - c G^0_{g} \\
  - c^2 \vec{G}_g \\
  0 \\
  \rho_{\mathrm{d},k} \frac{\vec{v} - \vec{v}_{\mathrm{d},k}}{T_{\mathrm{s},k}} + \rho_{\mathrm{d},k} \Omega_{\mathrm{L},k} \left(\vec{v}_{\mathrm{d},k} - \vec{v}\right) \times \hat{\vec{b}}
\end{array}\right],
\end{aligned}
</script>

where the total fluid energy is

<script type="math/tex; mode=display">
E = \rho e + \frac{1}{2} \rho v^2 + \frac{1}{2} B^2 \, ,
</script>

and the face-centered magnetic field is evolved according to the ideal MHD induction equation:

<script type="math/tex; mode=display">
\frac{\partial \vec{B}}{\partial t} - \nabla \times \left(\vec{v} \times \vec{B}\right) = 0.
</script>

Quokka also solves the non-conservative auxiliary internal energy equation:

<script type="math/tex; mode=display">
\begin{aligned}
\frac{\partial (\rho e_{\text{aux}})}{\partial t} = - \nabla \cdot (\rho e_{\text{aux}} \vec{v}) - p \nabla \cdot \vec{v} + S_{\text{rad}} + \mathcal{H} - \mathcal{C} + \omega_1 \sum_{k=1}^{N_{\mathrm{dust}}} \rho_{\mathrm{d},k} \frac{(\vec{v}_{\mathrm{d},k} - \vec{v})^{2}}{T_{\mathrm{s},k}}, \\
\Delta S_{\text{rad}} = \int \sum_g c G^0_g \ dt - \frac{1}{2} \Delta \left(\rho v^2 \right),
\end{aligned}
</script>

and the gravitational Poisson equation:

<script type="math/tex; mode=display">
\begin{aligned}
\nabla^2 \phi = -4 \pi G \left( \rho + \sum_i \rho_i \right), \\
\vec{g} \equiv -\nabla \phi,
\end{aligned}
</script>

where

-   \\(\rho\\) is the gas density,
-   \\(\vec{v}\\) is the gas velocity,
-   \\(E\\) is the total fluid energy density, including magnetic energy when MHD is enabled,
-   \\(\rho e_{\text{aux}}\\) is the auxiliary gas internal energy density,
-   \\(X_n\\) is the fractional concentration of species \\(n\\),
-   \\(\dot{X}_n\\) is the chemical reaction term for species \\(n\\),
-   \\(\mathcal{H}\\) is the optically-thin volumetric heating term (radiative and chemical),
-   \\(\mathcal{C}\\) is the optically-thin volumetric cooling term (radiative and chemical),
-   \\(p(\rho, e)\\) is the gas pressure derived from a general convex equation of state,
-   \\(\vec{B}\\) is the magnetic field,
-   \\(\mathsf{I}\\) is the identity tensor,
-   \\(E_g\\) is the radiation energy density for group \\(g\\),
-   \\(F_g\\) is the radiation flux for group \\(g\\),
-   \\(\mathsf{P}_g\\) is the radiation pressure tensor for group \\(g\\),
-   \\(G_g\\) is the radiation four-force \\([G^0_g, \vec{G}_g]\\) due to group \\(g\\),
-   \\(\Delta S_{\text{rad}}\\) is the change in gas internal energy due to radiation over a timestep,
-   \\(\phi\\) is the Newtonian gravitational potential,
-   \\(\vec{g}\\) is the gravitational acceleration,
-   \\(\rho_i\\) is the mass density due to particle \\(i\\),
-   \\(\rho_{\mathrm{d},k}\\) is the dust mass density for dust species \\(k\\) (\\(k \in [1, N_{\mathrm{dust}}]\\)),
-   \\(\vec{v}_{\mathrm{d},k}\\) is the dust velocity for dust species \\(k\\),
-   \\(T_{\mathrm{s},k}\\) is the aerodynamic stopping time for dust species \\(k\\),
-   \\(\xi_k\\) is the charge-to-mass ratio for dust species \\(k\\),
-   \\(\Omega_{\mathrm{L},k} = \xi_k |\vec{B}|\\) is the dust gyrofrequency for dust species \\(k\\),
-   \\(\hat{\vec{b}} = \vec{B}/|\vec{B}|\\) is the unit vector along the magnetic field,
-   \\(\omega_1\\) is the fraction of physical dust-drag dissipation deposited into the gas.

Note that since work done by radiation on the gas is included in the \\(c \sum_g G^0_g\\) term, \\(S_{\text{rad}}\\) is not the same as \\(c \sum_g G^0_g\\).


## Collisionless particles

Quokka solves the following equation of motion for collisionless particles:

<script type="math/tex; mode=display">
\frac{d^2 \vec{x}_i}{d t^2} = \vec{g} ,
</script>

where \\(\vec{x}_i\\) is the position vector of particle \\(i\\).
