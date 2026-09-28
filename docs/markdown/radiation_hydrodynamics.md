# Radiation Hydrodynamics

> **Warning: Beta feature**
>
> The radiation module has been verified against the test problems in [@Wibking_2022], [@He_2024], and [@He_2024b], but is still marked **beta** for science-use maturity. Please cite the relevant methods paper and record the exact commit hash used.
>

Quokka solves the equations of radiation hydrodynamics (RHD) with a two-moment (M1) method in the mixed-frame formulation, accurate to first order in \\(v/c\\). Radiation may be grey (a single frequency-integrated group) or multigroup, and the solver is *asymptotic-preserving*: it recovers the correct diffusion solution even when the photon mean free path is far smaller than a cell. This page summarises what is solved, how, and how to set it up. The methods are presented in full in three papers, cited throughout below: the Godunov radiation solver in [@Wibking_2022], the grey time-integration scheme in [@He_2024], and the multigroup formulation in [@He_2024b]. The stage-by-stage structure of the time integrator is documented separately in the [Radiation Integrator](radiation_integrator.md) page, and the full conservation-law system including MHD, dust, and gravity in [Equations](equations.md).

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

The four-force is written in the *mixed-frame* form: the opacities and emissivities are evaluated in the comoving frame, where they are isotropic and simple, while all radiation moments stay in the lab frame. Assuming the gas is in local thermodynamic equilibrium and neglecting scattering, the frequency-integrated (grey) four-force to order \\(v/c\\) is, following [@MihalasMihalas] and [@Krumholz2007],

<script type="math/tex; mode=display">
\begin{aligned}
- c G^0 &= 4 \pi \chi_{0P} B - c \chi_{0E} E + c^{-1} (2 \chi_{0E} - \chi_{0F}) v^i F^i \, , \\
- G^i &= - c^{-1} \chi_{0F} F^i + 4 \pi c^{-2} v^i \chi_{0P} B + c^{-1} \chi_{0F} v^j P^{ji} + c^{-1} (\chi_{0F} - \chi_{0E}) v^i E \, ,
\end{aligned}
</script>

where \\(\chi\_{0P}\\), \\(\chi\_{0E}\\), and \\(\chi\_{0F}\\) are the comoving-frame Planck-, energy-, and flux-mean absorption coefficients and \\(B\\) is the Planck function at the gas temperature. Quokka works with *mass* opacities \\(\kappa = \chi / \rho\\) in \\(\mathrm{cm^2\\,g^{-1}}\\), which is what a problem generator supplies. The leading terms are the familiar emission, absorption, and radiation force; the terms in \\(v/c\\) carry the work done by the radiation force on the gas and the frame-transformation ("frame-dragging") effects that become order unity in the dynamic diffusion regime. Terms of order \\(v^2/c^2\\) and higher are not documented here; the single-group solver can optionally include them (see `beta_order` below).

### The multigroup four-force

For multigroup, the expressions above are integrated over each group, from \\(\nu\_{g-}\\) to \\(\nu\_{g+}\\). Quokka solves the group-integrated four-force derived in [@He_2024b],

<script type="math/tex; mode=display">
\begin{aligned}
- c G_g^0 &= \underbrace{4 \pi \chi_{0B,g} B_g}_{\text{emission}} - \underbrace{c \, \chi_{0E,g} E_g}_{\text{absorption}} + \underbrace{c^{-1} (1 + \alpha_{\chi_0, g}) \chi_{0F,g} \, v^i F_g^i}_{\text{work on the gas}} \, , \\[4pt]
- G_g^i &= \underbrace{- c^{-1} \chi_{0F,g} F_g^i}_{\text{radiation force}} + \underbrace{\frac{4 \pi}{c^{2}} v^i \chi_{0B,g} B_g}_{\text{momentum of emission}} - \underbrace{\frac{4 \pi}{3 c^{2}} v^i \, \Delta_g (\nu \chi_0 B_{\nu})}_{\text{group coupling}} + \underbrace{c^{-1} (1 + \alpha_{\chi_0, g}) \chi_{0E,g} \, v^j P_g^{ji}}_{\text{frame dragging}} \, ,
\end{aligned}
</script>

where \\(B\_g\\), \\(E\_g\\), \\(\boldsymbol{F}\_g\\), and \\(\mathsf{P}\_g\\) are the Planck function and the three radiation moments integrated over group \\(g\\); \\(\chi\_{0B,g}\\), \\(\chi\_{0E,g}\\), and \\(\chi\_{0F,g}\\) are the comoving-frame absorption coefficients averaged across the group weighted by \\(B\_\nu\\), \\(E\_\nu\\), and \\(F\_\nu\\) respectively; \\(\alpha\_{\chi\_0,g}\\) is the power-law index of the opacity across the group; and

<script type="math/tex; mode=display">
\Delta_g(Q) \equiv Q(\nu_{g+}) - Q(\nu_{g-})
</script>

is the difference of a frequency-dependent quantity between the upper and lower edges of the group. See [Multigroup opacity models](#multigroup-opacity-models) below for how the three mean opacities and \\(\alpha\_{\chi\_0,g}\\) are evaluated, and [Multigroup opacities](#multigroup-opacities) for how to supply them from a problem generator.

Term by term:

- **Emission.** The rate at which the gas radiates thermally into group \\(g\\). Only the part of the blackbody spectrum that falls inside the group contributes, so \\(B\_g\\) depends on the gas temperature and on where the group edges sit relative to the spectral peak.
- **Absorption.** The rate at which group-\\(g\\) radiation is absorbed by the gas. Emission and absorption balance when the group is in radiative equilibrium with the matter, which is what forces \\(E\_g \to 4 \pi B\_g / c\\) at high optical depth.
- **Work on the gas.** The \\(O(v/c)\\) energy exchange that accompanies the radiation force: as the gas is pushed by the flux, the radiation does work on it. The factor \\((1 + \alpha\_{\chi\_0,g})\\) arises from the \\(\nu \\, \partial \chi\_0 / \partial \nu\\) term in the lab-frame opacity — a moving observer sees Doppler-shifted frequencies, and if the opacity varies across the group, that shift changes how strongly the group is absorbed. It reduces to unity for an opacity that is constant across the group.
- **Radiation force.** The momentum the gas absorbs from the group's flux. This is the leading term of the momentum exchange and the one responsible for radiation pressure on matter.
- **Momentum of emission.** Thermal emission from moving matter is beamed forward by the Doppler effect, so it carries net momentum even though it is isotropic in the comoving frame. This term is the corresponding recoil on the gas.
- **Group coupling.** This term has no counterpart in the grey equations. The same Doppler shift that gives the thermal emission net momentum also spreads that momentum over frequency differently from the way it spreads the energy; this term is that difference, and it is what moves photons across group boundaries. Because it telescopes, \\(\sum\_g \Delta\_g (\nu \chi\_0 B\_\nu)\\) collapses to the value of \\(\nu \chi\_0 B\_\nu\\) at the two ends of the whole frequency grid, so it contributes nothing to the total momentum exchange and the grey expressions are recovered exactly — it only redistributes photons among groups. This does place a requirement on `radBoundaries`: the grid must be wide enough that \\(\nu \chi\_0 B\_\nu\\) is negligible at both ends, otherwise the cancellation is incomplete and energy leaks out of the frequency domain.
- **Frame dragging.** The \\(O(v/c)\\) transformation of the group's radiation pressure between the lab and comoving frames. Together with the work term it becomes order unity in the dynamic diffusion regime, which is why both must be retained.

Under the piecewise constant opacity model, \\(\alpha\_{\chi\_0,g} = 0\\) and the three mean opacities collapse to a single value \\(\chi\_{0,g}\\), leaving

<script type="math/tex; mode=display">
\begin{aligned}
- c G_g^0 &= \chi_{0,g} \left( 4 \pi B_g - c E_g + c^{-1} v^i F_g^i \right) , \\[4pt]
- G_g^i &= \chi_{0,g} \left[ - c^{-1} F_g^i + \frac{4 \pi}{c^{2}} v^i \left( B_g - \frac{1}{3} \Delta_g (\nu B_{\nu}) \right) + c^{-1} v^j P_g^{ji} \right] .
\end{aligned}
</script>

Summing either form over all groups recovers the grey four-force of [Matter-radiation coupling](#matter-radiation-coupling) above.

### Radiation band types

Not every group in a multigroup run has to couple to the matter in the same way. Quokka recognises three *band types*, which differ only in which of the terms above are switched on. **Chemical** (ionizing) bands are declared individually and occupy the last groups of the frequency grid. The **dust-absorption** type is at present a property of the whole run rather than of a single band: setting `dust_absorption_only` converts every non-chemical group at once. Thermal and dust-absorption bands cannot yet be mixed in one run; supporting that is future work. The default is all-**thermal**, which is the case every equation so far describes.

| Band type       | Transport | Thermal emission | Absorbed energy heats the gas | Radiation force and work | Photochemistry |
| --------------- | --------- | ---------------- | ----------------------------- | ------------------------ | -------------- |
| Thermal         | yes       | yes              | yes                           | yes                      | no             |
| Dust-absorption | yes       | no               | **no**                        | yes                      | no             |
| Chemical        | yes       | no               | no (photochemistry instead)   | yes                      | yes            |

**Thermal bands** solve the full four-force of the previous section. Use them for any band in which the gas and dust radiate and reabsorb at the local temperature — the infrared, in practice.

**Chemical bands** carry ionizing photons. They are transported and absorbed, but the absorbed energy is passed to the photochemistry network rather than to the thermal solve, so that ionization and the associated heating are computed consistently with the chemical state. See [Photoionization](photoionization.md).

#### Dust-absorption-only mode

A dust-absorption band is one in which dust is the only absorber and the absorbed energy is promptly re-radiated at wavelengths that fall outside the frequency grid being followed. The far-ultraviolet and Lyman-Werner bands are the motivating case: they are absorbed by dust grains, which re-emit in the infrared, and they drive photoelectric heating and \\(\rm H\_2\\) dissociation rather than a thermal exchange with the gas.

Setting \\(B\_g = 0\\) removes the emission, momentum-of-emission, and group-coupling terms from the four-force, leaving

<script type="math/tex; mode=display">
\begin{aligned}
- c G_g^0 &= - \underbrace{c \, \chi_{0E,g} E_g}_{\text{absorbed by dust}} + \underbrace{c^{-1} (1 + \alpha_{\chi_0, g}) \chi_{0F,g} \, v^i F_g^i}_{\text{work on the gas}} \, , \\[4pt]
- G_g^i &= \underbrace{- c^{-1} \chi_{0F,g} F_g^i}_{\text{radiation force}} + \underbrace{c^{-1} (1 + \alpha_{\chi_0, g}) \chi_{0E,g} \, v^j P_g^{ji}}_{\text{frame dragging}} \, .
\end{aligned}
</script>

The momentum exchange is untouched, so the gas feels the full radiation force: dust and gas remain *dynamically* coupled even though they are not thermally coupled. This is the point of the mode — in a galaxy simulation the radiation pressure on dust is a first-order effect on the dynamics and must not be dropped along with the thermal exchange.

What distinguishes the mode is where the absorbed energy goes. For a thermal band the gas energy equation receives the whole of \\(c G^0\_g\\). For a dust-absorption band it receives only the work part,

<script type="math/tex; mode=display">
c G^0_{g,\,\rm gas} = - c^{-1} (1 + \alpha_{\chi_0, g}) \chi_{0F,g} \, v^i F_g^i \, ,
</script>

while the radiation moments still lose the full \\(- c G^0\_g\\) above: the photons really are absorbed, they simply do not heat the gas. The difference, \\(c \chi\_{0E,g} E\_g\\) per unit volume, leaves the simulation. **Total energy is therefore not conserved in this mode.** That is by construction, not an error: the energy has gone into the dust, which radiates it away in the infrared, and neither the dust temperature nor that infrared emission is followed.

Two consequences are worth stating plainly.

- **The gas is heated by a separate module, not by this band.** The physical heating channel for FUV photons is photoelectric heating off grains, whose efficiency depends on the grain charge and therefore on the local electron density and radiation field — not on the absorbed energy alone. A dust-absorption band delivers the radiation field \\(E\_g\\) to the cell; a chemistry and cooling module such as Grackle turns it into a heating rate. Adding the absorbed energy directly to the gas as well would double-count it.
- **The mode assumes weak gas-dust thermal coupling.** Dust and gas exchange heat at a rate \\(\propto n^2\\), so the assumption that the dust returns none of the absorbed energy to the gas holds only at low density. For the \\(\gtrsim 1\\,\rm pc\\) resolution of a galaxy simulation, where the resolved gas density stays below \\(\sim 10^3\\,\rm cm^{-3}\\), the coupling is weak everywhere and the approximation is safe. At the densities reached in a resolved star-forming core it is not, and the full dust model (`ISM_Traits::enable_dust_gas_thermal_coupling_model`, see the [Dust module](dust_module.md)) with thermal bands should be used instead. The two are mutually exclusive, and combining them is a compile-time error.

#### Photoelectric heating

Setting a non-zero `pe_heating_efficiency` heats the gas photoelectrically from the band's radiation field. \\(\epsilon\_g\\) is the dimensionless efficiency factor of the standard interstellar expression, about 0.05 for cold molecular gas, and the heating rate per unit volume is that of [@BateKeto_2015], Eq. 26:

<script type="math/tex; mode=display">
\Gamma_{\rm PE} = \sum_g \epsilon_g \, R \, n_{\rm H} E_g \, , \qquad R = \frac{1.33 \times 10^{-24}}{5.29 \times 10^{-14}} \ {\rm cm^3\,s^{-1}} \, ,
</script>

where \\(1.33 \times 10^{-24}\\,\rm erg\\,s^{-1}\\) is the heating rate per hydrogen nucleus in a unit Habing field and \\(5.29 \times 10^{-14}\\,\rm erg\\,cm^{-3}\\) is the energy density that defines that field, so that \\(E\_g\\) divided by the latter is the local \\(G\_0\\). A zero entry means the band drives no photoelectric heating, which is how non-ultraviolet bands are labelled. Because the expression is linear in \\(E\_g\\), splitting one band into two and giving both the same efficiency reproduces the unsplit result exactly.

Note what \\(\Gamma\_{\rm PE}\\) does **not** contain: the dust opacity of the band. Photoelectric heating is the photoelectric effect on grains, and the grain physics is folded into the empirical coefficient rather than taken from \\(\kappa\\). Two consequences follow, and both differ from what a fraction-of-absorbed-energy model would give.

- **A transparent band still heats the gas.** A band with \\(\kappa\_g = 0\\) is not attenuated and exerts no radiation force, but if its efficiency is non-zero it heats the gas exactly as much as an absorbed band carrying the same \\(E\_g\\).
- **The heating is not taken out of the radiation.** It is neither bounded by, nor debited from, the energy the band absorbs. In this respect it behaves like the thermal-band photoelectric model: it adds energy to the gas that the radiation does not lose, on top of the energy this mode already discards to the dust.

What the form does buy is that it costs no iteration. \\(\Gamma\_{\rm PE}\\) depends on \\(E\_g\\), \\(n\_{\rm H}\\) and two constants, none of which depend on the gas energy, so it is added to the closed-form update rather than solved for. It also carries no \\(\hat{c}\\): like the cosmic-ray heating, it is a direct physical heating rate on the gas, not a transport rate.

Three limitations are worth knowing before using this.

- **cgs only.** \\(R\\) is an empirical coefficient in cgs units, so a non-zero efficiency requires `Physics_Traits::unit_system == UnitSystem::CGS`. Other unit systems are rejected at compile time rather than silently given a cgs number.
- **The opacity is not updated for the heating.** The band solver evaluates \\(\kappa\\) once, at the gas temperature at the start of the step, and the outer iteration does not revise it — it converges the work term, not the temperature. Photoelectric heating can change the gas temperature materially within a step, so **this mode assumes the opacity does not depend on the gas temperature.** That holds for its intended use, since ultraviolet dust opacity is a property of the grains rather than of the gas, and it is exact whenever `DefineOpacityExponentsAndLowerValues` ignores its `Tgas` argument. Otherwise the opacity lags the heating by one step and the error is first order in \\(\Delta t\\).
- **\\(\epsilon\_g\\) is a compile-time constant**, so it cannot depend on the local electron density or grain charge. A problem needing an efficiency that varies with the electron density or the field strength is not served by this interface.

Because this deposits photoelectric heating inside Quokka, no other part of the calculation may do so as well. Two guards enforce that at startup: the Grackle cooling table must not itself include photoelectric heating, and `use_sfh_based_pe_heating` — which answers the same question from a global star formation rate instead of the local field — must be off.

This is a separate mechanism from `ISM_Traits::enable_photoelectric_heating`, which applies to thermal bands in the gas-dust thermal coupling model. The two are mutually exclusive; unifying them is future work.

### Reduced speed of light

To relax the radiation timestep, the radiation subsystem may be solved with a reduced speed of light \\(\hat{c} < c\\) (the RSLA), set through `c_hat_over_c`. This scales the transport term by \\(\hat{c}/c\\) and leaves the equations exact when \\(\hat{c} = c\\) (the default). \\(\hat{c}\\) must remain much larger than every hydrodynamic speed in the problem. Energy and momentum are conserved to machine precision only for \\(\hat{c} = c\\).

## Numerical method

The scheme is developed and tested in [@He_2024] for grey radiation and extended to multigroup in [@He_2024b]; the underlying Godunov radiation solver is that of [@Wibking_2022]. A full timestep is operator-split into two parts.

1. **Hydrodynamic transport**, advanced explicitly with the PPM + RK2-SSP Godunov scheme described in [@Wibking_2022].
2. **Radiation transport and matter-radiation coupling**, advanced with an implicit-explicit (IMEX) scheme, subcycled with respect to the hydro step at `radiation.cfl` (see [Runtime parameters](#runtime-parameters)).

Within the radiation step, the transport terms \\(\nabla \cdot \boldsymbol{F}\_g\\) and \\(\nabla \cdot \mathsf{P}\_g\\) are treated **explicitly** and the four-force terms **implicitly**. This split is what keeps the method cheap on GPUs: the implicit part contains no spatial derivatives, so every cell is solved independently and the radiation update needs no more communication than a pure hydro update.

### IMEX PD-ARS

The two parts are combined with the asymptotic-preserving IMEX PD-ARS integrator, which advances the state \\(\boldsymbol{U}\\) over \\(\Delta t\\) in two stages,

<script type="math/tex; mode=display">
\begin{aligned}
\boldsymbol{U}^{(n+1/2)} &= \boldsymbol{U}^{(n)} + \Delta t \, \mathsf{T}(\boldsymbol{U}^{(n)}) + \Delta t \, \mathsf{S}(\boldsymbol{U}^{(n+1/2)}) \, , \\
\boldsymbol{U}^{(n+1)} &= \boldsymbol{U}^{(n)} + \frac{\Delta t}{2} \left[ \mathsf{T}(\boldsymbol{U}^{(n)}) + \mathsf{T}(\boldsymbol{U}^{(n+1/2)}) \right] + \frac{\Delta t}{2} \left[ \mathsf{S}(\boldsymbol{U}^{(n+1/2)}) + \mathsf{S}(\boldsymbol{U}^{(n+1)}) \right] ,
\end{aligned}
</script>

with \\(\mathsf{T}\\) the transport terms and \\(\mathsf{S}\\) the source terms. Transport and source terms enter symmetrically at both stages, which is what allows the near-exact cancellation between them that the diffusion limit requires. The scheme is second-order accurate and reduces to RK2-SSP in the streaming limit. [@He_2024] give the Butcher tableaux, the formal asymptotic analysis of the diffusion limit, and the numerical tests; the stage-by-stage mapping onto the code is in [Radiation Integrator](radiation_integrator.md).

One practical consequence, demonstrated in [@He_2024]: because the scheme is asymptotic-preserving by construction, the *ad hoc* correction to the HLL wavespeeds that earlier explicit radiation schemes needed in order to recover the diffusion limit is no longer required. Quokka uses the uncorrected HLL fluxes with PPM reconstruction by default. The correction survives only as the diagnostic flag `use_wavespeed_correction_`, used by `RadMarshakAsymptotic` to demonstrate what it does.

### The implicit solve

Each implicit stage solves, cell by cell, a system of \\(4 + 4 N\_g\\) equations for the gas energy, the gas momentum, and the energy and flux of every group. Following [@Howell_2003] and [@Wibking_2022], and as generalised to multigroup in [@He_2024b], it is split into two nested iterations:

- an **inner** Newton-Raphson iteration over the \\(1 + N\_g\\) energy variables (gas energy and the group exchange terms \\(R\_g\\)), with \\(\boldsymbol{v}\\) and \\(\boldsymbol{F}\_g\\) frozen;
- an **outer** iteration that updates \\(\boldsymbol{F}\_g\\) and the gas momentum analytically, then returns to the inner solve if the velocity-dependent terms have changed.

The description above is that of the **thermal** groups, which are the only ones genuinely coupled to the gas. Their part of the inner Jacobian is sparse — each group couples to the gas but not directly to any other group — so [@He_2024b] invert it by Gauss-Jordan elimination in \\(O(N\_g)\\) operations rather than \\(O(N\_g^3)\\). Outside the dynamic diffusion limit the outer loop almost always converges in one pass. The gas energy is recovered from the converged exchange terms rather than solved for independently, which is what makes the update conservative to machine precision regardless of how tightly the iteration converged. Convergence tolerances are set by `radiation.iteration_tolerance` and `radiation.iteration_tolerance_rel`; the choice of per-group unknown and the round-off floor on the residual are discussed in [Radiation Integrator](radiation_integrator.md).

**Under `dust_absorption_only` there is no coupled iteration at all.** Because these bands do not emit, their exchange term does not depend on the gas temperature; and because their absorbed energy is not given to the gas, the gas energy does not depend on theirs. Nothing couples, so the Newton-Raphson solve is skipped outright and each group is updated in closed form, \\(E\_g \to (E\_g + S\_g + W\_g) / (1 + \hat{c} \\, \rho \kappa\_{0E,g} \\, \Delta t)\\), where \\(W\_g\\) is the work term. The outer iteration remains and is the only iteration left: it converges \\(W\_g\\), which is the one quantity these bands deliver to the gas, and is how radiation pressure keeps doing work on it even though no heat is exchanged.

## Multigroup opacity models

The group-integrated four-force given in [The multigroup four-force](#the-multigroup-four-force) involves integrals of the opacity multiplied by a radiation quantity over each group, so a model for the frequency dependence of \\(\chi\_0\\) *within* a group is needed. Quokka offers two, both introduced in [@He_2024b] and selected with the `opacity_model` trait. [Which model to use](#which-model-to-use) gives the recommendation, and [Multigroup opacities](#multigroup-opacities) shows how to supply either from a problem generator.

### Piecewise constant (PC)

The opacity is taken to be constant across each group, \\(\chi\_0(\nu) = \chi\_{0,g}\\). The Planck-, energy-, and flux-mean opacities of a group are then all equal to that one value, and the \\(\partial \chi\_0 / \partial \nu\\) terms vanish. This is the simplest model and the one used by most earlier multigroup codes. It is accurate when the frequency grid is fine enough to resolve the variation of the opacity, and inaccurate when it is not. It is the case for which the four-force reduces to the compact form at the end of [The multigroup four-force](#the-multigroup-four-force).

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

- **Fixed slope** (`PPL_opacity_fixed_slope_spectrum`): assume \\(\nu Q\_\nu\\) is constant across each group, i.e. \\(\alpha\_{Q,g} = -1\\). This is not arbitrary — weighted by the spectrum itself, the mean power-law index of any spectrum that integrates to a finite energy is exactly \\(-1\\). It costs almost nothing and performs well even with a handful of groups. This is the PPL variant to use.
- **Full spectrum** (`PPL_opacity_full_spectrum`): fit \\(\alpha\_{E,g}\\) and \\(\alpha\_{B,g}\\) to the actual radiation and Planck spectra on the fly, refitting during the first few Newton iterations. This is significantly more expensive and, in the tests of [@He_2024b], never more accurate than the fixed-slope method at any number of groups. It is kept for testing and is **not recommended for production at any frequency resolution**.

Note that \\(\alpha\_{Q,g}\\) matters only when \\(\alpha\_{\chi\_0,g} \ne 0\\): the spectrum shape within a group is relevant only if the opacity varies across that group. Setting all exponents to zero reduces PPL to PC.

### Which model to use

The choice is governed by frequency resolution, and the deciding comparison is between PC and PPL fixed slope — not between the two PPL variants. [@He_2024b] measure the convergence of all three models on a Marshak wave with a continuously varying opacity \\(\chi\_0 \propto \nu^{-2}\\) at 4, 8, and 16 groups, and recommend:

| Frequency resolution | Recommended model | Why |
| --- | --- | --- |
| \\(N\_g \lesssim 10\\) | `PPL_opacity_fixed_slope_spectrum` | Best accuracy. With few groups the opacity varies strongly across each bin, which is exactly what PC cannot represent. |
| \\(N\_g \gtrsim 10\\) | `piecewise_constant_opacity` | Bins are narrow enough that a constant opacity is a good approximation, so PPL buys no accuracy, and PC is slightly cheaper because no slopes are evaluated. |

The sharper form of the criterion is stated in terms of bin width rather than group count: use PC whenever the bin width in logarithmic frequency is smaller than about \\(W/10\\), where \\(W\\) is the logarithmic width of the whole frequency range, and PPL fixed slope otherwise.

Two things are worth knowing about the ends of this range. At high resolution PC is not merely adequate but marginally the *most* accurate of the three, because the flux-mean opacity relation of [Flux-mean opacity](#flux-mean-opacity) introduces a small error that the PPL group means do not cancel. At low resolution the advantage of PPL is large: in the advecting radiation pulse test of [@He_2024b], PPL fixed slope with 4 groups has errors two to three times smaller than PC with 4 groups — comparable to PC with 8 — and 4 groups with PPL fixed slope is already enough to keep every quantity accurate to better than 10 per cent. The full-spectrum variant is not recommended in either regime.

### Flux-mean opacity

The flux-mean opacity \\(\chi\_{0F,g}\\) that appears in the four-force is not supplied by the user. It is computed internally from \\(\chi\_{0E,g}\\), \\(\chi\_{0B,g}\\), and the gas temperature so that \\(\boldsymbol{G}\_g \to 0\\) in an optically thick moving medium. As shown in [@He_2024b], enforcing this relation is what guarantees that the multigroup scheme reaches the correct diffusion limit; in the PC case it reduces to \\(\chi\_{0F,g} = \chi\_{0,g}\\), as expected.

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
| `opacity_model`  | `OpacityModel`             | `single_group`   | `single_group` for grey radiation; when `nGroups > 1`, one of the multigroup models — see [Which model to use](#which-model-to-use). |

### Grey opacities

For `nGroups = 1`, define the mean opacities \\(\chi\_{0P}\\), \\(\chi\_{0F}\\), and \\(\chi\_{0E}\\) of [Matter-radiation coupling](#matter-radiation-coupling) as functions of density and gas temperature. Each returns a mass opacity in \\(\mathrm{cm^2\\,g^{-1}}\\). Only `ComputePlanckOpacity` is mandatory; the other two default to it.

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

For `nGroups > 1` the grey hooks are not used. Instead specialise a single function, `DefineOpacityExponentsAndLowerValues`, which returns two arrays of length `nGroups + 1`: the power-law exponents \\(\alpha\_{\chi\_0,g}\\) in element `[0]`, and the opacity at the lower edge of each group, \\(\kappa\_{0,g-}\\), in element `[1]`. These are the two quantities that define the power law in [Piecewise power law (PPL)](#piecewise-power-law-ppl); under the PC model the exponents are ignored and `[1][g]` is used directly as the constant opacity of group `g`. The three group means \\(\chi\_{0B,g}\\), \\(\chi\_{0E,g}\\), and \\(\chi\_{0F,g}\\) that enter the four-force are derived from them internally.

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

### Declaring band types

By default every group is a thermal band. Setting `dust_absorption_only` on `RadSystem_Traits` turns every non-chemical group into a dust-absorption band for the whole run:

```c++
template <> struct RadSystem_Traits<MyProblem> {
	static constexpr double c_hat_over_c = 1.0;
	static constexpr double energy_unit = C::ev2erg;
	// two bands: far-ultraviolet and Lyman-Werner
	static constexpr amrex::GpuArray<double, 3> radBoundaries = {6.0, 11.2, 13.6};
	static constexpr bool dust_absorption_only = true;
	static constexpr OpacityModel opacity_model = OpacityModel::piecewise_constant_opacity;
	// photoelectric efficiency of each band; omit it entirely for no photoelectric heating
	static constexpr amrex::GpuArray<double, 2> pe_heating_efficiency = {0.01, 0.01};
};
```

The flag defaults to `false`, requires `nGroups > 1`, and cannot be combined with `ISM_Traits::enable_dust_gas_thermal_coupling_model` or `ISM_Traits::enable_photoelectric_heating` — each of those assumes a thermal exchange this mode deliberately removes, so the combination is rejected at compile time. Chemical bands are declared separately with `ChemBands()`, which returns their boundaries because the photochemistry network needs them; see [Photoionization](photoionization.md).

`pe_heating_efficiency` defaults to all zeros, in which case no photoelectric heating is applied; see [Photoelectric heating](#photoelectric-heating). Each entry must lie in \\([0, 1]\\), and a non-zero entry requires both `dust_absorption_only` and cgs units; all three are checked at compile time.

The opacity hook is unchanged: `DefineOpacityExponentsAndLowerValues` supplies \\(\kappa\\) for every group as usual. Return the dust absorption opacity of each band; the solver uses it for the absorption sink, the radiation force, and the work term, and never asks for an emissivity.

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
- `RadDustAbsorption` — dust-absorption-only bands: beam attenuation, radiation force, and photoelectric heating of an interstellar slab against the analytic profile.
- `RadDustAbsorptionPPL` — the same problem under a piecewise power-law opacity model.

## References

Because the three Quokka methods papers share authors and year, the short citations above do not distinguish them on sight. They are, in the order a reader should approach them:

- [@Wibking_2022] — the original Quokka paper: the Godunov radiation solver, PPM reconstruction and HLL fluxes for the radiation moments, and the M1 closure. Start here.
- [@He_2024] — *An asymptotically correct implicit-explicit time integration scheme for finite volume radiation-hydrodynamics*. The IMEX PD-ARS scheme of [Numerical method](#numerical-method), its asymptotic analysis in the static and dynamic diffusion limits, and the removal of the wavespeed correction.
- [@He_2024b] — *A novel numerical method for mixed-frame multigroup radiation-hydrodynamics with GPU acceleration implemented in the QUOKKA code*. Everything multigroup: the group-integrated four-force of [The multigroup four-force](#the-multigroup-four-force), the PC and PPL [opacity models](#multigroup-opacity-models), and the sparse Newton solve.

The mixed-frame formulation itself follows [@MihalasMihalas] and [@Krumholz2007], and the inner Newton-Raphson iteration follows [@Howell_2003]. If you use the radiation module, please cite the papers that apply to your work — see [Citation](citation.md).
