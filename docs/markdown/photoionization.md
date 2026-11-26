# Photoionization Module

> **Warning: Beta feature**
>
> The photoionization module has not yet been exercised in a published science application with Quokka and should currently be treated as **beta**.
>

**Reference:** Aubert & Teyssier (2008), "A radiative transfer scheme for cosmological
reionization based on a local Eddington tensor" (ATON paper, arXiv:0709.1544)

## 1. Governing Equations

### 1.1 M1 Radiative Transfer for Ionizing Photons

Taking the first two moments of the radiative transfer equation gives conservation laws
for the ionizing photon number density \\(N\_\gamma\\) and flux density \\(\mathbf{F}\_\gamma\\):

<script type="math/tex; mode=display">
\frac{\partial N_\gamma}{\partial t} + \nabla \cdot \mathbf{F}_\gamma =
- n_{\rm H^0} c \sigma_\gamma N_\gamma
+ n_e n_{\rm H^+} (\alpha_A - \alpha_B)
+ \dot{N}^*_\gamma,
</script>

<script type="math/tex; mode=display">
\frac{\partial \mathbf{F}_\gamma}{\partial t} + c^2 \nabla \cdot \mathsf{P}_\gamma =
- n_{\rm H^0} c \sigma_\gamma \mathbf{F}_\gamma.
</script>

| Symbol | Definition |
|--------|------------|
| \\(N\_\gamma\\) | Ionizing photon number density (\\(\mathrm{cm}^{-3}\\)) |
| \\(\mathbf{F}\_\gamma\\) | Ionizing photon number flux density (\\(\mathrm{cm}^{-2}\ \mathrm{s}^{-1}\\)) |
| \\(\mathsf{P}\_\gamma\\) | Radiation pressure tensor (\\(= \mathsf{D}\,F\_\gamma\\), \\(\mathrm{cm}^{-3}\\)) |
| \\(n\_{\rm H^0}\\) | Neutral hydrogen number density |
| \\(n\_{\rm H^+} = n\_e\\) | Ionized hydrogen / electron number density |
| \\(\sigma\_\gamma\\) | Frequency-averaged photoionization cross-section |
| \\(\alpha\_A, \alpha\_B\\) | Case A / B recombination coefficients (\\(\mathrm{cm}^{3}\ \mathrm{s}^{-1}\\)) |
| \\(\beta\\) | Collisional ionization rate coefficient (\\(\mathrm{cm}^{3}\ \mathrm{s}^{-1}\\)) |
| \\(\dot{N}^*\_\gamma\\) | Stellar ionizing photon emission rate (\\(\mathrm{cm}^{-3}\ \mathrm{s}^{-1}\\)) |

The source term \\(n\_e n\_{\rm H^+}(\alpha\_A - \alpha\_B)\\) represents diffuse recombination
radiation — photons re-emitted when H recombines directly to the ground state (case A
minus case B correction).

### 1.2 Hydrogen Thermochemistry

The neutral hydrogen fraction evolves as:

<script type="math/tex; mode=display">
\frac{D n_{\rm H^0}}{Dt} = \alpha_A n_e n_{\rm H^+} - \beta n_e n_{\rm H^0} - \Gamma_{\gamma {\rm H}^0} n_{\rm H^0},
</script>

with \\(n\_{\rm H^+} = n\_e\\) (charge conservation), \\(n\_{\rm H^+} + n\_{\rm H^0} = n\_{\rm H}\\)
(nuclei conservation), and the photoionization rate \\(\Gamma\_{\gamma {\rm H}^0} = c \sigma\_\gamma N\_\gamma\\).

The gas thermal energy evolves as:

<script type="math/tex; mode=display">
\rho \frac{D}{Dt}\!\left(\frac{e}{\rho}\right) = \mathcal{H}_{\rm photo} - \mathcal{L},
</script>

where \\(\mathcal{H}\_{\rm photo} = n\_{\rm H^0} c \sigma\_\gamma \epsilon\_\gamma N\_\gamma\\) is
the photoheating rate and \\(\epsilon\_\gamma = h(\bar{\nu} - \nu\_{{\rm H}^0})\\) is the mean
excess photon energy above the ionization threshold (29.65 eV for a \\(10^5\\) K blackbody).
For heating and cooling that are not directly due to hydrogen photoionization, we
reimplement the optically thin prescription of [@Krumholz2007], which proceeds as
follows. In molecular gas, the approximate cooling and heating functions of [@Koyama2002]
are used. In partially ionized gas, the cooling rate is computed following
[@Osterbrock1989], which includes cooling by ion-electron collisions involving the first
and second ionized states of O, N, and Ne — the dominant coolants in H II regions at
solar metallicity. A future PR will extend the cooling model to use the RIGEL 
prescription [@Deng2024].

### 1.3 On-the-Spot Approximation (OTSA)

When a hydrogen ion recombines directly to the ground state (n = 1), the emitted
Lyman-continuum photon is immediately capable of re-ionizing a nearby neutral hydrogen
atom. The **on-the-spot approximation** assumes this photon is re-absorbed locally —
within the same resolution element — so recombinations to n = 1 have no net effect on
the ionization state.

Under OTSA, one replaces \\(\alpha\_A \to \alpha\_B\\) everywhere and drops the diffuse
recombination source term from the radiation equation:

<script type="math/tex; mode=display">
\frac{\partial N_\gamma}{\partial t} + \nabla \cdot \mathbf{F}_\gamma =
- n_{\rm H^0} c \sigma_\gamma N_\gamma + \dot{N}^*_\gamma,
</script>

<script type="math/tex; mode=display">
\frac{D n_{\rm H^0}}{Dt} = \alpha_B n_e n_{\rm H^+} - \beta n_e n_{\rm H^0} - \Gamma_{\gamma {\rm H}^0} n_{\rm H^0}.
</script>

OTSA is valid when the mean free path of a recombination photon is much smaller than
the size of the ionized region — a good approximation deep inside large HII regions.
It breaks down near ionization fronts and in low-density, nearly fully ionized gas.

Quokka currently uses OTSA. The full \\((\alpha\_A - \alpha\_B)\\) diffuse source term is
planned for a future phase.

## 2. Numerical Scheme

The update is decomposed into three sequential operators per timestep, following ATON:

```
1. Stellar source step     Particle injection -> radEnergySource
2. Transport step           Explicit RK stages: advanceRadiation*
3. Thermochemical step      VODE ODE integration over the coupled
                            photoionization network
```

### 2.1 Thermochemical Implicit Solve via VODE

The stiffest part is the coupled, non-linear evolution of the photoionization network
in each cell. Quokka replaces the analytic cubic-polynomial solve used in ATON (which
cannot generalize to more complex networks) with a call to **VODE**, a variable-order,
variable-step stiff ODE integrator.

Under OTSA, VODE integrates the following system over the implicit timestep \\(\Delta t\\):

<script type="math/tex; mode=display">
\begin{aligned}
\frac{d N_\gamma}{d t} &= - n_{\rm H^0} \hat{c} \sigma_\gamma N_\gamma + \dot{N}, \\[2pt]
\frac{d F_{\gamma,i}}{d t} &= - n_{\rm H^0} \hat{c} \sigma_\gamma F_{\gamma,i} \quad (i = x, y, z), \\[2pt]
\frac{d n_{\rm H^0}}{d t} &= \alpha_B n_e n_{\rm H^+} - \beta n_e n_{\rm H^0} - \hat{c} \sigma_\gamma N_\gamma n_{\rm H^0}, \\[2pt]
\frac{d n_{\rm H^+}}{d t} &= -\alpha_B n_e n_{\rm H^+} + \beta n_e n_{\rm H^0} + \hat{c} \sigma_\gamma N_\gamma n_{\rm H^0}, \\[2pt]
\frac{d n_e}{d t} &= -\alpha_B n_e n_{\rm H^+} + \beta n_e n_{\rm H^0} + \hat{c} \sigma_\gamma N_\gamma n_{\rm H^0}, \\[2pt]
\frac{d e}{d t} &= n_{\rm H^0} \hat{c} \sigma_\gamma \epsilon_\gamma N_\gamma - \text{[cooling terms]}.
\end{aligned}
</script>

where \\(\hat{c}\\) is the reduced speed of light. Only one directional flux component is
integrated (normalized to 1.0 before the burn); the other two are scaled proportionally
after the ODE solve. The state vector has 6 components for a single chemical band:
\\((n\_e, n\_{\rm H^0}, n\_{\rm H^+}, e, N\_\gamma, F\_\gamma)\\).

Note that \\(n\_{\rm H^+} = n\_{\rm H} - n\_{\rm H^0}\\) and \\(n\_e = n\_{\rm H^+}\\) by
construction. Although only one of \\(n\_{\rm H^0}\\) or \\(n\_{\rm H^+}\\) is an independent
variable, both are integrated for symmetry.

The flux ODE \\(dF\_\gamma/dt = -n\_{\rm H^0} \hat{c} \sigma\_\gamma F\_\gamma\\) is similar
to the absorption term in \\(N\_\gamma\\), but \\(N\_\gamma\\) has an additional isotropic source
term (stellar emission \\(\dot{N}\\)). Isotropic sources add photons uniformly in all
directions — they contribute to \\(N\_\gamma\\) but produce no net flux. Flux must be
integrated to track the attenuation of the directional radiation field across the
timestep.

## 3. VODE Tolerances

### 3.1 Overview

Quokka uses VODE (via Microphysics) to integrate the chemistry and internal energy
source terms. The integrator requires absolute tolerances (`atol`) for each solution
variable. These are hand-tuned for each problem and specified directly in the input
file (see § 3.2). The `SetAtolFromPhysics` machinery (PR #1980) will derive tolerances
from physical scales automatically in a future PR.

### 3.2 Input parameters

| Parameter                               | Description                                                       |
| --------------------------------------- | ----------------------------------------------------------------- |
| `integrator.atol_spec`                  | Absolute tolerance for chemical species (\\(\mathrm{cm}^{-3}\\))                   |
| `integrator.atol_enuc`                  | Absolute tolerance for gas internal energy (\\(\mathrm{erg}\ \mathrm{g}^{-1}\\))             |
| `integrator.atol_rad_num`               | Absolute tolerance for photon number density (\\(\mathrm{cm}^{-3}\\))              |
| `integrator.rtol_spec`                  | Relative tolerance for chemical species                           |
| `integrator.rtol_enuc`                  | Relative tolerance for gas internal energy                        |
| `integrator.rtol_rad_num`               | Relative tolerance for photon number density                      |
| `integrator.species_failure_tolerance`  | VODE internal substep rejection threshold for negative species (\\(\mathrm{cm}^{-3}\\), see § 3.7) |
| `integrator.radiation_failure_tolerance`| VODE internal substep rejection threshold for negative photon density (\\(\mathrm{cm}^{-3}\\), see § 3.5) |

### 3.3 Why flux is excluded from convergence

The radiation flux \\(F\_\gamma\\) (normalized to 1.0 before the ODE) is integrated alongside the
other variables, but does not participate in any VODE convergence or error checks.

**Why flux is in the ODE.** The flux ODE is \\(dF/dt = -(\hat{c}\sigma)\,n\_{\rm H^0} F\\).
This is similar to the absorption term in \\(N\_\gamma\\), but \\(N\_\gamma\\) also has an
isotropic source term (stellar emission in OTSA, or \\(n\_e n\_{\rm H^+}(\alpha\_A - \alpha\_B)\\)
recombination radiation in case A). Since isotropic sources contribute photons uniformly
in all directions, they add to the photon number density but produce no net flux. Flux
must be integrated separately to track the attenuation of the directional radiation field.

**Why flux is excluded from convergence.** Flux is a passive scalar — its RHS depends
on \\(n\_{\rm H^0}\\) but flux does *not* appear in any other equation (species, energy, or
\\(N\_\gamma\\)). Convergence should be driven by the physically consequential quantities, not
by a diagnostic variable. In dark cells where flux goes to 0, demanding 1% accuracy on a
near-zero value wastes VODE steps with no physical benefit.

Excluding flux from convergence gave a **3.8× speedup** in photochemistry on CPU and a
**2.2× speedup** on GPU for the DTypeFront_JAFF test.

### 3.4 Physical constants

| Symbol  | Value                                   | Description                                                      |
| ------- | --------------------------------------- | ---------------------------------------------------------------- |
| `a_rad` | \\(7.5657 \times 10^{-15}\ \mathrm{erg}\ \mathrm{cm}^{-3}\ \mathrm{K}^{-4}\\) | Radiation constant (from `fundamental_constants.H`)              |
| `k_B`   | \\(1.380649 \times 10^{-16}\ \mathrm{erg}\ \mathrm{K}^{-1}\\)                 | Boltzmann constant                                               |
| `m_p`   | \\(1.67262192 \times 10^{-24}\ \mathrm{g}\\)                      | Proton mass                                                      |
| `c_v`   | \\(\frac{3}{2} k\_B / m\_p \sim 1.24 \times 10^{8}\ \mathrm{erg}\ \mathrm{g}^{-1}\ \mathrm{K}^{-1}\\) | Specific heat of monatomic hydrogen gas                           |

### 3.5 radiation_failure_tolerance

VODE uses this threshold in two places:

1. **Internal substeps:** if the photon number density becomes more negative than
   `radiation_failure_tolerance`, VODE rejects the substep and retries with a smaller
   timestep. This is the primary use.
2. **Final state:** after interpolating to the output time, if the photon number density
   is more negative than \\(1.5 \times\\) `radiation_failure_tolerance`, the burn is
   declared failed. The 1.5× factor (via `vode_final_state_radiation_failure_tolerance_factor`
   in `vode_type.H`) accounts for VODE's non-monotonic interpolation, preventing false
   failures from interpolation noise.

Set `radiation_failure_tolerance` equal to `atol_rad_num` (the photon negligibility
floor). The \\(1.5\times\\) final-state factor absorbs BDF interpolation overshoot without
manual inflation.

Physically, the amount of spurious ionization that can be produced by a negative photon
overshoot is at most `radiation_failure_tolerance` / \\(n\_{\rm H}\\). Whether this
matters depends on two regimes:

1. **Bright cells** (\\(N\_\gamma \gtrsim n\_{\rm H}\\)): the cell is fully ionized. A few
   percent error in photon count does not change the outcome.
2. **Dark cells** (\\(N\_\gamma \ll n\_{\rm H}\\)): the ratio is negligible as long as
   `radiation_failure_tolerance` \\(\ll n\_{\rm H}\\).

For low-density environments such as the CGM or IGM, where the ionized gas density can
be \\(\sim 10^{-4}\\)–\\(10^{-3}\ \mathrm{cm}^{-3}\\), the default value of this parameter may compete
with the physical ionization equilibrium — override it in the input file.

### 3.6 Erad_floor

`Erad_floor` is a compile-time `constexpr` in `RadSystem_Traits<problem_t>` that sets
the M1 hyperbolic solver floor — it prevents the radiation moment solver from
encountering zero energy density. It is **independent** of the VODE tolerances.

Define the equivalent floor temperature \\(T\_{\rm floor}\\) by \\(E\_{\rm rad, floor} \equiv a\_{\rm rad} T\_{\rm floor}^4\\).
The photon number density at the floor is \\(N\_{\gamma,{\rm floor}} = E\_{\rm rad, floor} / E\_{\rm photon}\\).

Dark cells (where \\(E\_{\rm rad} \approx E\_{\rm rad, floor}\\)) converge in one VODE step when
\\(\texttt{atol\_rad\_num} \gg N\_{\gamma,{\rm floor}}\\). A ratio of \\(\geq 10^4\\) is sufficient:

<script type="math/tex; mode=display">
\frac{\texttt{atol\_rad\_num}}{N_{\gamma,{\rm floor}}} \geq 10^4.
</script>

For typical `Erad_floor` values corresponding to \\(T\_{\rm floor} = 0.01\\)–\\(1\\) K, an
`atol_rad_num` on the order of \\(10^{-6}\\)–\\(10^{-2}\ \mathrm{cm}^{-3}\\) satisfies this constraint.

### 3.7 species_failure_tolerance

VODE uses this threshold in two places:

1. **Internal substeps (primary):** if a species number density becomes more negative
   than `species_failure_tolerance`, VODE rejects the substep and retries with a smaller
   timestep.
2. **Final state (secondary):** after interpolating to the output time, if a species is
   more negative than \\(1.5 \times\\) `species_failure_tolerance`, the burn is declared
   failed. The 1.5× factor (via `vode_final_state_species_failure_tolerance_factor` in
   `vode_type.H`) accounts for VODE's non-monotonic interpolation, preventing false
   failures from interpolation noise.

Set `integrator.species_failure_tolerance` equal to `atol_spec` (the species
negligibility floor). The \\(1.5\times\\) final-state factor absorbs BDF interpolation
overshoot without manual inflation.

## 4. Compatibility

### 4.1 Incompatibility with resampled cooling

Setting both `photochemistry.enabled = 1` and `cooling.enabled = 1` in the same
input file is a **fatal error** — Quokka will abort at startup.

The resampled Cloudy cooling table (`cooling.cooling_table_type = "resampled"`)
encodes the full H/He thermochemistry for an optically thin gas in a UV background,
including:

- photoheating by the UV background,
- recombination cooling,
- collisional ionization and excitation cooling.

The photoionization module computes the same processes from first principles using
the M1 radiation field and the hydrogen chemistry network (§ 1). Enabling both
simultaneously would double-count every one of these rates, producing physically
incorrect temperatures and ionization fractions.

**Correct setup:** use `photochemistry.enabled = 1` and leave `cooling.enabled = 0`
(the default). The photoionization chemistry network handles all heating and cooling
internally.

## 5. Cooling and heating for galaxy simulations

The network `thermophotochemistry_ism_handwritten` adds a cooling and heating model for isolated galaxy simulations at a resolution of a few pc, with peak densities of \\(n\_{\rm H}\sim100\ \mathrm{cm}^{-3}\\).
It covers hot supernova-heated gas, H II regions, and the warm and cold neutral medium.
It has no H\\(\_2\\) or CO chemistry and no gas–dust thermal coupling.
The model follows the RIGEL [@Deng2024] and TIGRESS-NCR [@Kim2023] prescriptions, with the rate fits of [@Gong2017], [@Wolfire2003] and [@Weingartner2001].

A problem opts in by using this network and by defining `PHOTOCHEM_ISM_COOLING` at compile time (see `src/problems/ISMCoolingOneZone/CMakeLists.txt`).
The new cooling and heating code is under that definition, so the other photoionization problems are unchanged. The dust absorption without the gas–dust coupling model (Sect. 5.5) is selected by the problem's `ISM_Traits` instead.
All rates below are volumetric, in \\(\mathrm{erg}\ \mathrm{cm}^{-3}\ \mathrm{s}^{-1}\\), with cooling and heating both counted as positive numbers.
Number densities always come from the species: \\(n\_{\rm H} = n\_{\rm HI} + n\_{\rm H^+}\\), never \\(\rho/m\_{\rm H}\\).
Helium is not in the network or in the photochemistry equation of state, so the helium-to-hydrogen ratio is set to \\(y\_{\rm He}=0\\) in all rates.

### 5.1 Two regimes

In each hydro step on each level, after the hydro update and before the radiation substeps, every cell gets a temperature from the photochemistry equation of state.
A cell is **hot** if \\(T \geq T\_{\rm th} = 2\times10^4\ \mathrm{K}\\) and **cold** otherwise.
The label is fixed for the whole hydro step, including all radiation substeps.
Cells with a density below `photochemistry.min_density_allowed` are skipped by the burn and by the hot-cell step, and are never hot.

- **Cold cells** run the usual photochemistry burn in every radiation substep. The energy equation of the network includes the terms of Sect. 5.2.
- **Hot cells** skip the burn. The cell is set fully ionized, and its internal energy is integrated over the hydro step with the Grackle cooling table (Sect. 5.4). The ionizing photons that cross the cell are attenuated by dust only.

A cell that cools through \\(T\_{\rm th}\\) during the hot-cell step starts the next step as a cold cell that is fully ionized, and recombines out of equilibrium.

### 5.2 Cold-gas terms

The terms of the network energy equation are listed below.
All terms that use an electron density use \\(n\_e^{\max}\\) (Sect. 5.3), not the network electron density.
The H chemistry itself (recombination, collisional ionization) keeps the network electrons, so species are conserved.
The analytic Jacobian contains the derivative of every term with respect to \\(T\\) and to each species.
Below, \\(T\_4 = T/10^4\ \mathrm{K}\\), \\(Z' = Z/Z\_\odot\\) is the uniform metallicity, and \\(G\_0\\) is the far-UV field in Habing units.

| Term | Status | Source |
|------|--------|--------|
| Photoionization heating | kept | existing network |
| H recombination cooling | kept | existing network, network \\(n\_e\\) |
| H collisional ionization cooling | kept | existing network, network \\(n\_e\\) |
| Free-free cooling, \\(1.3\times1.427\times10^{-27}\sqrt{T}\\,n\_e n\_{\rm H^+}\\) | kept | existing network, network \\(n\_e\\) |
| Koyama–Inutsuka heating and cooling | removed | replaced by the terms below |
| Metal cooling in ionized gas (`get_cle_term`) | removed | replaced by nebular lines |
| Ly\\(\alpha\\) cooling | added | [@Kim2023] Eq. 42 |
| Nebular metal lines | added | [@Kim2023] Eq. 47 |
| C II 158 \\(\mu\\)m and O I 63, 146 \\(\mu\\)m cooling | added | [@Deng2024] App. A.4, collision rates of [@Gong2017] |
| Photoelectric (PE) heating | added | [@Deng2024] App. A.7, after [@Wolfire2003] |
| Grain-recombination cooling | added | [@Kim2023] Sect. 4.6.3, after [@Weingartner2001] |
| Cosmic-ray (CR) heating | added | [@Deng2024] App. A.8 |

**Ly\\(\alpha\\).**

<script type="math/tex; mode=display">
\mathcal{L}_{\rm Ly\alpha} = E_{21}\, n_{\rm HI}\, n_e^{\max}\, k_{12}, \qquad
k_{12} = 5.31\times10^{-8}\, \frac{T_4^{0.15}}{1 + (T_4/5)^{0.65}}\, e^{-11.84/T_4}\ \mathrm{cm}^3\,\mathrm{s}^{-1},
</script>

with \\(E\_{21} = 10.2\ \mathrm{eV}\\).

**Nebular lines.** The coefficients are those printed in [@Kim2023]. The value \\(3.86\times10^{-23}\\) in the RIGEL paper differs from the \\(3.68\times10^{-23}\\) printed in NCR [@Kim2023]; we use the NCR value.

<script type="math/tex; mode=display">
\mathcal{L}_{\rm neb} = Z'\, n_e^{\max}\, n_{\rm H^+}\, \frac{3.68\times10^{-23}\, e^{-3.86/T_4}\, f_{\rm neb}(T)}{T_4^{1/2}\left(1 + 0.12\, n_{e,2}^{\,0.38 - 0.12\ln T_4}\right)}, \qquad
\log_{10} f_{\rm neb} = \sum_{i=0}^{6} a_i\, (\ln T_4)^i,
</script>

with \\(n\_{e,2} = n\_e^{\max}/(100\ \mathrm{cm}^{-3})\\) and \\((a\_0,\dots,a\_6) = (0.692, -0.586, 0.816, -0.505, 0.118, 7.66\times10^{-3}, -5.08\times10^{-3})\\).

**C II and O I fine-structure lines.** All carbon is C\\(^+\\), with \\(n\_{\rm C^+} = x\_{\rm C,tot}\\, n\_{\rm H}\\) and \\(x\_{\rm C,tot} = 1.6\times10^{-4} Z'\\).
Neutral oxygen follows neutral hydrogen, \\(n\_{\rm O} = x\_{\rm O,tot}\\, n\_{\rm HI}\\) with \\(x\_{\rm O,tot} = 3.2\times10^{-4} Z'\\).
C II is a two-level atom and O I a three-level atom, both with H I and electrons as collision partners and no H\\(\_2\\).
For C II, with level 1 above level 0,

<script type="math/tex; mode=display">
\mathcal{L}_{\rm CII} = n_{\rm C^+}\, A_{10}\, E_{10}\, \frac{q_{01}}{q_{01} + q_{10} + A_{10}}, \qquad
q_{ij} = n_{\rm HI}\, k^{\rm H}_{ij}(T) + n_e^{\max}\, k^{e}_{ij}(T),
</script>

and \\(q\_{01}\\) follows from \\(q\_{10}\\) by detailed balance.
The collision-rate fits \\(k\_{ij}\\) are those of [@Gong2017], as implemented in the Athena++ chemistry module.

**PE heating.**

<script type="math/tex; mode=display">
\mathcal{G}_{\rm PE} = 1.3\times10^{-24}\, \frac{Z_d}{0.01}\, \epsilon\, G_0\, n_{\rm H}, \qquad
\epsilon = \frac{0.0487}{1 + 0.004\,\psi^{0.73}} + \frac{0.0365\,(\tilde T/10^4)^{0.7}}{1 + 2\times10^{-4}\,\psi}, \qquad
\psi = \frac{G_0 \sqrt{\tilde T}}{0.5\, n_e^{\max}},
</script>

with \\(Z\_d = 0.5\\, Z = 0.5 \times 0.0136\\, Z'\\), so \\(Z\_d/0.01 = 0.68\\, Z'\\).
The temperature in \\(\epsilon\\) and \\(\psi\\) is capped, \\(\tilde T = \min(T, 10^4\ \mathrm{K})\\), the upper limit of the fit.
Without the cap, the \\(T^{0.7}\\) term makes PE heating exceed cooling in hot, dilute gas.

**Grain-recombination cooling.**

<script type="math/tex; mode=display">
\mathcal{L}_{\rm gr} = 10^{-28}\, Z'\, n_e^{\max}\, n_{\rm H}\, T^{\,D_0 + D_1/\ln x}\, \exp\!\left[D_2 + (D_3 - D_4 \ln x)\ln x\right], \qquad
x = \frac{G_0 \sqrt{T}}{n_e^{\max}} + 50,
</script>

with \\((D\_0,\dots,D\_4) = (0.4535, 2.234, -6.266, 1.442, 0.05089)\\), the coefficients of the Athena++ implementation by the TIGRESS-NCR authors.

**CR heating.**

<script type="math/tex; mode=display">
\mathcal{G}_{\rm CR} = 1.6022\times10^{-11}\ \mathrm{erg} \times \left(\zeta_{\rm HI}\, n_{\rm HI} + \zeta_{\rm HeI}\, n_{\rm HeI}\right), \qquad
\zeta_{\rm HeI} = 1.1\,\zeta_{\rm HI}, \quad n_{\rm HeI} = y_{\rm He}\, n_{\rm HI}.
</script>

The helium term is zero because \\(y\_{\rm He}=0\\).

Each of the new terms has its own switch (Sect. 5.6).
The network registers no thermal radiation bands, so all the energy these terms remove leaves the simulation.

### 5.3 Electron density for the rates

The cooling and heating rates above use

<script type="math/tex; mode=display">
n_e^{\max} = \max\!\left( n_{\rm H^+},\ x_{e,\rm CIE}(n_{\rm H}, T)\, n_{\rm H},\ x_{\rm C,tot}\, n_{\rm H},\ \sqrt{\zeta_{\rm HI}\, n_{\rm H} / \alpha_B(T)} \right).
</script>

The four terms are:

1. the electron density of the network;
2. the collisional-ionization-equilibrium (CIE) electron fraction from the Grackle table (Sect. 5.4), which includes ionization by the UV background;
3. the electrons that come from ionized carbon;
4. a cosmic-ray floor, from CR ionization balanced by case-B recombination with the network's \\(\alpha\_B(T)\\).

The Jacobian uses the derivatives of the term that is largest.
The maximum is needed because in neutral gas the network electron density is far below the electron density that carbon and the background UV field provide, and the rates would otherwise be too small.

### 5.4 Hot cells

In each hot cell, once per hydro step before the radiation substeps:

1. All hydrogen is moved to H\\(^+\\), keeping the number of H nuclei, and \\(n\_e = n\_{\rm H^+}\\). The gas internal energy is not changed.
2. The internal energy is integrated over the hydro step at fixed density with an adaptive Runge–Kutta method, using the equation below the list.
3. The temperature is taken from the photochemistry equation of state at each evaluation, and the table rate keeps applying if the cell cools below \\(T\_{\rm th}\\) within the step. At or below the larger of `temperature_floor` and the lowest table temperature, \\(10\ \mathrm{K}\\), net cooling by the table term is set to zero and only heating remains, so that a cell without a heating source cannot cool to zero internal energy. Because the cell is fully ionized, \\(\mathcal{G}\_{\rm CR} = 0\\). The result is clamped to the energy at `temperature_floor`. If `network.energy_switch = 0`, the integration is skipped and the internal energy is unchanged. The two heating terms follow `network.pe_heating_switch` and `network.cr_heating_switch`.
4. If the integration needs more than `maxStepsODEIntegrate` steps, the code prints the cell and aborts. There is no retry, because the step runs after the hydro update. Each call prints the number of hot cells and their average and maximum number of substeps.

The equation integrated in step 2 is

<script type="math/tex; mode=display">
\frac{dE_{\rm int}}{dt} = -n_{\rm H}^2\, \Lambda_{\rm net}(n_{\rm H}, T) + \mathcal{G}_{\rm PE} + \mathcal{G}_{\rm CR}.
</script>

The net cooling function \\(\Lambda\_{\rm net}\\) is the Grackle primordial plus metal cooling minus heating, from the Cloudy table `CloudyData_UVB=HM2012_shielded` at redshift 0, with the metal part scaled by \\(Z'\\), and without PE heating or Compton cooling.
It is stored on Grackle's native \\((\log n\_{\rm H}, \log T)\\) grid, which covers \\(-10 < \log\_{10}(n\_{\rm H}/\mathrm{cm}^{-3}) < 4\\) and \\(1 < \log\_{10}(T/\mathrm{K}) < 9\\), together with the CIE electron fraction \\(x\_{e,\rm CIE}\\), derived from the tabulated mean molecular weight.
Lookups are bilinear in the logarithms and clamp to the table edge, so they are finite for any positive input (the derivatives are zero outside the table).
The table for \\(Z' = 1\\) is `extern/cooling/CloudyData_UVB=HM2012_shielded_nT_Z1.h5`.
To make a table for another metallicity, run `extern/cooling/make_grackle_nT_table.py` (see the [cooling module](cooling_module.md)).
At startup the code aborts if `network.ism_metallicity` differs from the metallicity stored in the table.

In every radiation substep, if `DUST_CHEMICAL_BAND_ABSORPTION` is defined, the chemical-band photons in a hot cell are attenuated by dust only,

<script type="math/tex; mode=display">
N_\gamma \to N_\gamma\, e^{-\hat{c}\,\kappa_d \rho\, \Delta t}, \qquad \mathbf{F} \to \mathbf{F}\, e^{-\hat{c}\,\kappa_d \rho\, \Delta t},
</script>

with \\(\kappa\_d\\) = `network.dust_kappa`, and the photon number is kept at or above the floor `small_x`.
Hot gas is fully ionized, so neutral hydrogen does not absorb.
The gas receives the momentum of the absorbed photons, as in cold cells.

### 5.5 Dust absorption without gas–dust coupling

Dust absorption of ionizing photons no longer requires the gas–dust thermal coupling model (`enable_dust_gas_thermal_coupling_model`).
Without that model there are no thermal radiation groups to receive the absorbed energy, so the energy absorbed by dust leaves the simulation, in hot and in cold cells.
The gas still receives the momentum.
Total energy is therefore not conserved in this mode: the dust re-radiates the energy in the infrared, which is not followed.
When the coupling model is on, cold cells still pass the absorbed energy to the thermal groups, as before, and hot cells always discard it.

### 5.6 Parameters

The input parameters are listed in [Runtime parameters](parameters.md#photochemistry-cooling-for-galaxy-simulations).
Two are required when `PHOTOCHEM_ISM_COOLING` is defined, and the code aborts at startup, naming the parameter, if either is missing: `G0_external` and `ism_cooling.grackle_nT_table`.
`G0_external` is an AMReX parser expression for \\(G\_0\\) in Habing units, a function of `x`, `y`, `z`, `time` and `dt`, evaluated at the cell centre.
Any background value must be part of the expression.

### 5.7 Limitations

- The photochemistry equation of state is pure hydrogen, while the Grackle table assumes a hydrogen and helium mixture for its temperature axis and for \\(x\_{e,\rm CIE}\\). The \\(x\_{e,\rm CIE}\\) of fully ionized gas therefore includes helium electrons, and \\(T\\) is slightly inconsistent with the table until helium is added to the network.
- The metallicity is uniform and must equal the metallicity of the table.
- Hot, fully ionized cells receive no cosmic-ray heating, because \\(\mathcal{G}\_{\rm CR} \propto n\_{\rm HI}\\).
- The network has no H\\(\_2\\), CO or C I, and no gas–dust thermal coupling.

### 5.8 Verification

The tests are in `src/problems/ISMCoolingRates` (all rates and the table lookup against the Python reference `extern/cooling/ism_cooling_reference.py`), `src/problems/ISMCoolingOneZone` (Jacobian, startup checks, the temperature of 25 cells at \\(10^{-2} \leq n\_{\rm H} \leq 10^2\ \mathrm{cm}^{-3}\\) after 1 Gyr of evolution against the integrated ODE, the hot-cell step, a cell crossing \\(T\_{\rm th}\\), a dense hot cell without heating) and `src/problems/ISMCoolingSlab` (dust attenuation of a photon beam in a cold and a hot slab).
At \\(G\_0 = 1\\) and \\(Z' = 1\\), the temperature after 1 Gyr falls from 8200 K to 6600 K for \\(n\_{\rm H} \leq 0.15\ \mathrm{cm}^{-3}\\), drops to about 560 K at \\(n\_{\rm H} = 0.32\ \mathrm{cm}^{-3}\\), and falls from 72 K to 38 K between \\(n\_{\rm H} = 10\\) and \\(100\ \mathrm{cm}^{-3}\\).
