# Photoionization Module

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
**2.2× speedup** on GPU for the DTypeFront test.

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

## 5. Strömgren-Volume Subgrid Feedback

The M1 solver described above transports ionizing photons and solves the hydrogen chemistry explicitly. That is the accurate option, but it is expensive, and it only works when the H II region is resolved by several cells. In galaxy-scale and cloud-scale runs the H II region around a young star is often smaller than one cell, so the M1 solve costs a great deal and still cannot represent the feedback. For those cases Quokka provides a much cheaper subgrid alternative in `src/particles/particle_photoionization.hpp`, which transports no photons at all.

The module follows the Strömgren-volume technique of Kessel-Deynet & Burkert (2000) and Dale et al. (2007b), in the form used for stellar feedback by Hopkins et al. (2018, FIRE-2). For each star particle it finds the volume whose recombinations exactly consume the particle's ionizing photon budget $Q$, marks the gas inside as ionized, and holds it at a fixed temperature $T_{\rm HII}$. The resulting overpressure is what drives the H II region expansion. Like the M1 module, it assumes the on-the-spot approximation, which is why the case-B recombination coefficient appears.

FIRE-2 sorts the cells near a star by distance and walks outward consuming the budget. Because cells are consumed in order of increasing distance, the consumed set is always a distance-prefix, which is exactly a ball. The walk is therefore equivalent to finding the single radius $R_{\rm St}$ at which the enclosed recombination rate equals $Q$, and that equivalence is exact for an arbitrary, non-uniform density field. Quokka uses this to replace the cross-rank cell sort with a radial-bin histogram and one small MPI reduction per source, which parallelizes cleanly over AMReX boxes and ranks.

The ionizing photon rate is a per-particle property assigned once at birth from the stellar mass. `ToyStellarModel` takes it from the Sternberg, Hoffmann & Pauldrach (2003) grid as recalibrated by Martins, Schaerer & Hillier (2005). Over the O-star main sequence, roughly $15$--$60\,M_\odot$ at $Z \approx Z_\odot$, the rate is well represented by a cubic in $x \equiv \log_{10}(m/M_\odot)$:

$$\log_{10} Q_0\ [\mathrm{s}^{-1}] = c_0 + c_1 x + c_2 x^2 + c_3 x^3,$$

with $c_0 = 40.0765$, $c_1 = 10.7952$, $c_2 = -4.0785$, $c_3 = 0.5822$. These reproduce the published anchor points $\log_{10} Q_0 = 48.5$, $49.0$ and $49.5$ at $20$, $30$ and $50\,M_\odot$ respectively. The cubic is monotonic in mass everywhere, so it cannot produce the non-physical inversion in which a more massive star ionizes less.

Below $15\,M_\odot$ -- the lower edge of the fit, around spectral type B0 -- the rate is set to zero rather than extrapolating the polynomial. Extrapolation is badly behaved there: it would credit a $5\,M_\odot$ star with $\sim10^{46}$ photons per second against a true value near $10^{38}$, and in a sampled IMF low-mass stars vastly outnumber O stars, so the spurious contribution would dominate the budget.

The rate is stored in a dedicated particle component and is not refreshed as the particle accretes. The one exception is a star born below the cutoff, which carries $Q = 0$ and is re-evaluated until accretion carries it into the O-star range; at that point $Q$ is assigned from the mass it has then and frozen.

**Important:** the stellar model assigns $Q$ on the first update, and detects the "not yet assigned" state by the component still being zero. `amrex::ParticleContainer::InitFromAsciiFile` only fills the components present in the file and leaves the rest indeterminate, so a problem that creates star particles that way must explicitly zero the remaining real components. See `src/problems/StromgrenVolumeFeedback/` and `src/problems/ParticleStarEvolution/` for worked examples.

### 5.1 The ionized-fraction output slot

The ionized fraction is written into a passive scalar so that it appears in plotfiles. Two properties of that slot matter.

First, passive scalars are stored as **conserved densities**, so the slot holds $\rho\, x_{\rm ion}$, not $x_{\rm ion}$. Divide by the density to recover the fraction.

Second, the passive-scalar block **begins with the mass scalars** — `numPassiveScalars` counts the mass scalars as well as any extra ones. Pointing `stromgren.x_ion_scalar_index` at one of those slots would overwrite a species partial density every step, and `EnforceLimits` would then renormalize the corrupted value back into the composition. The module therefore rejects an index below `numMassScalars` or at or above `numPassiveScalars`, aborting with an explanatory message rather than silently corrupting the chemistry or silently writing nothing.

The field is recomputed from scratch each step from the current density and photon budgets, so it is a diagnostic output rather than an advected state variable.

### 5.2 Input parameters

| Parameter | Default | Meaning |
|---|---|---|
| `stromgren.enabled` | `0` | Master switch. |
| `stromgren.T_HII` | `1.0e4` | Temperature imposed on fully ionized gas (K). |
| `stromgren.alpha_B` | `2.59e-13` | Case-B recombination coefficient ($\mathrm{cm}^3\ \mathrm{s}^{-1}$). |
| `stromgren.hydrogen_mass_fraction` | `1.0` | Converts mass density to hydrogen number density. |
| `stromgren.R_max_cells` | `32.0` | Cap on the search radius, in cells. |
| `stromgren.x_ion_scalar_index` | `-1` | Passive scalar slot receiving the ionized fraction; negative disables the output. Must be $\geq$ `numMassScalars` and $<$ `numPassiveScalars` (validated at runtime; see below). |
| `stromgren.Q_ion` | `-1.0` | When positive, overrides the per-particle rate from the stellar model. |

### 5.3 Limitations

1. **The ionized region is always spherical.** Champagne flows, breakout along low-density channels, and shadowing behind dense clumps are not represented. This is inherent to the Strömgren-volume approximation, not to Quokka's implementation of it; capturing anisotropy would require angular binning.
2. **Finest level only.** Like the supernova deposition, the module acts on the finest level, where star particles are assumed to live. An H II region extending beyond the finest-level grids is truncated.
3. **Photons are not conserved globally.** A budget not exhausted within `R_max_cells` is discarded, as in FIRE-1. There is no long-range transfer step to receive the remainder; the module warns when this happens.
4. **The mean molecular weight does not track ionization.** Quokka's EOS uses a fixed $\mu$ unless mass scalars are evolved, so ionized gas keeps its neutral $\mu$. Since $P = \rho k T / (\mu m_H)$, the overpressure driving the expansion is underestimated by roughly $\mu_{\rm neutral} / \mu_{\rm ionized} \approx 2$. Compensate by setting $T_{\rm HII}$ to an effective value near $2\times10^4$ K rather than $10^4$ K.
5. **No radiation pressure and no MHD.** Only thermal feedback is applied. The module aborts at runtime if enabled for a problem with face-centred (MHD) variables, because the internal energy update does not remove the magnetic contribution.

### 5.4 Validation

Four tests run the same binary against different paths through the module.

| Test | Configuration | Checks |
|---|---|---|
| `StromgrenVolumeFeedback` | one source, uniform medium | equivalent radius matches the analytic $R_{\rm St} = (3Q / 4\pi \alpha_B n_H^2)^{1/3}$ to better than $0.01$ cells |
| `StromgrenVolumeGradient` | linear density gradient, off-centre source | the analytic radius no longer applies, so photon conservation is checked instead: recombinations inside the region equal $Q$ |
| `StromgrenVolumeSubgrid` | $Q$ so small that $R_{\rm St} \approx 0.005$ cells | the unresolved regime the module exists for; a one-cell tolerance is vacuous here, so relative accuracy is required (measured $\sim 2\times10^{-7}$) |
| `StromgrenVolumeOverlap` | two co-located sources | combined region matches $2^{1/3} R_{\rm St}$ and recombinations equal $2Q$, confirming photons are not spent twice | The dynamical benchmark for this class of model is the StarBench D-type expansion comparison (Bisbas et al. 2015), which should be checked against the Hosokawa & Inutsuka (2006) solution rather than the Spitzer (1978) one; it is not part of the automated test suite.
