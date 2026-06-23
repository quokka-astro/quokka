# Photoionization Module

**Reference:** Aubert & Teyssier (2008), "A radiative transfer scheme for cosmological
reionization based on a local Eddington tensor" (ATON paper, arXiv:0709.1544)

## 1. Governing Equations

### 1.1 M1 Radiative Transfer for Ionizing Photons

Taking the first two moments of the radiative transfer equation gives conservation laws
for the ionizing photon number density $N_\gamma$ and flux density $\mathbf{F}_\gamma$:

$$
\frac{\partial N_\gamma}{\partial t} + \nabla \cdot \mathbf{F}_\gamma =
- n_{\rm H^0} c \sigma_\gamma N_\gamma
+ n_e n_{\rm H^+} (\alpha_A - \alpha_B)
+ \dot{N}^*_\gamma,
$$

$$
\frac{\partial \mathbf{F}_\gamma}{\partial t} + c^2 \nabla \cdot \mathsf{P}_\gamma =
- n_{\rm H^0} c \sigma_\gamma \mathbf{F}_\gamma.
$$

| Symbol | Definition |
|--------|------------|
| $N_\gamma$ | Ionizing photon number density (cm^-3) |
| $\mathbf{F}_\gamma$ | Ionizing photon number flux density (cm^-2 s^-1) |
| $\mathsf{P}_\gamma$ | Radiation pressure tensor ($= \mathsf{D}\,F_\gamma$, cm^-3) |
| $n_{\rm H^0}$ | Neutral hydrogen number density |
| $n_{\rm H^+} = n_e$ | Ionized hydrogen / electron number density |
| $\sigma_\gamma$ | Frequency-averaged photoionization cross-section |
| $\alpha_A, \alpha_B$ | Case A / B recombination coefficients (cm^3 s^-1) |
| $\beta$ | Collisional ionization rate coefficient (cm^3 s^-1) |
| $\dot{N}^*_\gamma$ | Stellar ionizing photon emission rate (cm^-3 s^-1) |

The source term $n_e n_{\rm H^+}(\alpha_A - \alpha_B)$ represents diffuse recombination
radiation — photons re-emitted when H recombines directly to the ground state (case A
minus case B correction).

### 1.2 Hydrogen Thermochemistry

The neutral hydrogen fraction evolves as:

$$
\frac{D n_{\rm H^0}}{Dt} = \alpha_A n_e n_{\rm H^+} - \beta n_e n_{\rm H^0} - \Gamma_{\gamma {\rm H}^0} n_{\rm H^0},
$$

with $n_{\rm H^+} = n_e$ (charge conservation), $n_{\rm H^+} + n_{\rm H^0} = n_{\rm H}$
(nuclei conservation), and the photoionization rate $\Gamma_{\gamma {\rm H}^0} = c \sigma_\gamma N_\gamma$.

The gas thermal energy evolves as:

$$
\rho \frac{D}{Dt}\!\left(\frac{e}{\rho}\right) = \mathcal{H}_{\rm photo} - \mathcal{L},
$$

where $\mathcal{H}_{\rm photo} = n_{\rm H^0} c \sigma_\gamma \epsilon_\gamma N_\gamma$ is
the photoheating rate and $\epsilon_\gamma = h(\bar{\nu} - \nu_{{\rm H}^0})$ is the mean
excess photon energy above the ionization threshold (29.65 eV for a $10^5$ K blackbody).
For heating and cooling that are not directly due to hydrogen photoionization, we
follow the optically thin prescription of [@Krumholz2007]: In molecular
gas, the approximate cooling and heating functions of [@Koyama2002] are used. 
In partially ionized gas, the cooling rate is computed
following [@Osterbrock1989], which includes cooling by ion-electron
collisions involving the first and second ionized states of O, N, and Ne — the
dominant coolants in H II regions at solar metallicity. A future PR will extend the
cooling model to use the RIGEL prescription (Deng et al. 2024).

### 1.3 On-the-Spot Approximation (OTSA)

When a hydrogen ion recombines directly to the ground state (n = 1), the emitted
Lyman-continuum photon is immediately capable of re-ionizing a nearby neutral hydrogen
atom. The **on-the-spot approximation** assumes this photon is re-absorbed locally —
within the same resolution element — so recombinations to n = 1 have no net effect on
the ionization state.

Under OTSA, one replaces $\alpha_A \to \alpha_B$ everywhere and drops the diffuse
recombination source term from the radiation equation:

$$
\frac{\partial N_\gamma}{\partial t} + \nabla \cdot \mathbf{F}_\gamma =
- n_{\rm H^0} c \sigma_\gamma N_\gamma + \dot{N}^*_\gamma,
$$

$$
\frac{D n_{\rm H^0}}{Dt} = \alpha_B n_e n_{\rm H^+} - \beta n_e n_{\rm H^0} - \Gamma_{\gamma {\rm H}^0} n_{\rm H^0}.
$$

OTSA is valid when the mean free path of a recombination photon is much smaller than
the size of the ionized region — a good approximation deep inside large HII regions.
It breaks down near ionization fronts and in low-density, nearly fully ionized gas.

Quokka currently uses OTSA. The full $(\alpha_A - \alpha_B)$ diffuse source term is
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

Under OTSA, VODE integrates the following system over the implicit timestep $\Delta t$:

$$
\begin{aligned}
\frac{d N_\gamma}{d t} &= - n_{\rm H^0} \hat{c} \sigma_\gamma N_\gamma + \dot{N}, \\[2pt]
\frac{d F_{\gamma,i}}{d t} &= - n_{\rm H^0} \hat{c} \sigma_\gamma F_{\gamma,i} \quad (i = x, y, z), \\[2pt]
\frac{d n_{\rm H^0}}{d t} &= \alpha_B n_e n_{\rm H^+} - \beta n_e n_{\rm H^0} - \hat{c} \sigma_\gamma N_\gamma n_{\rm H^0}, \\[2pt]
\frac{d n_{\rm H^+}}{d t} &= -\alpha_B n_e n_{\rm H^+} + \beta n_e n_{\rm H^0} + \hat{c} \sigma_\gamma N_\gamma n_{\rm H^0}, \\[2pt]
\frac{d n_e}{d t} &= -\alpha_B n_e n_{\rm H^+} + \beta n_e n_{\rm H^0} + \hat{c} \sigma_\gamma N_\gamma n_{\rm H^0}, \\[2pt]
\frac{d e}{d t} &= n_{\rm H^0} \hat{c} \sigma_\gamma \epsilon_\gamma N_\gamma - \text{[cooling terms]}.
\end{aligned}
$$

where $\hat{c}$ is the reduced speed of light. Only one directional flux component is
integrated (normalized to 1.0 before the burn); the other two are scaled proportionally
after the ODE solve. The state vector has 6 components for a single chemical band:
$(n_e, n_{\rm H^0}, n_{\rm H^+}, e, N_\gamma, F_\gamma)$.

Note that $n_{\rm H^+} = n_{\rm H} - n_{\rm H^0}$ and $n_e = n_{\rm H^+}$ by
construction. Although only one of $n_{\rm H^0}$ or $n_{\rm H^+}$ is an independent
variable, both are integrated for symmetry.

The flux ODE $dF_\gamma/dt = -n_{\rm H^0} \hat{c} \sigma_\gamma F_\gamma$ is similar
to the absorption term in $N_\gamma$, but $N_\gamma$ has an additional isotropic source
term (stellar emission $\dot{N}$). Isotropic sources add photons uniformly in all
directions — they contribute to $N_\gamma$ but produce no net flux. Flux must be
integrated to track the attenuation of the directional radiation field across the
timestep.

## 3. VODE Tolerances

### 3.1 Overview

Quokka uses VODE (via Microphysics) to integrate the chemistry and internal energy
source terms. The integrator requires absolute tolerances (`atol`) for each solution
variable. These are hand-tuned for each problem and specified directly in the input
file (see SS 3.2). The `SetAtolFromPhysics` machinery (PR #1980) will derive tolerances
from physical scales automatically in a future PR.

### 3.2 Input parameters

| Parameter                               | Description                                                       |
| --------------------------------------- | ----------------------------------------------------------------- |
| `integrator.atol_spec`                  | Absolute tolerance for chemical species (cm^-3)                   |
| `integrator.atol_enuc`                  | Absolute tolerance for gas internal energy (erg g^-1)             |
| `integrator.atol_rad_num`               | Absolute tolerance for photon number density (cm^-3)              |
| `integrator.rtol_spec`                  | Relative tolerance for chemical species                           |
| `integrator.rtol_enuc`                  | Relative tolerance for gas internal energy                        |
| `integrator.rtol_rad_num`               | Relative tolerance for photon number density                      |
| `integrator.species_failure_tolerance`  | VODE internal substep rejection threshold for negative species (cm^-3, see SS 3.7) |
| `integrator.radiation_failure_tolerance`| VODE internal substep rejection threshold for negative photon density (cm^-3, see SS 3.5) |

### 3.3 Why flux is excluded from convergence

The radiation flux $F_\gamma$ (normalized to 1.0 before the ODE) is integrated alongside the
other variables, but does not participate in any VODE convergence or error checks.

**Why flux is in the ODE.** The flux ODE is $dF/dt = -(\hat{c}\sigma)\,n_{\rm H^0} F$.
This is similar to the absorption term in $N_\gamma$, but $N_\gamma$ also has an
isotropic source term (stellar emission in OTSA, or $n_e n_{\rm H^+}(\alpha_A - \alpha_B)$
recombination radiation in case A). Since isotropic sources contribute photons uniformly
in all directions, they add to the photon number density but produce no net flux. Flux
must be integrated separately to track the attenuation of the directional radiation field.

**Why flux is excluded from convergence.** Flux is a passive scalar — its RHS depends
on $n_{\rm H^0}$ but flux does *not* appear in any other equation (species, energy, or
$N_\gamma$). Convergence should be driven by the physically consequential quantities, not
by a diagnostic variable. In dark cells where flux goes to 0, demanding 1% accuracy on a
near-zero value wastes VODE steps with no physical benefit.

Excluding flux from convergence gave a **3.8x speedup** in photochemistry on CPU and a
**2.2x speedup** on GPU for the DTypeFront test.

### 3.4 Physical constants

| Symbol  | Value                                   | Description                                                      |
| ------- | --------------------------------------- | ---------------------------------------------------------------- |
| `a_rad` | `7.5657e-15 erg cm^-3 K^-4`             | Radiation constant (from `fundamental_constants.H`)              |
| `k_B`   | `1.380649e-16 erg K^-1`                 | Boltzmann constant                                               |
| `m_p`   | `1.67262192e-24 g`                      | Proton mass                                                      |
| `c_v`   | `3/2 x k_B / m_p ~ 1.24e8 erg g^-1 K^-1` | Specific heat of monatomic hydrogen gas                           |

### 3.5 radiation_failure_tolerance

VODE uses this threshold in two places:

1. **Internal substeps:** if the photon number density becomes more negative than
   `radiation_failure_tolerance`, VODE rejects the substep and retries with a smaller
   timestep. This is the primary use.
2. **Final state:** after interpolating to the output time, if the photon number density
   is more negative than $1.5 \times$ `radiation_failure_tolerance`, the burn is
   declared failed. The 1.5× factor (via `vode_final_state_radiation_failure_tolerance_factor`
   in `vode_type.H`) accounts for VODE's non-monotonic interpolation, preventing false
   failures from interpolation noise.

Set `radiation_failure_tolerance` equal to `atol_rad_num` (the photon negligibility
floor). The $1.5\times$ final-state factor absorbs BDF interpolation overshoot without
manual inflation.

Physically, the amount of spurious ionization that can be produced by a negative photon
overshoot is at most $\texttt{radiation\_failure\_tolerance} / n_{\rm H}$. Whether this
matters depends on two regimes:

1. **Bright cells** ($N_\gamma \gtrsim n_{\rm H}$): the cell is fully ionized. A few
   percent error in photon count does not change the outcome.
2. **Dark cells** ($N_\gamma \ll n_{\rm H}$): the ratio is negligible as long as
   $\texttt{radiation\_failure\_tolerance} \ll n_{\rm H}$.

For low-density environments such as the CGM or IGM, where the ionized gas density can
be $\sim 10^{-4}$–$10^{-3}$ cm$^{-3}$, the default value of this parameter may compete
with the physical ionization equilibrium — override it in the input file.

### 3.6 Erad_floor

`Erad_floor` is a compile-time `constexpr` in `RadSystem_Traits<problem_t>` that sets
the M1 hyperbolic solver floor — it prevents the radiation moment solver from
encountering zero energy density. It is **independent** of the VODE tolerances.

Define the equivalent floor temperature $T_{\rm floor}$ by $E_{\rm rad, floor} \equiv a_{\rm rad} T_{\rm floor}^4$.
The photon number density at the floor is $N_{\gamma,{\rm floor}} = E_{\rm rad, floor} / E_{\rm photon}$.

Dark cells (where $E_{\rm rad} \approx E_{\rm rad, floor}$) converge in one VODE step when
$\texttt{atol\_rad\_num} \gg N_{\gamma,{\rm floor}}$. A ratio of $\geq 10^4$ is sufficient:

$$
\frac{\texttt{atol\_rad\_num}}{N_{\gamma,{\rm floor}}} \geq 10^4.
$$

For typical `Erad_floor` values corresponding to $T_{\rm floor} = 0.01$–$1$ K, an
`atol_rad_num` on the order of $10^{-6}$–$10^{-2}$ cm^-3 satisfies this constraint.

### 3.7 species_failure_tolerance

VODE uses this threshold in two places:

1. **Internal substeps (primary):** if a species number density becomes more negative
   than `species_failure_tolerance`, VODE rejects the substep and retries with a smaller
   timestep.
2. **Final state (secondary):** after interpolating to the output time, if a species is
   more negative than $1.5 \times$ `species_failure_tolerance`, the burn is declared
   failed. The 1.5× factor (via `vode_final_state_species_failure_tolerance_factor` in
   `vode_type.H`) accounts for VODE's non-monotonic interpolation, preventing false
   failures from interpolation noise.

Set `integrator.species_failure_tolerance` equal to `atol_spec` (the species
negligibility floor). The $1.5\times$ final-state factor absorbs BDF interpolation
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
the M1 radiation field and the hydrogen chemistry network (SS 1). Enabling both
simultaneously would double-count every one of these rates, producing physically
incorrect temperatures and ionization fractions.

**Correct setup:** use `photochemistry.enabled = 1` and leave `cooling.enabled = 0`
(the default). The photoionization chemistry network handles all heating and cooling
internally.
