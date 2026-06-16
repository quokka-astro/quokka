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
| $N_\gamma$ | Ionizing photon number density (cm⁻³) |
| $\mathbf{F}_\gamma$ | Ionizing photon number flux density (cm⁻² s⁻¹) |
| $\mathsf{P}_\gamma$ | Radiation pressure tensor ($= \mathsf{D}\,F_\gamma$, cm⁻³) |
| $n_{\rm H^0}$ | Neutral hydrogen number density |
| $n_{\rm H^+} = n_e$ | Ionized hydrogen / electron number density |
| $\sigma_\gamma$ | Frequency-averaged photoionization cross-section |
| $\alpha_A, \alpha_B$ | Case A / B recombination coefficients |
| $\dot{N}^*_\gamma$ | Stellar ionizing photon emission rate (cm⁻³ s⁻¹) |

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
The cooling rate $\mathcal{L}$ includes case A recombination cooling, collisional
excitation of H, collisional ionization cooling, and Bremsstrahlung. Rate coefficients
follow Hui & Gnedin (1997) and Maselli et al. (2003).

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
1. Stellar source step     Particle injection → radEnergySource
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

Quokka uses CVODE/VODE (via Microphysics) to integrate the chemistry and internal energy
source terms.  The integrator requires absolute tolerances (`atol`) for each solution
variable.  Rather than hand-tuning these tolerances, `SetAtolFromPhysics<problem_t>()`
(in `src/radiation/photochem_atol.H`) derives them from high-level physical
parameters specified in the input file.

### 3.2 Input parameters


| Parameter                                         | Required | Default | Description                                                                              |
| ------------------------------------------------- | -------- | ------- | ---------------------------------------------------------------------------------------- |
| `integrator.typical_n_H`                          | yes      | —       | Representative total H number density (cm⁻³)                                             |
| `integrator.typical_minimal_radiation_T`          | yes      | —       | Typical temperature of the cold (neutral) gas in the domain (K). Sets the photon density below which radiation is numerically negligible. |
| `integrator.desired_accuracy_on_T_at_typical_n_H` | no       | 1.0 K   | Desired temperature accuracy at `typical_n_H`                                            |
| `integrator.spec_abundance_tol`                   | no       | 1e-5    | Species negligibility threshold, as a fraction of `typical_n_H`                          |
| `integrator.radiation_failure_tolerance`          | no       | 0.01    | Maximum allowed negative photon number density (cm⁻³) before a burn is flagged as failed |


The relative tolerances (`rtol_spec`, `rtol_enuc`, `rtol_rad_num`) are specified directly
in the input file as usual.

### 3.3 Why flux is excluded from convergence

The radiation flux `F_γ` (normalized to 1.0 before the ODE) is integrated alongside the
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
by a diagnostic variable. In dark cells where flux → 0, demanding 1% accuracy on a
near-zero value wastes VODE steps with no physical benefit.

Excluding flux from convergence gave a **3.8× speedup** in photochemistry on CPU and a
**2.2× speedup** on GPU for the DTypeFront test.

### 3.4 Physical constants


| Symbol     | Value                                  | Description                                                                                                    |
| ---------- | -------------------------------------- | -------------------------------------------------------------------------------------------------------------- |
| `a_rad`    | `7.5657e-15 erg cm⁻³ K⁻⁴`              | Radiation constant (from `fundamental_constants.H`)                                                            |
| `k_B`      | `1.380649e-16 erg K⁻¹`                 | Boltzmann constant                                                                                             |
| `m_p`      | `1.67262192e-24 g`                     | Proton mass                                                                                                    |
| `c_v`      | `3/2 × k_B / m_p ≈ 1.24e8 erg g⁻¹ K⁻¹` | Specific heat of monatomic hydrogen gas                                                                        |
| `E_photon` | problem-dependent                      | Midpoint energy of the first chemistry radiation band (erg), from `RadSystem<problem_t>::GetChemBandQuanta(0)` |


### 3.5 Derived atol values

Let $T_{\rm min} \equiv$ `typical_minimal_radiation_T` for brevity.
`SetAtolFromPhysics` computes:

| Variable | Formula | Rationale |
|---|---|---|
| `atol_spec` | $\texttt{spec\_abundance\_tol} \times \texttt{typical\_n\_H}$ | Species below this fraction of $n_{\rm H}$ are negligible |
| `atol_enuc` | $c_v \times \texttt{desired\_accuracy\_on\_T\_at\_typical\_n\_H}$ | Converts temperature accuracy to internal energy tolerance ($c_v = \tfrac{3}{2} k_B / m_p$) |
| `atol_rad_num` | $10^{-6} \times a_{\rm rad} \times T_{\rm min}^4 / E_{\rm photon}$ | One millionth of the blackbody photon density at $T_{\rm min}$ — radiation below this is negligible |
| `radiation_failure_tolerance` | 0.05 (fixed default) | Physical guard, not derived. Overridable (see §3.7). |


### 3.6 The $10^{-6}$ prefactor

The factor $10^{-6}$ in `atol_rad_num` has a specific physical meaning:

- It sets the tolerance to 1 part per million of the blackbody photon density at $T_{\rm min}$.
- After roughly $10^6$ VODE steps, the accumulated local error in photon number remains
  below the physically meaningful radiation level at the minimum temperature.
- Cells with radiation below this threshold are considered "dark" and VODE
  returns in a single BDF step.

### 3.7 radiation_failure_tolerance

This is a **physical guard**, not a numerical tolerance. It defines the maximum allowed
negative photon number density (cm⁻³) before a burn is declared failed — at most this
many spurious photons can be "created from nothing" by VODE's Newton overshoot.

Whether this matters depends on two regimes:

1. **Bright cells** ($N_\gamma \gtrsim n_{\rm H}$): the cell is fully ionized. A few
   percent error in photon count does not change the outcome.
2. **Dark cells** ($N_\gamma \ll n_{\rm H}$): the spurious ionization is at most
   $\texttt{radiation\_failure\_tolerance} / n_{\rm H}$. If this ratio is $\ll 1\%$,
   it is negligible.

The default of 0.05 cm⁻³ is appropriate for galactic disk or GMC environments, where the
typical ionized gas density is ~10²–10⁴ cm⁻³ (ratio ≤ 5×10⁻⁴). For low-density environments
such as the CGM or IGM, where the ionized gas density can be ~10⁻⁴–10⁻³ cm⁻³, this
default competes with the physical ionization equilibrium — override it in the input file.

**Rule of thumb:** set `radiation_failure_tolerance` to at least two orders of magnitude
below the **typical density of ionized gas** in the problem. This ensures that spurious
photon creation cannot measurably affect the ionization fraction. The value does not
scale with `atol_rad_num` because the Newton overshoot in the stiff radiation-chemistry
system has a floor independent of the tolerance.

### 3.8 Relationship between Erad_floor and typical_minimal_radiation_T

These are two distinct parameters that serve different purposes:


| Parameter                     | Where set                                             | Purpose                                                                                                 |
| ----------------------------- | ----------------------------------------------------- | ------------------------------------------------------------------------------------------------------- |
| `Erad_floor`                  | `constexpr` in `RadSystem_Traits<problem_t>`          | M1 hyperbolic solver floor — prevents the radiation moment solver from encountering zero energy density |
| `typical_minimal_radiation_T` | Input file (`integrator.typical_minimal_radiation_T`) | VODE tolerance — set to the typical temperature of the cold (neutral) gas in the domain |


Define the equivalent floor temperature $T_{\rm floor}$ by $E_{\rm rad, floor} \equiv a_{\rm rad} T_{\rm floor}^4$.
The photon number density at the floor is $N_{\gamma,{\rm floor}} = E_{\rm rad, floor} / E_{\rm photon}$.

Dark cells (where $E_{\rm rad} \approx E_{\rm rad, floor}$) converge in one VODE step when
$\texttt{atol\_rad\_num} \gg N_{\gamma,{\rm floor}}$.  Since $E_{\rm photon}$ cancels,
the ratio simplifies to a function of the two temperatures alone:

$$
\frac{\texttt{atol\_rad\_num}}{N_{\gamma,{\rm floor}}}
= \frac{10^{-6} \, a_{\rm rad} \, T_{\rm min}^4 / E_{\rm photon}}
       {a_{\rm rad} \, T_{\rm floor}^4 / E_{\rm photon}}
= 10^{-6} \left( \frac{T_{\rm min}}{T_{\rm floor}} \right)^{\!4}.
$$

A ratio of $\geq 10^4$ is sufficient, which requires $T_{\rm floor} \leq T_{\rm min} / 316$.

**Example (DTypeFront):** $T_{\rm min} = 10\ {\rm K}$, $E_{\rm rad, floor} = a_{\rm rad} (0.01\ {\rm K})^4$ → $T_{\rm floor} = 0.01\ {\rm K}$.
Ratio = $10^{-6} \times (10 / 0.01)^4 = 10^6$ ✓.

### 3.9 Mutual exclusivity

The `integrator.typical_*` parameters and hand-tuned `integrator.atol_*` parameters
are **mutually exclusive** — using both triggers an error.  Specifying neither also
triggers an error, because VODE's built-in defaults ($\sim 10^{-10}$) are unusably
tight for photochemistry and will cause the integrator to stall.

### 3.10 Setting up a new problem

1. Set `Erad_floor` in `RadSystem_Traits<problem_t>` to a blackbody temperature low
   enough that it does not produce spurious ionization (typical: $0.01$–$1$ K).
2. Choose `typical_n_H` as the representative hydrogen density of the problem.
3. Choose `typical_minimal_radiation_T` as the typical temperature of the cold
   (neutral) gas in the domain.
4. Check $10^{-6} \, (T_{\rm min} / T_{\rm floor})^4 \geq 10^4$, i.e.
   $T_{\rm floor} \leq T_{\rm min} / 316$.
   If this fails, either lower `Erad_floor` or raise `typical_minimal_radiation_T`.
5. The $10^{-6}$ prefactor and `desired_accuracy_on_T_at_typical_n_H = 1.0 K` are
   reasonable defaults for most problems.

