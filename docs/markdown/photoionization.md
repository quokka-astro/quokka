# Photoionization Module

**Reference:** Aubert & Teyssier (2008), "A radiative transfer scheme for cosmological
reionization based on a local Eddington tensor" (ATON paper, arXiv:0709.1544)

## 1. Governing Equations

### 1.1 M1 Radiative Transfer for Ionizing Photons

Taking the first two moments of the radiative transfer equation gives conservation laws
for the ionizing photon number density `N_γ` and flux density `F_γ`:

```
∂N_γ/∂t + ∇·F_γ = -n_HI c σ_γ N_γ + n_e n_H+(α_A − α_B) + Ṅ*_γ
∂F_γ/∂t + c² ∇·P_γ = -n_HI c σ_γ F_γ
```

| Symbol | Definition |
|--------|------------|
| `N_γ` | Ionizing photon number density (cm⁻³) |
| `F_γ` | Ionizing photon number flux density (cm⁻² s⁻¹) |
| `P_γ` | Radiation pressure tensor (= D F_γ, cm⁻³) |
| `n_HI` | Neutral hydrogen number density |
| `n_H+` = `n_e` | Ionized hydrogen / electron number density |
| `σ_γ` | Frequency-averaged photoionization cross-section |
| `α_A, α_B` | Case A / B recombination coefficients |
| `Ṅ*_γ` | Stellar ionizing photon emission rate (cm⁻³ s⁻¹) |

The source term `n_e n_H+(α_A − α_B)` represents diffuse recombination radiation —
photons re-emitted when H recombines directly to the ground state (case A minus case B).

### 1.2 Hydrogen Thermochemistry

```
D n_HI / Dt = α_A n_e n_H+ − β n_e n_HI − Γ_γHI n_HI
```

with `n_H+ = n_e` (charge conservation), `n_H+ + n_HI = n_H` (nuclei conservation),
and the photoionization rate `Γ_γHI = c σ_γ N_γ`.

The gas thermal energy evolves as:

```
ρ D(e/ρ)/Dt = H_photo − L
```

where `H_photo = n_HI c σ_γ ε_γ N_γ` is the photoheating rate and `ε_γ = h(ν̄ − ν_HI)`
is the mean excess photon energy above the ionization threshold (29.65 eV for a 10⁵ K
blackbody). The cooling rate `L` includes case A recombination cooling, collisional
excitation of H, collisional ionization cooling, and Bremsstrahlung. Rate coefficients
follow Hui & Gnedin (1997) and Maselli et al. (2003).

### 1.3 On-the-Spot Approximation (OTSA)

When a hydrogen ion recombines directly to the ground state (n = 1), the emitted
Lyman-continuum photon is immediately capable of re-ionizing a nearby neutral hydrogen
atom. The **on-the-spot approximation** assumes this photon is re-absorbed locally —
within the same resolution element — so recombinations to n = 1 have no net effect on
the ionization state.

Under OTSA, one replaces `α_A → α_B` everywhere and drops the diffuse recombination
source term from the radiation equation:

```
∂N_γ/∂t + ∇·F_γ = -n_HI c σ_γ N_γ + Ṅ*_γ
D n_HI/Dt = α_B n_e n_H+ − β n_e n_HI − Γ_γHI n_HI
```

OTSA is valid when the mean free path of a recombination photon is much smaller than
the size of the ionized region — a good approximation deep inside large HII regions.
It breaks down near ionization fronts and in low-density, nearly fully ionized gas.

Quokka currently uses OTSA. The full `(α_A − α_B)` diffuse source term is planned for
a future phase.

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

Under OTSA, VODE integrates the following system over the implicit timestep `Δt`:

```
dN_γ/dt  = -n_HI ĉ σ_γ N_γ + Ṅ
dF_γ/dt  = -n_HI ĉ σ_γ F_γ
dn_HI/dt = α_B n_e n_H+ − β n_e n_HI − ĉ σ_γ N_γ n_HI
dn_H+/dt = −α_B n_e n_H+ + β n_e n_HI + ĉ σ_γ N_γ n_HI
de/dt    = n_HI ĉ σ_γ ε_γ N_γ − [cooling terms]
```

where `ĉ` is the reduced speed of light. The state vector has 6 components for a
single chemical band: `(n_e, n_HI, n_H+, e, N_γ, F_γ)`.

Note that `n_H+ = n_H − n_HI` and `n_e = n_H+` by construction. Although only one of
`n_HI` or `n_H+` is an independent variable, both are integrated for symmetry.

The flux ODE `dF_γ/dt = −n_HI ĉ σ_γ F_γ` is similar to the absorption term in N_gamma,
but N_gamma has an additional isotropic source term (stellar emission Ṅ). Isotropic
sources add photons uniformly in all directions — they contribute to N_gamma but
produce no net flux. Flux must be integrated to track the attenuation of the
directional radiation field across the timestep.

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

**Why flux is in the ODE.** The flux ODE is `dF/dt = -(ĉ σ) n_HI F`. This is similar
to the absorption term in N_gamma, but N_gamma also has an isotropic source term
(stellar emission in OTSA, or `n_e n_H+(α_A − α_B)` recombination radiation in case A).
Since isotropic sources contribute photons uniformly in all directions, they add to the
photon number density but produce no net flux. Flux must be integrated separately to
track the attenuation of the directional radiation field.

**Why flux is excluded from convergence.** Flux is a passive scalar — its RHS depends
on `n_HI` but flux does *not* appear in any other equation (species, energy, or N_gamma).
Convergence should be driven by the physically consequential quantities, not by a
diagnostic variable. In dark cells where flux → 0, demanding 1% accuracy on a near-zero
value wastes VODE steps with no physical benefit.

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

Let `T_min ≡ typical_minimal_radiation_T` for brevity.
`SetAtolFromPhysics` computes:


| Variable                      | Formula                                      | Rationale                                                                                                              |
| ----------------------------- | -------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| `atol_spec`                   | `spec_abundance_tol × typical_n_H`           | Species below this fraction of `n_H` are negligible                                                                    |
| `atol_enuc`                   | `c_v × desired_accuracy_on_T_at_typical_n_H` | Converts temperature accuracy to internal energy tolerance (`c_v = 3/2 × k_B / m_p`)                                   |
| `atol_rad_num`                | `1e-6 × a_rad × T_min⁴ / E_photon`           | One millionth of the blackbody photon density at `T_min` — radiation below this is negligible                          |
| `radiation_failure_tolerance` | 0.05 (fixed default)                         | Physical guard, not derived. Overridable via input file (see below for the rule of thumb). |


### 3.6 The 1e-6 prefactor

The factor `1e-6` in `atol_rad_num` has a specific physical meaning:

- It sets the tolerance to 1 part per million of the blackbody photon density at `T_min`.
- After roughly 10⁶ VODE steps, the accumulated local error in photon number remains
below the physically meaningful radiation level at the minimum temperature.
- Cells with radiation below this threshold are considered "dark" and VODE
returns in a single BDF step.

### 3.7 radiation_failure_tolerance

This is a **physical guard**, not a numerical tolerance. It defines the maximum allowed
negative photon number density (cm⁻³) before a burn is declared failed — at most this
many spurious photons can be "created from nothing" by VODE's Newton overshoot.

Whether this matters depends on two regimes:

1. **Bright cells** (`N_gamma ≳ n_H`): the cell is fully ionized. A few percent error in
  photon count does not change the outcome.
2. **Dark cells** (`N_gamma ≪ n_H`): the spurious ionization is at most
  `radiation_failure_tolerance / n_H`. If this ratio is ≪ 1 %, it is negligible.

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


Define the equivalent floor temperature `T_floor` by `Erad_floor ≡ a_rad × T_floor⁴`.
The photon number density at the floor is `N_gamma_floor = Erad_floor / E_photon`.

Dark cells (where `Erad ≈ Erad_floor`) converge in one VODE step when
`atol_rad_num ≫ N_gamma_floor`.  Since `E_photon` cancels, the ratio simplifies to
a function of the two temperatures alone:

```
atol_rad_num / N_gamma_floor = (1e-6 × a_rad × T_min⁴ / E_photon) / (a_rad × T_floor⁴ / E_photon)
                              = 1e-6 × (T_min / T_floor)⁴
```

A ratio of ≥ 10⁴ is sufficient, which requires `T_floor ≤ T_min / 316`.

**Example (DTypeFront):** `T_min = 10 K`, `Erad_floor = a_rad × (0.01 K)⁴` → `T_floor = 0.01 K`.
Ratio = `1e-6 × (10 / 0.01)⁴ = 10⁶` ✓.

### 3.9 Mutual exclusivity

The `integrator.typical_`* parameters and hand-tuned `integrator.atol_*` parameters
are **mutually exclusive** — using both triggers an error.  Specifying neither also
triggers an error, because VODE's built-in defaults (~1e-10) are unusably tight for
photochemistry and will cause the integrator to stall.

### 3.10 Setting up a new problem

1. Set `Erad_floor` in `RadSystem_Traits<problem_t>` to a blackbody temperature low
  enough that it does not produce spurious ionization (typical: 0.01–1 K).
2. Choose `typical_n_H` as the representative hydrogen density of the problem.
3. Choose `typical_minimal_radiation_T` as the typical temperature of the cold
  (neutral) gas in the domain.
4. Check `1e-6 × (T_min / T_floor)⁴ ≥ 10⁴`, i.e. `T_floor ≤ T_min / 316`.
  If this fails, either lower `Erad_floor` or raise `typical_minimal_radiation_T`.
5. The `1e-6` prefactor and `desired_accuracy_on_T_at_typical_n_H = 1.0 K` are
  reasonable defaults for most problems.

