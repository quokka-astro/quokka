# Photoionization Module

> **Note:** This page currently documents only the VODE tolerance (atol/rtol) configuration.
> Full documentation of the photoionization module (chemistry network, cross sections,
> radiation coupling) will be added later.

## VODE Absolute Tolerances

Quokka uses CVODE/VODE (via Microphysics) to integrate the chemistry and internal energy
source terms.  The integrator requires absolute tolerances (`atol`) for each solution
variable.  Rather than hand-tuning these tolerances, `SetAtolFromPhysics<problem_t>()`
(in `src/radiation/photochem_atol.H`) derives them from high-level physical
parameters specified in the input file.

### Input parameters

| Parameter                                         | Required | Default | Description                                                                              |
| ------------------------------------------------- | -------- | ------- | ---------------------------------------------------------------------------------------- |
| `integrator.typical_n_H`                          | yes      | —       | Representative total H number density (cm⁻³)                                             |
| `integrator.typical_minimal_radiation_T`          | yes      | —       | Minimum temperature (K) at which the radiation field is physically meaningful            |
| `integrator.desired_accuracy_on_T_at_typical_n_H` | no       | 1.0 K   | Desired temperature accuracy at `typical_n_H`                                            |
| `integrator.spec_abundance_tol`                   | no       | 1e-5    | Species negligibility threshold, as a fraction of `typical_n_H`                          |
| `integrator.radiation_failure_tolerance`          | no       | 0.01    | Maximum allowed negative photon number density (cm⁻³) before a burn is flagged as failed |

The relative tolerances (`rtol_spec`, `rtol_enuc`, `rtol_rad_num`, `rtol_rad_flux`) are
specified directly in the input file as usual.

### Physical constants

| Symbol | Value | Description |
| ------ | ----- | ----------- |
| `a_rad` | `7.5657e-15 erg cm⁻³ K⁻⁴` | Radiation constant (from `fundamental_constants.H`) |
| `k_B` | `1.380649e-16 erg K⁻¹` | Boltzmann constant |
| `m_p` | `1.67262192e-24 g` | Proton mass |
| `c_v` | `3/2 × k_B / m_p ≈ 1.24e8 erg g⁻¹ K⁻¹` | Specific heat of monatomic hydrogen gas |
| `E_photon` | problem-dependent | Midpoint energy of the first chemistry radiation band (erg), from `RadSystem<problem_t>::GetChemBandQuanta(0)` |

### Derived atol values

Let `T_min ≡ typical_minimal_radiation_T` for brevity.
`SetAtolFromPhysics` computes:

| Variable                      | Formula                                      | Rationale                                                                                                              |
| ----------------------------- | -------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| `atol_spec`                   | `spec_abundance_tol × typical_n_H`           | Species below this fraction of `n_H` are negligible                                                                    |
| `atol_enuc`                   | `c_v × desired_accuracy_on_T_at_typical_n_H` | Converts temperature accuracy to internal energy tolerance (`c_v = 3/2 × k_B / m_p`)                                   |
| `atol_rad_num`                | `1e-6 × a_rad × T_min⁴ / E_photon`           | One millionth of the blackbody photon density at `T_min` — radiation below this is negligible                         |
| `atol_rad_flux`               | `= rtol_rad_flux`                              | Normalized flux in [0,1]; setting atol = rtol gives crossover `y_cross = 1` (always under absolute control)            |
| `radiation_failure_tolerance` | 0.01 (fixed default)                         | Physical guard, not derived — 0.01 photons/cm³ is negligible in any astrophysical context. Overridable via input file. |

### The 1e-6 prefactor

The factor `1e-6` in `atol_rad_num` has a specific physical meaning:

- It sets the tolerance to 1 part per million of the blackbody photon density at `T_min`.
- After roughly 10⁶ VODE steps, the accumulated local error in photon number remains
  below the physically meaningful radiation level at the minimum temperature.
- Cells with radiation below this threshold are considered "dark" and VODE
  returns in a single BDF step.

### radiation_failure_tolerance

This is a **physical guard**, not a numerical tolerance. It defines the maximum allowed
negative photon number density (cm⁻³) before a burn is declared failed. The default of
0.01 cm⁻³ means that up to 0.01 photons per cm³ can be "created from nothing" by VODE's
Newton overshoot — negligible in any astrophysical context. This parameter does not scale
with `atol_rad_num` because the Newton overshoot in the stiff radiation-chemistry system
has a floor independent of the tolerance.

### Relationship between Erad_floor and typical_minimal_radiation_T

These are two distinct parameters that serve different purposes:

| Parameter                         | Where set                                             | Purpose                                                                                                 |
| --------------------------------- | ----------------------------------------------------- | ------------------------------------------------------------------------------------------------------- |
| `Erad_floor`                      | `constexpr` in `RadSystem_Traits<problem_t>`          | M1 hyperbolic solver floor — prevents the radiation moment solver from encountering zero energy density |
| `typical_minimal_radiation_T`     | Input file (`integrator.typical_minimal_radiation_T`) | VODE tolerance — defines the minimum physically meaningful radiation temperature                        |

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

### Mutual exclusivity

The `integrator.typical_*` parameters and hand-tuned `integrator.atol_*` parameters
are **mutually exclusive** — using both triggers an error.  Specifying neither also
triggers an error, because VODE's built-in defaults (~1e-10) are unusably tight for
photochemistry and will cause the integrator to stall.

### Setting up a new problem

1. Set `Erad_floor` in `RadSystem_Traits<problem_t>` to a blackbody temperature low
   enough that it does not produce spurious ionization (typical: 0.01–1 K).
2. Choose `typical_n_H` as the representative hydrogen density of the problem.
3. Choose `typical_minimal_radiation_T` as the lowest temperature at which radiation
   is physically important (e.g. the minimum gas temperature in the domain).
4. Check `1e-6 × (T_min / T_floor)⁴ ≥ 10⁴`, i.e. `T_floor ≤ T_min / 316`.
   If this fails, either lower `Erad_floor` or raise `typical_minimal_radiation_T`.
5. The `1e-6` prefactor and `desired_accuracy_on_T_at_typical_n_H = 1.0 K` are
   reasonable defaults for most problems.
