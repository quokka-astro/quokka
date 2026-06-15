# Photoionization Module

> **Note:** This page currently documents only the VODE tolerance (atol/rtol) configuration.
> Full documentation of the photoionization module (chemistry network, cross sections,
> radiation coupling) will be added later.

## VODE Absolute Tolerances

Quokka uses CVODE/VODE (via Microphysics) to integrate the chemistry and internal energy
source terms.  The integrator requires absolute tolerances (`atol`) for each solution
variable.  Rather than hand-tuning these tolerances, `SetAtolFromPhysics<problem_t>()`
(in `src/radiation/photochem_atol.H`) derives them from three high-level physical
parameters specified in the input file.

### Input parameters

| Parameter | Required | Default | Description |
|---|---|---|---|
| `integrator.typical_n_H` | yes | — | Representative total H number density (cm⁻³) |
| `integrator.typical_minimal_radiation_T` | yes | — | Minimum temperature (K) at which the radiation field is physically meaningful |
| `integrator.desired_accuracy_on_T_at_typical_n_H` | no | 1.0 K | Desired temperature accuracy at `typical_n_H` |
| `integrator.spec_abundance_tol` | no | 1e-5 | Species negligibility threshold, as a fraction of `typical_n_H` |
| `integrator.radiation_failure_tolerance` | no | 0.01 | Maximum allowed negative photon number density (cm⁻³) before a burn is flagged as failed |

The relative tolerances (`rtol_*`) are specified directly as usual.

### Derived atol values

`SetAtolFromPhysics` computes:

| Variable | Formula | Rationale |
|---|---|---|
| `atol_spec` | `spec_abundance_tol × typical_n_H` | Species below this fraction of `n_H` are negligible |
| `atol_enuc` | `c_v × desired_accuracy_on_T_at_typical_n_H` | Converts temperature accuracy to internal energy tolerance (c_v = 3/2 k_B/m_p) |
| `atol_rad_num` | `1e-6 × a_rad × T_min⁴ / E_photon` | Photon density at 1 ppm of the BB field at `T_min` — radiation below this is negligible |
| `atol_rad_flux` | `rtol_rad_flux` | Normalized flux in [0,1]; setting atol = rtol gives y_cross = 1 (always absolute control) |
| `radiation_failure_tolerance` | 0.01 (fixed default) | Physical guard, not derived — 0.01 photons/cm³ is negligible in any astrophysical context. Overridable via input file. |

### radiation_failure_tolerance

This is a **physical guard**, not a numerical tolerance. It defines the maximum allowed
negative photon number density (cm⁻³) before a burn is declared failed. The default of
0.01 cm⁻³ means that up to 0.01 photons per cm³ can be "created from nothing" by VODE's
Newton overshoot — negligible in any astrophysical context. This parameter does not scale
with `atol_rad_num` because the Newton overshoot in the stiff radiation-chemistry system
has a floor independent of the tolerance.

### The 1e-6 prefactor

The factor `1e-6` in `atol_rad_num` has a specific physical meaning:

- It sets the tolerance to 1 part per million of the blackbody photon density at
  `typical_minimal_radiation_T`.
- This means that after roughly 10⁶ VODE steps, the accumulated local error in
  photon number remains below the physically meaningful radiation level at the
  minimum temperature.
- Cells with radiation below this threshold are considered "dark" and VODE
  returns in a single BDF step.

### Relationship between Erad_floor and typical_minimal_radiation_T

These are two distinct parameters that serve different purposes:

| Parameter | Where set | Purpose |
|---|---|---|
| `Erad_floor` | `constexpr` in `RadSystem_Traits<problem_t>` | M1 hyperbolic solver floor — prevents the radiation moment solver from encountering zero energy density |
| `typical_minimal_radiation_T` | Input file (`integrator.typical_minimal_radiation_T`) | VODE tolerance — defines the minimum physically meaningful radiation temperature |

They must satisfy `N_gamma_floor = Erad_floor / E_photon ≪ atol_rad_num` so that dark
cells (Erad ≈ Erad_floor) converge in one VODE step.  A ratio of ~10⁴ is sufficient.

**Example (DTypeFront):** `Erad_floor = a_rad × (0.01 K)⁴` gives `N_gamma_floor ≈ 1.25e-10 cm⁻³`.
With `typical_minimal_radiation_T = 10 K`, `atol_rad_num ≈ 1.25e-6 cm⁻³`, giving a
ratio of ~10⁴ — dark cells return in one BDF step, and radiation below the 10 K
blackbody level is numerically negligible.

### Mutual exclusivity

The `integrator.typical_*` parameters and hand-tuned `integrator.atol_*` parameters
are **mutually exclusive**.  Using both triggers an error.  If neither is present,
`SetAtolFromPhysics` is a no-op and VODE uses its built-in defaults — this ensures
backwards compatibility with existing input files.

### Setting up a new problem

1. Set `Erad_floor` in `RadSystem_Traits<problem_t>` to a blackbody temperature low
   enough that it does not produce spurious ionization (typical: 0.01–1 K).
2. Choose `typical_n_H` as the representative hydrogen density of the problem.
3. Choose `typical_minimal_radiation_T` as the lowest temperature at which radiation
   is physically important (e.g. the minimum gas temperature in the domain).
4. Verify that `atol_rad_num / N_gamma_floor ≥ 10⁴`.
5. The `1e-6` prefactor and `desired_accuracy_on_T_at_typical_n_H = 1.0 K` are
   reasonable defaults for most problems.
