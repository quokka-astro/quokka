# Photoionization Module — VODE Tolerance Design

## Overview

Quokka uses VODE (via Microphysics) to integrate the chemistry and internal energy
source terms. The integrator requires absolute tolerances (`atol`) for each solution
variable. Rather than hand-tuning these tolerances, `SetAtolFromPhysics<problem_t>()`
(in `src/radiation/photochem_atol.H`) derives them from high-level physical
parameters specified in the input file.

## Input parameters

| Parameter                                         | Required | Default | Description                                                                              |
| ------------------------------------------------- | -------- | ------- | ---------------------------------------------------------------------------------------- |
| `integrator.typical_n_H`                          | yes      | —       | Representative total H number density (cm⁻³)                                             |
| `integrator.typical_minimal_radiation_T`          | yes      | —       | Typical temperature of the cold (neutral) gas in the domain (K). Sets the photon density below which radiation is numerically negligible. |
| `integrator.desired_accuracy_on_T_at_typical_n_H` | no       | 1.0 K   | Desired temperature accuracy at `typical_n_H`                                            |
| `integrator.spec_abundance_tol`                   | no       | 1e-5    | Species negligibility threshold, as a fraction of `typical_n_H`                          |
| `integrator.radiation_failure_tolerance`          | no       | 0.05    | Maximum allowed negative photon number density (cm⁻³) before a burn is flagged as failed |

The relative tolerances (`rtol_spec`, `rtol_enuc`, `rtol_rad_num`) are specified directly
in the input file as usual.

## Physical constants

| Symbol     | Value                                  | Description                                                                                                    |
| ---------- | -------------------------------------- | -------------------------------------------------------------------------------------------------------------- |
| `a_rad`    | `7.5657e-15 erg cm⁻³ K⁻⁴`              | Radiation constant (from `fundamental_constants.H`)                                                            |
| `k_B`      | `1.380649e-16 erg K⁻¹`                 | Boltzmann constant                                                                                             |
| `m_p`      | `1.67262192e-24 g`                     | Proton mass                                                                                                    |
| `c_v`      | `3/2 × k_B / m_p ≈ 1.24e8 erg g⁻¹ K⁻¹` | Specific heat of monatomic hydrogen gas                                                                        |
| `E_photon` | problem-dependent                      | Midpoint energy of the first chemistry radiation band (erg), from `RadSystem<problem_t>::GetChemBandQuanta(0)` |

## Derived atol values

Let $T_{\rm min} \equiv$ `typical_minimal_radiation_T` for brevity.
`SetAtolFromPhysics` computes:

| Variable | Formula | Rationale |
|---|---|---|
| `atol_spec` | $\texttt{spec\_abundance\_tol} \times \texttt{typical\_n\_H}$ | Species below this fraction of $n_{\rm H}$ are negligible |
| `atol_enuc` | $c_v \times \texttt{desired\_accuracy\_on\_T\_at\_typical\_n\_H}$ | Converts temperature accuracy to internal energy tolerance ($c_v = \tfrac{3}{2} k_B / m_p$) |
| `atol_rad_num` | $10^{-6} \times a_{\rm rad} \times T_{\rm min}^4 / E_{\rm photon}$ | One millionth of the blackbody photon density at $T_{\rm min}$ — radiation below this is negligible |

The `radiation_failure_tolerance` (default 0.05 cm⁻³) and `species_failure_tolerance`
(see §10) are not derived by `SetAtolFromPhysics`; they are set independently.

## The $10^{-6}$ prefactor

The factor $10^{-6}$ in `atol_rad_num` has a specific physical meaning:

- It sets the tolerance to 1 part per million of the blackbody photon density at $T_{\rm min}$.
- After roughly $10^6$ VODE steps, the accumulated local error in photon number remains
  below the physically meaningful radiation level at the minimum temperature.
- Cells with radiation below this threshold are considered "dark" and VODE
  returns in a single BDF step.

## radiation_failure_tolerance

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

## Relationship between `Erad_floor` and `typical_minimal_radiation_T`

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

## Mutual exclusivity

The `integrator.typical_*` parameters and hand-tuned `integrator.atol_*` parameters
are **mutually exclusive** — using both triggers an error. Specifying neither also
triggers an error, because VODE's built-in defaults ($\sim 10^{-10}$) are unusably
tight for photochemistry and will cause the integrator to stall.

## Setting up a new problem

1. Set `Erad_floor` in `RadSystem_Traits<problem_t>` to a blackbody temperature low
   enough that it does not produce spurious ionization (typical: $0.01$–$1$ K).
2. Choose `typical_n_H` as the representative hydrogen density of the problem.
3. Choose `typical_minimal_radiation_T` as the typical temperature of the cold
   (neutral) gas in the domain.
4. Check $10^{-6} \, (T_{\rm min} / T_{\rm floor})^4 \geq 10^4$, i.e.
   $T_{\rm floor} \leq T_{\rm min} / 316$.
   If this fails, either lower `Erad_floor` or raise `typical_minimal_radiation_T`.
5. The $10^{-6}$ prefactor and `desired_accuracy_on_T_at_typical_n_H = 1.0 K$ are
   reasonable defaults for most problems.

## `species_failure_tolerance`

VODE's internal step uses `species_failure_tolerance` directly; the final interpolated
state uses $1.5 \times$ `species_failure_tolerance` (via
`vode_final_state_species_failure_tolerance_factor` in `vode_type.H`). This accounts
for VODE's non-monotonic interpolation back to the output time, preventing false
burn failures from interpolation noise.

When using `SetAtolFromPhysics`, set `integrator.species_failure_tolerance` equal to
`atol_spec` (the species negligibility floor). The $1.5\times$ final-state factor
absorbs BDF interpolation overshoot without manual inflation.
