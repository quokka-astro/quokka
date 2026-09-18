#ifndef ACTUAL_EOS_H
#define ACTUAL_EOS_H

// This is a mult-gamma EOS for photochemistry.
// Each species can have its own gamma, but
// otherwise, they all act as an ideal gas.

#include "eos_type.H"
#include <AMReX.H>
#include <network.H>
#include <fundamental_constants.H>
#include <extern_parameters.H>
#include <cmath>
#include <actual_eos_data.H>

const std::string eos_name = "multigamma";

inline
void actual_eos_init ()
{

    // Set the gammas & masses for the species

    for (int n = 0; n < NumSpec; ++n) {
        gammas[n] = eos_rp::eos_gamma_default;
        spmasses[n] = 1.67353251819e-24;
    }
    // Electrons have actual electron mass, not proton mass
    int const e_idx = network_spec_index("Electron");
    if (e_idx >= 0) {
        spmasses[e_idx] = C::m_e;
    }

    int idx;
    // Set the gammas & masses for the species
    #define GET_SPECIES_PARAMS(num) do { \
        idx = network_spec_index(eos_rp::species_##num##_name); \
        if (idx >= 0) { \
            gammas[idx] = eos_rp::species_##num##_gamma; \
            spmasses[idx] = eos_rp::species_##num##_mass; \
        } \
    } while (0)

    GET_SPECIES_PARAMS(1);
    GET_SPECIES_PARAMS(2);
    GET_SPECIES_PARAMS(3);

    #undef GET_SPECIES_PARAMS

}


template <typename I>
AMREX_GPU_HOST_DEVICE AMREX_INLINE
bool is_input_valid (I input)
{
  static_assert(std::is_same_v<I, eos_input_t>, "input must be either eos_input_rt or eos_input_re");

  bool valid = false;

  if (input == eos_input_rt ||
      input == eos_input_re) {
    valid = true;
  }

  return valid;
}


template <typename I, typename T>
AMREX_GPU_HOST_DEVICE AMREX_INLINE
void actual_eos (I input, T& state)
{
    static_assert(std::is_same_v<I, eos_input_t>, "input must be either eos_input_rt or eos_input_re");
    constexpr amrex::Real protonmass = C::m_p;

    amrex::Real sum_Abarinv = 0.0_rt;
    amrex::Real sum_ni_fi_over_2 = 0.0_rt;
    amrex::Real rhotot = 0.0_rt;

    for (int n = 0; n < NumSpec; ++n) {
        rhotot += state.xn[n] * spmasses[n];
        sum_ni_fi_over_2 += state.xn[n] / (gammas[n] - 1.0);
        sum_Abarinv += state.xn[n];
    }
    // fi_over_2_avg is the per-particle average heat capacity (used for cs, G, pressure)
    // sum_ni_fi_over_2 remains the total (unnormalized) heat capacity (used for e, T, dedT)
    const amrex::Real fi_over_2_avg = sum_ni_fi_over_2 / sum_Abarinv;
    sum_Abarinv *= protonmass/rhotot;
    state.mu = 1.0 / sum_Abarinv;

    //-------------------------------------------------------------------------
    // For all EOS input modes EXCEPT eos_input_rt, first compute dens
    // and temp as needed from the inputs.
    //-------------------------------------------------------------------------

    amrex::Real temp = NAN;
    amrex::Real dens = NAN;
    amrex::Real eint = NAN;
    switch (input) {

    case eos_input_rt:

        // dens, temp and xmass are inputs
        // We don't need to do anything here
        temp = state.T;
        dens = state.rho;
        eint = sum_ni_fi_over_2 * C::k_B * temp / rhotot;
        break;

    case eos_input_re:

        // dens, energy, and xmass are inputs
        // Solve for the temperature

        if constexpr (has_energy<T>) {
            dens = state.rho;

            // stop the integration if the internal energy < 0
            AMREX_ASSERT(state.e > 0.);
            temp = state.e * rhotot / (sum_ni_fi_over_2 * C::k_B);
            eint = state.e;
        }

        break;

    case eos_input_rh:
        // dens, enthalpy, and xmass are inputs
#ifndef AMREX_USE_GPU
        amrex::Error("eos_input_rh is not supported");
#endif

          break;

    case eos_input_tp:
        // temp, pressure, and xmass are inputs
#ifndef AMREX_USE_GPU
        amrex::Error("eos_input_tp is not supported");
#endif

          break;

    case eos_input_rp:
        // dens, pressure, and xmass are inputs
        if constexpr (has_pressure<T>) {
            dens = state.rho;

            // stop the integration if the pressure < 0
            AMREX_ASSERT(state.p > 0.);
            eint = state.p * fi_over_2_avg / dens;
            temp = eint * rhotot / (sum_ni_fi_over_2 * C::k_B);
        }
        break;

    case eos_input_ps:
        // pressure, entropy, and xmass are inputs
#ifndef AMREX_USE_GPU
        amrex::Error("eos_input_ps is not supported");
#endif

        break;

    case eos_input_ph:
        // pressure, enthalpy, and xmass are inputs
#ifndef AMREX_USE_GPU
        amrex::Error("eos_input_ph is not supported");
#endif

          break;

    case eos_input_th:
        // temp, enthalpy, and xmass are inputs
#ifndef AMREX_USE_GPU
        amrex::Error("eos_input_th is not supported");
#endif

          break;

    default:

#ifndef AMREX_USE_GPU
        amrex::Error("EOS: invalid input.");
#endif

        break;

    }

    //-------------------------------------------------------------------------
    // Now we have the density and temperature (and mass fractions /
    // mu), regardless of the inputs.
    //-------------------------------------------------------------------------

    state.T   = temp;
    state.rho = dens;

    if constexpr (has_energy<T>) {
        state.e = eint;
    }

    if constexpr (has_pressure<T>) {
        amrex::Real pressure = state.rho * eint / fi_over_2_avg;
        state.p = pressure;

        state.dpdT = pressure / temp;
        state.dpdr = pressure / dens;
        state.cs = std::sqrt((1.0 + 1.0/fi_over_2_avg) * state.p /state.rho);
        if constexpr (has_G<T>) {
            state.G = 0.5 * (1.0 + (1.0 + 1.0/fi_over_2_avg));
        }
    }

    amrex::Real dedT = sum_ni_fi_over_2 * C::k_B / rhotot;
    amrex::Real dedr = 0.0_rt;
    if constexpr (has_energy<T>) {
        state.dedT = dedT;
        state.dedr = dedr;
        if constexpr (has_pressure<T>) {
            state.dpde = state.dpdT / state.dedT;
        }
    }

}



inline
void actual_eos_finalize ()
{
}

#endif
