#ifndef ACTUAL_EOS_H
#define ACTUAL_EOS_H

// This is a mult-gamma EOS for primordial ISM chemistry.
// Each species can have its own gamma, but
// otherwise, they all act as an ideal gas.

#include <AMReX.H>
#include <network.H>
#include <fundamental_constants.H>
#include <extern_parameters.H>
#include <cmath>
#include <actual_eos_data.H>
#include <string_view>

constexpr std::string_view eos_name = "multigamma";

inline
void actual_eos_init ()
{

    // Set the gammas & masses for the species

    for (int n = 0; n < NumSpec; ++n) {
        gammas[n] = eos_rp::eos_gamma_default;
        spmasses[n] = 1.67353251819e-24;
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
    GET_SPECIES_PARAMS(4);
    GET_SPECIES_PARAMS(5);
    GET_SPECIES_PARAMS(6);
    GET_SPECIES_PARAMS(7);
    GET_SPECIES_PARAMS(8);
    GET_SPECIES_PARAMS(9);
    GET_SPECIES_PARAMS(10);
    GET_SPECIES_PARAMS(11);
    GET_SPECIES_PARAMS(12);
    GET_SPECIES_PARAMS(13);
    GET_SPECIES_PARAMS(14);

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

    const amrex::Real gasconstant = C::n_A * C::k_B;
    const amrex::Real protonmass = C::m_p;
    // Special gamma factors
    amrex::Real sum_Abarinv = 0.0_rt;
    amrex::Real sum_gammasinv = 0.0_rt;
    amrex::Real rhotot = 0.0_rt;

    for (int n = 0; n < NumSpec; ++n) {
        rhotot += state.xn[n]*spmasses[n];
    }

    for (int n = 0; n < NumSpec; ++n) {
        sum_Abarinv += state.xn[n];
        sum_gammasinv += (state.xn[n]*protonmass/rhotot) * (1.0/(gammas[n]-1.0));
    }

    sum_Abarinv *= protonmass/rhotot;
    state.mu = 1.0 / sum_Abarinv;
    sum_gammasinv /= sum_Abarinv;

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
        eint = sum_gammasinv * sum_Abarinv * gasconstant * state.T;
        break;

    case eos_input_re:

        // dens, energy, and xmass are inputs
        // Solve for the temperature

        if constexpr (has_energy<T>) {
            dens = state.rho;

            // stop the integration if the internal energy < 0
            AMREX_ASSERT(state.e > 0.);
            temp = state.e /( sum_gammasinv * gasconstant * sum_Abarinv);
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
            eint = state.p * sum_gammasinv / dens;
            temp = eint / (sum_gammasinv * gasconstant * sum_Abarinv);
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
        amrex::Real pressure = state.rho * eint / sum_gammasinv;
        state.p = pressure;

        state.dpdT = pressure / temp;
        state.dpdr = pressure / dens;
        state.cs = std::sqrt((1.0 + 1.0/sum_gammasinv) * state.p /state.rho);
        if constexpr (has_G<T>) {
            state.G = 0.5 * (1.0 + (1.0 + 1.0/sum_gammasinv));
        }
    }

    amrex::Real dedT = sum_gammasinv * sum_Abarinv * gasconstant;
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
