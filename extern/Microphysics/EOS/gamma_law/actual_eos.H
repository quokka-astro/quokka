#ifndef ACTUAL_EOS_H
#define ACTUAL_EOS_H

#include <string>
#include <string_view>
#include <extern_parameters.H>
#include <fundamental_constants.H>
#include <cmath>

// This is a constant gamma equation of state, using an ideal gas.
//
// The gas may either be completely ionized or completely neutral.
//
// The ratio of specific heats (gamma) is allowed to vary.  NOTE: the
// expression for entropy is only valid for an ideal MONATOMIC gas
// (gamma = 5/3).

constexpr std::string_view eos_name = "gamma_law";

inline
void actual_eos_init() {

  // constant ratio of specific heats
  if (eos_rp::eos_gamma <= 0.0) {
    amrex::Error("gamma_const cannot be < 0");
  }

}



template <typename I>
AMREX_GPU_HOST_DEVICE AMREX_INLINE
bool is_input_valid (I input)
{
  static_assert(std::is_same_v<I, eos_input_t>, "input must be an eos_input_t");

  bool valid = true;

  if (input == eos_input_th) {
    valid = false;
  }

  return valid;
}


template <typename I, typename T>
AMREX_GPU_HOST_DEVICE AMREX_INLINE
void actual_eos (I input, T& state)
{
    static_assert(std::is_same_v<I, eos_input_t>, "input must be an eos_input_t");

    // Get the mass of a nucleon from m_u.
    const amrex::Real m_nucleon = C::m_u;

    if constexpr (has_xn<T>) {
        if (eos_rp::eos_assume_neutral) {
            state.mu = state.abar;
        } else {
            amrex::Real sum = 0.0;
            for (int n = 0; n < NumSpec; n++) {
                sum += (zion[n] + 1.0) * state.xn[n] * aion_inv[n];
            }
            state.mu = 1.0 / sum;
        }
    }

    // For all EOS input modes EXCEPT eos_input_rt, first compute dens
    // and temp as needed from the inputs.

    switch (input)
    {

      case eos_input_rt:

          // dens, temp and xmass are inputs

          // We don't need to do anything here
          break;

      case eos_input_rh:

        // dens, enthalpy, and xmass are inputs

        // Solve for the temperature:
        // h = e + p/rho = (p/rho)*[1 + 1/(gamma-1)] = (p/rho)*gamma/(gamma-1)

          if constexpr (has_enthalpy<T>) {
              state.T = (state.h * state.mu * m_nucleon / C::k_B)*(eos_rp::eos_gamma - 1.0)/eos_rp::eos_gamma;
          }

          break;

      case eos_input_tp:

          // temp, pres, and xmass are inputs

          // Solve for the density:
          // p = rho k T / (mu m_nucleon)

          if constexpr (has_pressure<T>) {
              state.rho =  state.p * state.mu * m_nucleon / (C::k_B * state.T);
          }

          break;

      case eos_input_rp:

          // dens, pres, and xmass are inputs

          // Solve for the temperature:
          // p = rho k T / (mu m_nucleon)

          if constexpr (has_pressure<T>) {
              state.T = state.p * state.mu * m_nucleon / (C::k_B * state.rho);
          }

          break;

      case eos_input_re:

          // dens, energy, and xmass are inputs

          // Solve for the temperature
          // e = k T / [(mu m_nucleon)*(gamma-1)]

          if constexpr (has_energy<T>) {
              state.T = state.e * state.mu * m_nucleon * (eos_rp::eos_gamma - 1.0) / C::k_B;
          }

          break;

      case eos_input_ps:

          // pressure, entropy, and xmass are inputs

          // Solve for the temperature
          // Invert Sackur-Tetrode eqn (below) using
          // rho = p mu m_nucleon / (k T)

          if constexpr (has_pressure<T> && has_entropy<T>) {
              state.T = std::pow(state.p, 2.0/5.0) *
                        std::pow(2.0 * M_PI * C::hbar * C::hbar / (state.mu * m_nucleon), 3.0/5.0) *
                        std::exp(2.0 * state.mu * m_nucleon * state.s / (5.0 * C::k_B) - 1.0) / C::k_B;
          }

          // Solve for the density
          // rho = p mu m_nucleon / (k T)

          if constexpr (has_pressure<T>) {
              state.rho = state.p * state.mu * m_nucleon / (C::k_B * state.T);
          }

          break;

      case eos_input_ph:

          // pressure, enthalpy and xmass are inputs

          // Solve for temperature and density

          if constexpr (has_pressure<T> && has_enthalpy<T>) {
              state.rho = state.p / state.h * eos_rp::eos_gamma / (eos_rp::eos_gamma - 1.0);
              state.T = state.p * state.mu * m_nucleon / (C::k_B * state.rho);
          }

          break;

      case eos_input_th:

          // temperature, enthalpy and xmass are inputs

          // This system is underconstrained.

  #ifndef AMREX_USE_GPU
          amrex::Error("EOS: eos_input_th is not a valid input for the gamma law EOS.");
  #endif

          break;

      default:

  #ifndef AMREX_USE_GPU
          amrex::Error("EOS: invalid input.");
  #endif

        break;

    }

    // Now we have the density and temperature (and mass fractions /
    // mu), regardless of the inputs.

    amrex::Real Tinv = 1.0 / state.T;
    amrex::Real rhoinv = 1.0 / state.rho;

    // Compute the pressure simply from the ideal gas law, and the
    // specific internal energy using the gamma-law EOS relation.
    amrex::Real pressure = state.rho * state.T * C::k_B / (state.mu * m_nucleon);
    amrex::Real energy = pressure / (eos_rp::eos_gamma - 1.0) * rhoinv;
    if constexpr (has_pressure<T>) {
        state.p = pressure;
    }
    if constexpr (has_energy<T>) {
        state.e = energy;
    }

    // enthalpy is h = e + p/rho
    if constexpr (has_enthalpy<T>) {
        state.h = energy + pressure * rhoinv;
    }

    // entropy (per gram) of an ideal monoatomic gas (the Sackur-Tetrode equation)
    // NOTE: this expression is only valid for gamma = 5/3.
    if constexpr (has_entropy<T>) {
        const amrex::Real fac = 1.0 / std::pow(2.0 * M_PI * C::hbar * C::hbar, 1.5);

        state.s = (C::k_B / (state.mu * m_nucleon)) *
                  (2.5 + std::log((std::pow(state.mu * m_nucleon, 2.5) * rhoinv) *
                                  std::pow(C::k_B * state.T, 1.5) * fac));
    }

    // Compute the thermodynamic derivatives and specific heats
    if constexpr (has_pressure<T>) {
        state.dpdT = state.p * Tinv;
        state.dpdr = state.p * rhoinv;
    }
    if constexpr (has_energy<T>) {
        state.dedT = state.e * Tinv;
        state.dedr = 0.0;
    }
    if constexpr (has_entropy<T>) {
        state.dsdT = 1.5 * (C::k_B / (state.mu * m_nucleon)) * Tinv;
        state.dsdr = - (C::k_B / (state.mu * m_nucleon)) * rhoinv;
    }
    if constexpr (has_enthalpy<T>) {
        state.dhdT = state.dedT + state.dpdT * rhoinv;
        state.dhdr = 0.0;
    }

    if constexpr (has_xne_xnp<T>) {
        state.xne = 0.0;
        state.xnp = 0.0;
    }
    if constexpr (has_eta<T>) {
        state.eta = 0.0;
    }
    if constexpr (has_pele_ppos<T>) {
        state.pele = 0.0;
        state.ppos = 0.0;
    }

    if constexpr (has_energy<T>) {
        state.cv = state.dedT;

        if constexpr (has_pressure<T>) {
            state.cp = eos_rp::eos_gamma * state.cv;

            state.gam1 = eos_rp::eos_gamma;

            state.dpdr_e = state.dpdr - state.dpdT * state.dedr * (1.0 / state.dedT);
            state.dpde = state.dpdT * (1.0 / state.dedT);

            // sound speed
            state.cs = std::sqrt(eos_rp::eos_gamma * state.p * rhoinv);
            if constexpr (has_G<T>) {
                state.G = 0.5 * (1.0 + eos_rp::eos_gamma);
            }
        }
    }

    if constexpr (has_dpdA<T>) {
        state.dpdA = - state.p * (1.0 / state.abar);
    }
    if constexpr (has_dedA<T>) {
        state.dedA = - state.e * (1.0 / state.abar);
    }

    if (eos_rp::eos_assume_neutral) {
        if constexpr (has_dpdZ<T>) {
            state.dpdZ = 0.0;
        }
        if constexpr (has_dedZ<T>) {
            state.dedZ = 0.0;
        }
    } else {
        if constexpr (has_dpdZ<T>) {
            state.dpdZ = state.p * (1.0 / (1.0 + state.zbar));
        }
        if constexpr (has_dedZ<T>) {
            state.dedZ = state.e * (1.0/(1.0 + state.zbar));
        }
    }
}

inline
void actual_eos_finalize() {

}

#endif
