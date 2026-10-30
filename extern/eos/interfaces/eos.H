#ifndef EOS_H
#define EOS_H

#include <iostream>
#include <network_properties.H>
#include <eos_data.H>
#include <eos_type.H>
#include <eos_composition.H>
#include <eos_override.H>
#include <actual_eos.H>
#include <AMReX_Algorithm.H>


// EOS initialization routine: read in general EOS parameters, then
// call any specific initialization used by the EOS.

AMREX_INLINE
void eos_init (amrex::Real& small_temp_in, amrex::Real& small_dens_in)
{
  // Allocate and set default values

  EOSData::mintemp = 1.e-200;
  EOSData::maxtemp = 1.e200;
  EOSData::mindens = 1.e-200;
  EOSData::maxdens = 1.e200;
  EOSData::minx = 1.e-200;
  EOSData::maxx = 1.0 + 1.e-12;
  EOSData::minye = 1.e-200;
  EOSData::maxye = 1.0 + 1.e-12;
  EOSData::mine = 1.e-200;
  EOSData::maxe = 1.e200;
  EOSData::minp = 1.e-200;
  EOSData::maxp = 1.e200;
  EOSData::mins = 1.e-200;
  EOSData::maxs = 1.e200;
  EOSData::minh = 1.e-200;
  EOSData::maxh = 1.e200;

  // Set up any specific parameters or initialization steps required by the EOS we are using.
  actual_eos_init();

  // Set EOSData::min{temp,dens} and small_{temp,dens} to the maximum of the two
  EOSData::mintemp = amrex::max(EOSData::mintemp, small_temp_in);
  small_temp_in = amrex::max(small_temp_in, EOSData::mintemp);

  EOSData::mindens = amrex::max(EOSData::mindens, small_dens_in);
  small_dens_in = amrex::max(small_dens_in, EOSData::mindens);

  EOSData::initialized = true;
}

// Overload of the above for cases where we didn't pass in small_temp and small_dens

AMREX_INLINE
void eos_init ()
{
    amrex::Real small_temp_tmp = -1.e200;
    amrex::Real small_dens_tmp = -1.e200;
    eos_init(small_temp_tmp, small_dens_tmp);
}

AMREX_INLINE
void eos_finalize ()
{
  actual_eos_finalize();
}

// Given an EOS state, ensure that rho and T are
// valid, then call with eos_input_rt.
template <typename T>
AMREX_GPU_HOST_DEVICE AMREX_INLINE
void eos_reset (T& state, bool& has_been_reset)
{
  state.T = amrex::min(EOSData::maxtemp, amrex::max(EOSData::mintemp, state.T));
  state.rho = amrex::min(EOSData::maxdens, amrex::max(EOSData::mindens, state.rho));
  actual_eos(eos_input_rt, state);

  has_been_reset = true;
}

// For density, just ensure that it is within mindens and maxdens.
template <typename T>
AMREX_GPU_HOST_DEVICE AMREX_INLINE
void reset_rho (T& state)
{
  state.rho = amrex::min(EOSData::maxdens, amrex::max(EOSData::mindens, state.rho));
}

// For temperature, just ensure that it is within mintemp and maxtemp.
template <typename T>
AMREX_GPU_HOST_DEVICE AMREX_INLINE
void reset_T (T& state)
{
  state.T = amrex::min(EOSData::maxtemp, amrex::max(EOSData::mintemp, state.T));
}

template <typename T>
AMREX_GPU_HOST_DEVICE AMREX_INLINE
void reset_e (T& state, bool& has_been_reset)
{
    if constexpr (has_energy<T>) {
        if (state.e < EOSData::mine || state.e > EOSData::maxe) {
            eos_reset(state, has_been_reset);
        }
    }
}

template <typename T>
AMREX_GPU_HOST_DEVICE AMREX_INLINE
void reset_h (T& state, bool& has_been_reset)
{
    if constexpr (has_enthalpy<T>) {
        if (state.h < EOSData::minh || state.h > EOSData::maxh) {
            eos_reset(state, has_been_reset);
        }
    }
}

template <typename T>
AMREX_GPU_HOST_DEVICE AMREX_INLINE
void reset_s (T& state, bool& has_been_reset)
{
    if constexpr (has_entropy<T>) {
        if (state.s < EOSData::mins || state.s > EOSData::maxs) {
            eos_reset(state, has_been_reset);
        }
    }
}

template <typename T>
AMREX_GPU_HOST_DEVICE AMREX_INLINE
void reset_p (T& state, bool& has_been_reset)
{
    if constexpr (has_pressure<T>) {
        if (state.p < EOSData::minp || state.p > EOSData::maxp) {
            eos_reset(state, has_been_reset);
        }
    }
}

template <typename I, typename T>
AMREX_GPU_HOST_DEVICE AMREX_INLINE
void reset_inputs(const I input, T& state, bool& has_been_reset)
{
  // Reset the input quantities to valid values. For inputs other than rho and T,
  // this will evolve an EOS call, which will negate the need to do the main EOS call.

  switch (input) {

  case eos_input_rt:

    reset_rho(state);
    reset_T(state);

    break;

  case eos_input_rh:

    reset_rho(state);
    reset_h(state, has_been_reset);

    break;

  case eos_input_tp:

    reset_T(state);
    reset_p(state, has_been_reset);

    break;

  case eos_input_rp:

    reset_rho(state);
    reset_p(state, has_been_reset);

    break;

  case eos_input_re:

    reset_rho(state);
    reset_e(state, has_been_reset);

    break;

  case eos_input_ps:

    reset_p(state, has_been_reset);
    reset_s(state, has_been_reset);

    break;

  case eos_input_ph:

    reset_p(state, has_been_reset);
    reset_h(state, has_been_reset);

    break;

  case eos_input_th:

    reset_T(state);
    reset_h(state, has_been_reset);

    break;
  }
}

#ifndef AMREX_USE_GPU
template <typename T>
AMREX_INLINE
void print_state (T& state) {
  std::cout << "DENS = " << state.rho << std::endl;
  std::cout << "TEMP = " << state.T << std::endl;
  std::cout << "X = ";
  for (int n = 0; n < NumSpec; n ++) {
    std::cout << state.xn[n] << " ";
  }
  std::cout << std::endl;
#ifdef PHOTOCHEMISTRY
  std::cout << "RAD = ";
  for (int n = 0; n < NumChemRadEqs; n ++) {
    std::cout << state.rn[n] << " ";
  }
  std::cout << std::endl;
  std::cout << "RAD_THERMAL = ";
  for (int n = 0; n < NumThermalBands; n ++) {
    std::cout << state.re_thermal[n] << " ";
  }
  std::cout << std::endl;
#endif

  std::cout << "Y_E = " << state.y_e;
}

template <typename T>
AMREX_INLINE
void check_rho (T& state)
{
  if (state.rho < EOSData::mindens) {
    print_state(state);
    amrex::Error("EOS: rho smaller than mindens.");

  } else if (state.rho > EOSData::maxdens) {
    print_state(state);
    amrex::Error("EOS: rho greater than maxdens.");
  }
}

template <typename T>
AMREX_INLINE
void check_T (T& state)
{
  if (state.T < EOSData::mintemp) {
    print_state(state);
    amrex::Error("EOS: T smaller than mintemp.");

  } else if (state.T > EOSData::maxtemp) {
    print_state(state);
    amrex::Error("EOS: T greater than maxtemp.");
  }
}

template <typename T>
AMREX_INLINE
void check_e (T& state)
{
    if constexpr (has_energy<T>) {
        if (state.e < EOSData::mine) {
            print_state(state);
            amrex::Error("EOS: e smaller than mine.");

        } else if (state.e > EOSData::maxe) {
            print_state(state);
            amrex::Error("EOS: e greater than maxe.");
        }
    }
}

template <typename T>
AMREX_INLINE
void check_h (T& state)
{
    if constexpr (has_enthalpy<T>) {
        if (state.h < EOSData::minh) {
            print_state(state);
            amrex::Error("EOS: h smaller than minh.");

        } else if (state.h > EOSData::maxh) {
            print_state(state);
            amrex::Error("EOS: h greater than maxh.");

        }
    }
}

template <typename T>
AMREX_INLINE
void check_s (T& state)
{
    if constexpr (has_entropy<T>) {
        if (state.s < EOSData::mins) {
            print_state(state);
            amrex::Error("EOS: s smaller than mins.");

        } else if (state.s > EOSData::maxs) {
            print_state(state);
            amrex::Error("EOS: s greater than maxs.");
        }
    }
}

template <typename T>
AMREX_INLINE
void check_p (T& state)
{
    if constexpr (has_pressure<T>) {
        if (state.p < EOSData::minp) {
            print_state(state);
            amrex::Error("EOS: p smaller than minp.");

        } else if (state.p > EOSData::maxp) {
            print_state(state);
            amrex::Error("EOS: p greater than maxp.");
        }
    }
}

template <typename I, typename T>
AMREX_INLINE
void check_inputs (const I input, T& state)
{
  // Check the inputs for validity.

  for (int n = 0; n < NumSpec; n++) {
    if (state.xn[n] < EOSData::minx) {
      print_state(state);
      amrex::Error("EOS: mass fraction less than minimum possible mass fraction.");

    } else if (state.xn[n] > EOSData::maxx) {
      print_state(state);
      amrex::Error("EOS: mass fraction more than maximum possible mass fraction.");
    }
  }

  if (state.y_e > EOSData::minye) {
    print_state(state);
    amrex::Error("EOS: y_e less than minimum possible electron fraction.");

  } else if (state.y_e > EOSData::maxye) {
    print_state(state);
    amrex::Error("EOS: y_e greater than maximum possible electron fraction.");
  }

  switch (input) {

  case eos_input_rt:

    check_rho(state);
    check_T(state);
    break;

  case eos_input_rh:

    check_rho(state);
    check_h(state);
    break;

  case eos_input_tp:

    check_T(state);
    check_p(state);
    break;

  case eos_input_rp:

    check_rho(state);
    check_p(state);
    break;

  case eos_input_re:

    check_rho(state);
    check_e(state);
    break;

  case eos_input_ps:

    check_p(state);
    check_s(state);
    break;

  case eos_input_ph:

    check_p(state);
    check_h(state);
    break;

  case eos_input_th:

    check_T(state);
    check_h(state);

  }
}
#endif

template <typename I, typename T>
AMREX_GPU_HOST_DEVICE AMREX_INLINE
void eos (const I input, T& state)
{
  static_assert(std::is_same_v<I, eos_input_t>, "input must be an eos_input_t");

  // Input arguments

  bool has_been_reset = false;

  // Local variables

#ifndef AMREX_USE_GPU
  if (!EOSData::initialized) {
    amrex::Error("EOS: not initialized");
  }
#endif

  if constexpr (has_xn<T>) {
      // Get abar, zbar, etc.
      composition(state);
  }

  // Force the inputs to be valid.
  reset_inputs(input, state, has_been_reset);

  // Allow the user to override any details of the
  // EOS state. This should generally occur right
  // before the actual_eos call.
  eos_override(state);

  // Call the EOS.
  if (!has_been_reset) {
    actual_eos(input, state);
  }
}

#endif
