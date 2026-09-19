#ifndef EOS_H
#define EOS_H

#include <iostream>
#include <network_properties.H>
#include <eos_data.H>
#include <eos_type.H>
#include <eos_composition.H>
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

  // Call the EOS.
  if (!has_been_reset) {
    actual_eos(input, state);
  }
}

#endif
