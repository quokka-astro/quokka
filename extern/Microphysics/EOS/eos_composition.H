#ifndef EOS_COMPOSITION_H
#define EOS_COMPOSITION_H

#include <AMReX_BLFort.H>
#include <network.H>
#include <eos_type.H>


using namespace amrex::literals;


template <typename T>
AMREX_GPU_HOST_DEVICE AMREX_INLINE
void composition (T& state)
{
  // Calculate abar, the mean nucleon number,
  // zbar, the mean proton number,
  // mu, the mean molecular weight,
  // mu_e, the mean number of nucleons per electron, and
  // y_e, the electron fraction.


  state.y_e = 0.0_rt;
  state.abar = 0.0_rt;
  for (int n = 0; n < NumSpec; n++) {
    state.y_e += state.xn[n] * zion[n] * aion_inv[n];
    state.abar += state.xn[n] * aion_inv[n];
  }
  state.abar = 1.0_rt / state.abar;

  state.mu_e = 1.0_rt / state.y_e;
  state.zbar = state.abar * state.y_e;
}

#endif
