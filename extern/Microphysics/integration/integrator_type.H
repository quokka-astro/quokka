#ifndef INTEGRATOR_TYPE_H
#define INTEGRATOR_TYPE_H

#include <eos.H>

///
/// update the thermodynamics in the burn_t state -- this may involve an EOS call.
/// we also pass in the int_state so we get the latest values of the mass fractions,
/// temperature, and internal energy
///
template <typename BurnT, typename I>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void update_thermodynamics (BurnT& state, const I& int_state)
{

    for (int n = 1; n <= NumSpec; ++n) {
        state.xn[n-1] = int_state.y(n);
    }
    state.e = int_state.y(net_ienuc);
#ifdef PHOTOCHEMISTRY
    for (int n = 1; n <= NumRadEqs; ++n) {
        state.rn[n-1] = int_state.y(net_ienuc + n);
    }
    state.c_hat = int_state.c_hat;
#endif

    if (integrator_rp::scale_system) {
        state.e *= state.e_scale;
    }

    // Get T from e (also updates composition quantities).

    if (integrator_rp::call_eos_in_rhs) {
        eos(eos_input_re, state);
    }

    // override T if we are fixing it (e.g. due to
    // drive_initial_convection)
    if (state.T_fixed > 0.0_rt) {
        state.T = state.T_fixed;
    }

}

#endif
