#ifndef INTEGRATOR_TYPE_STRANG_H
#define INTEGRATOR_TYPE_STRANG_H

#include <AMReX_Algorithm.H>

#include <eos.H>
#include <integrator_type.H>

template <typename I, typename BurnT>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void integrator_to_burn (const I& int_state, BurnT& state)
{
    // Copy the integration data to the burn state.

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

    state.time = int_state.t;
}


template <typename BurnT, typename I>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void burn_to_integrator (const BurnT& state, I& int_state)
{
    // Copy the integration data from the burn state.

    for (int n = 1; n <= NumSpec; ++n) {
        int_state.y(n) = state.xn[n-1];
    }

    int_state.y(net_ienuc) = state.e;

#ifdef PHOTOCHEMISTRY
    for (int n = 1; n <= NumRadEqs; ++n) {
        int_state.y(net_ienuc + n) = state.rn[n-1];
    }
    int_state.c_hat = state.c_hat;
#endif

    if (integrator_rp::scale_system) {
        int_state.y(net_ienuc) /= state.e_scale;
    }
}

template <typename BurnT, typename I>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void clean_state (const amrex::Real time, BurnT& state, I& int_state)
{
    amrex::ignore_unused(time);
    amrex::ignore_unused(state);

    // Ensure that mass fractions always stay positive and less than or
    // equal to 1.

    if (integrator_rp::do_species_clip) {
        for (int n = 1; n <= NumSpec; ++n) {
            int_state.y(n) = amrex::Clamp(int_state.y(n), integrator_rp::SMALL_X_SAFE, 1.0_rt);
        }
    }

    // for primordial chem, just enforce positivity
    if (integrator_rp::use_number_densities) {
        for (int n = 1; n <= NumSpec; ++n) {
            int_state.y(n) = amrex::max(int_state.y(n), integrator_rp::SMALL_X_SAFE);
        }
    }
#ifdef PHOTOCHEMISTRY
    for (int n = net_ienuc+1; n <= net_ienuc+NumRadEqs; n += MicrophysicsNumRadVarsPerGroup) {
        int_state.y(n) = amrex::max(int_state.y(n), integrator_rp::SMALL_X_SAFE);
    }
#endif

    // Renormalize the abundances as necessary.

    if (integrator_rp::renormalize_abundances) {
        amrex::Real sum = 0.0_rt;

        for (int n = 1; n <= NumSpec; ++n) {
            sum += int_state.y(n);
        }

        for (int n = 1; n <= NumSpec; ++n) {
            int_state.y(n) /= sum;
        }
    }
}

#endif
