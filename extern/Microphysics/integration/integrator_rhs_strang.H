#ifndef INTEGRATOR_RHS_STRANG_H
#define INTEGRATOR_RHS_STRANG_H

#include <network.H>
#include <actual_network.H>
#include <actual_rhs.H>
#include <burn_type.H>
#include <extern_parameters.H>
#include <integrator_data.H>
#include <integrator_type_strang.H>

// The rhs routine provides the right-hand-side for the Rosenbrock solver.
// This is a generic interface that calls the specific RHS routine in the
// network you're actually using.

template <typename BurnT, typename T>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void rhs (const amrex::Real time, BurnT& state, T& int_state, RArray1D& ydot, [[maybe_unused]] const bool in_jacobian=false)
{

    // We are integrating a system of
    //
    // y(1:NumSpec) = dX/dt
    // y(net_ienuc) = denuc/dt

    // we come in with (integrator type) int_state having the current
    // solution (or predicted) state and we need to copy this over to
    // the (burn_t) state to interface with the actual reaction
    // network

    // Fix the state as necessary -- this ensures that the mass
    // fractions that enter are valid (and optionally normalized)

    clean_state(time, state, int_state);

    // Update the thermodynamics as necessary -- this primarily takes
    // the information in int_state (X and e), copies it to the
    // (burn_t) state and calls the EOS to get state.T

    update_thermodynamics(state, int_state);

    // Only do the burn if the incoming temperature is within the temperature
    // bounds. Otherwise set the RHS to zero and return.

    if (state.T <= EOSData::mintemp || state.T >= integrator_rp::MAX_TEMP) {

        for (int n = 1; n <= INT_NEQS; ++n) {
            ydot(n) = 0.0_rt;
        }

        return;

    }

    // at this point, the (burn_t) state is synchronized with the integrator
    // and is thermodynamically consistent.

    // Call the specific network routine to get the RHS.

    actual_rhs(state, ydot);


    // We integrate X, not Y
    // turn it off for primordial chem
    if (! integrator_rp::use_number_densities) {
        for (int n = 1; n <= NumSpec; ++n) {
            ydot(n) *= aion[n-1];
        }
    }

    // scale the energy
    if (integrator_rp::scale_system) {
        ydot(net_ienuc) /= state.e_scale;
    }

    // Allow energy integration to be disabled.

    if (! integrator_rp::integrate_energy) {
        ydot(net_ienuc) = 0.0_rt;
    }

    // apply fudge factor:

    if (integrator_rp::react_boost > 0.0_rt) {
        for (int n = 1; n <= INT_NEQS; ++n) {
            ydot(n) *= integrator_rp::react_boost;
        }
    }

}


// Analytical Jacobian
template<typename BurnT, typename T, class MatrixType>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void jac ([[maybe_unused]] const amrex::Real time, BurnT& state, T& int_state, MatrixType& pd)
{

    // Only do the burn if the incoming temperature is within the temperature
    // bounds. Otherwise set the Jacobian to zero and return.

    if (state.T <= EOSData::mintemp || state.T >= integrator_rp::MAX_TEMP) {
        pd.zero();
        return;
    }

    // Call the specific network routine to get the Jacobian.

    integrator_to_burn(int_state, state);

    actual_jac(state, pd);

    // We integrate X, not Y
    // turn it off for primordial chem
    if (! integrator_rp::use_number_densities) {
        for (int j = 1; j <= NumSpec; ++j) {
            for (int i = 1; i <= INT_NEQS; ++i) {
                pd.mul(j, i, aion[j-1]);
                pd.mul(i, j, aion_inv[j-1]);
            }
        }
    }

    // scale the energy derivatives

    if (integrator_rp::scale_system) {
        // first the row de/dX
        for (int j = 1; j <= INT_NEQS; ++j) {
            pd(net_ienuc,j) /= state.e_scale;
        }

        // now the column dX/de
        for (int i = 1; i <= INT_NEQS; ++i) {
            pd(i,net_ienuc) *= state.e_scale;
        }
    }

    // apply fudge factor:
    if (integrator_rp::react_boost > 0.0_rt) {
        pd.mul(integrator_rp::react_boost);
    }

    // Allow temperature and energy integration to be disabled.

    if (! integrator_rp::integrate_energy) {
        for (int j = 1; j <= INT_NEQS; ++j) {
            pd(net_ienuc,j) = 0.0_rt;
        }
    }

}

#endif
