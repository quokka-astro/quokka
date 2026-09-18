#ifndef INTEGRATOR_SETUP_STRANG_H
#define INTEGRATOR_SETUP_STRANG_H

#include <AMReX_REAL.H>
#include <AMReX_ANSIEscCode.H>
#include <AMReX_GpuPrint.H>

#include <iomanip>

#include <network.H>
#include <burn_type.H>
#include <eos_type.H>
#include <eos.H>
#include <integrator_data.H>

#include <extern_parameters.H>

struct state_backup_t {
    amrex::Real T_in{};
    amrex::Real e_in{};
#ifndef AMREX_USE_GPU
    amrex::Real xn_in[NumSpec]{};
#ifdef PHOTOCHEMISTRY
    amrex::Real rn_in[NumRadEqs]{};
#endif
#endif
};

template <typename BurnT, typename IntegratorT>
AMREX_GPU_HOST_DEVICE AMREX_INLINE
IntegratorT integrator_setup (BurnT& state, amrex::Real dt, bool is_retry)
{

    IntegratorT int_state{};

    // Set the tolerances.

    if (!is_retry) {
        int_state.atol_spec = integrator_rp::atol_spec; // mass fractions
        int_state.atol_enuc = integrator_rp::atol_enuc; // energy generated

        int_state.rtol_spec = integrator_rp::rtol_spec; // mass fractions
        int_state.rtol_enuc = integrator_rp::rtol_enuc; // energy generated
#ifdef PHOTOCHEMISTRY
        int_state.atol_rad_num = integrator_rp::atol_rad_num; // radiation number density
        int_state.rtol_rad_num = integrator_rp::rtol_rad_num; // radiation number density
#endif
    } else {
        int_state.atol_spec = integrator_rp::retry_atol_spec > 0 ?
            integrator_rp::retry_atol_spec : integrator_rp::atol_spec; // mass fractions
        int_state.atol_enuc = integrator_rp::retry_atol_enuc > 0 ?
            integrator_rp::retry_atol_enuc : integrator_rp::atol_enuc; // energy generated

        int_state.rtol_spec = integrator_rp::retry_rtol_spec > 0 ?
            integrator_rp::retry_rtol_spec : integrator_rp::rtol_spec; // mass fractions
        int_state.rtol_enuc = integrator_rp::retry_rtol_enuc > 0 ?
            integrator_rp::retry_rtol_enuc : integrator_rp::rtol_enuc; // energy generated
#ifdef PHOTOCHEMISTRY
        int_state.atol_rad_num = integrator_rp::retry_atol_rad_num > 0 ?
            integrator_rp::retry_atol_rad_num : integrator_rp::atol_rad_num; // radiation number density
        int_state.rtol_rad_num = integrator_rp::retry_rtol_rad_num > 0 ?
            integrator_rp::retry_rtol_rad_num : integrator_rp::rtol_rad_num; // radiation number density
#endif
    }

    // set the Jacobian type
    if (is_retry && integrator_rp::retry_swap_jacobian) {
        int_state.jacobian_type = (integrator_rp::jacobian == 1) ? 2 : 1;
    } else {
        int_state.jacobian_type = static_cast<short>(integrator_rp::jacobian);
    }

    // Start off by assuming a successful burn.

    state.success = true;

    // Initialize the integration time.

    int_state.t = 0.0_rt;
    int_state.tout = dt;

    // We assume that (rho, T) coming in are valid, do an EOS call
    // to fill the rest of the thermodynamic variables.

    eos(eos_input_rt, state);

    // set the scaling for energy if we integrate it dimensionlessly
    state.e_scale = state.e;

    if (integrator_rp::scale_system) {
        // the absolute tol for energy needs to reflect the scaled
        // energy the integrator sees
        int_state.atol_enuc /= state.e_scale;
    }

    // Fill in the initial integration state.

    burn_to_integrator(state, int_state);

    return int_state;
}

///
/// Save the initial composition and thermodynamic state for
/// diagnostics.
///
template <typename BurnT>
AMREX_GPU_HOST_DEVICE AMREX_INLINE
state_backup_t integrator_backup (const BurnT& state) {

    state_backup_t state_save;

#ifndef AMREX_USE_GPU
    for (int n = 0; n < NumSpec; ++n) {
        state_save.xn_in[n] = state.xn[n];
    }
#ifdef PHOTOCHEMISTRY
    for (int n = 0; n < NumRadEqs; ++n) {
        state_save.rn_in[n] = state.rn[n];
    }
#endif
#endif
    state_save.T_in = state.T;
    state_save.e_in = state.e;

    return state_save;

}


template <typename BurnT, typename IntegratorT>
AMREX_GPU_HOST_DEVICE AMREX_INLINE
void integrator_cleanup (IntegratorT& int_state, BurnT& state,
                         int istate, const state_backup_t& state_save, amrex::Real dt)
{
    // Copy the integration data back to the burn state.

    integrator_to_burn(int_state, state);

    // Subtract off the initial energy (the application codes expect
    // to get back only the generated energy during the burn).

    if (integrator_rp::subtract_internal_energy) {
        state.e -= state_save.e_in;
    }

    // Normalize the final abundances (except if they are number
    // densities)

    if (! integrator_rp::use_number_densities) {
        normalize_abundances_burn(state);
    }

    // Get the number of RHS and Jacobian evaluations.

    state.n_rhs = int_state.n_rhs;
    state.n_jac = int_state.n_jac;
    state.n_step = int_state.n_step;

    state.error_code = static_cast<short>(istate);

    // The integrator may not always fail even though it can lead to
    // unphysical states.  Add some checks that indicate a burn fail
    // even if the integrator thinks the integration was successful.

    if (istate != IERR_SUCCESS) {
        state.success = false;
    }

    for (int n = 1; n <= NumSpec; ++n) {
        if (int_state.y(n) < -integrator_rp::species_failure_tolerance) {
            state.success = false;
        }

        // Don't enforce a max if we are evolving number densities

        if (! integrator_rp::use_number_densities) {
            if (int_state.y(n) > 1.0_rt + integrator_rp::species_failure_tolerance) {
                state.success = false;
            }
        }
    }
#ifdef PHOTOCHEMISTRY
    for (int n = net_ienuc+1; n <= net_ienuc+NumRadEqs; n += MicrophysicsNumRadVarsPerGroup) {
        if (int_state.y(n) < -integrator_rp::radiation_failure_tolerance) {
            state.success = false;
        }
    }
#endif

#ifndef AMREX_USE_GPU
    if (integrator_rp::burner_verbose) {
        // Print out some integration statistics, if desired.
        std::cout <<  "integration summary: " << std::endl;
        std::cout <<  "dens: " << state.rho << " temp: " << state.T << std::endl;
        std::cout << " energy released: " << state.e << std::endl;
        std::cout <<  "number of steps taken: " << state.n_step << std::endl;
        std::cout <<  "number of f evaluations: " << state.n_rhs << std::endl;
    }
#endif

    // If we failed, print out the current state of the integration.

    if (! state.success) {
#ifndef AMREX_USE_GPU
        std::cout << amrex::Font::Bold << amrex::FGColor::Red << "[ERROR] integration failed in net" << amrex::ResetDisplay << std::endl;
        std::cout << "istate = " << istate << std::endl;
        if (istate == IERR_SUCCESS) {
            std::cout << "  Integration exited successfully, but a check on the data values failed" << std::endl;
        }
        std::cout << "zone = (" << state.i << ", " << state.j << ", " << state.k << ")" << std::endl;
        std::cout << "time = " << int_state.t << std::endl;
        std::cout << "dt = " << std::setprecision(OUTDIGITS) << dt << std::endl;
        std::cout << "temp start = " << std::setprecision(OUTDIGITS) << state_save.T_in << std::endl;
        std::cout << "xn start = ";
        for (const double X : state_save.xn_in) {
            std::cout << std::scientific << std::setprecision(OUTDIGITS) << X << " ";
        }
#ifdef PHOTOCHEMISTRY
        std::cout << std::endl;
        std::cout << "rn start = ";
        for (const double X : state_save.rn_in) {
            std::cout << std::scientific << std::setprecision(OUTDIGITS) << X << " ";
        }
#endif
        std::cout << std::endl;
        std::cout << "dens current = " << std::setprecision(OUTDIGITS) << state.rho << std::endl;
        std::cout << "temp current = " << std::setprecision(OUTDIGITS) << state.T << std::endl;
        std::cout << "xn current = ";
        for (const double X : state.xn) {
            std::cout << std::scientific << std::setprecision(OUTDIGITS) << X << " ";
        }
#ifdef PHOTOCHEMISTRY
        std::cout << std::endl;
        std::cout << "rn current = ";
        for (const double X : state.rn) {
            std::cout << std::scientific << std::setprecision(OUTDIGITS) << X << " ";
        }
#endif
        std::cout << std::endl;
        std::cout << "energy generated = " << state.e << std::endl;
#elif defined(ALLOW_GPU_PRINTF)
        AMREX_DEVICE_PRINTF("[ERROR] integration failed in net, istate = %d, time = %g, dt = %g, dens = %g, temp_start = %g\n",
                            istate, int_state.t, dt, state.rho, state_save.T_in);
#endif
    }

}

#endif
