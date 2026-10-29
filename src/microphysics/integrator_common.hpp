#ifndef MICROPHYSICS_INTEGRATOR_COMMON_HPP_
#define MICROPHYSICS_INTEGRATOR_COMMON_HPP_
/// \file integrator_common.hpp
/// \brief The parts every Strang integrator shares. Ported from Microphysics' integration/integrator_type.H,
/// integration/integrator_type_strang.H, integration/integrator_rhs_strang.H and integration/integrator_setup_strang.H.

#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>

#include "AMReX_ANSIEscCode.H"
#include "AMReX_Algorithm.H"
#include "AMReX_Array.H"
#include "AMReX_Extension.H"
#include "AMReX_GpuQualifiers.H"
#include "AMReX_REAL.H"

#include "microphysics/integrator_errors.hpp"
#include "microphysics/integrator_layout.hpp"
#include "microphysics/integrator_params.hpp"
#include "microphysics/microphysics_traits.hpp"

// eos() is still Microphysics'. actual_rhs and actual_jac come from the problem's network.
#include "actual_rhs.H"
#include "eos.H"

namespace quokka::microphysics
{

using amrex::literals::operator""_rt;

/// Update the thermodynamics in the burn state from the integrator state. This may involve an EOS call.
template <typename Layout, typename BurnT, typename IntegratorT>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void update_thermodynamics(BurnT &state, const IntegratorT &int_state)
{
	for (int n = 1; n <= Layout::num_spec; ++n) {
		state.xn[n - 1] = int_state.y(n);
	}
	state.e = int_state.y(Layout::energy_row);
	if constexpr (Layout::photochemistry) {
		for (int n = 1; n <= Layout::num_chem_rad_eqs; ++n) {
			state.rn[n - 1] = int_state.y(Layout::energy_row + n);
		}
		for (int n = 1; n <= Layout::num_thermal_bands; ++n) {
			state.re_thermal[n - 1] = int_state.y(Layout::energy_row + Layout::num_chem_rad_eqs + n);
		}
		state.c_hat = int_state.c_hat;
	}

	// Get T from e (also updates composition quantities).

	if constexpr (Layout::traits::call_eos_in_rhs) {
		eos(eos_input_re, state);
	}
}

/// Copy the integration data to the burn state.
template <typename Layout, typename IntegratorT, typename BurnT>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void integrator_to_burn(const IntegratorT &int_state, BurnT &state)
{
	for (int n = 1; n <= Layout::num_spec; ++n) {
		state.xn[n - 1] = int_state.y(n);
	}

	state.e = int_state.y(Layout::energy_row);

	if constexpr (Layout::photochemistry) {
		for (int n = 1; n <= Layout::num_chem_rad_eqs; ++n) {
			state.rn[n - 1] = int_state.y(Layout::energy_row + n);
		}
		for (int n = 1; n <= Layout::num_thermal_bands; ++n) {
			state.re_thermal[n - 1] = int_state.y(Layout::energy_row + Layout::num_chem_rad_eqs + n);
		}
		state.c_hat = int_state.c_hat;
		if constexpr (Layout::q_dust) {
			state.e_dust_absorbed = int_state.y(Layout::energy_row + Layout::num_chem_rad_eqs + Layout::num_thermal_bands + 1);
		}
	}

	state.time = int_state.t;
}

/// Copy the integration data from the burn state.
template <typename Layout, typename BurnT, typename IntegratorT>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void burn_to_integrator(const BurnT &state, IntegratorT &int_state)
{
	for (int n = 1; n <= Layout::num_spec; ++n) {
		int_state.y(n) = state.xn[n - 1];
	}

	int_state.y(Layout::energy_row) = state.e;

	if constexpr (Layout::photochemistry) {
		for (int n = 1; n <= Layout::num_chem_rad_eqs; ++n) {
			int_state.y(Layout::energy_row + n) = state.rn[n - 1];
		}
		for (int n = 1; n <= Layout::num_thermal_bands; ++n) {
			int_state.y(Layout::energy_row + Layout::num_chem_rad_eqs + n) = state.re_thermal[n - 1];
		}
		int_state.c_hat = state.c_hat;
		if constexpr (Layout::q_dust) {
			int_state.y(Layout::energy_row + Layout::num_chem_rad_eqs + Layout::num_thermal_bands + 1) = state.e_dust_absorbed;
		}
	}
}

/// Keep the integrated variables within their valid ranges before the right-hand side is evaluated.
template <typename Layout, typename IntegratorT> AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void clean_state(IntegratorT &int_state)
{
	for (int n = 1; n <= Layout::neqs; ++n) {
		switch (Layout::role(n)) {
			case VarRole::Species:
				// Ensure that mass fractions always stay positive and less than or
				// equal to 1.
				if constexpr (Layout::traits::do_species_clip) {
					int_state.y(n) = amrex::Clamp(int_state.y(n), integrator_params::SMALL_X_SAFE, 1.0_rt);
				}
				// enforce positivity of the number densities
				int_state.y(n) = amrex::max(int_state.y(n), integrator_params::SMALL_X_SAFE);
				break;
			case VarRole::ChemPhotonNumber:
			case VarRole::ThermalBandEnergy:
				int_state.y(n) = amrex::max(int_state.y(n), integrator_params::SMALL_X_SAFE);
				break;
			case VarRole::DustAbsorbedEnergy:
				int_state.y(n) = amrex::max(int_state.y(n), 0.0_rt);
				break;
			case VarRole::Energy:
			case VarRole::ChemPhotonFlux:
				break;
		}
	}

	// Renormalize the abundances as necessary.

	if constexpr (Layout::traits::renormalize_abundances) {
		amrex::Real sum = 0.0_rt;

		for (int n = 1; n <= Layout::num_spec; ++n) {
			sum += int_state.y(n);
		}

		for (int n = 1; n <= Layout::num_spec; ++n) {
			int_state.y(n) /= sum;
		}
	}
}

/// The right-hand side of the integrated system: cleans the integrator state, syncs the burn state with it and calls
/// the network.
template <typename Layout, typename BurnT, typename IntegratorT>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void rhs(const amrex::Real time, BurnT &state, IntegratorT &int_state,
						  amrex::Array1D<amrex::Real, 1, Layout::neqs> &ydot)
{
	constexpr int int_neqs = Layout::neqs;

	// Fix the state as necessary -- this ensures that the mass
	// fractions that enter are valid (and optionally normalized)

	clean_state<Layout>(int_state);

	// Update the thermodynamics as necessary -- this primarily takes
	// the information in int_state (X and e), copies it to the
	// burn state and calls the EOS to get state.T

	update_thermodynamics<Layout>(state, int_state);

	// Only do the burn if the incoming temperature is within the temperature
	// bounds. Otherwise set the RHS to zero and return.

	if (state.T <= EOSData::mintemp || state.T >= Layout::traits::max_temp) {

		for (int n = 1; n <= int_neqs; ++n) {
			ydot(n) = 0.0_rt;
		}

		return;
	}

	state.time = time;

	// at this point, the burn state is synchronized with the integrator
	// and is thermodynamically consistent.

	// Call the specific network routine to get the RHS.

	actual_rhs(state, ydot);

	// Allow energy integration to be disabled.

	if constexpr (!Layout::traits::integrate_energy) {
		ydot(Layout::energy_row) = 0.0_rt;
	}
}

/// The analytic Jacobian of the integrated system. The matrix is zeroed before the network fills it, so a network only
/// has to write its nonzero entries.
template <typename Layout, typename BurnT, typename IntegratorT, typename MatrixType>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void jac(BurnT &state, IntegratorT &int_state, MatrixType &pd)
{
	constexpr int int_neqs = Layout::neqs;

	// Only do the burn if the incoming temperature is within the temperature
	// bounds. Otherwise set the Jacobian to zero and return.

	if (state.T <= EOSData::mintemp || state.T >= Layout::traits::max_temp) {
		pd.zero();
		return;
	}

	// Call the specific network routine to get the Jacobian.

	integrator_to_burn<Layout>(int_state, state);

	pd.zero();

	actual_jac(state, pd);

	// Allow temperature and energy integration to be disabled.

	if constexpr (!Layout::traits::integrate_energy) {
		for (int j = 1; j <= int_neqs; ++j) {
			pd(Layout::energy_row, j) = 0.0_rt;
		}
	}
}

/// for outputting the burn state failure, use enough precision to be reproducible (not guaranteed for std::fixed)
constexpr int OUTDIGITS = std::numeric_limits<amrex::Real>::max_digits10;

/// The initial composition and thermodynamic state, saved for diagnostics.
template <typename Layout> struct StateBackup {
	amrex::Real T_in{};
#ifndef AMREX_USE_GPU
	amrex::GpuArray<amrex::Real, Layout::num_spec> xn_in{};
	amrex::GpuArray<amrex::Real, Layout::num_chem_rad_eqs> rn_in{};
	amrex::GpuArray<amrex::Real, Layout::num_thermal_bands> re_thermal_in{};
	amrex::Real c_hat{};
	amrex::Real e_dust_absorbed_in{};
#endif
};

/// Set up the integrator state from the burn state.
template <typename Layout, typename IntegratorT, typename BurnT>
AMREX_GPU_HOST_DEVICE AMREX_INLINE auto integrator_setup(BurnT &state, const amrex::Real dt) -> IntegratorT
{
	IntegratorT int_state{};

	// Set the tolerances.

	int_state.atol_spec = integrator_params::atol_spec; // mass fractions
	int_state.atol_enuc = integrator_params::atol_enuc; // energy generated

	int_state.rtol_spec = integrator_params::rtol_spec; // mass fractions
	int_state.rtol_enuc = integrator_params::rtol_enuc; // energy generated
	if constexpr (Layout::photochemistry) {
		int_state.atol_rad_num = integrator_params::atol_rad_num; // radiation number density
		int_state.rtol_rad_num = integrator_params::rtol_rad_num; // radiation number density
	}

	// Start off by assuming a successful burn.

	state.success = true;

	// Initialize the integration time.

	int_state.t = 0.0_rt;
	int_state.tout = dt;

	// We assume that (rho, T) coming in are valid, do an EOS call
	// to fill the rest of the thermodynamic variables.

	eos(eos_input_rt, state);

	if constexpr (Layout::q_dust) {
		state.e_dust_absorbed = 0.0_rt;
	}

	// Fill in the initial integration state.

	burn_to_integrator<Layout>(state, int_state);

	return int_state;
}

/// Save the initial composition and thermodynamic state for diagnostics.
template <typename Layout, typename BurnT> AMREX_GPU_HOST_DEVICE AMREX_INLINE auto integrator_backup(const BurnT &state) -> StateBackup<Layout>
{
	StateBackup<Layout> state_save;

#ifndef AMREX_USE_GPU
	for (int n = 0; n < Layout::num_spec; ++n) {
		state_save.xn_in[n] = state.xn[n];
	}
	if constexpr (Layout::photochemistry) {
		for (int n = 0; n < Layout::num_chem_rad_eqs; ++n) {
			state_save.rn_in[n] = state.rn[n];
		}
		for (int n = 0; n < Layout::num_thermal_bands; ++n) {
			state_save.re_thermal_in[n] = state.re_thermal[n];
		}
		state_save.c_hat = state.c_hat;
		if constexpr (Layout::q_dust) {
			state_save.e_dust_absorbed_in = state.e_dust_absorbed;
		}
	}
#endif
	state_save.T_in = state.T;

	return state_save;
}

/// Copy the integrator state back to the burn state and check the result.
template <typename Layout, typename IntegratorT, typename BurnT>
AMREX_GPU_HOST_DEVICE AMREX_INLINE void integrator_cleanup(IntegratorT &int_state, BurnT &state, const int istate, const StateBackup<Layout> &state_save,
							   const amrex::Real dt)
{
	// Copy the integration data back to the burn state.

	integrator_to_burn<Layout>(int_state, state);

	// Get the number of RHS and Jacobian evaluations.

	state.n_rhs = int_state.n_rhs;
	state.n_jac = int_state.n_jac;
	state.n_step = int_state.n_step;

	state.error_code = static_cast<std::int16_t>(istate);

	// The integrator may not always fail even though it can lead to
	// unphysical states.  Add some checks that indicate a burn fail
	// even if the integrator thinks the integration was successful.

	if (istate != IERR_SUCCESS) {
		state.success = false;
	}

	for (int n = 1; n <= Layout::neqs; ++n) {
		switch (Layout::role(n)) {
			case VarRole::Species:
				if (int_state.y(n) < -integrator_params::species_failure_tolerance) {
					state.success = false;
				}
				break;
			case VarRole::ChemPhotonNumber:
				if (int_state.y(n) < -integrator_params::radiation_failure_tolerance) {
					state.success = false;
				}
				break;
			case VarRole::Energy:
			case VarRole::ChemPhotonFlux:
			case VarRole::ThermalBandEnergy:
			case VarRole::DustAbsorbedEnergy:
				break;
		}
	}

#ifndef AMREX_USE_GPU
	if (integrator_params::burner_verbose) {
		// Print out some integration statistics, if desired.
		std::cout << "integration summary: " << '\n';
		std::cout << "dens: " << state.rho << " temp: " << state.T << '\n';
		std::cout << " energy released: " << state.e << '\n';
		std::cout << "number of steps taken: " << state.n_step << '\n';
		std::cout << "number of f evaluations: " << state.n_rhs << '\n' << std::flush;
	}

	// If we failed, print out the current state of the integration.

	if (!state.success) {
		// NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-array-to-pointer-decay,hicpp-no-array-decay): AMReX's escape codes are char arrays
		std::cout << amrex::Font::Bold << amrex::FGColor::Red << "[ERROR] integration failed in net" << amrex::ResetDisplay << '\n';
		std::cout << "istate = " << istate << '\n';
		if (istate == IERR_SUCCESS) {
			std::cout << "  Integration exited successfully, but a check on the data values failed" << '\n';
		}
		std::cout << "zone = (" << state.i << ", " << state.j << ", " << state.k << ")" << '\n';
		std::cout << "time = " << int_state.t << '\n';
		std::cout << "dt = " << std::setprecision(OUTDIGITS) << dt << '\n';
		std::cout << "temp start = " << std::setprecision(OUTDIGITS) << state_save.T_in << '\n';
		std::cout << "xn start = ";
		for (int n = 0; n < Layout::num_spec; ++n) {
			std::cout << std::scientific << std::setprecision(OUTDIGITS) << state_save.xn_in[n] << " ";
		}
		if constexpr (Layout::photochemistry) {
			std::cout << '\n';
			std::cout << "rn start = ";
			for (int n = 0; n < Layout::num_chem_rad_eqs; ++n) {
				std::cout << std::scientific << std::setprecision(OUTDIGITS) << state_save.rn_in[n] << " ";
			}
			std::cout << '\n';
			std::cout << "re_thermal start = ";
			for (int n = 0; n < Layout::num_thermal_bands; ++n) {
				std::cout << std::scientific << std::setprecision(OUTDIGITS) << state_save.re_thermal_in[n] << " ";
			}
			if constexpr (Layout::q_dust) {
				std::cout << '\n';
				std::cout << "e_dust_absorbed start = " << std::scientific << std::setprecision(OUTDIGITS) << state_save.e_dust_absorbed_in;
			}
		}
		std::cout << '\n';
		std::cout << "dens current = " << std::setprecision(OUTDIGITS) << state.rho << '\n';
		std::cout << "temp current = " << std::setprecision(OUTDIGITS) << state.T << '\n';
		std::cout << "xn current = ";
		for (const double X : state.xn) {
			std::cout << std::scientific << std::setprecision(OUTDIGITS) << X << " ";
		}
		if constexpr (Layout::photochemistry) {
			std::cout << '\n';
			std::cout << "rn current = ";
			for (const double X : state.rn) {
				std::cout << std::scientific << std::setprecision(OUTDIGITS) << X << " ";
			}
			std::cout << '\n';
			std::cout << "re_thermal current = ";
			for (const double X : state.re_thermal) {
				std::cout << std::scientific << std::setprecision(OUTDIGITS) << X << " ";
			}
			if constexpr (Layout::q_dust) {
				std::cout << '\n';
				std::cout << "e_dust_absorbed current = " << std::scientific << std::setprecision(OUTDIGITS) << state.e_dust_absorbed;
			}
		}
		std::cout << '\n';
		std::cout << "energy generated = " << state.e << '\n' << std::flush;
	}
#endif
}

} // namespace quokka::microphysics

#endif // MICROPHYSICS_INTEGRATOR_COMMON_HPP_
