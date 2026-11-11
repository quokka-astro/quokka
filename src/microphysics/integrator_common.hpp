// SPDX-FileCopyrightText: © The Quokka Authors
// SPDX-License-Identifier: MIT

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

#include "hydro/EOS.hpp"
#include "hydro/EOSFloors.hpp"
#include "microphysics/integrator_errors.hpp"
#include "microphysics/integrator_params.hpp"
#include "microphysics/microphysics_traits.hpp"
#include "microphysics/ode_system.hpp"

// The burn state's EOS is the problem's (quokka::EOS, the one its network names). actual_rhs and actual_jac are members
// of the problem's network type, which the problem file includes.

namespace quokka::microphysics
{

using amrex::literals::operator""_rt;

/// Update the thermodynamics in the burn state from the integrator state. This may involve an EOS call.
template <typename System, typename BurnT, typename IntegratorT>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void update_thermodynamics(BurnT &state, const IntegratorT &int_state)
{
	for (int n = 0; n < System::num_spec; ++n) {
		state.xn[n] = int_state.y[n];
	}
	state.e = int_state.y[System::energy_row];
	if constexpr (System::photochemistry) {
		for (int n = 0; n < System::num_chem_rad_eqs; ++n) {
			state.rn[n] = int_state.y[System::energy_row + 1 + n];
		}
		for (int n = 0; n < System::num_thermal_bands; ++n) {
			state.re_thermal[n] = int_state.y[System::thermal_band_row + n];
		}
		state.c_hat = int_state.c_hat;
	}

	// Get T from e.

	if constexpr (System::traits::call_eos_in_rhs) {
		using GasEOS = quokka::EOS<typename BurnT::problem_type>;
		GasEOS::Solve(GasEOS::Input::RhoE, state);
	}
}

/// Copy the integration data to the burn state.
template <typename System, typename IntegratorT, typename BurnT>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void integrator_to_burn(const IntegratorT &int_state, BurnT &state)
{
	for (int n = 0; n < System::num_spec; ++n) {
		state.xn[n] = int_state.y[n];
	}

	state.e = int_state.y[System::energy_row];

	if constexpr (System::photochemistry) {
		for (int n = 0; n < System::num_chem_rad_eqs; ++n) {
			state.rn[n] = int_state.y[System::energy_row + 1 + n];
		}
		for (int n = 0; n < System::num_thermal_bands; ++n) {
			state.re_thermal[n] = int_state.y[System::thermal_band_row + n];
		}
		state.c_hat = int_state.c_hat;
		if constexpr (System::q_dust) {
			state.e_dust_absorbed = int_state.y[System::dust_row];
		}
	}

	state.time = int_state.t;
}

/// Copy the integration data from the burn state.
template <typename System, typename BurnT, typename IntegratorT>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void burn_to_integrator(const BurnT &state, IntegratorT &int_state)
{
	for (int n = 0; n < System::num_spec; ++n) {
		int_state.y[n] = state.xn[n];
	}

	int_state.y[System::energy_row] = state.e;

	if constexpr (System::photochemistry) {
		for (int n = 0; n < System::num_chem_rad_eqs; ++n) {
			int_state.y[System::energy_row + 1 + n] = state.rn[n];
		}
		for (int n = 0; n < System::num_thermal_bands; ++n) {
			int_state.y[System::thermal_band_row + n] = state.re_thermal[n];
		}
		int_state.c_hat = state.c_hat;
		if constexpr (System::q_dust) {
			int_state.y[System::dust_row] = state.e_dust_absorbed;
		}
	}
}

/// Keep the integrated variables within their valid ranges before the right-hand side is evaluated.
template <typename System, typename IntegratorT>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void clean_state(IntegratorT &int_state, const IntegratorParams &params)
{
	for (int n = 0; n < System::neqs; ++n) {
		switch (System::role(n)) {
			case VarRole::Species:
				// do_species_clip also caps each species at 1, which only makes sense for mass fractions
				if constexpr (System::traits::do_species_clip) {
					int_state.y[n] = amrex::Clamp(int_state.y[n], params.SMALL_X_SAFE, 1.0_rt);
				}
				// enforce positivity of the number densities
				int_state.y[n] = amrex::max(int_state.y[n], params.SMALL_X_SAFE);
				break;
			case VarRole::ChemPhotonNumber:
			case VarRole::ThermalBandEnergy:
				int_state.y[n] = amrex::max(int_state.y[n], params.SMALL_X_SAFE);
				break;
			case VarRole::DustAbsorbedEnergy:
				int_state.y[n] = amrex::max(int_state.y[n], 0.0_rt);
				break;
			case VarRole::Energy:
			case VarRole::ChemPhotonFlux:
				break;
		}
	}

	// Renormalize the abundances as necessary.

	if constexpr (System::traits::renormalize_abundances) {
		amrex::Real sum = 0.0_rt;

		for (int n = 0; n < System::num_spec; ++n) {
			sum += int_state.y[n];
		}

		for (int n = 0; n < System::num_spec; ++n) {
			int_state.y[n] /= sum;
		}
	}
}

/// The right-hand side of the integrated system: cleans the integrator state, syncs the burn state with it and calls
/// the network.
template <typename System, typename BurnT, typename IntegratorT>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void rhs(const amrex::Real time, BurnT &state, IntegratorT &int_state,
						  amrex::GpuArray<amrex::Real, System::neqs> &ydot, const IntegratorParams &params)
{
	constexpr int int_neqs = System::neqs;

	// Fix the state as necessary -- this ensures that the species
	// that enter are valid (and optionally normalized)

	clean_state<System>(int_state, params);

	// Update the thermodynamics as necessary -- this primarily takes
	// the information in int_state (X and e), copies it to the
	// burn state and calls the EOS to get state.T

	update_thermodynamics<System>(state, int_state);

	// Only do the burn if the incoming temperature is within the temperature
	// bounds. Otherwise set the RHS to zero.

	const bool in_bounds = !(state.T <= quokka::eos_floors::min_temp || state.T >= System::traits::max_temp);

	if (in_bounds) {
		state.time = time;

		// at this point, the burn state is synchronized with the integrator
		// and is thermodynamically consistent.

		// Call the specific network routine to get the RHS.

		System::traits::Network::actual_rhs(state, ydot);

		// Allow energy integration to be disabled.

		if constexpr (!System::traits::integrate_energy) {
			ydot[System::energy_row] = 0.0_rt;
		}
	}

	// The zeroing is a select after the branch, not stores inside an else branch: the compiler would merge such stores
	// with the network's last store into one store with a run-time index, which keeps the whole integrator state, ydot
	// included, in scratch memory on the GPU.
	for (int n = 0; n < int_neqs; ++n) {
		ydot[n] = in_bounds ? ydot[n] : 0.0_rt;
	}
}

/// Whether the network can evaluate its right-hand side and Jacobian together (Network::actual_rhs_and_jac).
template <typename System, typename BurnT, typename MatrixType>
constexpr bool network_has_rhs_and_jac = requires(BurnT & state, amrex::GpuArray<amrex::Real, System::neqs> &ydot, MatrixType &pd)
{
	System::traits::Network::actual_rhs_and_jac(state, ydot, pd);
};

/// The right-hand side and the analytic Jacobian at the same state, as rhs() followed by jac() would give them, from
/// one network call (Network::actual_rhs_and_jac) that evaluates the rate coefficients once.
template <typename System, typename BurnT, typename IntegratorT, typename MatrixType>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void rhs_and_jac(const amrex::Real time, BurnT &state, IntegratorT &int_state,
							  amrex::GpuArray<amrex::Real, System::neqs> &ydot, MatrixType &pd, const IntegratorParams &params)
{
	constexpr int int_neqs = System::neqs;

	clean_state<System>(int_state, params);
	update_thermodynamics<System>(state, int_state);

	const bool in_bounds = !(state.T <= quokka::eos_floors::min_temp || state.T >= System::traits::max_temp);

	pd = MatrixType{};
	if (in_bounds) {
		state.time = time;
		System::traits::Network::actual_rhs_and_jac(state, ydot, pd);
		if constexpr (!System::traits::integrate_energy) {
			ydot[System::energy_row] = 0.0_rt;
			for (int j = 0; j < int_neqs; ++j) {
				pd(System::energy_row, j) = 0.0_rt;
			}
		}
	}

	// a select after the branch, as in rhs()
	for (int n = 0; n < int_neqs; ++n) {
		ydot[n] = in_bounds ? ydot[n] : 0.0_rt;
	}
}

/// The analytic Jacobian of the integrated system. The matrix is zeroed before the network fills it, so a network only
/// has to write its nonzero entries.
template <typename System, typename BurnT, typename IntegratorT, typename MatrixType>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void jac(BurnT &state, IntegratorT &int_state, MatrixType &pd)
{
	constexpr int int_neqs = System::neqs;

	// Only do the burn if the incoming temperature is within the temperature
	// bounds. Otherwise set the Jacobian to zero and return.

	if (state.T <= quokka::eos_floors::min_temp || state.T >= System::traits::max_temp) {
		pd = MatrixType{};
		return;
	}

	// Call the specific network routine to get the Jacobian.

	integrator_to_burn<System>(int_state, state);

	pd = MatrixType{};

	System::traits::Network::actual_jac(state, pd);

	// Allow temperature and energy integration to be disabled.

	if constexpr (!System::traits::integrate_energy) {
		for (int j = 0; j < int_neqs; ++j) {
			pd(System::energy_row, j) = 0.0_rt;
		}
	}
}

/// for outputting the burn state failure, use enough precision to be reproducible (not guaranteed for std::fixed)
constexpr int OUTDIGITS = std::numeric_limits<amrex::Real>::max_digits10;

/// The initial composition and thermodynamic state, saved for diagnostics.
template <typename System> struct StateBackup {
	amrex::Real T_in{};
#ifndef AMREX_USE_GPU
	amrex::GpuArray<amrex::Real, System::num_spec> xn_in{};
	amrex::GpuArray<amrex::Real, System::num_chem_rad_eqs> rn_in{};
	amrex::GpuArray<amrex::Real, System::num_thermal_bands> re_thermal_in{};
	amrex::Real c_hat{};
	amrex::Real e_dust_absorbed_in{};
#endif
};

/// Set up the integrator state from the burn state.
template <typename System, typename IntegratorT, typename BurnT>
AMREX_GPU_HOST_DEVICE AMREX_INLINE auto integrator_setup(BurnT &state, const amrex::Real dt, const IntegratorParams &params) -> IntegratorT
{
	IntegratorT int_state{};

	// Set the tolerances.

	int_state.atol_spec = params.atol_spec; // species number densities
	int_state.atol_enuc = params.atol_enuc; // specific internal energy

	int_state.rtol_spec = params.rtol_spec; // species number densities
	int_state.rtol_enuc = params.rtol_enuc; // specific internal energy
	if constexpr (System::photochemistry) {
		int_state.atol_rad_num = params.atol_rad_num; // radiation number density
		int_state.rtol_rad_num = params.rtol_rad_num; // radiation number density
	}

	// Start off by assuming a successful burn.

	state.success = true;

	// Initialize the integration time.

	int_state.t = 0.0_rt;
	int_state.tout = dt;

	// The burn state comes in completed by the EOS (its caller solved it from rho and e), so no EOS call is needed here:
	// solving it again from (rho, T) would only recompute e from T.

	if constexpr (System::q_dust) {
		state.e_dust_absorbed = 0.0_rt;
	}

	// Fill in the initial integration state.

	burn_to_integrator<System>(state, int_state);

	return int_state;
}

/// Save the initial composition and thermodynamic state for diagnostics.
template <typename System, typename BurnT> AMREX_GPU_HOST_DEVICE AMREX_INLINE auto integrator_backup(const BurnT &state) -> StateBackup<System>
{
	StateBackup<System> state_save;

#ifndef AMREX_USE_GPU
	for (int n = 0; n < System::num_spec; ++n) {
		state_save.xn_in[n] = state.xn[n];
	}
	if constexpr (System::photochemistry) {
		for (int n = 0; n < System::num_chem_rad_eqs; ++n) {
			state_save.rn_in[n] = state.rn[n];
		}
		for (int n = 0; n < System::num_thermal_bands; ++n) {
			state_save.re_thermal_in[n] = state.re_thermal[n];
		}
		state_save.c_hat = state.c_hat;
		if constexpr (System::q_dust) {
			state_save.e_dust_absorbed_in = state.e_dust_absorbed;
		}
	}
#endif
	state_save.T_in = state.T;

	return state_save;
}

/// Copy the integrator state back to the burn state and check the result.
template <typename System, typename IntegratorT, typename BurnT>
AMREX_GPU_HOST_DEVICE AMREX_INLINE void integrator_cleanup(IntegratorT &int_state, BurnT &state, const int istate, const StateBackup<System> &state_save,
							   const amrex::Real dt, const IntegratorParams &params)
{
	// Copy the integration data back to the burn state.

	integrator_to_burn<System>(int_state, state);

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

	for (int n = 0; n < System::neqs; ++n) {
		switch (System::role(n)) {
			case VarRole::Species:
				if (int_state.y[n] < -params.species_failure_tolerance) {
					state.success = false;
				}
				break;
			case VarRole::ChemPhotonNumber:
				if (int_state.y[n] < -params.radiation_failure_tolerance) {
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
	if (params.burner_verbose) {
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
		for (int n = 0; n < System::num_spec; ++n) {
			std::cout << std::scientific << std::setprecision(OUTDIGITS) << state_save.xn_in[n] << " ";
		}
		if constexpr (System::photochemistry) {
			std::cout << '\n';
			std::cout << "rn start = ";
			for (int n = 0; n < System::num_chem_rad_eqs; ++n) {
				std::cout << std::scientific << std::setprecision(OUTDIGITS) << state_save.rn_in[n] << " ";
			}
			std::cout << '\n';
			std::cout << "re_thermal start = ";
			for (int n = 0; n < System::num_thermal_bands; ++n) {
				std::cout << std::scientific << std::setprecision(OUTDIGITS) << state_save.re_thermal_in[n] << " ";
			}
			if constexpr (System::q_dust) {
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
		if constexpr (System::photochemistry) {
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
			if constexpr (System::q_dust) {
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
