#ifndef MICROPHYSICS_INTEGRATOR_COMMON_HPP_
#define MICROPHYSICS_INTEGRATOR_COMMON_HPP_
/// \file integrator_common.hpp
/// \brief The parts every Strang integrator shares. Ported from Microphysics' integration/integrator_type.H,
/// integration/integrator_type_strang.H and integration/integrator_rhs_strang.H.

#include <type_traits>

#include "AMReX_Algorithm.H"
#include "AMReX_Array.H"
#include "AMReX_Extension.H"
#include "AMReX_GpuQualifiers.H"
#include "AMReX_REAL.H"

#include "microphysics/microphysics_traits.hpp"

// still from extern/Microphysics: burn_t, net_ienuc, aion, eos(), the network's actual_rhs and actual_jac, the
// integrator parameters and integrator_neqs
#include "actual_rhs.H"
#include "burn_type.H"
#include "eos.H"
#include "extern_parameters.H"
#include "integrator_data.H"
#include "network.H"

namespace quokka::microphysics
{

using amrex::literals::operator""_rt;

template <typename problem_t, typename BurnT> struct IntegratorLayout {
	static constexpr bool photochemistry = requires { &BurnT::rn; };

	static constexpr auto NumChemRadEqs() -> int
	{
		if constexpr (photochemistry) {
			return static_cast<int>(std::extent_v<decltype(BurnT::rn)>);
		} else {
			return 0;
		}
	}

	static constexpr auto NumThermalBands() -> int
	{
		if constexpr (photochemistry) {
			return static_cast<int>(std::extent_v<decltype(BurnT::re_thermal)>);
		} else {
			return 0;
		}
	}

	static constexpr int num_chem_rad_eqs = NumChemRadEqs();
	static constexpr int num_thermal_bands = NumThermalBands();
	static constexpr int vars_per_group = Microphysics_Traits<problem_t>::integrate_rad_flux ? 2 : 1;
	static constexpr bool q_dust = requires { &BurnT::e_dust_absorbed; };

	static_assert(num_chem_rad_eqs % vars_per_group == 0, "every chemical band must have vars_per_group variables");
	static_assert(integrator_neqs<BurnT>() == NumSpec + 1 + num_chem_rad_eqs + num_thermal_bands + (q_dust ? 1 : 0),
		      "the layout must match Microphysics' integrator_neqs");
};

/// Update the thermodynamics in the burn state from the integrator state. This may involve an EOS call.
template <typename Layout, typename BurnT, typename IntegratorT>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void update_thermodynamics(BurnT &state, const IntegratorT &int_state)
{
	for (int n = 1; n <= NumSpec; ++n) {
		state.xn[n - 1] = int_state.y(n);
	}
	state.e = int_state.y(net_ienuc);
	if constexpr (Layout::photochemistry) {
		for (int n = 1; n <= Layout::num_chem_rad_eqs; ++n) {
			state.rn[n - 1] = int_state.y(net_ienuc + n);
		}
		for (int n = 1; n <= Layout::num_thermal_bands; ++n) {
			state.re_thermal[n - 1] = int_state.y(net_ienuc + Layout::num_chem_rad_eqs + n);
		}
		state.c_hat = int_state.c_hat;
	}

	if (integrator_rp::scale_system) {
		state.e *= state.e_scale;
	}

	// Get T from e (also updates composition quantities).

	if (integrator_rp::call_eos_in_rhs) {
		eos(eos_input_re, state);
	}

	// override T if we are fixing it
	if (state.T_fixed > 0.0_rt) {
		state.T = state.T_fixed;
	}
}

/// Copy the integration data to the burn state.
template <typename Layout, typename IntegratorT, typename BurnT>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void integrator_to_burn(const IntegratorT &int_state, BurnT &state)
{
	for (int n = 1; n <= NumSpec; ++n) {
		state.xn[n - 1] = int_state.y(n);
	}

	state.e = int_state.y(net_ienuc);

	if constexpr (Layout::photochemistry) {
		for (int n = 1; n <= Layout::num_chem_rad_eqs; ++n) {
			state.rn[n - 1] = int_state.y(net_ienuc + n);
		}
		for (int n = 1; n <= Layout::num_thermal_bands; ++n) {
			state.re_thermal[n - 1] = int_state.y(net_ienuc + Layout::num_chem_rad_eqs + n);
		}
		state.c_hat = int_state.c_hat;
		if constexpr (Layout::q_dust) {
			state.e_dust_absorbed = int_state.y(net_ienuc + Layout::num_chem_rad_eqs + Layout::num_thermal_bands + 1);
		}
	}

	if (integrator_rp::scale_system) {
		state.e *= state.e_scale;
	}

	state.time = int_state.t;
}

/// Copy the integration data from the burn state.
template <typename Layout, typename BurnT, typename IntegratorT>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void burn_to_integrator(const BurnT &state, IntegratorT &int_state)
{
	for (int n = 1; n <= NumSpec; ++n) {
		int_state.y(n) = state.xn[n - 1];
	}

	int_state.y(net_ienuc) = state.e;

	if constexpr (Layout::photochemistry) {
		for (int n = 1; n <= Layout::num_chem_rad_eqs; ++n) {
			int_state.y(net_ienuc + n) = state.rn[n - 1];
		}
		for (int n = 1; n <= Layout::num_thermal_bands; ++n) {
			int_state.y(net_ienuc + Layout::num_chem_rad_eqs + n) = state.re_thermal[n - 1];
		}
		int_state.c_hat = state.c_hat;
		if constexpr (Layout::q_dust) {
			int_state.y(net_ienuc + Layout::num_chem_rad_eqs + Layout::num_thermal_bands + 1) = state.e_dust_absorbed;
		}
	}

	if (integrator_rp::scale_system) {
		int_state.y(net_ienuc) /= state.e_scale;
	}
}

/// Keep the integrated variables within their valid ranges before the right-hand side is evaluated.
template <typename Layout, typename IntegratorT> AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void clean_state(IntegratorT &int_state)
{
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
	if constexpr (Layout::photochemistry) {
		for (int n = net_ienuc + 1; n <= net_ienuc + Layout::num_chem_rad_eqs; n += Layout::vars_per_group) {
			int_state.y(n) = amrex::max(int_state.y(n), integrator_rp::SMALL_X_SAFE);
		}
		for (int n = net_ienuc + Layout::num_chem_rad_eqs + 1; n <= net_ienuc + Layout::num_chem_rad_eqs + Layout::num_thermal_bands; ++n) {
			int_state.y(n) = amrex::max(int_state.y(n), integrator_rp::SMALL_X_SAFE);
		}
		if constexpr (Layout::q_dust) {
			int_state.y(net_ienuc + Layout::num_chem_rad_eqs + Layout::num_thermal_bands + 1) =
			    amrex::max(int_state.y(net_ienuc + Layout::num_chem_rad_eqs + Layout::num_thermal_bands + 1), 0.0_rt);
		}
	}

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

/// The right-hand side of the integrated system: cleans the integrator state, syncs the burn state with it and calls
/// the network.
template <typename Layout, typename BurnT, typename IntegratorT, int int_neqs>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void rhs(const amrex::Real time, BurnT &state, IntegratorT &int_state, amrex::Array1D<amrex::Real, 1, int_neqs> &ydot)
{
	// Fix the state as necessary -- this ensures that the mass
	// fractions that enter are valid (and optionally normalized)

	clean_state<Layout>(int_state);

	// Update the thermodynamics as necessary -- this primarily takes
	// the information in int_state (X and e), copies it to the
	// (burn_t) state and calls the EOS to get state.T

	update_thermodynamics<Layout>(state, int_state);

	// Only do the burn if the incoming temperature is within the temperature
	// bounds. Otherwise set the RHS to zero and return.

	if (state.T <= EOSData::mintemp || state.T >= integrator_rp::MAX_TEMP) {

		for (int n = 1; n <= int_neqs; ++n) {
			ydot(n) = 0.0_rt;
		}

		return;
	}

	state.time = time;

	// at this point, the (burn_t) state is synchronized with the integrator
	// and is thermodynamically consistent.

	// Call the specific network routine to get the RHS.

	actual_rhs(state, ydot);

	// We integrate X, not Y
	// turn it off for primordial chem
	if (!integrator_rp::use_number_densities) {
		for (int n = 1; n <= NumSpec; ++n) {
			ydot(n) *= aion[n - 1];
		}
	}

	// scale the energy
	if (integrator_rp::scale_system) {
		ydot(net_ienuc) /= state.e_scale;
	}

	// Allow energy integration to be disabled.

	if (!integrator_rp::integrate_energy) {
		ydot(net_ienuc) = 0.0_rt;
	}

	// apply fudge factor:

	if (integrator_rp::react_boost > 0.0_rt) {
		for (int n = 1; n <= int_neqs; ++n) {
			ydot(n) *= integrator_rp::react_boost;
		}
	}
}

/// The analytic Jacobian of the integrated system. The matrix is zeroed before the network fills it, so a network only
/// has to write its nonzero entries.
template <typename Layout, typename BurnT, typename IntegratorT, typename MatrixType>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void jac(BurnT &state, IntegratorT &int_state, MatrixType &pd)
{
	constexpr int int_neqs = integrator_neqs<BurnT>();

	// Only do the burn if the incoming temperature is within the temperature
	// bounds. Otherwise set the Jacobian to zero and return.

	if (state.T <= EOSData::mintemp || state.T >= integrator_rp::MAX_TEMP) {
		pd.zero();
		return;
	}

	// Call the specific network routine to get the Jacobian.

	integrator_to_burn<Layout>(int_state, state);

	pd.zero();

	actual_jac(state, pd);

	// We integrate X, not Y
	// turn it off for primordial chem
	if (!integrator_rp::use_number_densities) {
		for (int j = 1; j <= NumSpec; ++j) {
			for (int i = 1; i <= int_neqs; ++i) {
				pd.mul(j, i, aion[j - 1]);
				pd.mul(i, j, aion_inv[j - 1]);
			}
		}
	}

	// scale the energy derivatives

	if (integrator_rp::scale_system) {
		// first the row de/dX
		for (int j = 1; j <= int_neqs; ++j) {
			pd(net_ienuc, j) /= state.e_scale;
		}

		// now the column dX/de
		for (int i = 1; i <= int_neqs; ++i) {
			pd(i, net_ienuc) *= state.e_scale;
		}
	}

	// apply fudge factor:
	if (integrator_rp::react_boost > 0.0_rt) {
		pd.mul(integrator_rp::react_boost);
	}

	// Allow temperature and energy integration to be disabled.

	if (!integrator_rp::integrate_energy) {
		for (int j = 1; j <= int_neqs; ++j) {
			pd(net_ienuc, j) = 0.0_rt;
		}
	}
}

} // namespace quokka::microphysics

#endif // MICROPHYSICS_INTEGRATOR_COMMON_HPP_
