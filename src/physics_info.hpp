#ifndef PHYSICS_INFO_HPP_ // NOLINT
#define PHYSICS_INFO_HPP_

#include "AMReX_REAL.H"
#include "fundamental_constants.H"
#include "physics_numVars.hpp"
#include <AMReX.H>

using Real = amrex::Real;

// enum for unit system, one of CGS, CONSTANTS, CUSTOM
enum class UnitSystem { CGS, CONSTANTS, CUSTOM };

// this struct is specialized by the user application code.
template <typename problem_t> struct Physics_Traits {
	static constexpr bool is_hydro_enabled = false;
	static constexpr int numMassScalars = 0;
	static constexpr int numPassiveScalars = numMassScalars + 0;
	static constexpr bool is_radiation_enabled = false;
	static constexpr bool is_dust_enabled = false;
	static constexpr bool is_self_gravity_enabled = false;
	static constexpr bool is_mhd_enabled = false;
	static constexpr int nGroups = 1; // number of radiation groups
	static constexpr int nDustGroups = 1; // number of dust groups
	static constexpr UnitSystem unit_system = UnitSystem::CGS;
	static constexpr double boltzmann_constant = C::k_B;	    // Hydro, EOS
	static constexpr double gravitational_constant = C::Gconst; // gravity
	static constexpr double c_light = C::c_light;		    // radiation
	static constexpr double radiation_constant = C::a_rad;	    // radiation
	static constexpr double unit_length = 1.0;
	static constexpr double unit_mass = 1.0;
	static constexpr double unit_time = 1.0;
	static constexpr double unit_temperature = 1.0;
};

// this struct stores the indices at which quantities start
template <typename problem_t> struct Physics_Indices {
	// number of cc quantities required for advection problems
	static constexpr int nvarTotal_cc_adv = 1;
	// number of cc quantities required for rad /+ hydro problem
	static constexpr int nvarTotal_cc = []() constexpr {
		if constexpr (!(Physics_Traits<problem_t>::is_hydro_enabled || Physics_Traits<problem_t>::is_radiation_enabled)) {
			return nvarTotal_cc_adv;
		}
		return Physics_Traits<problem_t>::numPassiveScalars + Physics_NumVars::numHydroVars + Physics_NumVars::numDustVarsPerGroup *
			Physics_Traits<problem_t>::nDustGroups * static_cast<int>(Physics_Traits<problem_t>::is_dust_enabled) +
			Physics_NumVars::numRadVarsPerGroup * Physics_Traits<problem_t>::nGroups *
			static_cast<int>(Physics_Traits<problem_t>::is_radiation_enabled);
	}();
	// cell-centered
	static constexpr int hydroFirstIndex = 0;
	static constexpr int pscalarFirstIndex = Physics_NumVars::numHydroVars;
	static constexpr int dustFirstIndex = pscalarFirstIndex + Physics_Traits<problem_t>::numPassiveScalars;
	static constexpr int radFirstIndex = dustFirstIndex + Physics_NumVars::numDustVarsPerGroup * Physics_Traits<problem_t>::nDustGroups * static_cast<int>(Physics_Traits<problem_t>::is_dust_enabled);
	// face-centered
	static constexpr int nvarPerDim_fc = Physics_NumVars::numMHDVars_per_dim * static_cast<int>(Physics_Traits<problem_t>::is_mhd_enabled);
	static constexpr int nvarTotal_fc = AMREX_SPACEDIM * nvarPerDim_fc;
	static constexpr int mhdFirstIndex = 0;
};

#endif // PHYSICS_INFO_HPP_
