#ifndef PHYSICS_INFO_HPP_ // NOLINT
#define PHYSICS_INFO_HPP_

#include "physics_numVars.hpp"
#include <AMReX.H>

// this struct is specialized by the user application code.
template <typename problem_t> struct Physics_Traits {
	// cell-centred
	static constexpr bool is_hydro_enabled = false;
	static constexpr int numMassScalars = 0;
	static constexpr int numPassiveScalars = numMassScalars + 0;
	static constexpr bool is_radiation_enabled = false;
	// face-centred
	static constexpr bool is_mhd_enabled = false;
	static constexpr int nGroups = 1; // number of radiation groups
};

// this struct stores the indices at which quantities start
template <typename problem_t> struct Physics_Indices {
	// number of cc quantities required for advection problems
	static const int nvarTotal_cc_adv = 1;
	// number of cc quantities required for rad /+ hydro problem
	static constexpr int nvarTotal_cc_radhydro = []() constexpr {
		if constexpr (Physics_Traits<problem_t>::is_radiation_enabled) {
			return Physics_Traits<problem_t>::numPassiveScalars +
			       Physics_NumVars::numHydroVars *
				   static_cast<int>(Physics_Traits<problem_t>::is_hydro_enabled || Physics_Traits<problem_t>::is_radiation_enabled) +
			       Physics_NumVars::numRadVars * Physics_Traits<problem_t>::nGroups;
		} else {
			return Physics_Traits<problem_t>::numPassiveScalars +
			       Physics_NumVars::numHydroVars *
				   static_cast<int>(Physics_Traits<problem_t>::is_hydro_enabled || Physics_Traits<problem_t>::is_radiation_enabled);
		}
	}();
	// cell-centered
	static const int nvarTotal_cc = nvarTotal_cc_radhydro > 0 ? nvarTotal_cc_radhydro : nvarTotal_cc_adv;
	static const int hydroFirstIndex = 0;
	static const int pscalarFirstIndex = Physics_NumVars::numHydroVars;
	static const int radFirstIndex = pscalarFirstIndex + Physics_Traits<problem_t>::numPassiveScalars;
	// face-centered
	static const int nvarPerDim_fc = Physics_NumVars::numVelVars_per_dim * static_cast<int>(Physics_Traits<problem_t>::is_hydro_enabled) +
					 Physics_NumVars::numMHDVars_per_dim * static_cast<int>(Physics_Traits<problem_t>::is_mhd_enabled);
	static const int nvarTotal_fc = AMREX_SPACEDIM * nvarPerDim_fc;
	static const int velFirstIndex = 0;
	static const int mhdFirstIndex = velFirstIndex + Physics_NumVars::numVelVars_per_dim;
};

#endif // PHYSICS_INFO_HPP_
