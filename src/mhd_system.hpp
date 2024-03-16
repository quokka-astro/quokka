#ifndef MHD_SYSTEM_HPP_ // NOLINT
#define MHD_SYSTEM_HPP_
//==============================================================================
// ...
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file mhd_system.hpp
/// \brief Defines a class for solving the MHD equations.
///

// c++ headers

// library headers

// internal headers
#include "physics_info.hpp"
#include "physics_numVars.hpp"

/// Class for a MHD system of conservation laws
template <typename problem_t> class MHDSystem
{
      public:
	static constexpr int nvar_per_dim_ = Physics_NumVars::numMHDVars_per_dim;
	static constexpr int nvar_tot_ = Physics_NumVars::numMHDVars_tot;

	enum varIndex_perDim {
		bfield_index = Physics_Indices<problem_t>::mhdFirstIndex,
	};
};

#endif // HYDRO_SYSTEM_HPP_
