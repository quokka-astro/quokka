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
#include "AMReX_Arena.H"
#include "AMReX_Array4.H"
#include "AMReX_BLassert.H"
#include "AMReX_FArrayBox.H"
#include "AMReX_Loop.H"
#include "AMReX_REAL.H"
#include "AMReX_TagBox.H"

// internal headers
#include "ArrayView.hpp"
#include "simulation.hpp"
#include "valarray.hpp"
#include "simulation.hpp"

/// Class for a MHD system of conservation laws
template <typename problem_t> class MHDSystem {
public:
  static constexpr int nvar_ = Physics_NumVars::numMHDVars; // number of quantities per face-centering

  enum varIndex {
    energy_index = Physics_Indices<problem_t>::mhdFirstIndex,
  };
};

#endif // HYDRO_SYSTEM_HPP_
