#ifndef PHYSICS_INFO_HPP_ // NOLINT
#define PHYSICS_INFO_HPP_


#include "physics_numVars.hpp"

// this struct is specialized by the user application code.
template <typename problem_t> struct Physics_Traits {
  static constexpr bool is_hydro_enabled = false;
  static constexpr bool is_radiation_enabled = false;
  static constexpr bool is_chemistry_enabled = false;

  static constexpr int numPassiveScalars = 0;
};

// this struct stores the indices at which quantities start
template <typename problem_t> struct Physics_Indices {
  // cell-centered quantities
  static const int hydroFirstIndex = 0;
  static const int pscalarFirstIndex = Physics_NumVars<problem_t>::numHydroVars;
  static const int radFirstIndex = pscalarFirstIndex + Physics_Traits<problem_t>::numPassiveScalars;
};

#endif // PHYSICS_INFO_HPP_