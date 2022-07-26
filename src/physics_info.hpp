#include "physics_numVars.hpp"

// this struct is specialized by the user application code.
template <typename problem_t> struct Physics_Traits {
  static constexpr bool is_hydro_enabled = false;
  static constexpr bool is_radiation_enabled = false;
  static constexpr bool is_pscalars_enabled = false;
  static constexpr bool is_mhd_enabled = false;

  static constexpr int numPassiveScalars = 0;
};

// this struct stores the indices at which quantities start
template <typename problem_t> struct Physics_Indices {
  // cell-centered quantities
  static const int hydroFirstIndex = 0;
  static const int radFirstIndex = 0;
  static const int pscalarFirstIndex =
      ((int)Physics_Traits<problem_t>::is_hydro_enabled * Physics_NumVars<problem_t>::numHydroVars) +
      ((int)Physics_Traits<problem_t>::is_radiation_enabled * Physics_NumVars<problem_t>::numRadVars);

  // face-centered quantities
  static const int mhdCompStarts = 0;
};
