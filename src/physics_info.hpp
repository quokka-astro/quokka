#include "physics_numVars.hpp"

// this struct is specialized by the user application code.
template <typename problem_t> struct Physics_Traits {
  static constexpr bool is_hydro_enabled = false;
  static constexpr bool is_radiation_enabled = false;
  static constexpr bool is_mhd_enabled = false;
  static constexpr bool is_primordial_chem_enabled = false;
  static constexpr bool is_metalicity_enabled = false;
};

template <typename problem_t> struct Physics_Indices {
  // cell-centered quantities
  static const int hydroCompStarts = 0;
  static const int radCompStarts = 0;
  static const int primordialChemCompStarts =
      ((int)Physics_Traits<problem_t>::is_hydro_enabled * Physics_NumVars<problem_t>::numHydroVars) + // create constants header file
      ((int)Physics_Traits<problem_t>::is_radiation_enabled * Physics_NumVars<problem_t>::numRadVars);
  static const int matalicityCompStarts =
      primordialChemCompStarts +
      (int)Physics_Traits<problem_t>::is_metalicity_enabled;
  // face-centered quantities
  static const int mhdCompStarts = 0;
};
