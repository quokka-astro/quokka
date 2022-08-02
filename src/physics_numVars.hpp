#ifndef PHYSICS_NUMVARS_HPP_ // NOLINT
#define PHYSICS_NUMVARS_HPP_

template <typename problem_t> struct Physics_NumVars {
  static const int numHydroVars = 6;
  static const int numRadVars = 4;
};

#endif // PHYSICS_NUMVARS_HPP_