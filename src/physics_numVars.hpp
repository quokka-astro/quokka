#ifndef PHYSICS_NUMVARS_HPP_ // NOLINT
#define PHYSICS_NUMVARS_HPP_

#include <AMReX.H>

struct Physics_NumVars {
  // cell-centred
  static const int numHydroVars = 6;
  static const int numRadVars = 4;
  // face-centred
  static const int numMHDVars_per_dim = 1;
  static const int numMHDVars_tot = AMREX_SPACEDIM * numMHDVars_per_dim;
};

#endif // PHYSICS_NUMVARS_HPP_