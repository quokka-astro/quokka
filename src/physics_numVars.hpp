#ifndef PHYSICS_NUMVARS_HPP_ // NOLINT
#define PHYSICS_NUMVARS_HPP_

struct Physics_NumVars {
  // cell-centred
  static const int numHydroVars = 6;
  static const int numRadVars = 4;
  // face-centred (specify the number of quantities stored per spatial dimension, i.e., each face-centering)
  static const int numMHDVars = 1;
};

#endif // PHYSICS_NUMVARS_HPP_