#include <mpi.h>
#include "mangle_names.h"
#include "TurbGen.h"

static TurbGen st_TurbGenStirICs;

// Initialise the turbulence generator to produce a single turbulent realisation based on input parameters.
// This applies for the turbulent initial conditions unit 'StirICs'
extern "C" void FTOC(st_stirics_init_single_realisation_c)(const int * ndim, const double L[3], const double * k_min, const double * k_max,
                                                           const int * spect_form, const double * power_law_exp, const double * angles_exp,
                                                           const double * sol_weight, const int * random_seed) {
  int MyPE = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &MyPE);
  st_TurbGenStirICs = TurbGen(MyPE); // create TurbGen obj for StirICs
  if (st_TurbGenStirICs.init_single_realisation(
        *ndim, L, *k_min, *k_max, *spect_form, *power_law_exp, *angles_exp, *sol_weight, *random_seed) != 0) exit(-1);
}

// Function returns the turbulent vector field (vx,vy,vz) given start and end coordinates
// pos_beg and pos_end, on a uniform grid of size n[0]*n[1]*n[2]. Returns single precision (float).
extern "C" void FTOC(st_stirics_get_turb_vector_unigrid_c)(const double pos_beg[3], const double pos_end[3],
                                                           const int n[3], float * vx, float * vy, float * vz) {
  float * grid_out[3] = {vx, vy, vz};
  st_TurbGenStirICs.get_turb_vector_unigrid(pos_beg, pos_end, n, grid_out);
}
