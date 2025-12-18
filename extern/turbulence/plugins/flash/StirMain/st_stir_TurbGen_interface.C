#include <string>
#include <mpi.h>
#include "mangle_names.h"
#include "TurbGen.h"

static TurbGen st_TurbGenStir;

// Initialise the turbulence generator with input parameters from 'parameter_file'.
// If MPI is present, we pass the MPI rank to TurbGen_init_driving; note that the MPI rank
// is simply to control printf to stdout, i.e., only rank 0 will print.
// This function should only be called once, i.e., to initialise the turbulence generator.
// Here we also return the delta time between updates of driving patterns, which can be
// used to constrain the simulation timestep (to make sure the code goes through all
// driving patterns and doesn't skip any).
extern "C" void FTOC(st_stir_init_driving_c)(char * parameter_file, double * time, double * dt_driv) {
  int MyPE = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &MyPE);
  st_TurbGenStir = TurbGen(MyPE); // create TurbGen obj for Stir
  std::string param_file = parameter_file;
  if (st_TurbGenStir.init_driving(param_file, *time) != 0) exit(-1);
  // return time between pattern updates
  *dt_driv = st_TurbGenStir.get_turnover_time() / st_TurbGenStir.get_nsteps_per_turnover_time();
}

// Function to update the turbulence driving mode coefficients.
// Based on input 'time', it checks if the pattern needs to be updated.
// If it was updated, return 1 in 'have_updated_pattern', else return 0.
// Note that this call to TurbGen_check_for_update(time) is very cheap
// and returns right away if the current input time means that no update
// of the turbulence driving pattern is necessary; therefore, we can safely
// call this in every hydro timestep without wasting compute time.
extern "C" void FTOC(st_stir_check_for_update_of_turb_pattern_c)(const double * time, int * have_updated_pattern,
                                                                 const double v_turb[3]) {
  if (st_TurbGenStir.check_for_update(*time, v_turb)) {
    *have_updated_pattern = 1;
  } else {
    *have_updated_pattern = 0;
  }
}
/*
extern "C" void FTOC(st_stir_check_for_update_of_turb_pattern_c)(double * time, int * have_updated_pattern) {
  if (st_TurbGenStir.check_for_update(*time)) {
    *have_updated_pattern = 1;
  } else {
    *have_updated_pattern = 0;
  }
}
*/

// Function returns the turbulent vector field (vx,vy,vz) given start and end coordinates
// pos_beg and pos_end, on a uniform grid of size n[0]*n[1]*n[2]. Returns single precision (float).
extern "C" void FTOC(st_stir_get_turb_vector_unigrid_c)(const double pos_beg[3], const double pos_end[3],
                                                        const int n[3], float * vx, float * vy, float * vz) {
  float * grid_out[3] = {vx, vy, vz};
  st_TurbGenStir.get_turb_vector_unigrid(pos_beg, pos_end, n, grid_out);
}
