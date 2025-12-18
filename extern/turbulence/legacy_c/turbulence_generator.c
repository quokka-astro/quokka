/**********************************************************
 *** Basic example code for using Turbulence Generator. ***
 *** Written by Christoph Federrath, 2022.              ***
***********************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include "turbulence_generator.h"

int main(void)
{
  // 1st example case (turbulence driving)
  if (true) {
    printf("turbulence_generator: ==========================================================\n");

    // turbulence_generator.h defines data structure 'TurbGenData'
    tgd.verbose = false; // in case we want additional output during operations of the turbulence generator

    // First, initialise the turbulence generator with input parameters from 'parameter_file'.
    // If MPI is present, we pass the MPI rank to TurbGen_init_turbulence_generator;
    // note that the MPI rank is simply to control printf to stdout, i.e., only rank 0 will print.
    // This function should only be called once, i.e., to initialise the turbulence generator.
    char * parameter_file = "turbulence_generator.inp";
    int MPI_rank = 0;
    TurbGen_init_driving(parameter_file, MPI_rank);

    // as an example, let's loop through a few time steps (e.g., of a hydro code)
    // to demonstrate how the turbulence generator is updated and how the physical turbulent
    // acceleration field is returned by "turbulence_generator.h

    // get the turbulent turnover time
    double t_turb = tgd.decay;
    // let's go through 5 turnover times based on the inputs in 'parameter_file' above
    double t_end = 5.0 * t_turb;

    // set some number of time steps that the hydro code would go through
    int nsteps = 123;
    // time step (this will normally be set by the hydro code;
    // ultimately, one simply feeds in the 'time' at which one wants the respective
    // physical turbulent acceleration field -- see below in call to TurbGen_check_for_update
    double dt = t_end / nsteps;

    // loop through time steps
    for (int step = 0; step <= nsteps; step++) {

      double time = step * dt;
      printf("turbulence_generator: time = %f\n", time);

      // Function to update the turbulence driving mode coefficients.
      // Based on input 'time', it checks if the pattern needs to be updated.
      // If it was updated, return true in 'have_updated_pattern', else return false.
      // Note that this call to TurbGen_update(time) is very cheap and returns right away if current input
      // time means that no update of the turbulence driving pattern is necessary; therefore, we can safely
      // call this in every hydro timestep without wasting compute time.
      bool have_updated_pattern = TurbGen_check_for_update(time);

      if (have_updated_pattern) {
        // say we want the turbulent vector field value at some physical position pos[3]
        double pos[3] = {0.1, -0.5, 0.3};
        double v[3]; // return vector
        TurbGen_get_turb_vector(pos, v); // get the vector
        printf("turbulence_generator: turbulent vector at position (%f %f %f) is vx, vy, vz = %f %f %f\n",
                pos[0], pos[1], pos[2], v[0], v[1], v[2]);
      }

    } // end of loop over timesteps

  } // end of 1st example


  // 2nd example case (return turbulent vector field on uniform grid)
  if (true) {
    printf("turbulence_generator: ==========================================================\n");

    // The above example returns the turbulent vector field at only a single position.
    // Here we want the turbulent vector field at all positions on a 3D uniform grid of size nx, nz, nz.
    // In grid codes, this would be function to call, because it is faster than calling TurbGen_get_turb_vector
    // at every requested position; here, it is done for an entire block (uniform grid) of data.

    // first init as in example 1
    char * parameter_file = "turbulence_generator.inp";
    int MPI_rank = 0;
    TurbGen_init_driving(parameter_file, MPI_rank);

    // now fast forward to target time (e.g., after 5 turbulent turnover times)
    bool have_updated_pattern = TurbGen_check_for_update(5*tgd.decay);

    // random example of physical coordinates of a target output grid (pos_beg must be < pos_end):
    double pos_beg[3] = {0.1, -0.5, 0.3}; // first cell coordinate in uniform grid (x,y,z)
    double pos_end[3] = {0.4, -0.3, 0.5}; // last  cell coordinate in uniform grid (x,y,z)
    int n[3] = {10, 20, 13}; // some random number of cells that define the output grid size
    long ntot = n[0]*n[1]*n[2]; // total number of grid cells
    float * grid_out[3]; // output grid (three cartesian components: vx, vy, vz), which receives the turbulent acceleration field

    // allocate
    for (int dim = 0; dim < 3; dim++) {
      grid_out[dim] = malloc(ntot*sizeof(*grid_out[dim]));
      if (grid_out[dim] == NULL) { printf("Allocation error for dim = %i.\n", dim); exit(-1); }
    }

    // call to return 3 uniform grids with the 3 components of the turbulent acceleration field at requested positions
    TurbGen_get_turb_vector_unigrid(pos_beg, pos_end, n, grid_out);

    // indices in x,y,z, i.e., i,j,k, and 1D index
    int i, j, k; long index;
    i = 3; j = 0; k = 10;
    index = k*n[1]*n[0] + j*n[0] + i;
    printf("turbulence_generator: e.g., x-component of turbulent vector field at index i,j,k = (%i %i %i) is %f\n", i,j,k, grid_out[0][index]);
    i = 9; j = 1; k = 4;
    index = k*n[1]*n[0] + j*n[0] + i;
    printf("turbulence_generator: e.g., z-component of turbulent vector field at index i,j,k = (%i %i %i) is %f\n", i,j,k, grid_out[2][index]);

    // clean up
    for (int dim = 0; dim < 3; dim++) {
      if (grid_out[dim]) free(grid_out[dim]); grid_out[dim] = NULL;
    }

  } // end of 2nd example


  // 3rd example case (single realisation)
  if (true) {
    printf("turbulence_generator: ==========================================================\n");

    // Example of how to generate a single realisation of a turbulent vector field.
    // One can use this for example to generate an intial turbulent velocity or magnetic field
    // with the spectral properties specified in the call to TurbGen_get_turb_vector_unigrid_single_realisation.

    double L[3] = {1.0, 1.0, 1.0}; // size of box in x, y, z
    double k_min = 2.0; // minimum wavenumber for turbulent field (in units of 2pi / L[X])
    double k_max = 20.0; // minimum wavenumber for turbulent field (in units of 2pi / L[X])
    int spect_form = 2; // 0: band/rectangle/constant, 1: paraboloid, 2: power law
    double power_law_exp = -2.0; // if spect_form == 2: power-law exponent (e.g., -2 would be Burgers,
                                 // or -5/3 would be Kolmogorov, or 1.5 would Kazantsev)
    double angles_exp = 1.0; // if spect_form == 2: spectral sampling of angles;
                             // number of modes (angles) in k-shell surface increases as k^angles_exp.
                             // For full sampling, angles_exp = 2.0; for healpix-type sampling, angles_exp = 0.0.
    double sol_weight = 0.5; // solenoidal weight: 1.0: solenoidal driving, 0.0: compressive driving, 0.5: natural mixture
    int random_seed = 140281; // random seed for this turbulent realisation
    int PE = 0; // in case we call this from an MPI program, we supply the MPI rank here, so we limited fprint to stdout to the MPI master rank

    // random example of physical coordinates of a target output grid (pos_beg must be < pos_end):
    int n[3] = {64, 64, 64}; // some random number of cells that define the output grid size
    double d[3]; for (int dim = 0; dim < 3; dim++) d[dim] = L[dim] / n[dim]; // define cell size of uniform grid
    double pos_beg[3] = {-0.5+d[X]/2, -0.5+d[Y]/2, -0.5+d[Z]/2}; // first cell coordinate in uniform grid (x,y,z); here cell-centered example
    double pos_end[3] = {+0.5-d[X]/2, +0.5-d[Y]/2, +0.5-d[Z]/2}; // last  cell coordinate in uniform grid (x,y,z); here cell-centered example
    long ntot = n[0]*n[1]*n[2]; // total number of grid cells
    float * grid_out[3]; // output grid (three cartesian components: vx, vy, vz), which receives the turbulent acceleration field

    // allocate
    for (int dim = 0; dim < 3; dim++) {
      grid_out[dim] = malloc(ntot*sizeof(*grid_out[dim]));
      if (grid_out[dim] == NULL) { printf("Allocation error for dim = %i.\n", dim); exit(-1); }
    }

    // initialise generator to return a single turbulent realisation based on input parameters
    TurbGen_init_single_realisation(L, k_min, k_max, spect_form, power_law_exp, angles_exp, sol_weight, random_seed, PE);

    // call to return 3 uniform grids with the 3 components of the turbulent acceleration field at requested positions
    TurbGen_get_turb_vector_unigrid(pos_beg, pos_end, n, grid_out);

    // indices in x,y,z, i.e., i,j,k, and 1D index
    int i, j, k; long index;
    i = 3; j = 0; k = 10;
    index = k*n[1]*n[0] + j*n[0] + i;
    printf("turbulence_generator: e.g., x-component of turbulent vector field at index i,j,k = (%i %i %i) is %f\n", i,j,k, grid_out[0][index]);
    i = 9; j = 1; k = 4;
    index = k*n[1]*n[0] + j*n[0] + i;
    printf("turbulence_generator: e.g., z-component of turbulent vector field at index i,j,k = (%i %i %i) is %f\n", i,j,k, grid_out[2][index]);

    // compute mean and RMS of vector field
    double mean[3] = {0.0, 0.0, 0.0};
    double rms[3] = {0.0, 0.0, 0.0};
    double std[3] = {0.0, 0.0, 0.0};
    for (int dim = 0; dim < 3; dim++) {
      for (long ni = 0; ni < ntot; ni++) {
        mean[dim] += grid_out[dim][ni];
        rms [dim] += pow(grid_out[dim][ni],2.0);
      }
      mean[dim] /= ntot; // mean
      rms[dim] = sqrt(rms[dim] / ntot); // root mean squared
      std[dim] = sqrt(rms[dim]*rms[dim] - mean[dim]*mean[dim]); // standard deviation
    }
    printf("turbulence_generator: Turbulent vector field mean (x,y,z) = (%e %e %e)\n", mean[0], mean[1], mean[2]);
    printf("turbulence_generator: Turbulent vector field  rms (x,y,z) = (%e %e %e)\n",  rms[0],  rms[1],  rms[2]);
    printf("turbulence_generator: Turbulent vector field  std (x,y,z) = (%e %e %e)\n",  std[0],  std[1],  std[2]);
    printf("turbulence_generator: Turbulent vector field total 3D std = %e\n", sqrt(std[0]*std[0]+std[1]*std[1]+std[2]*std[2]));

    // clean up
    for (int dim = 0; dim < 3; dim++) {
      if (grid_out[dim]) free(grid_out[dim]); grid_out[dim] = NULL;
    }

  } // end of 3rd example

  return 0;
}


