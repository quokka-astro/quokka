//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file test_poisson.cpp
/// \brief Defines a test problem for the free-space Poisson solver.
///

#include "AMReX_BC_TYPES.H"
#include "AMReX_BLassert.H"
#include "AMReX_Config.H"
#include "AMReX_ParallelDescriptor.H"
#include "AMReX_ParmParse.H"
#include "AMReX_Print.H"

#include "RadhydroSimulation.hpp"
#include "hydro_system.hpp"
#include "Gravity.hpp"

#include "test_poisson.hpp"

using Real = amrex::Real;

struct PoissonProblem {};

template <>
void RadhydroSimulation<PoissonProblem>::setInitialConditionsAtLevel(int lev) {
  amrex::GpuArray<Real, AMREX_SPACEDIM> dx = geom[lev].CellSizeArray();
  amrex::GpuArray<Real, AMREX_SPACEDIM> prob_lo = geom[lev].ProbLoArray();
  amrex::GpuArray<Real, AMREX_SPACEDIM> prob_hi = geom[lev].ProbHiArray();

  Real x0 = prob_lo[0] + 0.5 * (prob_hi[0] - prob_lo[0]);
  Real y0 = prob_lo[1] + 0.5 * (prob_hi[1] - prob_lo[1]);
  Real z0 = prob_lo[2] + 0.5 * (prob_hi[2] - prob_lo[2]);

  for (amrex::MFIter iter(state_new_[lev]); iter.isValid(); ++iter) {
    const amrex::Box &indexRange = iter.validbox();
    auto const &state = state_new_[lev].array(iter);

    amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
      Real const x = prob_lo[0] + (i + Real(0.5)) * dx[0];
      Real const y = prob_lo[1] + (j + Real(0.5)) * dx[1];
      Real const z = prob_lo[2] + (k + Real(0.5)) * dx[2];
      Real const R0 = 1./4.;
      Real const r = std::sqrt(std::pow(x - x0, 2) + std::pow(y - y0, 2) +
                               std::pow(z - z0, 2)) / R0;

      double rho = 0.;
      if (r < 1.0) {
        rho = std::pow(r - r*r, 4.0);
      }

      for (int n = 0; n < state.nComp(); ++n) {
        state(i, j, k, n) = 0.; // zero fill all components
      }

      AMREX_ASSERT(!std::isnan(rho));
      state(i, j, k, HydroSystem<PoissonProblem>::density_index) = rho;
    });
  }

  // set flag
  areInitialConditionsDefined_ = true;
}

auto problem_main() -> int {
  const int nvars = RadhydroSimulation<PoissonProblem>::nvarTotal_;
  amrex::Vector<amrex::BCRec> boundaryConditions(nvars);
  for (int n = 0; n < nvars; ++n) {
    for (int i = 0; i < AMREX_SPACEDIM; ++i) {
      boundaryConditions[n].setLo(i, amrex::BCType::foextrap);
      boundaryConditions[n].setHi(i, amrex::BCType::foextrap);
    }
  }

  // Problem initialization
  RadhydroSimulation<PoissonProblem> sim(boundaryConditions);
  sim.is_hydro_enabled_ = true;
  sim.is_radiation_enabled_ = false;
  sim.plotfileInterval_ = -1;
  sim.checkpointInterval_ = -1;

  // initialize
  sim.setInitialConditions();

  // set physical boundary conditions (periodic, free space, or reflecting)
  amrex::BCRec phys_bc;
  for (int i = 0; i < AMREX_SPACEDIM; ++i) {
    phys_bc.setLo(i, amrex::PhysBCType::outflow);
    phys_bc.setHi(i, amrex::PhysBCType::outflow);
  }

  // solve Poisson equation
  amrex::Print() << "Initializing gravity solver..." << std::endl;
  
  Gravity<PoissonProblem> gravity_solver(
      &sim, phys_bc, sim.coordCenter_,
      HydroSystem<PoissonProblem>::density_index);

  amrex::Print() << "Initialized gravity solver." << std::endl;

  // initialize gravity MultiFabs
  gravity_solver.phi_old_[0].define(sim.boxArray(0), sim.DistributionMap(0), 1, 1);
  gravity_solver.phi_new_[0].define(sim.boxArray(0), sim.DistributionMap(0), 1, 1);
  gravity_solver.g_old_[0].define(sim.boxArray(0), sim.DistributionMap(0), 3, 1);
  gravity_solver.g_new_[0].define(sim.boxArray(0), sim.DistributionMap(0), 3, 1);

  gravity_solver.phi_old_[0].setVal(0.);
  gravity_solver.phi_new_[0].setVal(0.);

  // create temporary arrays for (face-centered) \grad \phi
  gravity_solver.install_level(0);

  amrex::Print() <<  "Starting solve..." << std::endl;

  // this is necessary to call before solving, otherwise abs_tol = 0!
  gravity_solver.update_max_rhs();

  int is_new = 1;
  gravity_solver.solve_for_phi(
      0, gravity_solver.phi_new_[0],
      amrex::GetVecOfPtrs(gravity_solver.get_grad_phi_curr(0)), is_new);

  amrex::Print() << "... testing grad_phi_curr after doing single level solve " << '\n';

  gravity_solver.test_level_grad_phi_curr(0);

  // compare to exact solution for phi (as done in Ch 5.1 of Van Straalen thesis)
  //amrex::MultiFab phi_exact(sim->boxArray(0), sim->DistributionMap(0), 1, 0);
  //compute_exact_phi(phi_exact);

  // Cleanup and exit
  amrex::Print() << "Finished." << std::endl;
  return 0;
}
