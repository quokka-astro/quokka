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

#include "Gravity.hpp"
#include "RadhydroSimulation.hpp"
#include "hydro_system.hpp"

#include "test_poisson.hpp"

using Real = amrex::Real;

struct PoissonProblem {};

constexpr Real X0 = 0.4;
constexpr Real Y0 = 0.7;
constexpr Real Z0 = 0.3;
constexpr Real R0 = 1. / 4.;

template <>
void RadhydroSimulation<PoissonProblem>::setInitialConditionsAtLevel(int lev) {
  amrex::GpuArray<Real, AMREX_SPACEDIM> dx = geom[lev].CellSizeArray();
  amrex::GpuArray<Real, AMREX_SPACEDIM> prob_lo = geom[lev].ProbLoArray();

  for (amrex::MFIter iter(state_new_[lev]); iter.isValid(); ++iter) {
    const amrex::Box &indexRange = iter.validbox();
    auto const &state = state_new_[lev].array(iter);

    amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
      Real const x = prob_lo[0] + (i + Real(0.5)) * dx[0];
      Real const y = prob_lo[1] + (j + Real(0.5)) * dx[1];
      Real const z = prob_lo[2] + (k + Real(0.5)) * dx[2];
      Real const r = std::sqrt(std::pow(x - X0, 2) + std::pow(y - Y0, 2) +
                               std::pow(z - Z0, 2)) /
                     R0;

      double f = 0.;
      if (r < 1.0) {
        f = std::pow(r - r * r, 4.0);
      }

      for (int n = 0; n < state.nComp(); ++n) {
        state(i, j, k, n) = 0.; // zero fill all components
      }

      AMREX_ASSERT(!std::isnan(f));
      state(i, j, k, HydroSystem<PoissonProblem>::density_index) =
          f / (4.0 * M_PI * C::Gconst);
    });
  }

  // set flag
  areInitialConditionsDefined_ = true;
}

void computeReferenceSolution(
    amrex::MultiFab &ref,
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx,
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_lo,
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const & /*prob_hi*/) {

  // fill reference solution multifab
  for (amrex::MFIter iter(ref); iter.isValid(); ++iter) {
    const amrex::Box &indexRange = iter.validbox();
    auto const &phiExact = ref.array(iter);
    auto const ncomp = ref.nComp();
    AMREX_ASSERT(ncomp == 1);

    amrex::ParallelFor(
        indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
          // compute exact phi
          Real const x = prob_lo[0] + (i + Real(0.5)) * dx[0];
          Real const y = prob_lo[1] + (j + Real(0.5)) * dx[1];
          Real const z = prob_lo[2] + (k + Real(0.5)) * dx[2];
          Real const r = std::sqrt(std::pow(x - X0, 2) + std::pow(y - Y0, 2) +
                                   std::pow(z - Z0, 2)) /
                         R0;

          Real phi = NAN;
          if (r < 1.0) {
            phi = std::pow(r, 6) / 42. - std::pow(r, 7) / 14. +
                  std::pow(r, 8) / 12. - 2. * std::pow(r, 9) / 45. +
                  std::pow(r, 10) / 110. - (1. / 1260.);
          } else {
            phi = -1.0 / (2310. * r);
          }
          phiExact(i, j, k) = (R0 * R0) * phi;
        });
  }
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

  Gravity<PoissonProblem> grav(&sim, phys_bc, sim.coordCenter_,
                               HydroSystem<PoissonProblem>::density_index);

  // initialize gravity MultiFabs
  for (int i = 0; i <= sim.finestLevel(); ++i) {
    grav.phi_old_[i].define(sim.boxArray(i), sim.DistributionMap(i), 1, 1);
    grav.phi_new_[i].define(sim.boxArray(i), sim.DistributionMap(i), 1, 1);
    grav.g_old_[i].define(sim.boxArray(i), sim.DistributionMap(i), 3, 1);
    grav.g_new_[i].define(sim.boxArray(i), sim.DistributionMap(i), 3, 1);
    grav.phi_old_[i].setVal(0.);
    grav.phi_new_[i].setVal(0.);

    // create temporary arrays for (face-centered) \grad \phi
    grav.install_level(i);
  }

  amrex::Print() << "Starting solve..." << std::endl;

  // this is necessary to call before solving, otherwise abs_tol = 0!
  grav.update_max_rhs();

  // solve on root level
  int is_new = 1;
  grav.solve_for_phi(0, grav.phi_new_[0],
                     amrex::GetVecOfPtrs(grav.get_grad_phi_curr(0)), is_new);

  amrex::Print() << "... testing grad_phi_curr after doing single level solve "
                 << '\n';

  grav.test_level_grad_phi_curr(0);

  // compare to exact solution for phi
  // (for this test problem, see Ch 5.1 of Van Straalen thesis)
  int ncomp = 1;
  int nghost = 0;
  amrex::MultiFab phi_exact(sim.boxArray(0), sim.DistributionMap(0), ncomp,
                            nghost);
  auto dx = sim.Geom(0).CellSizeArray();
  auto prob_lo = sim.Geom(0).ProbLoArray();
  auto prob_hi = sim.Geom(0).ProbHiArray();
  computeReferenceSolution(phi_exact, dx, prob_lo, prob_hi);

  // compute error norm
  amrex::MultiFab residual(sim.boxArray(0), sim.DistributionMap(0), ncomp,
                           nghost);
  amrex::MultiFab::Copy(residual, phi_exact, 0, 0, ncomp, nghost);
  amrex::MultiFab::Saxpy(residual, -1., grav.phi_new_[0], 0, 0, ncomp, nghost);

  amrex::Real sol_norm = phi_exact.norm1(0);
  amrex::Real err_norm = residual.norm1(0);
  const double rel_error = err_norm / sol_norm;
  amrex::Print() << "\nRelative error norm = " << rel_error << "\n";

  int status = 0;
  const double err_tol = 1.2e-4; // for 2nd-order discretization, 64^3 grid
  if (rel_error > err_tol) {
    status = 1;
  }

  // Cleanup and exit
  return status;
}
