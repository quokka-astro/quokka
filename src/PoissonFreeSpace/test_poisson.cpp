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

#include "AMReX_SPACE.H"
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
  // (For this test problem, see Ch 5.1 of Van Straalen thesis)

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
    amrex::MultiFab &ref, amrex::GpuArray<Real, AMREX_SPACEDIM> const &dx,
    amrex::GpuArray<Real, AMREX_SPACEDIM> const &prob_lo,
    amrex::GpuArray<Real, AMREX_SPACEDIM> const & /*prob_hi*/) {
  // (For this test problem, see Ch 5.1 of Van Straalen thesis)

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

template <>
void RadhydroSimulation<PoissonProblem>::ErrorEst(int lev,
                                                  amrex::TagBoxArray &tags,
                                                  Real /*time*/,
                                                  int /*ngrow*/) {
  // tag cells for refinement
  Real const jeans_threshold = 0.25;

  auto dx = Geom(lev).CellSizeArray();
  Real const dx_min = std::min({AMREX_D_DECL(dx[0], dx[1], dx[2])});

  for (amrex::MFIter mfi(state_new_[lev]); mfi.isValid(); ++mfi) {
    const amrex::Box &box = mfi.validbox();
    const auto state = state_new_[lev].const_array(mfi);
    const auto tag = tags.array(mfi);

    amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
      Real const rho =
          state(i, j, k, HydroSystem<PoissonProblem>::density_index);
      //Real const cs = 1.0 / (4. * 1024. * M_PI); // arbitrary for this test problem
      Real const cs = 1.0 / (1024. * M_PI); // arbitrary for this test problem
      Real const lambda_jeans = cs * std::sqrt(M_PI / (C::Gconst * rho));
      Real const jeans_number = dx_min / lambda_jeans;

      if (jeans_number > jeans_threshold && rho > 0.0) {
        tag(i, j, k) = amrex::TagBox::SET;
      }
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

  // set physical boundary conditions (periodic or free space)
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
    // allocate cell-centered MultiFabs for potential, acceleration
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

  // multilevel solve
  grav.multilevel_solve_for_new_phi(0, sim.finestLevel());
  grav.test_composite_phi(0);

  // test single-level solves
  for (int i = 0; i <= sim.finestLevel(); ++i) {
    grav.construct_old_gravity(0., i);
    grav.construct_new_gravity(0., i);
  }

  // compare to exact solution for phi
  int ncomp = 1;
  int nghost = 0;
  int status = 0;
  const double err_tol = 1.2e-4; // for 2nd-order discretization, 64^3 grid

  for (int i = 0; i <= sim.finestLevel(); ++i) {
    // compute exact solution on level i
    amrex::MultiFab phi_exact(sim.boxArray(0), sim.DistributionMap(0), ncomp,
                              nghost);
    auto dx = sim.Geom(0).CellSizeArray();
    auto prob_lo = sim.Geom(0).ProbLoArray();
    auto prob_hi = sim.Geom(0).ProbHiArray();
    computeReferenceSolution(phi_exact, dx, prob_lo, prob_hi);

    // compute error norm on level i
    amrex::MultiFab residual(sim.boxArray(0), sim.DistributionMap(0), ncomp,
                             nghost);
    amrex::MultiFab::Copy(residual, phi_exact, 0, 0, ncomp, nghost);
    amrex::MultiFab::Saxpy(residual, -1., grav.phi_new_[0], 0, 0, ncomp,
                           nghost);

    Real sol_norm = phi_exact.norm1(0);
    Real err_norm = residual.norm1(0);
    // this gets larger when more AMR levels are used...
    const double rel_error = err_norm / sol_norm;
    amrex::Print() << "[level " << i << "] Relative error norm = " << rel_error
                   << "\n";

    if (rel_error > err_tol) {
      status = 1;
    }
  }

  amrex::Print() << std::endl;
  return status;
}
