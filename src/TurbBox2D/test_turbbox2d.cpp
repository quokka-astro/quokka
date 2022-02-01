//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file test_turbbox2d.cpp
/// \brief Defines a test problem for a box with stochastic forcing.
///

#include "AMReX_Array.H"
#include "AMReX_BCRec.H"
#include "AMReX_BC_TYPES.H"
#include "AMReX_BLassert.H"
#include "AMReX_Config.H"
#include "AMReX_Print.H"
#include "AMReX_REAL.H"

#include "RadhydroSimulation.hpp"
#include "hydro_system.hpp"
#include "Forcing.H"

#include "test_turbbox2d.hpp"

struct TurbBox {};

template <> struct EOS_Traits<TurbBox> {
  static constexpr double gamma = 1.0;
  static constexpr double cs_isothermal = 1.0;
};

template <>
void RadhydroSimulation<TurbBox>::setInitialConditionsAtLevel(int lev) {
  for (amrex::MFIter iter(state_old_[lev]); iter.isValid(); ++iter) {
    const amrex::Box &indexRange = iter.validbox();
    auto const &state = state_new_[lev].array(iter);

    amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
      double vx = 0.;
      double vy = 0.;
      double vz = 0.;
      double rho = 1.0;
      double P = 1.0;

      AMREX_ASSERT(!std::isnan(vx));
      AMREX_ASSERT(!std::isnan(vy));
      AMREX_ASSERT(!std::isnan(vz));
      AMREX_ASSERT(!std::isnan(rho));
      AMREX_ASSERT(!std::isnan(P));

      const auto v_sq = vx * vx + vy * vy + vz * vz;
      const auto gamma = HydroSystem<TurbBox>::gamma_;

      state(i, j, k, density_index) = rho;
      state(i, j, k, x1Momentum_index) = rho * vx;
      state(i, j, k, x2Momentum_index) = rho * vy;
      state(i, j, k, x3Momentum_index) = rho * vz;
      state(i, j, k, energy_index) =
          P / (gamma - 1.) + 0.5 * rho * v_sq;

      // initialize radiation variables to zero
      state(i, j, k, RadSystem<TurbBox>::radEnergy_index) = 0;
      state(i, j, k, RadSystem<TurbBox>::x1RadFlux_index) = 0;
      state(i, j, k, RadSystem<TurbBox>::x2RadFlux_index) = 0;
      state(i, j, k, RadSystem<TurbBox>::x3RadFlux_index) = 0;
    });
  }

  // set flag
  areInitialConditionsDefined_ = true;
}

auto problem_main() -> int {
  // Problem parameters
  const int nvars = RadhydroSimulation<TurbBox>::nvarTotal_;
  amrex::Vector<amrex::BCRec> boundaryConditions(nvars);
  for (int n = 0; n < nvars; ++n) {
    for (int i = 0; i < AMREX_SPACEDIM; ++i) {
      // periodic
      boundaryConditions[n].setLo(i, amrex::BCType::int_dir);
      boundaryConditions[n].setHi(i, amrex::BCType::int_dir);
    }
  }

  // Problem initialization
  RadhydroSimulation<TurbBox> sim(boundaryConditions);
  sim.is_hydro_enabled_ = true;
  sim.is_radiation_enabled_ = false;
  sim.stopTime_ = 0.1; // 1.5;
  sim.cflNumber_ = 0.3;
  sim.maxTimesteps_ = 20000;
  sim.plotfileInterval_ = 2000;

  // initialize
  sim.setInitialConditions();

  // evolve
  sim.evolve();

  // Cleanup and exit
  return 0;
}