//==============================================================================
// Copyright 2022 Neco Kriel.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file test_mhd_prelim.cpp
/// \brief Defines a test problem to make sure face-centred quantities are created correctly.
///

#include <valarray>

#include "AMReX_Array.H"
#include "AMReX_Array4.H"
#include "AMReX_REAL.H"

#include "RadhydroSimulation.hpp"
#include "fextract.hpp"
#include "grid.hpp"
#include "test_fc_quantities.hpp"

struct FCQuantities {};

template <> struct HydroSystem_Traits<FCQuantities> {
  static constexpr double gamma = 5. / 3.;
  static constexpr bool reconstruct_eint = true;
};

template <> struct Physics_Traits<FCQuantities> {
  // cell-centred
  static constexpr bool is_hydro_enabled = true;
  static constexpr bool is_chemistry_enabled = false;
  static constexpr int numPassiveScalars = 0; // number of passive scalars
  static constexpr bool is_radiation_enabled = false;
  // face-centred
  static constexpr bool is_mhd_enabled = true;
};

template <>
void RadhydroSimulation<FCQuantities>::setInitialConditionsOnGrid(
    quokka::grid grid_elem) {
  // extract grid information
  const amrex::Array4<double>& state = grid_elem.array;
  const amrex::Box &indexRange = grid_elem.indexRange;
  const quokka::centering cen = grid_elem.cen;
  const quokka::direction dir = grid_elem.dir;

  if (cen == quokka::centering::cc) {
    const int ncomp_cc = ncomp_cc_;
    // loop over the grid and set the initial condition
    amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
      for (int n = 0; n < ncomp_cc; ++n) {
        state(i, j, k, n) = n;
      }
    });
  } else if (cen == quokka::centering::fc) {
    if (dir == quokka::direction::x) {
      amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
        state(i, j, k, MHDSystem<FCQuantities>::energy_index) = 1.0 + (i % 2);
      });
    } else if (dir == quokka::direction::y) {
      amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
        state(i, j, k, MHDSystem<FCQuantities>::energy_index) = 2.0 + (i % 2);
      });
    } else if (dir == quokka::direction::z) {
      amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
        state(i, j, k, MHDSystem<FCQuantities>::energy_index) = 3.0 + (i % 2);
      });
    }
  }
}

auto problem_main() -> int {
  // Based on the ATHENA test page:
  // https://www.astro.princeton.edu/~jstone/Athena/tests/linear-waves/linear-waves.html

  // Problem parameters
  // const int nx = 100;
  // const double Lx = 1.0;
  const double CFL_number = 0.1;
  const double max_time = 1.0;
  const int max_timesteps = 2e4;

  // Problem initialization
  const int nvars = RadhydroSimulation<FCQuantities>::nvarTotal_cc_;
  amrex::Vector<amrex::BCRec> BCs_cc(nvars);
  for (int n = 0; n < nvars; ++n) {
    for (int i = 0; i < AMREX_SPACEDIM; ++i) {
      BCs_cc[n].setLo(i, amrex::BCType::int_dir); // periodic
      BCs_cc[n].setHi(i, amrex::BCType::int_dir);
    }
  }

  RadhydroSimulation<FCQuantities> sim(BCs_cc);
  
  sim.cflNumber_ = CFL_number;
  sim.stopTime_ = max_time;
  sim.maxTimesteps_ = max_timesteps;

  // set initial conditions
  sim.setInitialConditions();

  // Main time loop
  sim.evolve();

  return 0;
}