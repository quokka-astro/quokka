//==============================================================================
// Copyright 2022 Neco Kriel.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file test_fc_quantities.cpp
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

constexpr double rho0 = 1.0; // background density
constexpr double P0 =
    1.0 / HydroSystem<FCQuantities>::gamma_; // background pressure
constexpr double v0 = 0.;                   // background velocity
constexpr double amp = 1.0e-6;              // perturbation amplitude

AMREX_GPU_DEVICE void computeWaveSolution(
    int i, int j, int k, amrex::Array4<amrex::Real> const &state,
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx,
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_lo) {
  const amrex::Real x_L = prob_lo[0] + (i + static_cast<amrex::Real>(0.0)) * dx[0];
  const amrex::Real x_R = prob_lo[0] + (i + static_cast<amrex::Real>(1.0)) * dx[0];
  const amrex::Real A = amp;

  const quokka::valarray<double, 3> R = {
      1.0, -1.0, 1.5}; // right eigenvector of sound wave
  const quokka::valarray<double, 3> U_0 = {
      rho0, rho0 * v0,
      P0 / (HydroSystem<FCQuantities>::gamma_ - 1.0) +
          0.5 * rho0 * std::pow(v0, 2)};
  const quokka::valarray<double, 3> dU =
      (A * R / (2.0 * M_PI * dx[0])) *
      (std::cos(2.0 * M_PI * x_L) - std::cos(2.0 * M_PI * x_R));

  double rho = U_0[0] + dU[0];
  double xmom = U_0[1] + dU[1];
  double Etot = U_0[2] + dU[2];
  double Eint = Etot - 0.5 * (xmom * xmom) / rho;

  state(i, j, k, HydroSystem<FCQuantities>::density_index) = rho;
  state(i, j, k, HydroSystem<FCQuantities>::x1Momentum_index) = xmom;
  state(i, j, k, HydroSystem<FCQuantities>::x2Momentum_index) = 0;
  state(i, j, k, HydroSystem<FCQuantities>::x3Momentum_index) = 0;
  state(i, j, k, HydroSystem<FCQuantities>::energy_index) = Etot;
  state(i, j, k, HydroSystem<FCQuantities>::internalEnergy_index) = Eint;
}

template <>
void RadhydroSimulation<FCQuantities>::setInitialConditionsOnGrid(
    quokka::grid grid_elem) {
  // extract grid information
  amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = grid_elem.dx_;
  amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = grid_elem.prob_lo_;
  const amrex::Array4<double>& state = grid_elem.array_;
  const amrex::Box &indexRange = grid_elem.indexRange_;
  const quokka::centering cen = grid_elem.cen_;
  const quokka::direction dir = grid_elem.dir_;

  if (cen == quokka::centering::cc) {
    const int ncomp_cc = ncomp_cc_;
    // loop over the grid and set the initial condition
    amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
      for (int n = 0; n < ncomp_cc; ++n) {
        state(i, j, k, n) = 0; // fill unused quantities with zeros
      }
      computeWaveSolution(i, j, k, state, dx, prob_lo);
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
  // Problem initialization
  const int nvars = RadhydroSimulation<FCQuantities>::nvarTotal_cc_;
  amrex::Vector<amrex::BCRec> BCs_cc(nvars);
  for (int n = 0; n < nvars; ++n) {
    for (int i = 0; i < AMREX_SPACEDIM; ++i) {
      BCs_cc[n].setLo(i, amrex::BCType::int_dir); // periodic
      BCs_cc[n].setHi(i, amrex::BCType::int_dir);
    }
  }

  RadhydroSimulation<FCQuantities> sim_write(BCs_cc);
  sim_write.cflNumber_ = 0.1;
  sim_write.stopTime_ = 1.0;
  sim_write.maxTimesteps_ = 2e4;
  sim_write.setInitialConditions();
  sim_write.evolve();

  RadhydroSimulation<FCQuantities> sim_read(BCs_cc);
  sim_read.cflNumber_ = 0.1;
  sim_read.stopTime_ = 1.0;
  sim_read.maxTimesteps_ = 2e4;
  sim_read.setInitialConditions();

  return 0;
}