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
#include "test_mhd_prelim.hpp"

struct WaveProblem {};

template <> struct HydroSystem_Traits<WaveProblem> {
  static constexpr double gamma = 5. / 3.;
  static constexpr bool reconstruct_eint = true;
};

template <> struct Physics_Traits<WaveProblem> {
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
    1.0 / HydroSystem<WaveProblem>::gamma_; // background pressure
constexpr double v0 = 0.;                   // background velocity
constexpr double amp = 1.0e-6;              // perturbation amplitude

AMREX_GPU_DEVICE void computeWaveSolution(
    int i, int j, int k, amrex::Array4<amrex::Real> const &state,
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx,
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_lo) {
  const amrex::Real x_L = prob_lo[0] + (i + amrex::Real(0.0)) * dx[0];
  const amrex::Real x_R = prob_lo[0] + (i + amrex::Real(1.0)) * dx[0];
  const amrex::Real A = amp;

  const quokka::valarray<double, 3> R = {
      1.0, -1.0, 1.5}; // right eigenvector of sound wave
  const quokka::valarray<double, 3> U_0 = {
      rho0, rho0 * v0,
      P0 / (HydroSystem<WaveProblem>::gamma_ - 1.0) +
          0.5 * rho0 * std::pow(v0, 2)};
  const quokka::valarray<double, 3> dU =
      (A * R / (2.0 * M_PI * dx[0])) *
      (std::cos(2.0 * M_PI * x_L) - std::cos(2.0 * M_PI * x_R));

  double rho = U_0[0] + dU[0];
  double xmom = U_0[1] + dU[1];
  double Etot = U_0[2] + dU[2];
  double Eint = Etot - 0.5 * (xmom * xmom) / rho;

  state(i, j, k, HydroSystem<WaveProblem>::density_index) = rho;
  state(i, j, k, HydroSystem<WaveProblem>::x1Momentum_index) = xmom;
  state(i, j, k, HydroSystem<WaveProblem>::x2Momentum_index) = 0;
  state(i, j, k, HydroSystem<WaveProblem>::x3Momentum_index) = 0;
  state(i, j, k, HydroSystem<WaveProblem>::energy_index) = Etot;
  state(i, j, k, HydroSystem<WaveProblem>::internalEnergy_index) = Eint;
}

template <>
void RadhydroSimulation<WaveProblem>::setInitialConditionsOnGrid(
    quokka::grid grid_elem) {
  // extract grid information
  const amrex::Array4<double>& state = grid_elem.array;
  const amrex::Box &indexRange = grid_elem.indexRange;
  const quokka::centering cen = grid_elem.cen;
  const quokka::direction dir = grid_elem.dir;

  if (cen == quokka::centering::cc) {
    // extract variables required from the geom object
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = grid_elem.dx;
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = grid_elem.prob_lo;
    const int ncomp = ncomp_cc_;
    // loop over the grid and set the initial condition
    amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
      for (int n = 0; n < ncomp; ++n) {
        state(i, j, k, n) = 0; // fill unused components with zeros
      }
      computeWaveSolution(i, j, k, state, dx, prob_lo);
    });
  } else if (cen == quokka::centering::fc) {
    if (dir == quokka::direction::x) {
      amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
        state(i, j, k, MHDSystem<WaveProblem>::energy_index) = (i % 2);
      });
    } else if (dir == quokka::direction::y) {
      amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
        state(i, j, k, MHDSystem<WaveProblem>::energy_index) = 2.0;
      });
    } else if (dir == quokka::direction::z) {
      amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
        state(i, j, k, MHDSystem<WaveProblem>::energy_index) = 3.0;
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
  const int nvars = RadhydroSimulation<WaveProblem>::nvarTotal_cc_;
  amrex::Vector<amrex::BCRec> BCs_cc(nvars);
  for (int n = 0; n < nvars; ++n) {
    for (int i = 0; i < AMREX_SPACEDIM; ++i) {
      BCs_cc[n].setLo(i, amrex::BCType::int_dir); // periodic
      BCs_cc[n].setHi(i, amrex::BCType::int_dir);
    }
  }

  RadhydroSimulation<WaveProblem> sim(BCs_cc);
  
  sim.cflNumber_ = CFL_number;
  sim.stopTime_ = max_time;
  sim.maxTimesteps_ = max_timesteps;

  // set initial conditions
  sim.setInitialConditions();

  // Main time loop
  sim.evolve();

  return 0;
}