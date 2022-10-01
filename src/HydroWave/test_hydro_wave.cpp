//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file test_hydro_wave.cpp
/// \brief Defines a test problem for a linear hydro wave.
///

#include <valarray>

#include "AMReX_Array.H"
#include "AMReX_Array4.H"
#include "AMReX_REAL.H"

#include "RadhydroSimulation.hpp"
#include "fextract.hpp"
#include "hydro_system.hpp"
#include "test_hydro_wave.hpp"

struct WaveProblem {};

template <> struct HydroSystem_Traits<WaveProblem> {
  static constexpr double gamma = 5. / 3.;
  static constexpr bool reconstruct_eint = true;
  static constexpr int nscalars = 0;       // number of passive scalars
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
void RadhydroSimulation<WaveProblem>::setInitialConditionsAtLevel(int lev) {
  amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = geom[lev].CellSizeArray();
  amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo =
      geom[lev].ProbLoArray();
  const int ncomp = ncomp_;

  for (amrex::MFIter iter(state_new_[lev]); iter.isValid(); ++iter) {
    const amrex::Box &indexRange = iter.validbox(); // excludes ghost zones
    auto const &state = state_new_[lev].array(iter);
    amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
      for (int n = 0; n < ncomp; ++n) {
        state(i, j, k, n) = 0; // fill unused components with zeros
      }
      computeWaveSolution(i, j, k, state, dx, prob_lo);
    });
  }

  // set flag
  areInitialConditionsDefined_ = true;
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
  const int nvars = RadhydroSimulation<WaveProblem>::nvarTotal_;
  amrex::Vector<amrex::BCRec> boundaryConditions(nvars);
  for (int n = 0; n < nvars; ++n) {
    for (int i = 0; i < AMREX_SPACEDIM; ++i) {
      boundaryConditions[n].setLo(i, amrex::BCType::int_dir); // periodic
      boundaryConditions[n].setHi(i, amrex::BCType::int_dir);
    }
  }

  RadhydroSimulation<WaveProblem> sim(boundaryConditions, false);
  sim.is_hydro_enabled_ = true;
  sim.is_radiation_enabled_ = false;
  sim.cflNumber_ = CFL_number;
  sim.stopTime_ = max_time;
  sim.maxTimesteps_ = max_timesteps;
  sim.plotfileInterval_ = -1;

  // set initial conditions
  sim.setInitialConditions();
  auto [pos_exact, val_exact] =
      fextract(sim.state_new_[0], sim.geom[0], 0, 0.5);

  // Main time loop
  sim.evolve();

  auto [position, values] = fextract(sim.state_new_[0], sim.geom[0], 0, 0.5);
  int nx = static_cast<int>(position.size());
  std::vector<double> xs = position;

  // compute error norm
  amrex::Real err_sq = 0.;
  for (int n = 0; n < RadhydroSimulation<WaveProblem>::ncompHydro_; ++n) {
    if (n == HydroSystem<WaveProblem>::internalEnergy_index) {
      continue;
    }
    amrex::Real dU_k = 0.;
    for (int i = 0; i < nx; ++i) {
      // Δ Uk = ∑i |Uk,in - Uk,i0| / Nx
      const amrex::Real U_k0 = val_exact.at(n)[i];
      const amrex::Real U_k1 = values.at(n)[i];
      dU_k += std::abs(U_k1 - U_k0) / static_cast<double>(nx);
    }
    // ε = || Δ U || = [&sum_k (Δ Uk)2]^{1/2}
    err_sq += dU_k * dU_k;
  }
  const amrex::Real epsilon = std::sqrt(err_sq);
  amrex::Print() << "rms of component-wise L1 error norms = " << epsilon << std::endl;

#ifdef HAVE_PYTHON
  // plot results
  if (amrex::ParallelDescriptor::IOProcessor()) {
    // extract values
    std::vector<double> d(nx);
    std::vector<double> vx(nx);
    std::vector<double> P(nx);

    for (int i = 0; i < nx; ++i) {
      amrex::Real rho =
          values.at(HydroSystem<WaveProblem>::density_index)[i];
      amrex::Real xmom =
          values.at(HydroSystem<WaveProblem>::x1Momentum_index)[i];
      amrex::Real Egas =
          values.at(HydroSystem<WaveProblem>::energy_index)[i];

      amrex::Real xvel = xmom / rho;
      amrex::Real Eint = Egas - xmom * xmom / (2.0 * rho);
      amrex::Real pressure = Eint * (HydroSystem<WaveProblem>::gamma_ - 1.);

      d.at(i) = (rho - rho0) / amp;
      vx.at(i) = (xvel - v0) / amp;
      P.at(i) = (pressure - P0) / amp;
    }

    std::vector<double> density_exact(nx);
    std::vector<double> velocity_exact(nx);
    std::vector<double> pressure_exact(nx);

    for (int i = 0; i < nx; ++i) {
      amrex::Real rho =
          val_exact.at(HydroSystem<WaveProblem>::density_index)[i];
      amrex::Real xmom =
          val_exact.at(HydroSystem<WaveProblem>::x1Momentum_index)[i];
      amrex::Real Egas =
          val_exact.at(HydroSystem<WaveProblem>::energy_index)[i];

      amrex::Real xvel = xmom / rho;
      amrex::Real Eint = Egas - xmom * xmom / (2.0 * rho);
      amrex::Real pressure = Eint * (HydroSystem<WaveProblem>::gamma_ - 1.);

      density_exact.at(i) = (rho - rho0) / amp;
      velocity_exact.at(i) = (xvel - v0) / amp;
      pressure_exact.at(i) = (pressure - P0) / amp;
    }

    // Plot results
    amrex::Real const t = sim.tNew_[0];

    std::map<std::string, std::string> d_args;
    std::map<std::string, std::string> dinit_args;
    std::map<std::string, std::string> dexact_args;
    d_args["label"] = "density";
    dinit_args["label"] = "density (initial)";

    matplotlibcpp::clf();
    matplotlibcpp::plot(xs, d, d_args);
    matplotlibcpp::plot(xs, density_exact, dinit_args);
    matplotlibcpp::legend();
    matplotlibcpp::title(fmt::format("t = {:.4f}", t));
    matplotlibcpp::save(fmt::format("./density_{:.4f}.pdf", t));

    std::map<std::string, std::string> P_args;
    std::map<std::string, std::string> Pinit_args;
    std::map<std::string, std::string> Pexact_args;
    P_args["label"] = "pressure";
    Pinit_args["label"] = "pressure (initial)";

    matplotlibcpp::clf();
    matplotlibcpp::plot(xs, P, P_args);
    matplotlibcpp::plot(xs, pressure_exact, Pinit_args);
    matplotlibcpp::legend();
    matplotlibcpp::title(fmt::format("t = {:.4f}", t));
    matplotlibcpp::save(fmt::format("./pressure_{:.4f}.pdf", t));

    std::map<std::string, std::string> v_args;
    std::map<std::string, std::string> vinit_args;
    std::map<std::string, std::string> vexact_args;
    v_args["label"] = "velocity";
    vinit_args["label"] = "velocity (initial)";

    matplotlibcpp::clf();
    matplotlibcpp::plot(xs, vx, v_args);
    matplotlibcpp::plot(xs, velocity_exact, vinit_args);
    matplotlibcpp::legend();
    matplotlibcpp::title(fmt::format("t = {:.4f}", t));
    matplotlibcpp::save(fmt::format("./velocity_{:.4f}.pdf", t));
  }
#endif

  const double err_tol = 1.0e-8; // for Nx = 100
  int status = 0;
  if (epsilon > err_tol) {
    status = 1;
  }

  return status;
}