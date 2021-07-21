//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file test_hydro_shocktube.cpp
/// \brief Defines a test problem for a shock tube.
///

#include <cmath>

#include "AMReX_BLassert.H"

#include "RadhydroSimulation.hpp"
#include "fextract.hpp"
#include "hydro_system.hpp"
#include "test_hydro_vacuum.hpp"

struct ShocktubeProblem {};

template <> struct EOS_Traits<ShocktubeProblem> {
  static constexpr double gamma = 1.4;
};

template <>
void RadhydroSimulation<ShocktubeProblem>::setInitialConditionsAtLevel(
    int lev) {
  amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = geom[lev].CellSizeArray();
  amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo =
      geom[lev].ProbLoArray();

  for (amrex::MFIter iter(state_new_[lev]); iter.isValid(); ++iter) {
    const amrex::Box &indexRange = iter.validbox(); // excludes ghost zones
    auto const &state = state_new_[lev].array(iter);

    amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
      amrex::Real const x = prob_lo[0] + (i + amrex::Real(0.5)) * dx[0];
      double vx = NAN;
      double rho = NAN;
      double P = NAN;

      if (x < 0.5) {
        rho = 1.0;
        vx = -2.0;
        P = 0.4;
      } else {
        rho = 1.0;
        vx = 2.0;
        P = 0.4;
      }

      for (int n = 0; n < ncomp_; ++n) {
        state(i, j, k, n) = 0.;
      }

      auto const gamma = HydroSystem<ShocktubeProblem>::gamma_;
      state(i, j, k, HydroSystem<ShocktubeProblem>::density_index) = rho;
      state(i, j, k, HydroSystem<ShocktubeProblem>::x1Momentum_index) =
          rho * vx;
      state(i, j, k, HydroSystem<ShocktubeProblem>::x2Momentum_index) = 0.;
      state(i, j, k, HydroSystem<ShocktubeProblem>::x3Momentum_index) = 0.;
      state(i, j, k, HydroSystem<ShocktubeProblem>::energy_index) =
          P / (gamma - 1.) + 0.5 * rho * (vx * vx);
    });
  }

  // set flag
  areInitialConditionsDefined_ = true;
}

template <>
void RadhydroSimulation<ShocktubeProblem>::computeReferenceSolution(
    amrex::MultiFab &ref,
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx,
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_lo) {

  auto const box = geom[0].Domain();
  int nx = (box.hiVect3d()[0] - box.loVect3d()[0]) + 1;
  std::vector<double> xs(nx);
  for (int i = 0; i < nx; ++i) {
    xs.at(i) = prob_lo[0] + (i + amrex::Real(0.5)) * dx[0];
  }

  // read in exact solution
  std::vector<double> xs_exact;
  std::vector<double> density_exact;
  std::vector<double> pressure_exact;
  std::vector<double> velocity_exact;
  std::vector<double> eint_exact;

  std::string filename = "../extern/Toro/e1rpex.out";
  std::ifstream fstream(filename, std::ios::in);
  AMREX_ALWAYS_ASSERT(fstream.is_open());

  for (std::string line; std::getline(fstream, line);) {
    std::istringstream iss(line);
    std::vector<double> values;

    for (double value = NAN; iss >> value;) {
      values.push_back(value);
    }
    auto x = values.at(0);
    auto density = values.at(1);
    auto velocity = values.at(2);
    auto pressure = values.at(3);
    auto eint =
        pressure / ((HydroSystem<ShocktubeProblem>::gamma_ - 1.0) * density);

    xs_exact.push_back(x);
    density_exact.push_back(density);
    pressure_exact.push_back(pressure);
    velocity_exact.push_back(velocity);
    eint_exact.push_back(eint);
  }

  std::vector<double> density_exact_interp(xs.size());
  interpolate_arrays(xs.data(), density_exact_interp.data(), xs.size(),
                     xs_exact.data(), density_exact.data(), xs_exact.size());

  std::vector<double> velocity_exact_interp(xs.size());
  interpolate_arrays(xs.data(), velocity_exact_interp.data(), xs.size(),
                     xs_exact.data(), velocity_exact.data(), xs_exact.size());

  std::vector<double> pressure_exact_interp(xs.size());
  interpolate_arrays(xs.data(), pressure_exact_interp.data(), xs.size(),
                     xs_exact.data(), pressure_exact.data(), xs_exact.size());

  std::vector<double> eint_exact_interp(xs.size());
  interpolate_arrays(xs.data(), eint_exact_interp.data(), xs.size(),
                     xs_exact.data(), eint_exact.data(), xs_exact.size());

  // fill reference solution multifab
  for (amrex::MFIter iter(ref); iter.isValid(); ++iter) {
    const amrex::Box &indexRange = iter.validbox();
    auto const &stateExact = ref.array(iter);
    auto const ncomp = ref.nComp();

    amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j,
                                                        int k) noexcept {
      for (int n = 0; n < ncomp; ++n) {
        stateExact(i, j, k, n) = 0.;
      }
      amrex::Real rho = density_exact_interp.at(i);
      amrex::Real vx = velocity_exact_interp.at(i);
      amrex::Real P = pressure_exact_interp.at(i);

      const auto gamma = HydroSystem<ShocktubeProblem>::gamma_;
      stateExact(i, j, k, HydroSystem<ShocktubeProblem>::density_index) = rho;
      stateExact(i, j, k, HydroSystem<ShocktubeProblem>::x1Momentum_index) =
          rho * vx;
      stateExact(i, j, k, HydroSystem<ShocktubeProblem>::x2Momentum_index) = 0.;
      stateExact(i, j, k, HydroSystem<ShocktubeProblem>::x3Momentum_index) = 0.;
      stateExact(i, j, k, HydroSystem<ShocktubeProblem>::energy_index) =
          P / (gamma - 1.) + 0.5 * rho * (vx * vx);
    });
  }

  // Plot results
  auto [position, values] = fextract(state_new_[0], geom[0], 0, 0.5);
  auto [pos_exact, val_exact] = fextract(ref, geom[0], 0, 0.5);

  if (amrex::ParallelDescriptor::IOProcessor()) {
    // extract values
    std::vector<double> d(nx);
    std::vector<double> vx(nx);
    std::vector<double> e(nx);

    for (int i = 0; i < nx; ++i) {
      amrex::Real rho =
          values.at(HydroSystem<ShocktubeProblem>::density_index).at(i);
      amrex::Real xmom =
          values.at(HydroSystem<ShocktubeProblem>::x1Momentum_index).at(i);
      amrex::Real Egas =
          values.at(HydroSystem<ShocktubeProblem>::energy_index).at(i);

      amrex::Real xvel = xmom / rho;
      amrex::Real Eint = Egas - xmom * xmom / (2.0 * rho);
      amrex::Real eint = Eint / rho;

      d.at(i) = rho;
      vx.at(i) = xvel;
      e.at(i) = eint;
    }

    // Plot results
    matplotlibcpp::clf();
    std::map<std::string, std::string> d_args;
    std::map<std::string, std::string> dexact_args;
    d_args["label"] = "density";
    dexact_args["label"] = "density (exact solution)";
    matplotlibcpp::plot(xs, d, d_args);
    matplotlibcpp::plot(xs, density_exact_interp, dexact_args);
    matplotlibcpp::legend();
    matplotlibcpp::title(fmt::format("t = {:.4f}", tNew_[0]));
    matplotlibcpp::save(fmt::format("./hydro_vacuum_{:.4f}.pdf", tNew_[0]));

    // internal energy plot
    matplotlibcpp::clf();
    std::map<std::string, std::string> e_args;
    std::map<std::string, std::string> eexact_args;
    e_args["label"] = "specific internal energy";
    eexact_args["label"] = "exact solution";
    matplotlibcpp::plot(xs, e, e_args);
    matplotlibcpp::plot(xs, eint_exact_interp, eexact_args);
    matplotlibcpp::legend();
    matplotlibcpp::title(fmt::format("t = {:.4f}", tNew_[0]));
    matplotlibcpp::save(
        fmt::format("./hydro_vacuum_eint_{:.4f}.pdf", tNew_[0]));
  }
}

auto problem_main() -> int {
  // Problem parameters
  // const int nx = 100;
  // const double Lx = 1.0;
  const double CFL_number = 0.8;
  const double max_time = 0.15;
  const double max_dt = 1e-3;
  const int max_timesteps = 5000;

  // Problem initialization
  const int nvars = RadhydroSimulation<ShocktubeProblem>::nvarTotal_;
  amrex::Vector<amrex::BCRec> boundaryConditions(nvars);
  for (int n = 0; n < nvars; ++n) {
    boundaryConditions[0].setLo(0, amrex::BCType::foextrap); // extrapolate
    boundaryConditions[0].setHi(0, amrex::BCType::foextrap);
    for (int i = 1; i < AMREX_SPACEDIM; ++i) {
      boundaryConditions[n].setLo(i, amrex::BCType::int_dir); // periodic
      boundaryConditions[n].setHi(i, amrex::BCType::int_dir);
    }
  }

  RadhydroSimulation<ShocktubeProblem> sim(boundaryConditions);
  sim.is_hydro_enabled_ = true;
  sim.is_radiation_enabled_ = false;
  sim.cflNumber_ = CFL_number;
  sim.maxDt_ = max_dt;
  sim.stopTime_ = max_time;
  sim.maxTimesteps_ = max_timesteps;
  sim.computeReferenceSolution_ = true;
  sim.plotfileInterval_ = -1;

  // Main time loop
  sim.setInitialConditions();
  sim.evolve();

  // Compute test success condition
  int status = 0;
  const double error_tol = 0.015;
  if (sim.errorNorm_ > error_tol) {
    status = 1;
  }

  return status;
}