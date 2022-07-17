//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file test_hydro_contact.cpp
/// \brief Defines a test problem for a contact wave.
///

#include "AMReX_BC_TYPES.H"
#include "AMReX_BLassert.H"
#include "AMReX_MultiFab.H"
#include "AMReX_ParmParse.H"

#include "RadhydroSimulation.hpp"
#include "fextract.hpp"
#include "hydro_system.hpp"
#ifdef HAVE_PYTHON
#include "matplotlibcpp.h"
#endif
#include "radiation_system.hpp"
#include "test_hydro_highmach.hpp"
#include <fstream>
#include <unistd.h>

using amrex::Real;

struct HighMachProblem {};

template <> struct HydroSystem_Traits<HighMachProblem> {
  static constexpr double gamma = 5. / 3.;
  static constexpr bool reconstruct_eint = false;
  static constexpr int nscalars = 0;       // number of passive scalars
};

template <>
void RadhydroSimulation<HighMachProblem>::setInitialConditionsAtLevel(int lev) {
  int ncomp = ncomp_;
  amrex::GpuArray<Real, AMREX_SPACEDIM> dx = geom[lev].CellSizeArray();
  amrex::GpuArray<Real, AMREX_SPACEDIM> prob_lo = geom[lev].ProbLoArray();

  for (amrex::MFIter iter(state_new_[lev]); iter.isValid(); ++iter) {
    const amrex::Box &indexRange = iter.validbox(); // excludes ghost zones
    auto const &state = state_new_[lev].array(iter);

    amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
      Real const x = prob_lo[0] + (i + Real(0.5)) * dx[0];

      double norm = 1. / (2.0 * M_PI);
      double vx = norm * std::sin(2.0 * M_PI * x);
      double rho = 1.0;
      double P = 1.0e-10;

      AMREX_ASSERT(!std::isnan(vx));
      AMREX_ASSERT(!std::isnan(rho));
      AMREX_ASSERT(!std::isnan(P));

      const auto gamma = HydroSystem<HighMachProblem>::gamma_;
      for (int n = 0; n < ncomp; ++n) {
        state(i, j, k, n) = 0.;
      }

      state(i, j, k, HydroSystem<HighMachProblem>::density_index) = rho;
      state(i, j, k, HydroSystem<HighMachProblem>::x1Momentum_index) = rho * vx;
      state(i, j, k, HydroSystem<HighMachProblem>::x2Momentum_index) = 0.;
      state(i, j, k, HydroSystem<HighMachProblem>::x3Momentum_index) = 0.;
      state(i, j, k, HydroSystem<HighMachProblem>::energy_index) =
          P / (gamma - 1.) + 0.5 * rho * (vx * vx);
      state(i, j, k, HydroSystem<HighMachProblem>::internalEnergy_index) =
          P / (gamma - 1.);
    });
  }

  // set flag
  areInitialConditionsDefined_ = true;
}

template <>
void RadhydroSimulation<HighMachProblem>::computeReferenceSolution(
    amrex::MultiFab &ref, amrex::GpuArray<Real, AMREX_SPACEDIM> const &dx,
    amrex::GpuArray<Real, AMREX_SPACEDIM> const &prob_lo) {

  // extract solution
  auto [position, values] = fextract(state_new_[0], geom[0], 0, 0.5);
  auto const nx = position.size();
  std::vector<double> x;

  for (int i = 0; i < nx; ++i) {
    Real const this_x = position.at(i);
    x.push_back(this_x);
  }

  // read in exact solution
  std::vector<double> x_exact;
  std::vector<double> d_exact;
  std::vector<double> vx_exact;
  std::vector<double> P_exact;

  std::string filename = "../extern/highmach_reference.txt";
  std::ifstream fstream(filename, std::ios::in);
  AMREX_ALWAYS_ASSERT(fstream.is_open());
  std::string header;
  std::getline(fstream, header);

  for (std::string line; std::getline(fstream, line);) {
    std::istringstream iss(line);
    std::vector<double> values;

    for (double value = NAN; iss >> value;) {
      values.push_back(value);
    }
    Real x_val = values.at(0);
    Real d_val = values.at(1);
    Real vx_val = values.at(2);
    Real P_val = values.at(3);

    x_exact.push_back(x_val);
    d_exact.push_back(d_val);
    vx_exact.push_back(vx_val);
    P_exact.push_back(P_val);
  }

  // interpolate density onto mesh
  std::vector<double> d_interp(x.size());
  interpolate_arrays(x.data(), d_interp.data(), static_cast<int>(x.size()),
                     x_exact.data(), d_exact.data(),
                     static_cast<int>(x_exact.size()));

  // interpolate velocity onto mesh
  std::vector<double> vx_interp(x.size());
  interpolate_arrays(x.data(), vx_interp.data(), static_cast<int>(x.size()),
                     x_exact.data(), vx_exact.data(),
                     static_cast<int>(x_exact.size()));

  // interpolate pressure onto mesh
  std::vector<double> P_interp(x.size());
  interpolate_arrays(x.data(), P_interp.data(), static_cast<int>(x.size()),
                     x_exact.data(), P_exact.data(),
                     static_cast<int>(x_exact.size()));

  // save reference solution
  const Real gamma = HydroSystem<HighMachProblem>::gamma_;
  for (amrex::MFIter iter(ref); iter.isValid(); ++iter) {
    const amrex::Box &indexRange = iter.validbox(); // excludes ghost zones
    auto const &state = ref.array(iter);
    auto const ncomp = ref.nComp();

    amrex::LoopConcurrentOnCpu(indexRange, [=](int i, int j, int k) noexcept {
      for (int n = 0; n < ncomp; ++n) {
        state(i, j, k, n) = 0.;
      }

      Real rho = d_interp.at(i);
      Real vx = vx_interp.at(i);
      Real Pgas = P_interp.at(i);
      Real Eint = Pgas / (gamma - 1.);
      Real Etot = Eint + 0.5 * rho * (vx * vx);

      state(i, j, k, HydroSystem<HighMachProblem>::density_index) = rho;
      state(i, j, k, HydroSystem<HighMachProblem>::x1Momentum_index) = rho * vx;
      state(i, j, k, HydroSystem<HighMachProblem>::x2Momentum_index) = 0;
      state(i, j, k, HydroSystem<HighMachProblem>::x3Momentum_index) = 0;
      state(i, j, k, HydroSystem<HighMachProblem>::energy_index) = Etot;
      state(i, j, k, HydroSystem<HighMachProblem>::internalEnergy_index) = Eint;
    });
  }

  // save results to file
  if (amrex::ParallelDescriptor::IOProcessor()) {
    std::vector<double> d_final;
    std::vector<double> vx_final;
    std::vector<double> P_final;

    for (int i = 0; i < nx; ++i) {
      const auto frho =
          values.at(HydroSystem<HighMachProblem>::density_index).at(i);
      const auto fxmom =
          values.at(HydroSystem<HighMachProblem>::x1Momentum_index).at(i);
      const auto fE =
          values.at(HydroSystem<HighMachProblem>::energy_index).at(i);
      const auto fvx = fxmom / frho;
      const auto fEint = fE - 0.5 * frho * (fvx * fvx);
      const auto fP = (HydroSystem<HighMachProblem>::gamma_ - 1.) * fEint;

      d_final.push_back(frho);
      vx_final.push_back(fvx);
      P_final.push_back(fP);
    }

    // save solution values to csv file
    std::ofstream csvfile;
    csvfile.open("highmach_output.csv");
    csvfile << "# x density velocity pressure\n";
    for (int i = 0; i < nx; ++i) {
      csvfile << x.at(i) << " ";
      csvfile << d_final.at(i) << " ";
      csvfile << vx_final.at(i) << " ";
      csvfile << P_final.at(i) << "\n";
    }
    csvfile.close();

#ifdef HAVE_PYTHON
    // plot solution
    std::map<std::string, std::string> args_exact;
    std::unordered_map<std::string, std::string> args;
    args["color"] = "black";
    args_exact["color"] = "black";

    // density
    matplotlibcpp::scatter(x, d_final, 5.0, args);
    matplotlibcpp::plot(x_exact, d_exact, args_exact);
    matplotlibcpp::yscale("log");
    matplotlibcpp::ylim(0.1, 31.623);
    matplotlibcpp::title(fmt::format("density (t = {:.4f})", tNew_[0]));
    matplotlibcpp::save("./hydro_highmach_density.pdf");

    // velocity
    matplotlibcpp::clf();
    matplotlibcpp::scatter(x, vx_final, 5.0, args);
    matplotlibcpp::plot(x_exact, vx_exact, args_exact);
    matplotlibcpp::ylim(-0.3, 0.3);
    matplotlibcpp::title(fmt::format("velocity (t = {:.4f})", tNew_[0]));
    matplotlibcpp::save("./hydro_highmach_velocity.pdf");

    // pressure
    matplotlibcpp::clf();
    matplotlibcpp::scatter(x, P_final, 5.0, args);
    matplotlibcpp::plot(x_exact, P_exact, args_exact);
    matplotlibcpp::yscale("log");
    matplotlibcpp::ylim(1.0e-17, 1.0);
    matplotlibcpp::title(fmt::format("pressure (t = {:.4f})", tNew_[0]));
    matplotlibcpp::save("./hydro_highmach_pressure.pdf");
#endif
  }
}

auto problem_main() -> int {
  // Problem parameters
  const int nvars = RadhydroSimulation<HighMachProblem>::nvarTotal_;
  amrex::Vector<amrex::BCRec> boundaryConditions(nvars);
  for (int n = 0; n < nvars; ++n) {
    boundaryConditions[0].setLo(0, amrex::BCType::int_dir); // periodic
    boundaryConditions[0].setHi(0, amrex::BCType::int_dir);
    for (int i = 1; i < AMREX_SPACEDIM; ++i) {
      boundaryConditions[n].setLo(i, amrex::BCType::int_dir); // periodic
      boundaryConditions[n].setHi(i, amrex::BCType::int_dir);
    }
  }

  // Problem initialization
  RadhydroSimulation<HighMachProblem> sim(boundaryConditions, false);
  sim.is_hydro_enabled_ = true;
  sim.is_radiation_enabled_ = false;
  sim.computeReferenceSolution_ = true;

  // initialize and evolve
  sim.setInitialConditions();
  sim.evolve();

  const double error_tol = 0.25;
  int status = 0;
  if (sim.errorNorm_ > error_tol || std::isnan(sim.errorNorm_)) {
    status = 1;
  }

  // Cleanup and exit
  amrex::Print() << "Finished." << std::endl;
  return status;
}
