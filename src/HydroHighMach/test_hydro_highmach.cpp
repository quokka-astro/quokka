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
#include "matplotlibcpp.h"
#include "radiation_system.hpp"
#include "test_hydro_highmach.hpp"

using amrex::Real;

struct HighMachProblem {};

template <> struct EOS_Traits<HighMachProblem> {
  static constexpr double gamma = 5. / 3.;
  static constexpr bool reconstruct_eint = false;
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

#ifdef HAVE_PYTHON
  // Plot results
  auto [position, values] = fextract(state_new_[0], geom[0], 0, 0.5);
  auto const nx = position.size();

  if (amrex::ParallelDescriptor::IOProcessor()) {
    std::vector<double> x;
    std::vector<double> d_final;
    std::vector<double> vx_final;
    std::vector<double> P_final;

    for (int i = 0; i < nx; ++i) {
      Real const this_x = position.at(i);
      x.push_back(this_x);

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

    std::map<std::string, std::string> d_args;
    d_args["label"] = "density";
    d_args["color"] = "black";
    matplotlibcpp::plot(x, d_final, d_args);
    matplotlibcpp::yscale("log");
    matplotlibcpp::ylim(0.1, 31.623);
    matplotlibcpp::legend();
    matplotlibcpp::title(fmt::format("t = {:.4f}", tNew_[0]));
    matplotlibcpp::save("./hydro_highmach_density.pdf");

    matplotlibcpp::clf();
    std::map<std::string, std::string> vx_args;
    vx_args["label"] = "velocity";
    vx_args["color"] = "black";
    matplotlibcpp::plot(x, vx_final, vx_args);
    matplotlibcpp::ylim(-0.3, 0.3);
    matplotlibcpp::legend();
    matplotlibcpp::title(fmt::format("t = {:.4f}", tNew_[0]));
    matplotlibcpp::save("./hydro_highmach_velocity.pdf");

    matplotlibcpp::clf();
    std::map<std::string, std::string> P_args;
    P_args["label"] = "pressure";
    P_args["color"] = "black";
    matplotlibcpp::plot(x, P_final, P_args);
    matplotlibcpp::yscale("log");
    matplotlibcpp::ylim(1.0e-17, 1.0);
    matplotlibcpp::legend();
    matplotlibcpp::title(fmt::format("t = {:.4f}", tNew_[0]));
    matplotlibcpp::save("./hydro_highmach_pressure.pdf");
  }
#endif
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
  RadhydroSimulation<HighMachProblem> sim(boundaryConditions);
  sim.is_hydro_enabled_ = true;
  sim.is_radiation_enabled_ = false;
  sim.computeReferenceSolution_ = true;

  // initialize and evolve
  sim.setInitialConditions();
  sim.evolve();

  int status = 0;
  // Cleanup and exit
  amrex::Print() << "Finished." << std::endl;
  return status;
}