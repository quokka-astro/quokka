//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file test_hydro_shocktube.cpp
/// \brief Defines a test problem for a shock tube.
///

#include <cmath>
#include <string>
#include <unordered_map>

#include "AMReX_BC_TYPES.H"
#include "AMReX_GpuQualifiers.H"
#include "ArrayUtil.hpp"
#include "RadhydroSimulation.hpp"
#include "fextract.hpp"
#include "hydro_system.hpp"
#include "matplotlibcpp.h"
#include "radiation_system.hpp"
#include "test_hydro_highmach.hpp"
#include "valarray.hpp"

double get_array(amrex::Array4<const amrex::Real> const &a, int i, int j, int k,
                 int n) {
  return a(i, j, k, n);
}

struct ShocktubeProblem {};

template <> struct EOS_Traits<ShocktubeProblem> {
  static constexpr double gamma = 1.4;
  static constexpr bool reconstruct_eint = false;
};

AMREX_GPU_MANAGED static amrex::Real rho0 = 1.0;
AMREX_GPU_MANAGED static amrex::Real v0 = 1.0;
AMREX_GPU_MANAGED static amrex::Real E0 = 4.0e-6;

AMREX_GPU_DEVICE AMREX_FORCE_INLINE auto
ExactSolution(int i, double t, double prob_lo, double dx)
    -> quokka::valarray<double, 3> {
  const auto gamma = HydroSystem<ShocktubeProblem>::gamma_;

  // compute point values of primitive variables
  auto primVal = [=](int i, double t) {
    const double x = prob_lo + (i + 0.5) * dx;
    const double rho = rho0 / (1. + v0 * t);
    const double vx = v0 * x / (1. + v0 * t);
    const double Eint = E0 / std::pow(1. + v0 * t, gamma);
    quokka::valarray<double, 3> vars = {rho, vx, Eint};
    return vars;
  };

  // compute point values of conserved variables
  auto consVal = [=](quokka::valarray<double, 3> primVars) {
    const double rho = primVars[0];
    const double vx = primVars[1];
    const double Eint = primVars[2];
    quokka::valarray<double, 3> vars = {rho, rho * vx,
                                        Eint + 0.5 * rho * vx * vx};
    return vars;
  };

  // compute cell-averaged values of variables
  auto q0 = consVal(primVal(i, t));
  auto del_sq =
      consVal(primVal(i - 1, t)) - 2.0 * q0 + consVal(primVal(i + 1, t));
  quokka::valarray<double, 3> avg = q0 + del_sq / 24.;
  return avg;
}

template <>
void RadhydroSimulation<ShocktubeProblem>::setInitialConditionsAtLevel(
    int lev) {
  amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = geom[lev].CellSizeArray();
  amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo =
      geom[lev].ProbLoArray();
  const int ncomp = ncomp_;

  for (amrex::MFIter iter(state_new_[lev]); iter.isValid(); ++iter) {
    const amrex::Box &indexRange = iter.validbox(); // excludes ghost zones
    auto const &state = state_new_[lev].array(iter);

    amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
      auto const avg = ExactSolution(i, 0., prob_lo[0], dx[0]);

      for (int n = 0; n < ncomp; ++n) {
        state(i, j, k, n) = 0.;
      }
      state(i, j, k, HydroSystem<ShocktubeProblem>::density_index) = avg[0];
      state(i, j, k, HydroSystem<ShocktubeProblem>::x1Momentum_index) = avg[1];
      state(i, j, k, HydroSystem<ShocktubeProblem>::x2Momentum_index) = 0.;
      state(i, j, k, HydroSystem<ShocktubeProblem>::x3Momentum_index) = 0.;
      state(i, j, k, HydroSystem<ShocktubeProblem>::energy_index) = avg[2];
    });
  }

  // set flag
  areInitialConditionsDefined_ = true;
}

template <>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE void
AMRSimulation<ShocktubeProblem>::setCustomBoundaryConditions(
    const amrex::IntVect &iv, amrex::Array4<amrex::Real> const &consVar,
    int /*dcomp*/, int numcomp, amrex::GeometryData const &geom,
    const amrex::Real time, const amrex::BCRec * /*bcr*/, int /*bcomp*/,
    int /*orig_comp*/) {
#if (AMREX_SPACEDIM == 1)
  auto i = iv.toArray()[0];
  int j = 0;
  int k = 0;
#endif
#if (AMREX_SPACEDIM == 2)
  auto [i, j] = iv.toArray();
  int k = 0;
#endif
#if (AMREX_SPACEDIM == 3)
  auto [i, j, k] = iv.toArray();
#endif

  const double prob_lo = geom.ProbLo(0);
  const double dx = geom.CellSize(0);

  auto const avg = ExactSolution(i, time, prob_lo, dx);

  consVar(i, j, k, HydroSystem<ShocktubeProblem>::density_index) = avg[0];
  consVar(i, j, k, HydroSystem<ShocktubeProblem>::x1Momentum_index) = avg[1];
  consVar(i, j, k, HydroSystem<ShocktubeProblem>::x2Momentum_index) = 0.;
  consVar(i, j, k, HydroSystem<ShocktubeProblem>::x3Momentum_index) = 0.;
  consVar(i, j, k, HydroSystem<ShocktubeProblem>::energy_index) = avg[2];

  consVar(i, j, k, RadSystem<ShocktubeProblem>::radEnergy_index) = 0;
  consVar(i, j, k, RadSystem<ShocktubeProblem>::x1RadFlux_index) = 0;
  consVar(i, j, k, RadSystem<ShocktubeProblem>::x2RadFlux_index) = 0;
  consVar(i, j, k, RadSystem<ShocktubeProblem>::x3RadFlux_index) = 0;
}

template <>
void RadhydroSimulation<ShocktubeProblem>::computeReferenceSolution(
    amrex::MultiFab &ref,
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx,
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_lo) {

  const double t = stopTime_;

  // fill reference solution multifab
  for (amrex::MFIter iter(ref); iter.isValid(); ++iter) {
    const amrex::Box &indexRange = iter.validbox();
    auto const &stateExact = ref.array(iter);
    auto const ncomp = ref.nComp();

    amrex::ParallelFor(indexRange, [=](int i, int j, int k) noexcept {
      auto const avg = ExactSolution(i, t, prob_lo[0], dx[0]);

      for (int n = 0; n < ncomp; ++n) {
        stateExact(i, j, k, n) = 0.;
      }
      stateExact(i, j, k, HydroSystem<ShocktubeProblem>::density_index) =
          avg[0];
      stateExact(i, j, k, HydroSystem<ShocktubeProblem>::x1Momentum_index) =
          avg[1];
      stateExact(i, j, k, HydroSystem<ShocktubeProblem>::x2Momentum_index) = 0.;
      stateExact(i, j, k, HydroSystem<ShocktubeProblem>::x3Momentum_index) = 0.;
      stateExact(i, j, k, HydroSystem<ShocktubeProblem>::energy_index) = avg[2];
    });
  }

#ifdef HAVE_PYTHON

  // Plot results
  auto [position, values] = fextract(state_new_[0], geom[0], 0, 0.5);
  auto [pos_exact, val_exact] = fextract(ref, geom[0], 0, 0.5);
  auto const nx = position.size();

  if (amrex::ParallelDescriptor::IOProcessor()) {
    std::vector<double> x;
    std::vector<double> d_final;
    std::vector<double> vx_final;
    std::vector<double> Eint_final;
    std::vector<double> d_exact;
    std::vector<double> vx_exact;
    std::vector<double> Eint_exact;

    for (int i = 0; i < nx; ++i) {
      amrex::Real const this_x = position.at(i);
      x.push_back(this_x);

      {
        const auto rho =
            val_exact.at(HydroSystem<ShocktubeProblem>::density_index).at(i);
        const auto xmom =
            val_exact.at(HydroSystem<ShocktubeProblem>::x1Momentum_index).at(i);
        const auto E =
            val_exact.at(HydroSystem<ShocktubeProblem>::energy_index).at(i);
        const auto vx = xmom / rho;
        const auto Eint = E - 0.5 * rho * (vx * vx);
        d_exact.push_back(rho);
        vx_exact.push_back(vx);
        Eint_exact.push_back(Eint);
      }

      {
        const auto frho =
            values.at(HydroSystem<ShocktubeProblem>::density_index).at(i);
        const auto fxmom =
            values.at(HydroSystem<ShocktubeProblem>::x1Momentum_index).at(i);
        const auto fE =
            values.at(HydroSystem<ShocktubeProblem>::energy_index).at(i);
        const auto fvx = fxmom / frho;
        const auto fEint = fE - 0.5 * frho * (fvx * fvx);
        d_final.push_back(frho);
        vx_final.push_back(fvx);
        Eint_final.push_back(fEint);
      }
    }

    std::unordered_map<std::string, std::string> d_args;
    std::map<std::string, std::string> dexact_args;
    d_args["label"] = "density";
    d_args["color"] = "black";
    dexact_args["label"] = "density (exact)";
    matplotlibcpp::scatter(x, d_final, 10.0, d_args);
    matplotlibcpp::plot(x, d_exact, dexact_args);
    matplotlibcpp::legend();
    matplotlibcpp::title(fmt::format("t = {:.4f}", tNew_[0]));
    matplotlibcpp::save("./hydro_highmach.pdf");

    std::unordered_map<std::string, std::string> e_args;
    std::map<std::string, std::string> eexact_args;
    d_args["label"] = "internal energy";
    d_args["color"] = "black";
    dexact_args["label"] = "internal energy (exact)";
    matplotlibcpp::clf();
    matplotlibcpp::scatter(x, Eint_final, 10.0, d_args);
    matplotlibcpp::plot(x, Eint_exact, dexact_args);
    matplotlibcpp::legend();
    matplotlibcpp::title(fmt::format("t = {:.4f}", tNew_[0]));
    matplotlibcpp::save("./hydro_highmach_eint.pdf");
  }
#endif
}

auto problem_main() -> int {
  // Problem parameters
  // const int nx = 64;
  // const double Lx = 2.0;
  const double CFL_number = 0.1;
  const double max_time = 1.0;
  const int max_timesteps = 1000;

  // Problem initialization
  const int nvars = RadhydroSimulation<ShocktubeProblem>::nvarTotal_;
  amrex::Vector<amrex::BCRec> boundaryConditions(nvars);
  for (int n = 0; n < nvars; ++n) {
    boundaryConditions[0].setLo(0, amrex::BCType::int_dir); // periodic
    boundaryConditions[0].setHi(0, amrex::BCType::int_dir);
    for (int i = 1; i < AMREX_SPACEDIM; ++i) {
      boundaryConditions[n].setLo(i, amrex::BCType::int_dir); // periodic
      boundaryConditions[n].setHi(i, amrex::BCType::int_dir);
    }
  }

  RadhydroSimulation<ShocktubeProblem> sim(boundaryConditions);
  sim.is_hydro_enabled_ = true;
  sim.is_radiation_enabled_ = false;
  sim.cflNumber_ = CFL_number;
  sim.reconstructionOrder_ = 3;
  sim.stopTime_ = max_time;
  sim.maxTimesteps_ = max_timesteps;
  sim.computeReferenceSolution_ = true;
  sim.plotfileInterval_ = -1;

  // Main time loop
  sim.setInitialConditions();
  sim.evolve();

  // Compute test success condition
  int status = 0;
  const double error_tol = 1.0e-4;
  if (sim.errorNorm_ > error_tol) {
    status = 1;
  }
  return status;
}
