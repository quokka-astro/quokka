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
#include "radiation_system.hpp"
#include "test_scalars.hpp"

using amrex::Real;

struct ScalarProblem {};

template <> struct HydroSystem_Traits<ScalarProblem> {
  static constexpr double gamma = 1.4;
  static constexpr bool reconstruct_eint = true;
};

template <> struct Physics_Traits<ScalarProblem> {
  static constexpr bool is_hydro_enabled = true;
  static constexpr bool is_radiation_enabled = false;
  static constexpr bool is_chemistry_enabled = false;

  static constexpr int numPassiveScalars = 1; // number of passive scalars
};

constexpr double v_contact = 2.0; // contact wave velocity

template <>
void RadhydroSimulation<ScalarProblem>::setInitialConditionsOnGrid(
    quokka::grid grid_elem) {

  // extract variables required from the geom object
  amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = grid_elem.dx;
  amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = grid_elem.prob_lo;
  const amrex::Box &indexRange = grid_elem.indexRange;
  const amrex::Array4<double>& state_cc = grid_elem.array;

  // loop over the grid and set the initial condition
  amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
    Real const x = prob_lo[0] + (i + Real(0.5)) * dx[0];
    double vx = NAN;
    double rho = NAN;
    double P = NAN;
    double scalar = NAN;

    if (x < 0.5) {
      rho = 1.4;
      vx = v_contact;
      P = 1.0;
      scalar = 1.0;
    } else {
      rho = 1.0;
      vx = v_contact;
      P = 1.0;
      scalar = 0.0;
    }

    const auto gamma = HydroSystem<ScalarProblem>::gamma_;
    for (int n = 0; n < state_cc.nComp(); ++n) {
      state_cc(i, j, k, n) = 0.;
    }
    state_cc(i, j, k, HydroSystem<ScalarProblem>::density_index) = rho;
    state_cc(i, j, k, HydroSystem<ScalarProblem>::x1Momentum_index) = rho * vx;
    state_cc(i, j, k, HydroSystem<ScalarProblem>::x2Momentum_index) = 0.;
    state_cc(i, j, k, HydroSystem<ScalarProblem>::x3Momentum_index) = 0.;
    state_cc(i, j, k, HydroSystem<ScalarProblem>::energy_index) =
        P / (gamma - 1.) + 0.5 * rho * (vx * vx);
    state_cc(i, j, k, HydroSystem<ScalarProblem>::internalEnergy_index) =
        P / (gamma - 1.);
    state_cc(i, j, k, HydroSystem<ScalarProblem>::scalar0_index) = scalar;
  });
}

template <>
void RadhydroSimulation<ScalarProblem>::computeReferenceSolution(
    amrex::MultiFab &ref, amrex::GpuArray<Real, AMREX_SPACEDIM> const &dx,
    amrex::GpuArray<Real, AMREX_SPACEDIM> const &prob_lo) {
  for (amrex::MFIter iter(ref); iter.isValid(); ++iter) {
    const amrex::Box &indexRange = iter.validbox();
    auto const &stateExact = ref.array(iter);
    auto const ncomp = ref.nComp();

    amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j,
                                                        int k) noexcept {
      Real const x = prob_lo[0] + (i + Real(0.5)) * dx[0];
      double vx = NAN;
      double rho = NAN;
      double P = NAN;
      double scalar = NAN;

      if (x < 0.5) {
        rho = 1.4;
        vx = v_contact;
        P = 1.0;
        scalar = 1.0;
      } else {
        rho = 1.0;
        vx = v_contact;
        P = 1.0;
        scalar = 0.0;
      }

      const auto gamma = HydroSystem<ScalarProblem>::gamma_;
      for (int n = 0; n < ncomp; ++n) {
        stateExact(i, j, k, n) = 0.;
      }
      stateExact(i, j, k, HydroSystem<ScalarProblem>::density_index) = rho;
      stateExact(i, j, k, HydroSystem<ScalarProblem>::x1Momentum_index) =
          rho * vx;
      stateExact(i, j, k, HydroSystem<ScalarProblem>::x2Momentum_index) = 0.;
      stateExact(i, j, k, HydroSystem<ScalarProblem>::x3Momentum_index) = 0.;
      stateExact(i, j, k, HydroSystem<ScalarProblem>::energy_index) =
          P / (gamma - 1.) + 0.5 * rho * (vx * vx);
      stateExact(i, j, k, HydroSystem<ScalarProblem>::internalEnergy_index) =
          P / (gamma - 1.);
      stateExact(i, j, k, HydroSystem<ScalarProblem>::scalar0_index) = scalar;
    });
  }

#ifdef HAVE_PYTHON
  // Plot results
  auto [position, values] = fextract(state_new_cc_[0], geom[0], 0, 0.5);
  auto [pos_exact, val_exact] = fextract(ref, geom[0], 0, 0.5);
  auto const nx = position.size();

  if (amrex::ParallelDescriptor::IOProcessor()) {
    std::vector<double> x;
    std::vector<double> d_final;
    std::vector<double> vx_final;
    std::vector<double> P_final;
    std::vector<double> s_final;
    std::vector<double> d_exact;
    std::vector<double> vx_exact;
    std::vector<double> P_exact;
    std::vector<double> s_exact;

    for (int i = 0; i < nx; ++i) {
      Real const this_x = position[i];
      x.push_back(this_x);

      {
        const auto rho =
            val_exact.at(HydroSystem<ScalarProblem>::density_index)[i];
        const auto xmom =
            val_exact.at(HydroSystem<ScalarProblem>::x1Momentum_index)[i];
        const auto E =
            val_exact.at(HydroSystem<ScalarProblem>::energy_index)[i];
        const auto s =
            val_exact.at(HydroSystem<ScalarProblem>::scalar0_index)[i];
        const auto vx = xmom / rho;
        const auto Eint = E - 0.5 * rho * (vx * vx);
        const auto P = (HydroSystem<ScalarProblem>::gamma_ - 1.) * Eint;
        d_exact.push_back(rho);
        vx_exact.push_back(vx);
        P_exact.push_back(P);
        s_exact.push_back(s);
      }

      {
        const auto frho =
            values.at(HydroSystem<ScalarProblem>::density_index)[i];
        const auto fxmom =
            values.at(HydroSystem<ScalarProblem>::x1Momentum_index)[i];
        const auto fE =
            values.at(HydroSystem<ScalarProblem>::energy_index)[i];
        const auto fs =
            values.at(HydroSystem<ScalarProblem>::scalar0_index)[i];
        const auto fvx = fxmom / frho;
        const auto fEint = fE - 0.5 * frho * (fvx * fvx);
        const auto fP = (HydroSystem<ScalarProblem>::gamma_ - 1.) * fEint;
        d_final.push_back(frho);
        vx_final.push_back(fvx);
        P_final.push_back(fP);
        s_final.push_back(fs);
      }
    }

    std::unordered_map<std::string, std::string> d_args;
    std::unordered_map<std::string, std::string> s_args;
    std::map<std::string, std::string> dexact_args;
    std::map<std::string, std::string> sexact_args;
    d_args["label"] = "density";
    d_args["color"] = "black";
    dexact_args["color"] = "black";
    s_args["label"] = "passive scalar";
    s_args["color"] = "blue";
    sexact_args["color"] = "blue";
    
    matplotlibcpp::scatter(x, d_final, 10.0, d_args);
    matplotlibcpp::plot(x, d_exact, dexact_args);
    matplotlibcpp::scatter(x, s_final, 10.0, s_args);
    matplotlibcpp::plot(x, s_exact, sexact_args);

    matplotlibcpp::legend();
    matplotlibcpp::title(fmt::format("t = {:.4f}", tNew_[0]));
    matplotlibcpp::save("./passive_scalar.pdf");
  }
#endif
}

auto problem_main() -> int {
  // Problem parameters
  const int nvars = RadhydroSimulation<ScalarProblem>::nvarTotal_;
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
  RadhydroSimulation<ScalarProblem> sim(boundaryConditions);
  
  sim.computeReferenceSolution_ = true;

  // initialize and evolve
  sim.setInitialConditions();
  sim.evolve();

  const double error_tol = 0.008;
  int status = 0;
  if (sim.errorNorm_ > error_tol) {
    status = 1;
  }

  // Cleanup and exit
  amrex::Print() << "Finished." << std::endl;
  return status;
}