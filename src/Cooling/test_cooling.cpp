//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file test_cooling.cpp
/// \brief Defines a test problem for SUNDIALS cooling.
///

#include <vector>

#include "AMReX_BC_TYPES.H"
#include "AMReX_ParallelDescriptor.H"
#include "AMReX_Sundials.H"

#include "RadhydroSimulation.hpp"
#include "fextract.hpp"
#include "hydro_system.hpp"
#include "test_cooling.hpp"

struct CouplingProblem {
}; // dummy type to allow compile-type polymorphism via template specialization

constexpr double Egas0 = 1.0e2;  // erg cm^-3
constexpr double rho0 = 1.0e-7;  // g cm^-3

template <>
void RadhydroSimulation<CouplingProblem>::setInitialConditionsAtLevel(int lev) {
  for (amrex::MFIter iter(state_old_[lev]); iter.isValid(); ++iter) {
    const amrex::Box &indexRange = iter.validbox();
    auto const &state = state_new_[lev].array(iter);

    amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
      state(i, j, k, RadSystem<CouplingProblem>::radEnergy_index) = 0;
      state(i, j, k, RadSystem<CouplingProblem>::x1RadFlux_index) = 0;
      state(i, j, k, RadSystem<CouplingProblem>::x2RadFlux_index) = 0;
      state(i, j, k, RadSystem<CouplingProblem>::x3RadFlux_index) = 0;

      state(i, j, k, RadSystem<CouplingProblem>::gasEnergy_index) = Egas0;
      state(i, j, k, RadSystem<CouplingProblem>::gasDensity_index) = rho0;
      state(i, j, k, RadSystem<CouplingProblem>::x1GasMomentum_index) = 0;
      state(i, j, k, RadSystem<CouplingProblem>::x2GasMomentum_index) = 0;
      state(i, j, k, RadSystem<CouplingProblem>::x3GasMomentum_index) = 0;
    });
  }

  // set flag
  areInitialConditionsDefined_ = true;
}

template <> void RadhydroSimulation<CouplingProblem>::computeAfterTimestep() {
  auto [position, values] = fextract(state_new_[0], Geom(0), 0, 0.5);

  if (amrex::ParallelDescriptor::IOProcessor()) {
    t_vec_.push_back(tNew_[0]);

    const amrex::Real Etot_i =
        values.at(HydroSystem<CouplingProblem>::energy_index).at(0);
    const amrex::Real x1GasMom =
        values.at(HydroSystem<CouplingProblem>::x1Momentum_index).at(0);
    const amrex::Real x2GasMom =
        values.at(HydroSystem<CouplingProblem>::x2Momentum_index).at(0);
    const amrex::Real x3GasMom =
        values.at(HydroSystem<CouplingProblem>::x3Momentum_index).at(0);
    const amrex::Real rho =
        values.at(HydroSystem<CouplingProblem>::density_index).at(0);

    const amrex::Real Egas_i = RadSystem<CouplingProblem>::ComputeEintFromEgas(
        rho, x1GasMom, x2GasMom, x3GasMom, Etot_i);

    Tgas_vec_.push_back(
        RadSystem<CouplingProblem>::ComputeTgasFromEgas(rho, Egas_i));
  }
}

auto problem_main() -> int {
  // Problem parameters

  // const int nx = 4;
  // const double Lx = 1e5; // cm
  const double CFL_number = 1.0;
  const double max_time = 1.0e-2; // s
  const int max_timesteps = 1e3;
  const double constant_dt = 1.0e-8; // s

  // Problem initialization
  constexpr int nvars = RadhydroSimulation<CouplingProblem>::nvarTotal_;
  amrex::Vector<amrex::BCRec> boundaryConditions(nvars);
  for (int n = 0; n < nvars; ++n) {
    for (int i = 0; i < AMREX_SPACEDIM; ++i) {
      boundaryConditions[n].setLo(i, amrex::BCType::foextrap); // extrapolate
      boundaryConditions[n].setHi(i, amrex::BCType::foextrap);
    }
  }

  RadhydroSimulation<CouplingProblem> sim(boundaryConditions);
  sim.is_hydro_enabled_ = true;
  sim.is_radiation_enabled_ = false;
  sim.cflNumber_ = CFL_number;
  //sim.constantDt_ = constant_dt;
  sim.maxTimesteps_ = max_timesteps;
  sim.stopTime_ = max_time;
  sim.plotfileInterval_ = -1;

  // initialize
  sim.setInitialConditions();

  // evolve
  sim.evolve();

  // copy solution slice to vector
  int status = 0;

#ifdef HAVE_PYTHON
  if (amrex::ParallelDescriptor::IOProcessor()) {
    // Plot results
    std::vector<double> &Tgas = sim.Tgas_vec_;
    std::vector<double> &t = sim.t_vec_;

    std::map<std::string, std::string> Tgas_args;
    Tgas_args["label"] = "gas temperature (numerical)";
    matplotlibcpp::plot(t, Tgas, Tgas_args);

    matplotlibcpp::yscale("log");
    matplotlibcpp::xscale("log");
    matplotlibcpp::ylim(0.1 * std::min(Tgas.front(), Trad.front()),
                        10.0 * std::max(Trad.back(), Tgas.back()));
    matplotlibcpp::legend();
    matplotlibcpp::xlabel("time t (s)");
    matplotlibcpp::ylabel("temperature T (K)");
    matplotlibcpp::save(fmt::format("./cooling.pdf"));

    std::vector<double> frac_err(t.size());
    for (int i = 0; i < t.size(); ++i) {
      frac_err.at(i) = Tgas_exact_interp.at(i) / Tgas.at(i) - 1.0;
    }
    matplotlibcpp::clf();
    matplotlibcpp::plot(t, frac_err);
    matplotlibcpp::xlabel("time t (s)");
    matplotlibcpp::ylabel("fractional error in gas temperature");
    matplotlibcpp::save(fmt::format("./cooling_error.pdf"));
  }
#endif

  // Cleanup and exit
  amrex::Print() << "Finished." << std::endl;
  return status;
}
