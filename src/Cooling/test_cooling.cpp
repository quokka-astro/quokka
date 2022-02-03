//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file test_cooling.cpp
/// \brief Defines a test problem for SUNDIALS cooling.
///

#include <cvode/cvode.h>
#include <nvector/nvector_manyvector.h>
#include <sundials/sundials_context.h>
#include <sundials/sundials_nonlinearsolver.h>
#include <sundials/sundials_nvector.h>
#include <sundials/sundials_types.h>
#include <sunnonlinsol/sunnonlinsol_fixedpoint.h>
#include <vector>

#include "AMReX_BC_TYPES.H"
#include "AMReX_BLassert.H"
#include "AMReX_MultiFab.H"
#include "AMReX_NVector_MultiFab.H"
#include "AMReX_ParallelContext.H"
#include "AMReX_ParallelDescriptor.H"
#include "AMReX_Sundials.H"

#include "RadhydroSimulation.hpp"
#include "fextract.hpp"
#include "hydro_system.hpp"
#include "test_cooling.hpp"

struct CouplingProblem {
}; // dummy type to allow compile-type polymorphism via template specialization

constexpr double Egas0 = 1.0e2; // erg cm^-3
constexpr double rho0 = 1.0e-7; // g cm^-3

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

struct SundialsUserData {
  std::function<int(realtype, N_Vector, N_Vector, void *)> f;
};

static auto f(realtype t, N_Vector y_data, N_Vector y_rhs, void *user_data)
    -> int {
  auto *udata = static_cast<SundialsUserData *>(user_data);
  return udata->f(t, y_data, y_rhs, user_data);
}

auto problem_main() -> int {
  // Problem parameters

  // const int nx = 4;
  // const double Lx = 1e5; // cm
  const double CFL_number = 1.0;
  const double max_time = 1.0e-2; // s
  const int max_timesteps = 1e3;

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
  // sim.constantDt_ = constant_dt;
  sim.maxTimesteps_ = max_timesteps;
  sim.stopTime_ = max_time;
  sim.plotfileInterval_ = -1;

  // Initialise SUNContext
  SUNContext sundialsContext = nullptr;
  auto *mpi_comm = amrex::ParallelContext::CommunicatorSub();
  SUNContext_Create(mpi_comm, &sundialsContext);

  // Create an N_Vector wrapper for the state MultiFab
  amrex::MultiFab &base_mf = sim.state_new_[0];
  int nComp = base_mf.n_comp;
  sunindextype length = nComp * base_mf.boxArray().numPts();
  N_Vector nv_sol = amrex::sundials::N_VMake_MultiFab(length, &base_mf);

  // initialize
  sim.setInitialConditions();

  // use CVODE to integrate the internal energy (ignoring other MultiFab
  // components)
  void *cvode_mem = CVodeCreate(CV_ADAMS, sundialsContext);
  amrex::Real time = 0.;
  amrex::Real dt = 1.0;
  N_Vector y_vec = nullptr;
  SundialsUserData user_data;

  user_data.f = [&](realtype rhs_time, N_Vector y_data, N_Vector y_rhs, void *
                    /* user_data */) -> int {
    amrex::Vector<amrex::MultiFab> S_data;
    amrex::Vector<amrex::MultiFab> S_rhs;

    const int num_vecs = N_VGetNumSubvectors_ManyVector(y_data);
    S_data.resize(num_vecs);
    S_rhs.resize(num_vecs);

    for (int i = 0; i < num_vecs; i++) {
      S_data.at(i) = amrex::MultiFab(
          *amrex::sundials::getMFptr(N_VGetSubvector_ManyVector(y_data, i)),
          amrex::make_alias, 0,
          amrex::sundials::getMFptr(N_VGetSubvector_ManyVector(y_data, i))
              ->nComp());
      S_rhs.at(i) = amrex::MultiFab(
          *amrex::sundials::getMFptr(N_VGetSubvector_ManyVector(y_rhs, i)),
          amrex::make_alias, 0,
          amrex::sundials::getMFptr(N_VGetSubvector_ManyVector(y_rhs, i))
              ->nComp());
    }

    // BaseT::post_update(S_data, rhs_time);
    // BaseT::rhs(S_rhs, S_data, rhs_time);

    return 0;
  };

  // set RHS function and initial conditions t0, y0
  AMREX_ALWAYS_ASSERT(CVodeInit(cvode_mem, f, time, y_vec) == CV_SUCCESS);

  // set integration tolerances
  amrex::Real reltol = 1.0e-6;
  // absolute tolerances should be set by consideration of the minimum physical
  // temperature
  amrex::Real abstol = 0;
  CVodeSStolerances(cvode_mem, reltol, abstol);

  // set nonlinear solver to fixed-point
  int m_accel = 0; // (optional) use Anderson acceleration
  SUNNonlinearSolver NLS =
      SUNNonlinSol_FixedPoint(y_vec, m_accel, sundialsContext);
  CVodeSetNonlinearSolver(cvode_mem, NLS);

  // solve ODEs
  realtype time_reached = NAN;
  int ierr = CVode(cvode_mem, time + dt, y_vec, &time_reached, CV_NORMAL);
  AMREX_ALWAYS_ASSERT_WITH_MESSAGE(ierr == CV_SUCCESS,
                                   "Cooling solve with CVODE failed!");

  // free SUNDIALS objects
  CVodeFree(&cvode_mem);
  SUNContext_Free(&sundialsContext);

  // plot results

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
    if (Tgas.size() > 0) {
      matplotlibcpp::ylim(0.1 * Tgas.front(), 10.0 * Tgas.back());
    }
    matplotlibcpp::legend();
    matplotlibcpp::xlabel("time t (s)");
    matplotlibcpp::ylabel("temperature T (K)");
    matplotlibcpp::save(fmt::format("./cooling.pdf"));
  }
#endif

  // Cleanup and exit
  amrex::Print() << "Finished." << std::endl;
  int status = 0;
  return status;
}
