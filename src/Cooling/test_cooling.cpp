//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file test_cooling.cpp
/// \brief Defines a test problem for SUNDIALS cooling.
///

#include <cvode/cvode.h>
#include <sundials/sundials_context.h>
#include <sundials/sundials_nonlinearsolver.h>
#include <sundials/sundials_nvector.h>
#include <sundials/sundials_types.h>
#include <sunnonlinsol/sunnonlinsol_fixedpoint.h>
#include <vector>

#include "AMReX_BC_TYPES.H"
#include "AMReX_BLassert.H"
#include "AMReX_FabArray.H"
#include "AMReX_MultiFab.H"
#include "AMReX_NVector_MultiFab.H"
#include "AMReX_ParallelContext.H"
#include "AMReX_ParallelDescriptor.H"
#include "AMReX_Sundials.H"

#include "RadhydroSimulation.hpp"
#include "fextract.hpp"
#include "hydro_system.hpp"
#include "test_cooling.hpp"

struct CoolingTest {
}; // dummy type to allow compile-type polymorphism via template specialization

constexpr double Egas0 = 1.0e2; // erg cm^-3
constexpr double rho0 = 1.0e-7; // g cm^-3

template <>
void RadhydroSimulation<CoolingTest>::setInitialConditionsAtLevel(int lev) {
  for (amrex::MFIter iter(state_old_[lev]); iter.isValid(); ++iter) {
    const amrex::Box &indexRange = iter.validbox();
    auto const &state = state_new_[lev].array(iter);

    amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
      state(i, j, k, RadSystem<CoolingTest>::radEnergy_index) = 0;
      state(i, j, k, RadSystem<CoolingTest>::x1RadFlux_index) = 0;
      state(i, j, k, RadSystem<CoolingTest>::x2RadFlux_index) = 0;
      state(i, j, k, RadSystem<CoolingTest>::x3RadFlux_index) = 0;

      state(i, j, k, RadSystem<CoolingTest>::gasEnergy_index) = Egas0;
      state(i, j, k, RadSystem<CoolingTest>::gasDensity_index) = rho0;
      state(i, j, k, RadSystem<CoolingTest>::x1GasMomentum_index) = 0;
      state(i, j, k, RadSystem<CoolingTest>::x2GasMomentum_index) = 0;
      state(i, j, k, RadSystem<CoolingTest>::x3GasMomentum_index) = 0;
    });
  }

  // set flag
  areInitialConditionsDefined_ = true;
}

template <> void RadhydroSimulation<CoolingTest>::computeAfterTimestep() {
  auto [position, values] = fextract(state_new_[0], Geom(0), 0, 0.5);

  if (amrex::ParallelDescriptor::IOProcessor()) {
    t_vec_.push_back(tNew_[0]);

    const amrex::Real Etot_i =
        values.at(HydroSystem<CoolingTest>::energy_index).at(0);
    const amrex::Real x1GasMom =
        values.at(HydroSystem<CoolingTest>::x1Momentum_index).at(0);
    const amrex::Real x2GasMom =
        values.at(HydroSystem<CoolingTest>::x2Momentum_index).at(0);
    const amrex::Real x3GasMom =
        values.at(HydroSystem<CoolingTest>::x3Momentum_index).at(0);
    const amrex::Real rho =
        values.at(HydroSystem<CoolingTest>::density_index).at(0);

    const amrex::Real Egas_i = RadSystem<CoolingTest>::ComputeEintFromEgas(
        rho, x1GasMom, x2GasMom, x3GasMom, Etot_i);

    Tgas_vec_.push_back(
        RadSystem<CoolingTest>::ComputeTgasFromEgas(rho, Egas_i));
  }
}

struct SundialsUserData {
  std::function<int(realtype, N_Vector, N_Vector, void *)> f;
};

static auto userdata_f(realtype t, N_Vector y_data, N_Vector y_rhs,
                       void *user_data) -> int {
  auto *udata = static_cast<SundialsUserData *>(user_data);
  return udata->f(t, y_data, y_rhs, user_data);
}

void rhs_cooling(amrex::MultiFab &S_rhs, amrex::MultiFab & /*S_data*/,
                 realtype /*t*/) {
  // compute cooling ODE right-hand side (== dy/dt) at time t
  for (amrex::MFIter iter(S_rhs); iter.isValid(); ++iter) {
    const amrex::Box &indexRange = iter.validbox();
    // auto const &state = S_data.const_array(iter);
    auto const &rhs = S_rhs.array(iter);

    amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
      rhs(i, j, k) = 1.0e-5;
    });
  }
}

void computeCooling(amrex::MultiFab &mf, amrex::Real dt) {
  // Initialise SUNContext
  SUNContext sundialsContext = nullptr;
  auto *mpi_comm = amrex::ParallelContext::CommunicatorSub();
  SUNContext_Create(mpi_comm, &sundialsContext);

  // Create MultiFab 'S_eint' with only gas internal energy
  const auto &ba = mf.boxArray();
  const auto &dmap = mf.DistributionMap();
  int nghost = mf.nGrow();
  amrex::MultiFab S_eint(ba, dmap, 1, nghost);

  // Copy gas internal energy to S_eint
  amrex::MultiFab::Copy(S_eint, mf, HydroSystem<CoolingTest>::energy_index, 0,
                        1, nghost);

  // Create an N_Vector wrapper for S_eint
  sunindextype length = S_eint.n_comp * S_eint.boxArray().numPts();
  N_Vector y_vec = amrex::sundials::N_VMake_MultiFab(length, &S_eint);

  // create CVode object
  void *cvode_mem = CVodeCreate(CV_ADAMS, sundialsContext);

  // create user data object
  SundialsUserData user_data;
  user_data.f = [&](realtype rhs_time, N_Vector y_data, N_Vector y_rhs, void *
                    /* user_data */) -> int {
    amrex::MultiFab S_data;
    amrex::MultiFab S_rhs;

    S_data =
        amrex::MultiFab(*amrex::sundials::getMFptr(y_data), amrex::make_alias,
                        0, amrex::sundials::getMFptr(y_data)->nComp());
    S_rhs =
        amrex::MultiFab(*amrex::sundials::getMFptr(y_rhs), amrex::make_alias, 0,
                        amrex::sundials::getMFptr(y_rhs)->nComp());

    rhs_cooling(S_rhs, S_data, rhs_time);
    return 0;
  };

  // set user data
  AMREX_ALWAYS_ASSERT(CVodeSetUserData(cvode_mem, &user_data) == CV_SUCCESS);

  // set RHS function and initial conditions t0, y0
  // (NOTE: CVODE allocates the rhs MultiFab itself!)
  AMREX_ALWAYS_ASSERT(CVodeInit(cvode_mem, userdata_f, 0, y_vec) == CV_SUCCESS);

  // set integration tolerances
  amrex::Real reltol = 1.0e-6;
  amrex::Real abstol = 1.0e-6;
  AMREX_ALWAYS_ASSERT(reltol > 0.);
  AMREX_ALWAYS_ASSERT(abstol > 0.); // CVODE requires this to be nonzero
  CVodeSStolerances(cvode_mem, reltol, abstol);

  // set nonlinear solver to fixed-point
  int m_accel = 0; // (optional) use Anderson acceleration with m_accel iterates
  SUNNonlinearSolver NLS =
      SUNNonlinSol_FixedPoint(y_vec, m_accel, sundialsContext);
  CVodeSetNonlinearSolver(cvode_mem, NLS);

  // solve ODEs
  realtype time_reached = NAN;
  int ierr = CVode(cvode_mem, dt, y_vec, &time_reached, CV_NORMAL);
  AMREX_ALWAYS_ASSERT_WITH_MESSAGE(ierr == CV_SUCCESS,
                                   "Cooling solve with CVODE failed!");

  // copy solution back to MultiFab 'mf'
  amrex::MultiFab::Copy(mf, S_eint, 0, HydroSystem<CoolingTest>::energy_index,
                        1, nghost);

  // free SUNDIALS objects
  N_VDestroy(y_vec);
  CVodeFree(&cvode_mem);
  SUNContext_Free(&sundialsContext);
}

auto problem_main() -> int {
  // Problem parameters
  const double CFL_number = 1.0;
  const double max_time = 1.0e-2; // s
  const int max_timesteps = 1e3;

  // Problem initialization
  constexpr int nvars = RadhydroSimulation<CoolingTest>::nvarTotal_;
  amrex::Vector<amrex::BCRec> boundaryConditions(nvars);
  for (int n = 0; n < nvars; ++n) {
    for (int i = 0; i < AMREX_SPACEDIM; ++i) {
      boundaryConditions[n].setLo(i, amrex::BCType::foextrap); // extrapolate
      boundaryConditions[n].setHi(i, amrex::BCType::foextrap);
    }
  }

  RadhydroSimulation<CoolingTest> sim(boundaryConditions);
  sim.is_hydro_enabled_ = true;
  sim.is_radiation_enabled_ = false;
  sim.cflNumber_ = CFL_number;
  sim.maxTimesteps_ = max_timesteps;
  sim.stopTime_ = max_time;
  sim.plotfileInterval_ = -1;

  // initialize
  sim.setInitialConditions();

  // compute cooling
  amrex::Real level_dt = 1.0;
  computeCooling(sim.state_new_[0], level_dt);

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
    if (!Tgas.empty()) {
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
