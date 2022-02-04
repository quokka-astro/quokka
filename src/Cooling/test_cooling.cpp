//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file test_cooling.cpp
/// \brief Defines a test problem for SUNDIALS cooling.
///
#include <vector>

#include <cvode/cvode.h>
#include <sundials/sundials_context.h>
#include <sundials/sundials_nonlinearsolver.h>
#include <sundials/sundials_nvector.h>
#include <sundials/sundials_types.h>
#include <sunnonlinsol/sunnonlinsol_fixedpoint.h>

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
#include "matplotlibcpp.h"
#include "radiation_system.hpp"
#include "test_cooling.hpp"

struct CoolingTest {
}; // dummy type to allow compile-type polymorphism via template specialization

constexpr double Tgas0 = 1.0e5; // K
constexpr double m_H = hydrogen_mass_cgs_;
constexpr double rho0 = 1.0e-3 * m_H; // g cm^-3
constexpr double seconds_in_year = 3.154e7;

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

      amrex::Real xmom = rho0 * 1.0e5;
      amrex::Real ymom = rho0 * 2.0e5;
      amrex::Real zmom = rho0 * 1.0e5;

      amrex::Real Eint0 =
          RadSystem<CoolingTest>::ComputeEgasFromTgas(rho0, Tgas0);
      amrex::Real Egas0 = RadSystem<CoolingTest>::ComputeEgasFromEint(
          rho0, xmom, ymom, zmom, Eint0);
      state(i, j, k, RadSystem<CoolingTest>::gasEnergy_index) = Egas0;
      state(i, j, k, RadSystem<CoolingTest>::gasDensity_index) = rho0;
      state(i, j, k, RadSystem<CoolingTest>::x1GasMomentum_index) = xmom;
      state(i, j, k, RadSystem<CoolingTest>::x2GasMomentum_index) = ymom;
      state(i, j, k, RadSystem<CoolingTest>::x3GasMomentum_index) = zmom;
    });
  }

  // set flag
  areInitialConditionsDefined_ = true;
}

template <> void RadhydroSimulation<CoolingTest>::computeAfterTimestep() {
  auto [position, values] = fextract(state_new_[0], Geom(0), 0, 0.5);

  if (amrex::ParallelDescriptor::IOProcessor()) {
    t_vec_.push_back(tNew_[0] / seconds_in_year);

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

auto cooling_function(amrex::Real const rho, amrex::Real const T) {
  // use fitting function from Koyama & Inutsuka (2002)
  amrex::Real gamma_heat = 2.0e-26; // Koyama & Inutsuka value
  amrex::Real lambda_cool =
      gamma_heat * (1.0e7 * std::exp(-114800. / (T + 1000.)) +
                    14. * std::sqrt(T) * std::exp(-92. / T));
  amrex::Real rho_over_mh = rho / m_H;
  amrex::Real cooling_source_term =
      rho_over_mh * gamma_heat - (rho_over_mh * rho_over_mh) * lambda_cool;
  return cooling_source_term;
}

void rhs_cooling(amrex::MultiFab &S_rhs, amrex::MultiFab &S_data,
                 amrex::MultiFab &hydro_state_mf, realtype /*t*/) {
  // compute cooling ODE right-hand side (== dy/dt) at time t
  for (amrex::MFIter iter(S_rhs); iter.isValid(); ++iter) {
    const amrex::Box &indexRange = iter.validbox();
    auto const &Eint_arr = S_data.const_array(iter);
    auto const &hydro_state = hydro_state_mf.const_array(iter);
    auto const &rhs = S_rhs.array(iter);

    amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
      const amrex::Real Eint = Eint_arr(i, j, k);
      const amrex::Real rho =
          hydro_state(i, j, k, HydroSystem<CoolingTest>::density_index);
      const amrex::Real Tgas =
          RadSystem<CoolingTest>::ComputeTgasFromEgas(rho, Eint);
      rhs(i, j, k) = cooling_function(rho, Tgas);
    });
  }
}

void computeEintFromMultiFab(amrex::MultiFab &S_eint, amrex::MultiFab &mf) {
  // compute gas internal energy for each cell, save in S_eint
  for (amrex::MFIter iter(S_eint); iter.isValid(); ++iter) {
    const amrex::Box &indexRange = iter.validbox();
    auto const &Eint = S_eint.array(iter);
    auto const &state = mf.const_array(iter);

    amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
      const amrex::Real rho =
          state(i, j, k, HydroSystem<CoolingTest>::density_index);
      const amrex::Real x1Mom =
          state(i, j, k, HydroSystem<CoolingTest>::x1Momentum_index);
      const amrex::Real x2Mom =
          state(i, j, k, HydroSystem<CoolingTest>::x2Momentum_index);
      const amrex::Real x3Mom =
          state(i, j, k, HydroSystem<CoolingTest>::x3Momentum_index);
      const amrex::Real Egas =
          state(i, j, k, HydroSystem<CoolingTest>::energy_index);

      Eint(i, j, k) = RadSystem<CoolingTest>::ComputeEintFromEgas(
          rho, x1Mom, x2Mom, x3Mom, Egas);
    });
  }
}

void updateEgasToMultiFab(amrex::MultiFab &S_eint, amrex::MultiFab &mf) {
  // copy solution back to MultiFab 'mf'
  for (amrex::MFIter iter(S_eint); iter.isValid(); ++iter) {
    const amrex::Box &indexRange = iter.validbox();
    auto const &Eint = S_eint.const_array(iter);
    auto const &state = mf.array(iter);

    amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
      const amrex::Real rho =
          state(i, j, k, HydroSystem<CoolingTest>::density_index);
      const amrex::Real x1Mom =
          state(i, j, k, HydroSystem<CoolingTest>::x1Momentum_index);
      const amrex::Real x2Mom =
          state(i, j, k, HydroSystem<CoolingTest>::x2Momentum_index);
      const amrex::Real x3Mom =
          state(i, j, k, HydroSystem<CoolingTest>::x3Momentum_index);
      const amrex::Real Eint_new = Eint(i, j, k);

      state(i, j, k, HydroSystem<CoolingTest>::energy_index) =
          RadSystem<CoolingTest>::ComputeEgasFromEint(rho, x1Mom, x2Mom, x3Mom,
                                                      Eint_new);
    });
  }
}

void computeCooling(amrex::MultiFab &mf, amrex::Real dt, void *cvode_mem,
                    SUNContext sundialsContext) {
  // Create MultiFab 'S_eint' with only gas internal energy
  const auto &ba = mf.boxArray();
  const auto &dmap = mf.DistributionMap();
  amrex::MultiFab S_eint(ba, dmap, 1, mf.nGrow());

  // Get gas internal energy from hydro state
  computeEintFromMultiFab(S_eint, mf);

  // Create an N_Vector wrapper for S_eint
  sunindextype length = S_eint.n_comp * S_eint.boxArray().numPts();
  N_Vector y_vec = amrex::sundials::N_VMake_MultiFab(length, &S_eint);

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

    rhs_cooling(S_rhs, S_data, mf, rhs_time);
    return 0;
  };

  // set user data
  AMREX_ALWAYS_ASSERT(CVodeSetUserData(cvode_mem, &user_data) == CV_SUCCESS);

  // set RHS function and initial conditions t0, y0
  // (NOTE: CVODE allocates the rhs MultiFab itself!)
  AMREX_ALWAYS_ASSERT(CVodeInit(cvode_mem, userdata_f, 0, y_vec) == CV_SUCCESS);

  // set integration tolerances
  amrex::Real reltol = 1.0e-6;
  amrex::Real abstol =
      reltol * RadSystem<CoolingTest>::ComputeEgasFromTgas(rho0, Tgas0);
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

  // update hydro state with new Eint
  updateEgasToMultiFab(S_eint, mf);

  // free N_Vec objects
  N_VDestroy(y_vec);
}

auto problem_main() -> int {
  // Problem parameters
  const double CFL_number = 1.0;
  const double max_time = 1.0e15; // s
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
  sim.computeAfterTimestep();

  // compute cooling time
  amrex::Real Eint0 = RadSystem<CoolingTest>::ComputeEgasFromTgas(rho0, Tgas0);
  amrex::Real t_cool = std::abs(Eint0 / cooling_function(rho0, Tgas0));
  amrex::Print() << "cooling time = " << t_cool << "\n\n";

  // Initialise SUNContext
  SUNContext sundialsContext = nullptr;
  auto *mpi_comm = amrex::ParallelContext::CommunicatorSub();
  SUNContext_Create(mpi_comm, &sundialsContext);

  // create CVode object
  void *cvode_mem = CVodeCreate(CV_ADAMS, sundialsContext);

  // compute cooling
  amrex::Real fixed_dt = 1.0e12; // s
  for (int i = 0; i < max_timesteps; ++i) {
    if (sim.tNew_[0] > max_time) {
      break;
    }
    computeCooling(sim.state_new_[0], fixed_dt, cvode_mem, sundialsContext);
    sim.tNew_[0] += fixed_dt;
    sim.computeAfterTimestep();
  }

  // free sundials objects
  CVodeFree(&cvode_mem);
  SUNContext_Free(&sundialsContext);

  if (amrex::ParallelDescriptor::IOProcessor()) {
    // Plot results
    std::vector<double> &Tgas = sim.Tgas_vec_;
    std::vector<double> &t = sim.t_vec_;

    amrex::Print() << "final temperature: " << Tgas.back() << "\n";

#ifdef HAVE_PYTHON
    std::map<std::string, std::string> Tgas_args;
    Tgas_args["label"] = "gas temperature (numerical)";
    matplotlibcpp::plot(t, Tgas, Tgas_args);

    matplotlibcpp::ylim(1.0e3, 1.0e5);
    matplotlibcpp::yscale("log");
    matplotlibcpp::xscale("log");
    matplotlibcpp::legend();
    matplotlibcpp::xlabel("time t (yr)");
    matplotlibcpp::ylabel("temperature T (K)");
    matplotlibcpp::title(fmt::format("density = {:.3e}", rho0 / m_H));
    matplotlibcpp::tight_layout();
    matplotlibcpp::save(fmt::format("./cooling.pdf"));
#endif
  }

  // Cleanup and exit
  amrex::Print() << "Finished." << std::endl;
  int status = 0;
  return status;
}
