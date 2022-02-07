//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file test_cooling.cpp
/// \brief Defines a test problem for SUNDIALS cooling.
///
#include <random>
#include <vector>

#include <cvode/cvode.h>
#include <sundials/sundials_context.h>
#include <sundials/sundials_nonlinearsolver.h>
#include <sundials/sundials_nvector.h>
#include <sundials/sundials_types.h>
#include <sunnonlinsol/sunnonlinsol_fixedpoint.h>

#include "AMReX_BC_TYPES.H"
#include "AMReX_BLProfiler.H"
#include "AMReX_BLassert.H"
#include "AMReX_FabArray.H"
#include "AMReX_GpuDevice.H"
#include "AMReX_MultiFab.H"
#include "AMReX_NVector_MultiFab.H"
#include "AMReX_ParallelContext.H"
#include "AMReX_ParallelDescriptor.H"
#include "AMReX_Sundials.H"
#include "AMReX_TableData.H"

#include "RadhydroSimulation.hpp"
#include "fextract.hpp"
#include "hydro_system.hpp"
#include "radiation_system.hpp"
#include "test_cooling.hpp"

using amrex::Real;

struct CoolingTest {
}; // dummy type to allow compile-type polymorphism via template specialization

constexpr double Tgas0 = 6000.; // K
constexpr double m_H = hydrogen_mass_cgs_;
constexpr double rho0 = 0.6 * m_H; // g cm^-3
constexpr double seconds_in_year = 3.154e7;

template <>
void RadhydroSimulation<CoolingTest>::setInitialConditionsAtLevel(int lev) {
  amrex::GpuArray<Real, AMREX_SPACEDIM> dx = geom[lev].CellSizeArray();
  amrex::GpuArray<Real, AMREX_SPACEDIM> prob_lo = geom[lev].ProbLoArray();
  amrex::GpuArray<Real, AMREX_SPACEDIM> prob_hi = geom[lev].ProbHiArray();
  Real const Lx = (prob_hi[0] - prob_lo[0]);
  Real const Ly = (prob_hi[1] - prob_lo[1]);
  Real const y0 = prob_lo[1] + 0.5 * (prob_hi[1] - prob_lo[1]);

  // perturbation parameters
  const int kmin = 0;
  const int kmax = 16;
  Real const A = 0.05 / kmax;

  // generate random phases
  amrex::Array<int, 2> tlo{kmin, kmin}; // lower bounds
  amrex::Array<int, 2> thi{kmax, kmax}; // upper bounds
  amrex::TableData<Real, 2> table_data(tlo, thi);
#ifdef AMREX_USE_GPU
  amrex::TableData<Real, 2> h_table_data(tlo, thi, The_Pinned_Arena());
  auto const &h_table = h_table_data.table();
#else
  auto const &h_table = table_data.table();
#endif
  // 64-bit Mersenne Twister (do not use 32-bit version!)
  std::mt19937_64 rng(1); // NOLINT
  std::uniform_real_distribution<double> sample_phase(0., 2.0 * M_PI);

  // Initialize data on the host
  for (int j = tlo[0]; j <= thi[0]; ++j) {
    for (int i = tlo[1]; i <= thi[1]; ++i) {
      h_table(i, j) = sample_phase(rng);
    }
  }

#ifdef AMREX_USE_GPU
  // Copy data to GPU memory
  table_data.copy(h_table_data);
  amrex::Gpu::streamSynchronize(); // not needed if the kernel using it is on
                                   // the same stream
#endif
  auto const &phase = table_data.const_table(); // const makes it read only

  for (amrex::MFIter iter(state_old_[lev]); iter.isValid(); ++iter) {
    const amrex::Box &indexRange = iter.validbox();
    auto const &state = state_new_[lev].array(iter);

    amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
      Real const x = prob_lo[0] + (i + Real(0.5)) * dx[0];
      Real const y = prob_lo[1] + (j + Real(0.5)) * dx[1];

      state(i, j, k, RadSystem<CoolingTest>::radEnergy_index) = 0;
      state(i, j, k, RadSystem<CoolingTest>::x1RadFlux_index) = 0;
      state(i, j, k, RadSystem<CoolingTest>::x2RadFlux_index) = 0;
      state(i, j, k, RadSystem<CoolingTest>::x3RadFlux_index) = 0;

      // compute perturbations
      Real delta_rho = 0;
      for (int ki = kmin; ki < kmax; ++ki) {
        for (int kj = kmin; kj < kmax; ++kj) {
          if ((ki == 0) && (kj == 0)) {
            continue;
          }
          Real const kx = 2.0 * M_PI * Real(ki) / Lx;
          Real const ky = 2.0 * M_PI * Real(kj) / Lx;
          delta_rho += A * std::sin(x * kx + y * ky + phase(ki, kj));
        }
      }
      AMREX_ALWAYS_ASSERT(delta_rho > -1.0);

      Real rho = 0.12 * m_H * (1.0 + delta_rho); // g cm^-3
      Real xmom = 0;
      Real ymom = 0;
      Real zmom = 0;
      Real const P = 4.0e4 * boltzmann_constant_cgs_; // erg cm^-3
      Real Eint = (HydroSystem<CoolingTest>::gamma_ - 1.) * P;
      //Real Eint = RadSystem<CoolingTest>::ComputeEgasFromTgas(rho, 1.0e4);

      Real const Egas = RadSystem<CoolingTest>::ComputeEgasFromEint(
          rho, xmom, ymom, zmom, Eint);

      state(i, j, k, RadSystem<CoolingTest>::gasEnergy_index) = Egas;
      state(i, j, k, RadSystem<CoolingTest>::gasDensity_index) = rho;
      state(i, j, k, RadSystem<CoolingTest>::x1GasMomentum_index) = xmom;
      state(i, j, k, RadSystem<CoolingTest>::x2GasMomentum_index) = ymom;
      state(i, j, k, RadSystem<CoolingTest>::x3GasMomentum_index) = zmom;
    });
  }

  // set flag
  areInitialConditionsDefined_ = true;
}

template <>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE void
AMRSimulation<CoolingTest>::setCustomBoundaryConditions(
    const amrex::IntVect &iv, amrex::Array4<amrex::Real> const &consVar,
    int /*dcomp*/, int /*numcomp*/, amrex::GeometryData const &geom,
    const amrex::Real /*time*/, const amrex::BCRec * /*bcr*/, int /*bcomp*/,
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

  amrex::Box const &box = geom.Domain();
  amrex::GpuArray<int, 3> lo = box.loVect3d();
  amrex::GpuArray<int, 3> hi = box.hiVect3d();

  if (j >= hi[1]) {
    // x2 upper boundary -- constant
    Real rho = rho0;
    Real xmom = 0;
    Real ymom = rho * (-26.0e5); // [-26 km/s]
    Real zmom = 0;
    Real Eint = RadSystem<CoolingTest>::ComputeEgasFromTgas(rho, Tgas0);
    Real const Egas = RadSystem<CoolingTest>::ComputeEgasFromEint(
        rho, xmom, ymom, zmom, Eint);

    consVar(i, j, k, RadSystem<CoolingTest>::gasDensity_index) = rho;
    consVar(i, j, k, RadSystem<CoolingTest>::x1GasMomentum_index) = xmom;
    consVar(i, j, k, RadSystem<CoolingTest>::x2GasMomentum_index) = ymom;
    consVar(i, j, k, RadSystem<CoolingTest>::x3GasMomentum_index) = zmom;
    consVar(i, j, k, RadSystem<CoolingTest>::gasEnergy_index) = Egas;
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

AMREX_GPU_HOST_DEVICE AMREX_INLINE auto cooling_function(Real const rho,
                                                         Real const T) -> Real {
  // use fitting function from Koyama & Inutsuka (2002)
  Real gamma_heat = 2.0e-26; // Koyama & Inutsuka value
  Real lambda_cool = gamma_heat * (1.0e7 * std::exp(-114800. / (T + 1000.)) +
                                   14. * std::sqrt(T) * std::exp(-92. / T));
  Real rho_over_mh = rho / m_H;
  Real cooling_source_term =
      rho_over_mh * gamma_heat - (rho_over_mh * rho_over_mh) * lambda_cool;
  return cooling_source_term;
}

void rhs_cooling(amrex::MultiFab &S_rhs, amrex::MultiFab &S_data,
                 amrex::MultiFab &hydro_state_mf, realtype /*t*/) {
  // compute cooling ODE right-hand side (== dy/dt) at time t

  auto const &Eint_arr = S_data.const_arrays();
  auto const &state = hydro_state_mf.const_arrays();
  auto const &rhs = S_rhs.arrays();

  amrex::ParallelFor(S_rhs, [=] AMREX_GPU_DEVICE(int box_no, int i, int j,
                                                 int k) noexcept {
    const Real Eint = Eint_arr[box_no](i, j, k);
    const Real rho =
        state[box_no](i, j, k, HydroSystem<CoolingTest>::density_index);
    const Real Tgas = RadSystem<CoolingTest>::ComputeTgasFromEgas(rho, Eint);

    rhs[box_no](i, j, k) = cooling_function(rho, Tgas);
  });
  amrex::Gpu::streamSynchronize();
}

void computeEintFromMultiFab(amrex::MultiFab &S_eint, amrex::MultiFab &mf) {
  // compute gas internal energy for each cell, save in S_eint
  auto const &Eint = S_eint.arrays();
  auto const &state = mf.const_arrays();

  amrex::ParallelFor(
      S_eint, [=] AMREX_GPU_DEVICE(int box_no, int i, int j, int k) noexcept {
        const Real rho =
            state[box_no](i, j, k, HydroSystem<CoolingTest>::density_index);
        const Real x1Mom =
            state[box_no](i, j, k, HydroSystem<CoolingTest>::x1Momentum_index);
        const Real x2Mom =
            state[box_no](i, j, k, HydroSystem<CoolingTest>::x2Momentum_index);
        const Real x3Mom =
            state[box_no](i, j, k, HydroSystem<CoolingTest>::x3Momentum_index);
        const Real Egas =
            state[box_no](i, j, k, HydroSystem<CoolingTest>::energy_index);

        Eint[box_no](i, j, k) = RadSystem<CoolingTest>::ComputeEintFromEgas(
            rho, x1Mom, x2Mom, x3Mom, Egas);
      });
  amrex::Gpu::streamSynchronize();
}

void updateEgasToMultiFab(amrex::MultiFab &S_eint, amrex::MultiFab &mf) {
  // copy solution back to MultiFab 'mf'
  auto const &Eint = S_eint.const_arrays();
  auto const &state = mf.arrays();

  amrex::ParallelFor(
      mf, [=] AMREX_GPU_DEVICE(int box_no, int i, int j, int k) noexcept {
        const Real rho =
            state[box_no](i, j, k, HydroSystem<CoolingTest>::density_index);
        const Real x1Mom =
            state[box_no](i, j, k, HydroSystem<CoolingTest>::x1Momentum_index);
        const Real x2Mom =
            state[box_no](i, j, k, HydroSystem<CoolingTest>::x2Momentum_index);
        const Real x3Mom =
            state[box_no](i, j, k, HydroSystem<CoolingTest>::x3Momentum_index);

        const Real Eint_new = Eint[box_no](i, j, k);

        state[box_no](i, j, k, HydroSystem<CoolingTest>::energy_index) =
            RadSystem<CoolingTest>::ComputeEgasFromEint(rho, x1Mom, x2Mom,
                                                        x3Mom, Eint_new);
      });
  amrex::Gpu::streamSynchronize();
}

void computeCooling(amrex::MultiFab &mf, Real dt, void *cvode_mem,
                    SUNContext sundialsContext) {
  BL_PROFILE("RadhydroSimulation::computeCooling()")

  // create CVode object
  cvode_mem = CVodeCreate(CV_ADAMS, sundialsContext);

  // Create MultiFab 'S_eint' with only gas internal energy
  const auto &ba = mf.boxArray();
  const auto &dmap = mf.DistributionMap();
  amrex::MultiFab S_eint(ba, dmap, 1, mf.nGrow());

  // Get gas internal energy from hydro state
  computeEintFromMultiFab(S_eint, mf);
  const Real Eint_min = S_eint.min(0);
  AMREX_ALWAYS_ASSERT(Eint_min > 0.);

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
  Real reltol = 1.0e-5;
  Real abstol = reltol * Eint_min;
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

  // free SUNDIALS objects
  N_VDestroy(y_vec);
  CVodeFree(&cvode_mem);
}

template <>
void RadhydroSimulation<CoolingTest>::computeAfterLevelAdvance(
    int lev, amrex::Real /*time*/, amrex::Real dt_lev, int /*iteration*/,
    int /*ncycle*/) {
  // compute operator split physics
  computeCooling(state_new_[lev], dt_lev, cvodeObject, sundialsContext);
}

auto problem_main() -> int {
  // Problem parameters
  const double CFL_number = 0.4;
  const double max_time = 5.0e4 * seconds_in_year; // 50 kyr
  const int max_timesteps = 2e4;

  // Problem initialization
  constexpr int nvars = RadhydroSimulation<CoolingTest>::nvarTotal_;
  amrex::Vector<amrex::BCRec> boundaryConditions(nvars);
  for (int n = 0; n < nvars; ++n) {
    boundaryConditions[n].setLo(0, amrex::BCType::int_dir); // periodic
    boundaryConditions[n].setHi(0, amrex::BCType::int_dir);
    for (int i = 1; i < AMREX_SPACEDIM; ++i) {
      boundaryConditions[n].setLo(i, amrex::BCType::foextrap); // extrapolate
      boundaryConditions[n].setHi(i, amrex::BCType::ext_dir);  // Dirichlet
    }
  }

  RadhydroSimulation<CoolingTest> sim(boundaryConditions);
  sim.is_hydro_enabled_ = true;
  sim.is_radiation_enabled_ = false;
  sim.reconstructionOrder_ = 2; // PLM
  sim.cflNumber_ = CFL_number;
  sim.maxTimesteps_ = max_timesteps;
  sim.stopTime_ = max_time;
  sim.plotfileInterval_ = 100;

  // Set initial conditions
  sim.setInitialConditions();

  // Initialise SUNContext
  SUNContext_Create(amrex::ParallelContext::CommunicatorSub(),
                    &sim.sundialsContext);

  // run simulation
  sim.evolve();

  // free sundials objects
  SUNContext_Free(&sim.sundialsContext);

  // Cleanup and exit
  int status = 0;
  return status;
}
