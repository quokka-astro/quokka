//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file test_hydro3d_blast.cpp
/// \brief Defines a test problem for a 3D explosion.
///

#include <limits>

#include "AMReX_Array.H"
#include "AMReX_Array4.H"
#include "AMReX_BC_TYPES.H"
#include "AMReX_BLassert.H"
#include "AMReX_Config.H"
#include "AMReX_Extension.H"
#include "AMReX_FArrayBox.H"
#include "AMReX_FabArrayUtility.H"
#include "AMReX_Loop.H"
#include "AMReX_ParallelDescriptor.H"
#include "AMReX_ParmParse.H"
#include "AMReX_Print.H"
#include "AMReX_REAL.H"

#include "RadhydroSimulation.hpp"
#include "hydro_system.hpp"
#include "radiation_system.hpp"
#include "test_radhydro_shell.hpp"

struct ShellProblem {};

constexpr double a_rad = 7.5646e-15;  // erg cm^-3 K^-4
constexpr double c = 2.99792458e10;   // cm s^-1
constexpr double cs0 = 0.633e5;       // (0.633 km/s) [cm s^-1]
constexpr double a0 = 2.0e5;          // ('reference' sound speed) [cm s^-1]
//constexpr double chat = c;            // cm s^-1
constexpr double chat = 1720. * a0;    // cm s^-1
constexpr double k_B = 1.380658e-16;  // erg K^-1
constexpr double m_H = 1.6726231e-24; // mass of hydrogen atom [g]
constexpr double gamma_gas = 5. / 3.; // monoatomic ideal gas

template <> struct RadSystem_Traits<ShellProblem> {
  static constexpr double c_light = c;
  static constexpr double c_hat = chat;
  static constexpr double radiation_constant = a_rad;
  static constexpr double mean_molecular_mass = 2.2 * m_H;
  static constexpr double boltzmann_constant = k_B;
  static constexpr double gamma = gamma_gas;
  static constexpr double Erad_floor = 0.;
  static constexpr bool compute_v_over_c_terms = true;
};

template <> struct EOS_Traits<ShellProblem> {
  static constexpr double gamma = gamma_gas;
};

constexpr amrex::Real Msun = 2.0e33;           // g
constexpr amrex::Real parsec_in_cm = 3.086e18; // cm

constexpr amrex::Real specific_luminosity = 2000.;             // erg s^-1 g^-1
constexpr amrex::Real GMC_mass = 1.0e6 * Msun;                 // g
constexpr amrex::Real epsilon = 0.5;                           // dimensionless
constexpr amrex::Real M_shell = (1 - epsilon) * GMC_mass;      // g
constexpr amrex::Real L_star = GMC_mass * specific_luminosity; // erg s^-1

constexpr amrex::Real r_0 = 5.0 * parsec_in_cm; // cm
constexpr amrex::Real sigma_star = 0.0625 * r_0;
constexpr amrex::Real H_shell = 0.1 * r_0; // cm

constexpr amrex::Real rho_0 =
    M_shell / ((4. / 3.) * M_PI * r_0 * r_0 * r_0);          // g cm^-3
constexpr amrex::Real P_0 = gamma_gas * rho_0 * (cs0 * cs0); // erg cm^-3
constexpr double c_v = k_B / ((2.2 * m_H) * (gamma_gas - 1.0));

constexpr amrex::Real kappa0 = 20.0; // specific opacity [cm^2 g^-1]
//constexpr amrex::Real tau0 = M_shell * kappa0 / (4.0 * M_PI * r_0 * r_0);

template <>
void RadSystem<ShellProblem>::SetRadEnergySource(
    array_t &radEnergy, const amrex::Box &indexRange,
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx,
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_lo,
    amrex::Real /*time*/) {
  // point-like radiation source
  amrex::Real const x0 = 0.;
  amrex::Real const y0 = 0.;
  amrex::Real const z0 = 0.;

  const amrex::Real source_norm =
      (1.0 / c) * L_star /
      std::pow(2.0 * M_PI * sigma_star * sigma_star, 1.5);

  amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
    amrex::Real const x = prob_lo[0] + (i + amrex::Real(0.5)) * dx[0];
    amrex::Real const y = prob_lo[1] + (j + amrex::Real(0.5)) * dx[1];
    amrex::Real const z = prob_lo[2] + (k + amrex::Real(0.5)) * dx[2];
    amrex::Real const r = std::sqrt(std::pow(x - x0, 2) + std::pow(y - y0, 2) +
                                    std::pow(z - z0, 2));

    radEnergy(i, j, k) =
        source_norm * std::exp(-(r * r) / (2.0 * sigma_star * sigma_star));
  });
}

template <>
auto RadSystem<ShellProblem>::ComputePlanckOpacity(const double /*rho*/,
                                                   const double /*Tgas*/)
    -> double {
  return kappa0;
}

template <>
auto RadSystem<ShellProblem>::ComputeRosselandOpacity(const double /*rho*/,
                                                      const double /*Tgas*/)
    -> double {
  return kappa0;
}

template <>
void RadhydroSimulation<ShellProblem>::setInitialConditionsAtLevel(int lev) {
  amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = geom[lev].CellSizeArray();
  amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo =
      geom[lev].ProbLoArray();

  amrex::Real const x0 = 0.;
  amrex::Real const y0 = 0.;
  amrex::Real const z0 = 0.;

  for (amrex::MFIter iter(state_new_[lev]); iter.isValid(); ++iter) {
    const amrex::Box &indexRange = iter.validbox();
    auto const &state = state_new_[lev].array(iter);

    amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
      amrex::Real const x = prob_lo[0] + (i + amrex::Real(0.5)) * dx[0];
      amrex::Real const y = prob_lo[1] + (j + amrex::Real(0.5)) * dx[1];
      amrex::Real const z = prob_lo[2] + (k + amrex::Real(0.5)) * dx[2];
      amrex::Real const r = std::sqrt(
          std::pow(x - x0, 2) + std::pow(y - y0, 2) + std::pow(z - z0, 2));

      double sigma_sh = H_shell / (2.0 * std::sqrt(2.0 * std::log(2.0)));
      double rho_norm = M_shell / (4.0 * M_PI * r * r *
                                   std::sqrt(2.0 * M_PI * sigma_sh * sigma_sh));
      double rho_shell = rho_norm * std::exp(-std::pow(r - r_0, 2) /
                                             (2.0 * sigma_sh * sigma_sh));
      double rho = std::max(rho_shell, 1.0e-8 * rho_0);

      const double F0 = L_star / (4.0 * M_PI * r * r);
      const double sigma_star_sq = sigma_star * sigma_star;

      // compute steady-state Frad (exact)
      const double Frad =
          F0 * (std::erf(r / (std::sqrt(2.0) * sigma_star)) -
                (2.0 * r) / std::sqrt(2.0 * M_PI * sigma_star_sq) *
                    std::exp(-(r * r) / (2.0 * sigma_star_sq)));

      // compute Erad from (isothermal) sound speed a0
      const double Eint = rho * (a0*a0);
      constexpr double Tgas0 = (a0*a0) / c_v;
      constexpr double Trad0 = Tgas0; // assume radiative equilibrium
      const double Erad = a_rad * std::pow(Trad0, 4);

      AMREX_ASSERT(!std::isnan(vx));
      AMREX_ASSERT(!std::isnan(vy));
      AMREX_ASSERT(!std::isnan(vz));
      AMREX_ASSERT(!std::isnan(rho));
      AMREX_ASSERT(!std::isnan(Eint));
      AMREX_ASSERT(!std::isnan(Erad));
      AMREX_ASSERT(!std::isnan(Frad));

      state(i, j, k, HydroSystem<ShellProblem>::density_index) = rho;
      state(i, j, k, HydroSystem<ShellProblem>::x1Momentum_index) = 0;
      state(i, j, k, HydroSystem<ShellProblem>::x2Momentum_index) = 0;
      state(i, j, k, HydroSystem<ShellProblem>::x3Momentum_index) = 0;
      state(i, j, k, HydroSystem<ShellProblem>::energy_index) = Eint;

      const double Frad_xyz = Frad / std::sqrt(3.0);
      state(i, j, k, RadSystem<ShellProblem>::radEnergy_index) = Erad;
      state(i, j, k, RadSystem<ShellProblem>::x1RadFlux_index) = Frad_xyz;
      state(i, j, k, RadSystem<ShellProblem>::x2RadFlux_index) = Frad_xyz;
      state(i, j, k, RadSystem<ShellProblem>::x3RadFlux_index) = Frad_xyz;
    });
  }

  // set flag
  areInitialConditionsDefined_ = true;
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE auto
vec_dot_r(amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> vec, int i, int j, int k)
    -> amrex::Real {
  // compute dot product of vec into rhat
  amrex::Real xhat = (i + amrex::Real(0.5));
  amrex::Real yhat = (j + amrex::Real(0.5));
  amrex::Real zhat = (k + amrex::Real(0.5));
  amrex::Real const norminv =
      1.0 / std::sqrt(xhat * xhat + yhat * yhat + zhat * zhat);

  xhat *= norminv;
  yhat *= norminv;
  zhat *= norminv;

  amrex::Real const dotproduct = vec[0] * xhat + vec[1] * yhat + vec[2] * zhat;
  return dotproduct;
}

#if 0
template <> void RadhydroSimulation<ShellProblem>::computeAfterTimestep() {
  // compute radial momentum for gas, radiation on level 0
  // (assuming octant symmetry)

  amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx0 =
      geom[0].CellSizeArray();
  amrex::Real const vol = AMREX_D_TERM(dx0[0], *dx0[1], *dx0[2]);
  auto const &state = state_new_[0];

  double radialMom =
      vol *
      amrex::ReduceSum(
          state, 0,
          [=] AMREX_GPU_DEVICE(amrex::Box const &bx,
                               amrex::Array4<amrex::Real const> const &arr) {
            amrex::Real result = 0.;
            amrex::Loop(bx, [&](int i, int j, int k) {
              amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> vec{
                  arr(i, j, k, RadSystem<ShellProblem>::x1GasMomentum_index),
                  arr(i, j, k, RadSystem<ShellProblem>::x2GasMomentum_index),
                  arr(i, j, k, RadSystem<ShellProblem>::x3GasMomentum_index)};
              result += vec_dot_r(vec, i, j, k);
            });
            return result;
          });

  amrex::ParallelAllReduce::Sum(radialMom,
                                amrex::ParallelContext::CommunicatorSub());

  double radialRadMom =
      (vol / c) *
      amrex::ReduceSum(
          state, 0,
          [=] AMREX_GPU_DEVICE(amrex::Box const &bx,
                               amrex::Array4<amrex::Real const> const &arr) {
            amrex::Real result = 0.;
            amrex::Loop(bx, [&](int i, int j, int k) {
              amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> vec{
                  arr(i, j, k, RadSystem<ShellProblem>::x1RadFlux_index),
                  arr(i, j, k, RadSystem<ShellProblem>::x2RadFlux_index),
                  arr(i, j, k, RadSystem<ShellProblem>::x3RadFlux_index)};
              result += vec_dot_r(vec, i, j, k);
            });
            return result;
          });

  amrex::ParallelAllReduce::Sum(radialRadMom,
                                amrex::ParallelContext::CommunicatorSub());

  amrex::Print() << "radial gas momentum = " << radialMom << std::endl;
  amrex::Print() << "radial radiation momentum = " << radialRadMom << std::endl;
}
#endif

template <>
void RadhydroSimulation<ShellProblem>::ErrorEst(int lev,
                                                amrex::TagBoxArray &tags,
                                                amrex::Real /*time*/,
                                                int /*ngrow*/) {
  // tag cells for refinement

  const amrex::Real eta_threshold = 0.2;      // gradient refinement threshold
  const amrex::Real rho_min = 1.0e-2 * rho_0; // minimum density for refinement

  for (amrex::MFIter mfi(state_new_[lev]); mfi.isValid(); ++mfi) {
    const amrex::Box &box = mfi.validbox();
    const auto state = state_new_[lev].const_array(mfi);
    const auto tag = tags.array(mfi);

    amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
      const int n = HydroSystem<ShellProblem>::density_index;
      amrex::Real const rho = state(i, j, k, n);
      amrex::Real const rho_xplus = state(i + 1, j, k, n);
      amrex::Real const rho_xminus = state(i - 1, j, k, n);
      amrex::Real const rho_yplus = state(i, j + 1, k, n);
      amrex::Real const rho_yminus = state(i, j - 1, k, n);

      amrex::Real const del_x =
          std::max(std::abs(rho_xplus - rho), std::abs(rho - rho_xminus));
      amrex::Real const del_y =
          std::max(std::abs(rho_yplus - rho), std::abs(rho - rho_yminus));

      amrex::Real const gradient_indicator =
          std::max(del_x, del_y) / std::max(rho, rho_min);

      if (gradient_indicator > eta_threshold) {
        tag(i, j, k) = amrex::TagBox::SET;
      }
    });
  }
}

auto problem_main() -> int {
  // This problem can only be run in 3D
  static_assert(AMREX_SPACEDIM == 3);

  auto isNormalComp = [=](int n, int dim) {
    // it is critical to reflect both the radiation and gas momenta!
    if ((n == RadSystem<ShellProblem>::x1GasMomentum_index) && (dim == 0)) {
      return true;
    }
    if ((n == RadSystem<ShellProblem>::x2GasMomentum_index) && (dim == 1)) {
      return true;
    }
    if ((n == RadSystem<ShellProblem>::x3GasMomentum_index) && (dim == 2)) {
      return true;
    }
    if ((n == RadSystem<ShellProblem>::x1RadFlux_index) && (dim == 0)) {
      return true;
    }
    if ((n == RadSystem<ShellProblem>::x2RadFlux_index) && (dim == 1)) {
      return true;
    }
    if ((n == RadSystem<ShellProblem>::x3RadFlux_index) && (dim == 2)) {
      return true;
    }
    return false;
  };

  const int nvars = RadhydroSimulation<ShellProblem>::nvarTotal_;
  amrex::Vector<amrex::BCRec> boundaryConditions(nvars);
  for (int n = 0; n < nvars; ++n) {
    for (int i = 0; i < AMREX_SPACEDIM; ++i) {
      if (isNormalComp(n, i)) {
        boundaryConditions[n].setLo(i, amrex::BCType::reflect_odd);
        boundaryConditions[n].setHi(i, amrex::BCType::foextrap); // outflow
      } else {
        boundaryConditions[n].setLo(i, amrex::BCType::reflect_even);
        boundaryConditions[n].setHi(i, amrex::BCType::foextrap); // outflow
      }
    }
  }

  // Problem initialization
  RadhydroSimulation<ShellProblem> sim(boundaryConditions);
  sim.is_hydro_enabled_ = true;
  sim.is_radiation_enabled_ = true;
  sim.densityFloor_ = 1.0e-8 * rho_0;
  sim.pressureFloor_ = 1.0e-8 * P_0;
  sim.reconstructionOrder_ =
      2; // 1 == donor cell, 2 == PLM, 3 == PPM (not recommended)
  sim.integratorOrder_ = 2; // RK2

  constexpr amrex::Real t0_hydro = r_0 / a0; // seconds
  //constexpr amrex::Real t0_light = r_0 / c; // seconds
  sim.stopTime_ = 0.124 * t0_hydro;
  //sim.stopTime_ = 20.0 * t0_light;
  sim.cflNumber_ = 0.3;
  //sim.initDt_ = 1.0e9; // seconds

  sim.maxTimesteps_ = 100; // for scaling tests

  sim.checkpointInterval_ = -1;
  sim.plotfileInterval_ = 100;

  // initialize
  sim.setInitialConditions();
  sim.computeAfterTimestep(); // compute radial momentum at t=0

  // evolve
  sim.evolve();

  // Cleanup and exit
  amrex::Print() << "Finished." << std::endl;
  return 0;
}