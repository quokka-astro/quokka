//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file test_radiation_force.cpp
/// \brief Defines a test problem for radiation force terms.
///

#include "test_radiation_force.hpp"
#include "AMReX.H"
#include "AMReX_BC_TYPES.H"
#include "AMReX_REAL.H"
#include "RadhydroSimulation.hpp"
#include "fextract.hpp"
#include "hydro_system.hpp"
#include "matplotlibcpp.h"
#include <string>
extern "C" {
#include "interpolate.h"
}
#include "radiation_system.hpp"

struct TubeProblem {};

constexpr double kappa0 = 1.0e5;                 // cm^2 g^-1
constexpr double mu = 2.33 * hydrogen_mass_cgs_; // g
constexpr double gamma_gas = 1.0001;             // quasi-isothermal

constexpr double a0 = 1.0e5;     // cm s^-1
constexpr double Frad0 = 1.0e14; // erg cm^-2 s^-1
constexpr double tau = 1.0e-3;   // optical depth (dimensionless)
constexpr double Sigma0 = tau / kappa0;
constexpr double Lx = 1.0; // cm

template <> struct RadSystem_Traits<TubeProblem> {
  static constexpr double c_light = c_light_cgs_;
  static constexpr double c_hat = 300. * a0;
  static constexpr double radiation_constant = radiation_constant_cgs_;
  static constexpr double mean_molecular_mass = mu;
  static constexpr double boltzmann_constant = boltzmann_constant_cgs_;
  static constexpr double gamma = gamma_gas;
  static constexpr double Erad_floor = 0.;
  static constexpr bool compute_v_over_c_terms = true;
};

template <>
AMREX_GPU_HOST_DEVICE auto
RadSystem<TubeProblem>::ComputePlanckOpacity(const double /*rho*/,
                                             const double /*Tgas*/) -> double {
  return 0.; // no heating/cooling
}

template <>
AMREX_GPU_HOST_DEVICE auto RadSystem<TubeProblem>::ComputeRosselandOpacity(
    const double /*rho*/, const double /*Tgas*/) -> double {
  return kappa0;
}

template <>
void RadhydroSimulation<TubeProblem>::setInitialConditionsAtLevel(int lev) {
  // set initial conditions
  amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = geom[lev].CellSizeArray();
  amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo =
      geom[lev].ProbLoArray();

  for (amrex::MFIter iter(state_old_[lev]); iter.isValid(); ++iter) {
    const amrex::Box &indexRange = iter.validbox(); // excludes ghost zones
    auto const &state = state_new_[lev].array(iter);

    amrex::ParallelFor(indexRange, [=](int i, int j, int k) noexcept {
      amrex::Real const x = prob_lo[0] + (i + amrex::Real(0.5)) * dx[0];

      amrex::Real const B = (kappa0 * Frad0) / (a0 * a0 * c_light_cgs_);
      amrex::Real const A = B * Sigma0 / (std::exp(B * Lx) - 1.0);
      amrex::Real const rho = A * std::exp(B * x);
      amrex::Real const Pgas = rho * (a0 * a0);

      state(i, j, k, RadSystem<TubeProblem>::radEnergy_index) =
          Frad0 / c_light_cgs_;
      state(i, j, k, RadSystem<TubeProblem>::x1RadFlux_index) = Frad0;
      state(i, j, k, RadSystem<TubeProblem>::x2RadFlux_index) = 0;
      state(i, j, k, RadSystem<TubeProblem>::x3RadFlux_index) = 0;

      state(i, j, k, RadSystem<TubeProblem>::gasEnergy_index) =
          Pgas / (gamma_gas - 1.0);
      state(i, j, k, RadSystem<TubeProblem>::gasDensity_index) = rho;
      state(i, j, k, RadSystem<TubeProblem>::x1GasMomentum_index) = 0.;
      state(i, j, k, RadSystem<TubeProblem>::x2GasMomentum_index) = 0.;
      state(i, j, k, RadSystem<TubeProblem>::x3GasMomentum_index) = 0.;
    });
  }

  // set flag
  areInitialConditionsDefined_ = true;
}

template <>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE void
AMRSimulation<TubeProblem>::setCustomBoundaryConditions(
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

  amrex::Real const Erad = Frad0 / c_light_cgs_;
  amrex::Real const Frad = Frad0;

  amrex::Real const *dx = geom.CellSize();
  amrex::Real const *prob_lo = geom.ProbLo();
  amrex::Real const x = prob_lo[0] + (i + amrex::Real(0.5)) * dx[0];
  amrex::Real const B = (kappa0 * Frad0) / (a0 * a0 * c_light_cgs_);
  amrex::Real const A = B * Sigma0 / (std::exp(B * Lx) - 1.0);
  amrex::Real const rho = A * std::exp(B * x);
  amrex::Real const Pgas = rho * (a0 * a0);

  if ((i < lo[0]) || (i > hi[0])) {
    // Dirichlet
    consVar(i, j, k, RadSystem<TubeProblem>::radEnergy_index) = Erad;
    consVar(i, j, k, RadSystem<TubeProblem>::x1RadFlux_index) = Frad;
    consVar(i, j, k, RadSystem<TubeProblem>::x2RadFlux_index) = 0;
    consVar(i, j, k, RadSystem<TubeProblem>::x3RadFlux_index) = 0;

    //#if 0
    consVar(i, j, k, RadSystem<TubeProblem>::gasDensity_index) = rho;
    consVar(i, j, k, RadSystem<TubeProblem>::x1GasMomentum_index) = 0.;
    consVar(i, j, k, RadSystem<TubeProblem>::x2GasMomentum_index) = 0.;
    consVar(i, j, k, RadSystem<TubeProblem>::x3GasMomentum_index) = 0.;
    consVar(i, j, k, RadSystem<TubeProblem>::gasEnergy_index) =
        Pgas / (gamma_gas - 1.0);
    //#endif
  }
}

template <>
void RadhydroSimulation<TubeProblem>::computeAfterLevelAdvance(
    int lev, amrex::Real /*time*/, amrex::Real dt_lev, int, int) {
  // reset sound speed to constant
  amrex::Real const t_damp = 1.0e-5 * (Lx / a0);

  for (amrex::MFIter iter(state_old_[lev]); iter.isValid(); ++iter) {
    const amrex::Box &indexRange = iter.validbox();
    auto const &state = state_new_[lev].array(iter);

    amrex::ParallelFor(indexRange, [=](int i, int j, int k) noexcept {
      amrex::Real const rho =
          state(i, j, k, RadSystem<TubeProblem>::gasDensity_index);
      amrex::Real px =
          state(i, j, k, RadSystem<TubeProblem>::x1GasMomentum_index);
      amrex::Real py =
          state(i, j, k, RadSystem<TubeProblem>::x2GasMomentum_index);
      amrex::Real pz =
          state(i, j, k, RadSystem<TubeProblem>::x3GasMomentum_index);

      px *= std::exp(-dt_lev / t_damp);
      py *= std::exp(-dt_lev / t_damp);
      pz *= std::exp(-dt_lev / t_damp);

      amrex::Real const Ekin = (px * px + py * py + pz * pz) / (2.0 * rho);
      amrex::Real const Pgas = rho * (a0 * a0);
      state(i, j, k, RadSystem<TubeProblem>::x1GasMomentum_index) = px;
      state(i, j, k, RadSystem<TubeProblem>::x2GasMomentum_index) = py;
      state(i, j, k, RadSystem<TubeProblem>::x3GasMomentum_index) = pz;
      state(i, j, k, RadSystem<TubeProblem>::gasEnergy_index) =
          (Pgas / (gamma_gas - 1.0)) + Ekin;
    });
  }
}

#if 0
template <> void RadhydroSimulation<TubeProblem>::computeAfterTimestep() {
  // compute x-momentum for gas, radiation on level 0

  amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx0 =
      geom[0].CellSizeArray();
  amrex::Real const vol = AMREX_D_TERM(dx0[0], *dx0[1], *dx0[2]);
  auto const &state = state_new_[0];

  double x1Mom =
      vol *
      amrex::ReduceSum(
          state, 0,
          [=] AMREX_GPU_DEVICE(amrex::Box const &bx,
                               amrex::Array4<amrex::Real const> const &arr) {
            amrex::Real result = 0.;
            amrex::Loop(bx, [&](int i, int j, int k) {
              result +=
                  arr(i, j, k, RadSystem<TubeProblem>::x1GasMomentum_index);
            });
            return result;
          });

  amrex::ParallelAllReduce::Sum(x1Mom,
                                amrex::ParallelContext::CommunicatorSub());

  double x1RadMom =
      (vol / c_light_cgs_) *
      amrex::ReduceSum(
          state, 0,
          [=] AMREX_GPU_DEVICE(amrex::Box const &bx,
                               amrex::Array4<amrex::Real const> const &arr) {
            amrex::Real result = 0.;
            amrex::Loop(bx, [&](int i, int j, int k) {
              result += arr(i, j, k, RadSystem<TubeProblem>::x1RadFlux_index);
            });
            return result;
          });

  amrex::ParallelAllReduce::Sum(x1RadMom,
                                amrex::ParallelContext::CommunicatorSub());

  amrex::Print() << "gas x-momentum = " << x1Mom << std::endl;
  amrex::Print() << "radiation x-momentum = " << x1RadMom << std::endl;
}
#endif

auto problem_main() -> int {
  // Problem parameters
  // const int nx = 128;
  constexpr double CFL_number = 0.05;
  constexpr double tmax = 1.0 * (Lx / a0);
  constexpr int max_timesteps = 2e6;

  // Boundary conditions
  auto isNormalComp = [=](int n, int dim) {
    if ((n == HydroSystem<TubeProblem>::x1Momentum_index) && (dim == 0)) {
      return true;
    }
    if ((n == HydroSystem<TubeProblem>::x2Momentum_index) && (dim == 1)) {
      return true;
    }
    if ((n == HydroSystem<TubeProblem>::x3Momentum_index) && (dim == 2)) {
      return true;
    }
    return false;
  };

  auto isHydroVar = [=](int n) {
    if (n == HydroSystem<TubeProblem>::density_index) {
      return true;
    }
    if (n == HydroSystem<TubeProblem>::x1Momentum_index) {
      return true;
    }
    if (n == HydroSystem<TubeProblem>::x2Momentum_index) {
      return true;
    }
    if (n == HydroSystem<TubeProblem>::x3Momentum_index) {
      return true;
    }
    if (n == HydroSystem<TubeProblem>::energy_index) {
      return true;
    }
    return false;
  };

  constexpr int nvars = RadSystem<TubeProblem>::nvar_;
  amrex::Vector<amrex::BCRec> boundaryConditions(nvars);
  bool reflectHydroVars = false;

  for (int n = 0; n < nvars; ++n) {
    // for x-axis:
    if (isHydroVar(n)) {
      if (reflectHydroVars) { // reflecting hydro
        if (isNormalComp(n, 0)) {
          boundaryConditions[n].setLo(0, amrex::BCType::reflect_odd);
          boundaryConditions[n].setHi(0, amrex::BCType::reflect_odd);
        } else {
          boundaryConditions[n].setLo(0, amrex::BCType::reflect_even);
          boundaryConditions[n].setHi(0, amrex::BCType::reflect_even);
        }
      } else { // Dirichlet hydro
        boundaryConditions[n].setLo(0, amrex::BCType::ext_dir);
        boundaryConditions[n].setHi(0, amrex::BCType::ext_dir);
      }
    } else { // Dirichlet radiation
      boundaryConditions[n].setLo(0, amrex::BCType::ext_dir);
      boundaryConditions[n].setHi(0, amrex::BCType::ext_dir);
    }

    // for y-, z- axes:
    for (int i = 1; i < AMREX_SPACEDIM; ++i) {
      // all periodic
      boundaryConditions[n].setLo(i, amrex::BCType::int_dir);
      boundaryConditions[n].setHi(i, amrex::BCType::int_dir);
    }
  }

  // Problem initialization
  RadhydroSimulation<TubeProblem> sim(boundaryConditions);
  sim.is_hydro_enabled_ = true;
  sim.is_radiation_enabled_ = true;
  sim.radiationReconstructionOrder_ = 2; // PLM
  sim.reconstructionOrder_ = 2;          // PLM
  sim.stopTime_ = tmax;
  sim.cflNumber_ = CFL_number;
  sim.radiationCflNumber_ = CFL_number;
  sim.maxTimesteps_ = max_timesteps;
  sim.plotfileInterval_ = -1;

  // initialize
  sim.setInitialConditions();
  auto [position0, values0] = fextract(sim.state_new_[0], sim.Geom(0), 0, 0.0);

  // evolve
  sim.evolve();

  // read output variables
  auto [position, values] = fextract(sim.state_new_[0], sim.Geom(0), 0, 0.0);
  const int nx = static_cast<int>(position0.size());

  // compute error norm
  std::vector<double> rho_err(nx);
  std::vector<double> rho_arr(nx);
  std::vector<double> rho_exact_arr(nx);
  std::vector<double> Frad_err(nx);
  std::vector<double> cs_err(nx);
  std::vector<double> vx_arr(nx);
  std::vector<double> Pgas_arr(nx);
  std::vector<double> Pgas_exact_arr(nx);
  std::vector<double> xs(nx);

  for (int i = 0; i < nx; ++i) {
    xs.at(i) = position.at(i);
    double rho_exact =
        values0.at(RadSystem<TubeProblem>::gasDensity_index).at(i);
    double rho = values.at(RadSystem<TubeProblem>::gasDensity_index).at(i);
    double Frad = values.at(RadSystem<TubeProblem>::x1RadFlux_index).at(i);

    double Egas = values.at(RadSystem<TubeProblem>::gasEnergy_index).at(i);
    double x1GasMom =
        values.at(RadSystem<TubeProblem>::x1GasMomentum_index).at(i);
    double x2GasMom =
        values.at(RadSystem<TubeProblem>::x2GasMomentum_index).at(i);
    double x3GasMom =
        values.at(RadSystem<TubeProblem>::x3GasMomentum_index).at(i);
    double Eint = RadSystem<TubeProblem>::ComputeEintFromEgas(
        rho_exact, x1GasMom, x2GasMom, x3GasMom, Egas);

    double Pgas = Eint * (gamma_gas - 1.0);
    double cs = std::sqrt(Pgas / (gamma_gas * rho));
    double vx = x1GasMom / rho;

    vx_arr.at(i) = vx;
    cs_err.at(i) = (cs - a0) / a0;
    Frad_err.at(i) = (Frad - Frad0) / Frad0;
    rho_err.at(i) = (rho - rho_exact) / rho_exact;
    rho_exact_arr.at(i) = rho_exact;
    rho_arr.at(i) = rho;
    Pgas_arr.at(i) = Pgas;
    Pgas_exact_arr.at(i) = rho_exact * (a0 * a0);
  }

  double err_norm = 0.;
  double sol_norm = 0.;
  for (int i = 0; i < nx; ++i) {
    err_norm += std::abs(rho_arr[i] - rho_exact_arr[i]);
    sol_norm += std::abs(rho_exact_arr[i]);
  }

  const double rel_err_norm = err_norm / sol_norm;
  const double rel_err_tol = 1.0e-4;
  int status = 1;
  if (rel_err_norm < rel_err_tol) {
    status = 0;
  }
  amrex::Print() << "Relative L1 norm = " << rel_err_norm << std::endl;

  // Plot results
  std::map<std::string, std::string> rho_args;
  std::map<std::string, std::string> rhoexact_args;
  rho_args["label"] = "simulation";
  rhoexact_args["label"] = "exact solution";
  matplotlibcpp::plot(xs, rho_arr, rho_args);
  matplotlibcpp::plot(xs, rho_exact_arr, rhoexact_args);
  matplotlibcpp::legend();
  matplotlibcpp::title(fmt::format("t = {:.4g} s", sim.tNew_[0]));
  matplotlibcpp::xlabel("x (cm)");
  matplotlibcpp::ylabel("density");
  matplotlibcpp::save("./radiation_force_tube.pdf");

#if 0
  // plot pressure profile
  std::map<std::string, std::string> P_args;
  std::map<std::string, std::string> Pexact_args;
  P_args["label"] = "simulation";
  Pexact_args["label"] = "exact solution";
  matplotlibcpp::clf();
  matplotlibcpp::plot(xs, Pgas_arr, P_args);
  matplotlibcpp::plot(xs, Pgas_exact_arr, Pexact_args);
  matplotlibcpp::legend();
  matplotlibcpp::title(fmt::format("t = {:.4g} s", sim.tNew_[0]));
  matplotlibcpp::xlabel("x (cm)");
  matplotlibcpp::ylabel("pressure");
  matplotlibcpp::save("./radiation_force_gaspressure.pdf");

  // plot radiation flux
  std::map<std::string, std::string> Frad_args;
  Frad_args["label"] = "radiation flux";
  matplotlibcpp::clf();
  matplotlibcpp::plot(xs, Frad_err, Frad_args);
  matplotlibcpp::legend();
  matplotlibcpp::title(fmt::format("t = {:.4g} s", sim.tNew_[0]));
  matplotlibcpp::xlabel("x (cm)");
  matplotlibcpp::ylabel("relative error");
  matplotlibcpp::save("./radiation_force_tube_flux.pdf");

  // plot sound speed
  std::map<std::string, std::string> cs_args;
  cs_args["label"] = "sound speed";
  matplotlibcpp::clf();
  matplotlibcpp::plot(xs, cs_err, cs_args);
  matplotlibcpp::legend();
  matplotlibcpp::title(fmt::format("t = {:.4g} s", sim.tNew_[0]));
  matplotlibcpp::xlabel("x (cm)");
  matplotlibcpp::ylabel("relative error");
  matplotlibcpp::save("./radiation_force_tube_cs.pdf");

  // plot velocity
  std::map<std::string, std::string> vx_args;
  vx_args["label"] = "gas velocity";
  matplotlibcpp::clf();
  matplotlibcpp::plot(xs, vx_arr, vx_args);
  matplotlibcpp::legend();
  matplotlibcpp::title(fmt::format("t = {:.4g} s", sim.tNew_[0]));
  matplotlibcpp::xlabel("x (cm)");
  matplotlibcpp::ylabel("x-velocity (cm/s)");
  matplotlibcpp::save("./radiation_force_tube_vel.pdf");
#endif

  // Cleanup and exit
  amrex::Print() << "Finished." << std::endl;
  return status;
}