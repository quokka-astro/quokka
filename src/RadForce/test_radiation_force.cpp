//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file test_radiation_force.cpp
/// \brief Defines a test problem for radiation force terms.
///

#include <string>

#include "AMReX.H"
#include "AMReX_BC_TYPES.H"
#include "AMReX_BLassert.H"
#include "AMReX_REAL.H"

#include "RadhydroSimulation.hpp"
#include "fextract.hpp"
#include "hydro_system.hpp"
#include "radiation_system.hpp"
#include "test_radiation_force.hpp"
#include "ArrayUtil.hpp"
extern "C" {
#include "interpolate.h"
}
#ifdef HAVE_PYTHON
#include "matplotlibcpp.h"
#endif

struct TubeProblem {};

constexpr double kappa0 = 5.0;                   // cm^2 g^-1
constexpr double mu = 2.33 * hydrogen_mass_cgs_; // g
constexpr double gamma_gas = 1.0;                // isothermal gas EOS
constexpr double a0 = 0.2e5;                     // cm s^-1
constexpr double tau = 1.0e-6; // optical depth (dimensionless)

constexpr double rho0 = 1.0e5 * mu; // g cm^-3
constexpr double Mach0 = 1.1;       // Mach number at wind base
constexpr double Mach1 = 2.128410288469465339;
constexpr double rho1 = (Mach0 / Mach1) * rho0;

constexpr double Frad0 = rho0 * a0 * c_light_cgs_ / tau; // erg cm^-2 s^-1
constexpr double g0 = kappa0 * Frad0 / c_light_cgs_;     // cm s^{-2}
constexpr double Lx = (a0 * a0) / g0;                    // cm

template <> struct RadSystem_Traits<TubeProblem> {
  static constexpr double c_light = c_light_cgs_;
  static constexpr double c_hat = 10. * (Mach1 * a0);
  static constexpr double radiation_constant = radiation_constant_cgs_;
  static constexpr double mean_molecular_mass = mu;
  static constexpr double boltzmann_constant = boltzmann_constant_cgs_;
  static constexpr double gamma = gamma_gas;
  static constexpr double Erad_floor = 0.;
  static constexpr bool compute_v_over_c_terms = true;
};

template <> struct EOS_Traits<TubeProblem> {
  static constexpr double gamma = gamma_gas;
  static constexpr double cs_isothermal = a0; // only used when gamma = 1
  static constexpr bool reconstruct_eint = false; // unused if isothermal
  static constexpr int nscalars = 0;       // number of passive scalars
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
  // read initial conditions from file
  amrex::Vector<double> x_arr;
  amrex::Vector<double> rho_arr;
  amrex::Vector<double> Mach_arr;

  std::string filename = "../extern/pressure_tube/optically_thin_wind.txt";
  std::ifstream fstream(filename, std::ios::in);
  AMREX_ALWAYS_ASSERT(fstream.is_open());
  std::string header;
  std::getline(fstream, header);

  for (std::string line; std::getline(fstream, line);) {
    std::istringstream iss(line);
    std::vector<double> values;
    for (double value = NAN; iss >> value;) {
      values.push_back(value);
    }
    auto x = values.at(0);    // position
    auto rho = values.at(1);  // density
    auto Mach = values.at(2); // Mach number

    x_arr.push_back(x);
    rho_arr.push_back(rho);
    Mach_arr.push_back(Mach);
  }

  // set initial conditions
  amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = geom[lev].CellSizeArray();
  amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo =
      geom[lev].ProbLoArray();

  for (amrex::MFIter iter(state_old_[lev]); iter.isValid(); ++iter) {
    const amrex::Box &indexRange = iter.validbox(); // excludes ghost zones
    auto const &state = state_new_[lev].array(iter);

    amrex::LoopConcurrentOnCpu(indexRange, [=](int i, int j, int k) noexcept {
      amrex::Real const x = (prob_lo[0] + (i + amrex::Real(0.5)) * dx[0]) / Lx;
      amrex::Real const D =
          interpolate_value(x, x_arr.dataPtr(), rho_arr.dataPtr(),
                            static_cast<int>(x_arr.size()));
      AMREX_ALWAYS_ASSERT(D > 0.);

      amrex::Real const Mach =
          interpolate_value(x, x_arr.dataPtr(), Mach_arr.dataPtr(),
                            static_cast<int>(x_arr.size()));
      AMREX_ALWAYS_ASSERT(!std::isnan(Mach));

      amrex::Real const rho = D * rho0;
      amrex::Real const vel = Mach * a0;

      state(i, j, k, RadSystem<TubeProblem>::radEnergy_index) =
          Frad0 / c_light_cgs_;
      state(i, j, k, RadSystem<TubeProblem>::x1RadFlux_index) = Frad0;
      state(i, j, k, RadSystem<TubeProblem>::x2RadFlux_index) = 0;
      state(i, j, k, RadSystem<TubeProblem>::x3RadFlux_index) = 0;

      state(i, j, k, RadSystem<TubeProblem>::gasDensity_index) = rho;
      state(i, j, k, RadSystem<TubeProblem>::x1GasMomentum_index) = rho * vel;
      state(i, j, k, RadSystem<TubeProblem>::x2GasMomentum_index) = 0;
      state(i, j, k, RadSystem<TubeProblem>::x3GasMomentum_index) = 0;
      state(i, j, k, RadSystem<TubeProblem>::gasEnergy_index) = 0;
      state(i, j, k, RadSystem<TubeProblem>::passiveScalar_index) = 0;
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
  amrex::Real rho = NAN;
  amrex::Real vel = NAN;

  if (i < lo[0]) {
    // left side
    rho = rho0;
    vel = Mach0 * a0;
  } else if (i > hi[0]) {
    // right side
    rho = rho1;
    vel = Mach1 * a0;
  }

  if ((i < lo[0]) || (i > hi[0])) {
    // Dirichlet
    consVar(i, j, k, RadSystem<TubeProblem>::radEnergy_index) = Erad;
    consVar(i, j, k, RadSystem<TubeProblem>::x1RadFlux_index) = Frad;
    consVar(i, j, k, RadSystem<TubeProblem>::x2RadFlux_index) = 0;
    consVar(i, j, k, RadSystem<TubeProblem>::x3RadFlux_index) = 0;

    consVar(i, j, k, RadSystem<TubeProblem>::gasDensity_index) = rho;
    consVar(i, j, k, RadSystem<TubeProblem>::x1GasMomentum_index) = rho * vel;
    consVar(i, j, k, RadSystem<TubeProblem>::x2GasMomentum_index) = 0;
    consVar(i, j, k, RadSystem<TubeProblem>::x3GasMomentum_index) = 0;
    consVar(i, j, k, RadSystem<TubeProblem>::gasEnergy_index) = 0;
    consVar(i, j, k, RadSystem<TubeProblem>::passiveScalar_index) = 0;
  }
}

auto problem_main() -> int {
  // Problem parameters
  // const int nx = 128;
  constexpr double CFL_number = 0.4;
  constexpr double tmax = 10.0 * (Lx / a0);
  constexpr int max_timesteps = 1e6;

  // Boundary conditions
  constexpr int nvars = RadSystem<TubeProblem>::nvar_;
  amrex::Vector<amrex::BCRec> boundaryConditions(nvars);
  for (int n = 0; n < nvars; ++n) {
    // for x-axis:
    boundaryConditions[n].setLo(0, amrex::BCType::ext_dir);
    boundaryConditions[n].setHi(0, amrex::BCType::ext_dir);
    // for y-, z- axes:
    for (int i = 1; i < AMREX_SPACEDIM; ++i) {
      // periodic
      boundaryConditions[n].setLo(i, amrex::BCType::int_dir);
      boundaryConditions[n].setHi(i, amrex::BCType::int_dir);
    }
  }

  // Problem initialization
  RadhydroSimulation<TubeProblem> sim(boundaryConditions);
  sim.is_hydro_enabled_ = true;
  sim.is_radiation_enabled_ = true;
  sim.radiationReconstructionOrder_ = 3; // PPM
  sim.reconstructionOrder_ = 3;          // PPM
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
  std::vector<double> xs(nx);
  std::vector<double> rho_arr(nx);
  std::vector<double> rho_exact_arr(nx);
  std::vector<double> rho_err(nx);
  std::vector<double> vx_arr(nx);
  std::vector<double> vx_exact_arr(nx);
  std::vector<double> Frad_err(nx);

  for (int i = 0; i < nx; ++i) {
    xs.at(i) = position.at(i);
    double rho_exact =
        values0.at(RadSystem<TubeProblem>::gasDensity_index).at(i);
    double x1GasMom_exact =
        values0.at(RadSystem<TubeProblem>::x1GasMomentum_index).at(i);
    double rho = values.at(RadSystem<TubeProblem>::gasDensity_index).at(i);
    double Frad = values.at(RadSystem<TubeProblem>::x1RadFlux_index).at(i);
    double x1GasMom =
        values.at(RadSystem<TubeProblem>::x1GasMomentum_index).at(i);
    double vx = x1GasMom / rho;
    double vx_exact = x1GasMom_exact / rho_exact;

    vx_arr.at(i) = vx / a0;
    vx_exact_arr.at(i) = vx_exact / a0;
    Frad_err.at(i) = (Frad - Frad0) / Frad0;
    rho_err.at(i) = (rho - rho_exact) / rho_exact;
    rho_exact_arr.at(i) = rho_exact;
    rho_arr.at(i) = rho;
  }

  double err_norm = 0.;
  double sol_norm = 0.;
  for (int i = 0; i < nx; ++i) {
    err_norm += std::abs(rho_arr[i] - rho_exact_arr[i]);
    sol_norm += std::abs(rho_exact_arr[i]);
  }

  const double rel_err_norm = err_norm / sol_norm;
  const double rel_err_tol = 0.002;
  int status = 1;
  if (rel_err_norm < rel_err_tol) {
    status = 0;
  }
  amrex::Print() << "Relative L1 norm = " << rel_err_norm << std::endl;

#ifdef HAVE_PYTHON
  // Plot density
  std::map<std::string, std::string> rho_args;
  std::unordered_map<std::string, std::string> rhoexact_args;
  rho_args["label"] = "simulation";
  rhoexact_args["label"] = "exact solution";
  matplotlibcpp::plot(xs, rho_arr, rho_args);
  matplotlibcpp::scatter(xs, rho_exact_arr, 1.0, rhoexact_args);
  matplotlibcpp::legend();
  matplotlibcpp::title(fmt::format("t = {:.4g} s", sim.tNew_[0]));
  matplotlibcpp::xlabel("x (cm)");
  matplotlibcpp::ylabel("density");
  matplotlibcpp::tight_layout();
  matplotlibcpp::save("./radiation_force_tube.pdf");

  // plot velocity
  int s = 4; // stride
  std::map<std::string, std::string> vx_args;
  std::unordered_map<std::string, std::string> vxexact_args;
  vxexact_args["label"] = "exact solution";
  vx_args["label"] = "simulation";
  vx_args["color"] = "C3";
  vxexact_args["color"] = "C3";
  vxexact_args["marker"] = "o";
  //vxexact_args["edgecolors"] = "k";
  matplotlibcpp::clf();
  matplotlibcpp::plot(xs, vx_arr, vx_args);
  matplotlibcpp::scatter(strided_vector_from(xs, s), strided_vector_from(vx_exact_arr, s), 10.0, vxexact_args);
  matplotlibcpp::legend();
  //matplotlibcpp::title(fmt::format("t = {:.4g} s", sim.tNew_[0]));
  matplotlibcpp::xlabel("length x (cm)");
  matplotlibcpp::ylabel("Mach number");
  matplotlibcpp::tight_layout();
  matplotlibcpp::save("./radiation_force_tube_vel.pdf");
#endif // HAVE_PYTHON

  // Cleanup and exit
  amrex::Print() << "Finished." << std::endl;
  return status;
}