//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file test_radiation_tube.cpp
/// \brief Defines a test problem for radiation pressure terms.
///

#include <string>

#include "AMReX.H"
#include "AMReX_BC_TYPES.H"

#include "test_radiation_tube.hpp"
#include "RadhydroSimulation.hpp"
#include "radiation_system.hpp"
#include "fextract.hpp"
#include "ArrayUtil.hpp"
#include "interpolate.hpp"
#ifdef HAVE_PYTHON
#include "matplotlibcpp.h"
#endif

struct TubeProblem {};

constexpr double kappa0 = 100.;                  // cm^2 g^-1
constexpr double mu = 2.33 * hydrogen_mass_cgs_; // g
constexpr double gamma_gas = 5. / 3.;

constexpr double rho0 = 1.0;                // g cm^-3
constexpr double T0 = 2.75e7;               // K
constexpr double rho1 = 2.1940476649492044; // g cm^-3
constexpr double T1 = 2.2609633884436745e7; // K

constexpr double a0 = 4.0295519855200705e7; // cm s^-1

template <> struct RadSystem_Traits<TubeProblem> {
  static constexpr double c_light = c_light_cgs_;
  static constexpr double c_hat = 10.0 * a0;
  static constexpr double radiation_constant = radiation_constant_cgs_;
  static constexpr double mean_molecular_mass = mu;
  static constexpr double boltzmann_constant = boltzmann_constant_cgs_;
  static constexpr double gamma = gamma_gas;
  static constexpr double Erad_floor = 0.;
  static constexpr bool compute_v_over_c_terms = true;
};

template <> struct Physics_Traits<TubeProblem> {
  static constexpr bool is_hydro_enabled = true;
  static constexpr bool is_radiation_enabled = true;
  static constexpr bool is_chemistry_enabled = false;

  static constexpr int numPassiveScalars = 0; // number of passive scalars
};

template <>
AMREX_GPU_HOST_DEVICE auto
RadSystem<TubeProblem>::ComputePlanckOpacity(const double /*rho*/,
                                             const double /*Tgas*/) -> double {
  return kappa0;
}

template <>
AMREX_GPU_HOST_DEVICE auto RadSystem<TubeProblem>::ComputeRosselandOpacity(
    const double /*rho*/, const double /*Tgas*/) -> double {
  return kappa0;
}

// declare global variables
// initial conditions read from file
amrex::Gpu::HostVector<double> x_arr;
amrex::Gpu::HostVector<double> rho_arr;
amrex::Gpu::HostVector<double> Pgas_arr;
amrex::Gpu::HostVector<double> Erad_arr;

amrex::Gpu::DeviceVector<double> x_arr_g;
amrex::Gpu::DeviceVector<double> rho_arr_g;
amrex::Gpu::DeviceVector<double> Pgas_arr_g;
amrex::Gpu::DeviceVector<double> Erad_arr_g;

template <> void RadhydroSimulation<TubeProblem>::preCalculateInitialConditions() {
  // map initial conditions to the global variables
  std::string filename = "../extern/pressure_tube/initial_conditions.txt";
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
    auto Pgas = values.at(2); // gas pressure
    auto Erad = values.at(3); // radiation energy density

    x_arr.push_back(x);
    rho_arr.push_back(rho);
    Pgas_arr.push_back(Pgas);
    Erad_arr.push_back(Erad);
  }

  x_arr_g.resize(x_arr.size());
  rho_arr_g.resize(rho_arr.size());
  Pgas_arr_g.resize(Pgas_arr.size());
  Erad_arr_g.resize(Erad_arr.size());

  // copy to device
  amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, x_arr.begin(), x_arr.end(), x_arr_g.begin());
  amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, rho_arr.begin(), rho_arr.end(), rho_arr_g.begin());
  amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, Pgas_arr.begin(), Pgas_arr.end(), Pgas_arr_g.begin());
  amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, Erad_arr.begin(), Erad_arr.end(), Erad_arr_g.begin());
  amrex::Gpu::streamSynchronizeAll();
}

template <>
void RadhydroSimulation<TubeProblem>::setInitialConditionsOnGrid(
    quokka::grid grid_elem) {
  // extract variables required from the geom object
  amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = grid_elem.dx;
  amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = grid_elem.prob_lo;
  const amrex::Box &indexRange = grid_elem.indexRange;
  const amrex::Array4<double>& state_cc = grid_elem.array;

  auto const &x_ptr = x_arr_g.dataPtr();
  auto const &rho_ptr = rho_arr_g.dataPtr();
  auto const &Pgas_ptr = Pgas_arr_g.dataPtr();
  auto const &Erad_ptr = Erad_arr_g.dataPtr();
  int x_size = static_cast<int>(x_arr_g.size());

  // loop over the grid and set the initial condition
  amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
    amrex::Real const x = prob_lo[0] + (i + amrex::Real(0.5)) * dx[0];

    amrex::Real const rho = interpolate_value(x, x_ptr, rho_ptr, x_size);
    amrex::Real const Pgas = interpolate_value(x, x_ptr, Pgas_ptr, x_size);
    amrex::Real const Erad = interpolate_value(x, x_ptr, Erad_ptr, x_size);

    state_cc(i, j, k, RadSystem<TubeProblem>::radEnergy_index) = Erad;
    state_cc(i, j, k, RadSystem<TubeProblem>::x1RadFlux_index) = 0;
    state_cc(i, j, k, RadSystem<TubeProblem>::x2RadFlux_index) = 0;
    state_cc(i, j, k, RadSystem<TubeProblem>::x3RadFlux_index) = 0;
    state_cc(i, j, k, RadSystem<TubeProblem>::gasEnergy_index) =
        Pgas / (gamma_gas - 1.0);
    state_cc(i, j, k, RadSystem<TubeProblem>::gasDensity_index) = rho;
    state_cc(i, j, k, RadSystem<TubeProblem>::gasInternalEnergy_index) =
        Pgas / (gamma_gas - 1.0);
    state_cc(i, j, k, RadSystem<TubeProblem>::x1GasMomentum_index) = 0.;
    state_cc(i, j, k, RadSystem<TubeProblem>::x2GasMomentum_index) = 0.;
    state_cc(i, j, k, RadSystem<TubeProblem>::x3GasMomentum_index) = 0.;
  });
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

  if (i < lo[0]) {
    // left side boundary -- constant
    const double Erad =
        RadSystem<TubeProblem>::radiation_constant_ * std::pow(T0, 4);
    const double Frad =
        consVar(lo[0], j, k, RadSystem<TubeProblem>::x1RadFlux_index);
    consVar(i, j, k, RadSystem<TubeProblem>::radEnergy_index) = Erad;
    consVar(i, j, k, RadSystem<TubeProblem>::x1RadFlux_index) = Frad;
    consVar(i, j, k, RadSystem<TubeProblem>::x2RadFlux_index) = 0.;
    consVar(i, j, k, RadSystem<TubeProblem>::x3RadFlux_index) = 0.;

    const double Egas =
        (boltzmann_constant_cgs_ / mu) * rho0 * T0 / (gamma_gas - 1.0);
    const double x1Mom =
        consVar(lo[0], j, k, RadSystem<TubeProblem>::x1GasMomentum_index);
    const double Ekin = 0.5 * (x1Mom * x1Mom) / rho0;
    consVar(i, j, k, RadSystem<TubeProblem>::gasEnergy_index) = Egas + Ekin;
    consVar(i, j, k, RadSystem<TubeProblem>::gasDensity_index) = rho0;
    consVar(i, j, k, RadSystem<TubeProblem>::gasInternalEnergy_index) = Egas;
    consVar(i, j, k, RadSystem<TubeProblem>::x1GasMomentum_index) = x1Mom;
    consVar(i, j, k, RadSystem<TubeProblem>::x2GasMomentum_index) = 0.;
    consVar(i, j, k, RadSystem<TubeProblem>::x3GasMomentum_index) = 0.;

  } else if (i > hi[0]) {
    // right-side boundary -- constant
    const double Erad =
        RadSystem<TubeProblem>::radiation_constant_ * std::pow(T1, 4);
    const double Frad =
        consVar(hi[0], j, k, RadSystem<TubeProblem>::x1RadFlux_index);
    consVar(i, j, k, RadSystem<TubeProblem>::radEnergy_index) = Erad;
    consVar(i, j, k, RadSystem<TubeProblem>::x1RadFlux_index) = Frad;
    consVar(i, j, k, RadSystem<TubeProblem>::x2RadFlux_index) = 0;
    consVar(i, j, k, RadSystem<TubeProblem>::x3RadFlux_index) = 0;

    const double Egas =
        (boltzmann_constant_cgs_ / mu) * rho1 * T1 / (gamma_gas - 1.0);
    const double x1Mom =
        consVar(hi[0], j, k, RadSystem<TubeProblem>::x1GasMomentum_index);
    const double Ekin = 0.5 * (x1Mom * x1Mom) / rho1;
    consVar(i, j, k, RadSystem<TubeProblem>::gasEnergy_index) = Egas + Ekin;
    consVar(i, j, k, RadSystem<TubeProblem>::gasDensity_index) = rho1;
    consVar(i, j, k, RadSystem<TubeProblem>::gasInternalEnergy_index) = Egas;
    consVar(i, j, k, RadSystem<TubeProblem>::x1GasMomentum_index) = x1Mom;
    consVar(i, j, k, RadSystem<TubeProblem>::x2GasMomentum_index) = 0.;
    consVar(i, j, k, RadSystem<TubeProblem>::x3GasMomentum_index) = 0.;
  }
}

auto problem_main() -> int {
  // Problem parameters
  // const int nx = 128;
  constexpr double Lx = 128.0;
  constexpr double CFL_number = 0.4;
  constexpr double tmax = Lx / a0;
  constexpr int max_timesteps = 2000;

  // Boundary conditions
  constexpr int nvars = RadSystem<TubeProblem>::nvar_;
  amrex::Vector<amrex::BCRec> BCs_cc(nvars);
  for (int n = 0; n < nvars; ++n) {
    BCs_cc[n].setLo(0, amrex::BCType::ext_dir); // Dirichlet x1
    BCs_cc[n].setHi(0, amrex::BCType::ext_dir); // Dirichlet x1
    for (int i = 1; i < AMREX_SPACEDIM; ++i) {
      BCs_cc[n].setLo(i, amrex::BCType::int_dir); // periodic
      BCs_cc[n].setHi(i, amrex::BCType::int_dir);
    }
  }

  // Problem initialization
  RadhydroSimulation<TubeProblem> sim(BCs_cc);
  
  sim.radiationReconstructionOrder_ = 2; // PLM
  sim.reconstructionOrder_ = 2; // PLM
  sim.stopTime_ = tmax;
  sim.cflNumber_ = CFL_number;
  sim.radiationCflNumber_ = CFL_number;
  sim.maxTimesteps_ = max_timesteps;
  sim.plotfileInterval_ = -1;

  // initialize
  sim.setInitialConditions();
  auto [position0, values0] = fextract(sim.state_new_cc_[0], sim.Geom(0), 0, 0.0);

  // evolve
  sim.evolve();

  // read output variables
  auto [position, values] = fextract(sim.state_new_cc_[0], sim.Geom(0), 0, 0.0);
  const int nx = static_cast<int>(position0.size());

  // compute error norm
  std::vector<double> Trad_arr(nx);
  std::vector<double> Trad_exact_arr(nx);
  std::vector<double> Trad_err(nx);
  std::vector<double> Tgas_arr(nx);
  std::vector<double> Tgas_err(nx);
  std::vector<double> rho_err(nx);
  std::vector<double> xs(nx);

  for (int i = 0; i < nx; ++i) {
    xs[i] = position[i];

    double rho_exact =
        values0.at(RadSystem<TubeProblem>::gasDensity_index)[i];
    double rho = values.at(RadSystem<TubeProblem>::gasDensity_index)[i];
    rho_err[i] = (rho - rho_exact) / rho_exact;

    double Trad_exact =
        std::pow(values0.at(RadSystem<TubeProblem>::radEnergy_index)[i] /
                     radiation_constant_cgs_,
                 1. / 4.);
    double Trad =
        std::pow(values.at(RadSystem<TubeProblem>::radEnergy_index)[i] /
                     radiation_constant_cgs_,
                 1. / 4.);
    Trad_arr[i] = Trad;
    Trad_exact_arr[i] = Trad_exact;
    Trad_err[i] = (Trad - Trad_exact) / Trad_exact;

    double Egas_exact =
        values0.at(RadSystem<TubeProblem>::gasEnergy_index)[i];
    double x1GasMom_exact =
        values0.at(RadSystem<TubeProblem>::x1GasMomentum_index)[i];
    double x2GasMom_exact =
        values0.at(RadSystem<TubeProblem>::x2GasMomentum_index)[i];
    double x3GasMom_exact =
        values0.at(RadSystem<TubeProblem>::x3GasMomentum_index)[i];

    double Egas = values.at(RadSystem<TubeProblem>::gasEnergy_index)[i];
    double x1GasMom =
        values.at(RadSystem<TubeProblem>::x1GasMomentum_index)[i];
    double x2GasMom =
        values.at(RadSystem<TubeProblem>::x2GasMomentum_index)[i];
    double x3GasMom =
        values.at(RadSystem<TubeProblem>::x3GasMomentum_index)[i];

    double Eint_exact = RadSystem<TubeProblem>::ComputeEintFromEgas(
        rho_exact, x1GasMom_exact, x2GasMom_exact, x3GasMom_exact, Egas_exact);
    double Tgas_exact =
        RadSystem<TubeProblem>::ComputeTgasFromEgas(rho_exact, Eint_exact);

    double Eint = RadSystem<TubeProblem>::ComputeEintFromEgas(
        rho, x1GasMom, x2GasMom, x3GasMom, Egas);
    double Tgas = RadSystem<TubeProblem>::ComputeTgasFromEgas(rho, Eint);

    Tgas_arr[i] = Tgas;
    Tgas_err[i] = (Tgas - Tgas_exact) / Tgas_exact;
  }

  double err_norm = 0.;
  double sol_norm = 0.;
  for (int i = 0; i < nx; ++i) {
    err_norm += std::abs(Trad_arr[i] - Trad_exact_arr[i]);
    sol_norm += std::abs(Trad_exact_arr[i]);
  }

  const double rel_err_norm = err_norm / sol_norm;
  const double rel_err_tol = 0.0008;
  int status = 1;
  if (rel_err_norm < rel_err_tol) {
    status = 0;
  }
  amrex::Print() << "Relative L1 norm = " << rel_err_norm << std::endl;

#ifdef HAVE_PYTHON
  // Plot results
  int s = 4; // stride
  std::map<std::string, std::string> Trad_args;
  std::map<std::string, std::string> Tgas_args;
  std::unordered_map<std::string, std::string> Texact_args;
  Trad_args["label"] = "radiation";
  Trad_args["color"] = "C1";
  Tgas_args["label"] = "gas";
  Tgas_args["color"] = "C2";
  Texact_args["label"] = "exact";
  Texact_args["marker"] = "o";
  Texact_args["color"] = "black";

  matplotlibcpp::plot(xs, Trad_arr, Trad_args);
  matplotlibcpp::plot(xs, Tgas_arr, Tgas_args);
  matplotlibcpp::scatter(strided_vector_from(xs, s), strided_vector_from(Trad_exact_arr, s), 10.0, Texact_args);

  matplotlibcpp::legend();
  //matplotlibcpp::title(fmt::format("t = {:.4g} s", sim.tNew_[0]));
  matplotlibcpp::xlabel("length x (cm)");
  matplotlibcpp::ylabel("temperature (Kelvins)");
  matplotlibcpp::tight_layout();
  matplotlibcpp::save("./radiation_pressure_tube.pdf");
#endif // HAVE_PYTHON

  // Cleanup and exit
  amrex::Print() << "Finished." << std::endl;
  return status;
}