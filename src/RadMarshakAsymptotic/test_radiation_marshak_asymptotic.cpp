//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file test_radiation_marshak_asymptotic.cpp
/// \brief Defines a test problem for radiation in the asymptotic diffusion regime.
///

#include "AMReX_BLassert.H"

#include "test_radiation_marshak_asymptotic.hpp"
#include "RadhydroSimulation.hpp"
#include "fextract.hpp"

struct SuOlsonProblemCgs {
}; // dummy type to allow compile-type polymorphism via template specialization

constexpr double kappa = 300.0;                   // cm^-1 (opacity)
constexpr double rho0 = 2.0879373766122384;       // g cm^-3 (matter density)
constexpr double T_hohlraum = 1.1604448449e7;     // K (1 keV)
constexpr double T_initial = 300.; // K

// constexpr double kelvin_to_eV = 8.617385e-5;
constexpr double a_rad = radiation_constant_cgs_;
constexpr double c_v =
    (boltzmann_constant_cgs_ / hydrogen_mass_cgs_) / (5. / 3. - 1.);

template <> struct RadSystem_Traits<SuOlsonProblemCgs> {
  static constexpr double c_light = c_light_cgs_;
  static constexpr double c_hat = c_light_cgs_;
  static constexpr double radiation_constant = radiation_constant_cgs_;
  static constexpr double mean_molecular_mass = hydrogen_mass_cgs_;
  static constexpr double boltzmann_constant = boltzmann_constant_cgs_;
  static constexpr double gamma = 5. / 3.;
  static constexpr double Erad_floor = 0.;
  static constexpr bool compute_v_over_c_terms = true;
};

template <>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto
RadSystem<SuOlsonProblemCgs>::ComputePlanckOpacity(
    const double rho, const double Tgas) -> double {
  auto sigma = kappa * std::pow(Tgas / T_hohlraum, -3); // cm^-1
  return (sigma / rho);                                 // cm^2 g^-1
}

template <>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto
RadSystem<SuOlsonProblemCgs>::ComputeRosselandOpacity(const double rho,
                                                      const double Tgas)
    -> double {
  auto sigma = kappa * std::pow(Tgas / T_hohlraum, -3); // cm^-1
  return (sigma / rho);                                 // cm^2 g^-1
}

template <>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto
RadSystem<SuOlsonProblemCgs>::ComputePlanckOpacityTempDerivative(
    const double rho, const double Tgas) -> double {
  auto sigma_dT =
      (-3.0 * kappa / Tgas) * std::pow(Tgas / T_hohlraum, -3); // cm^-1
  return (sigma_dT / rho);
}

template <>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto
RadSystem<SuOlsonProblemCgs>::ComputeEddingtonFactor(double /*f*/) -> double {
  return (1. / 3.); // Eddington approximation
}

template <>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE void
AMRSimulation<SuOlsonProblemCgs>::setCustomBoundaryConditions(
    const amrex::IntVect &iv, amrex::Array4<amrex::Real> const &consVar,
    int /*dcomp*/, int /*numcomp*/, amrex::GeometryData const & /*geom*/,
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

  // This boundary condition is only first-order accurate!
  // (Not quite good enough for simulating an optically-thick Marshak wave;
  //  the temperature becomes T_H in the first zone, rather than just at the
  //  face.)
  // [Solution: enforce Marshak boundary condition in Riemann solver]

  if (i < 0) {
    // Marshak boundary condition
    const double T_H = T_hohlraum;
    const double E_inc = radiation_constant_cgs_ * std::pow(T_H, 4);
    const double c = c_light_cgs_;
    // const double F_inc = c * E_inc / 4.0; // incident flux

    const double E_0 =
        consVar(0, j, k, RadSystem<SuOlsonProblemCgs>::radEnergy_index);
    const double F_0 =
        consVar(0, j, k, RadSystem<SuOlsonProblemCgs>::x1RadFlux_index);
    // const double E_1 = consVar(1, j, k,
    // RadSystem<SuOlsonProblemCgs>::radEnergy_index); const double F_1 =
    // consVar(1, j, k, RadSystem<SuOlsonProblemCgs>::x1RadFlux_index);

    // use PPM stencil at interface to solve for F_rad in the ghost zones
    // const double F_bdry = 0.5 * c * E_inc - (7. / 12.) * (c * E_0 + 2.0 *
    // F_0) +
    //		      (1. / 12.) * (c * E_1 + 2.0 * F_1);

    // use value at interface to solve for F_rad in the ghost zones
    const double F_bdry = 0.5 * c * E_inc - 0.5 * (c * E_0 + 2.0 * F_0);

    AMREX_ASSERT(std::abs(F_bdry / (c * E_inc)) < 1.0);

    // x1 left side boundary (Marshak)
    consVar(i, j, k, RadSystem<SuOlsonProblemCgs>::radEnergy_index) = E_inc;
    consVar(i, j, k, RadSystem<SuOlsonProblemCgs>::x1RadFlux_index) = F_bdry;
    consVar(i, j, k, RadSystem<SuOlsonProblemCgs>::x2RadFlux_index) = 0.;
    consVar(i, j, k, RadSystem<SuOlsonProblemCgs>::x3RadFlux_index) = 0.;
  } else {
    // right-side boundary -- constant
    const double Erad = radiation_constant_cgs_ * std::pow(T_initial, 4);

    consVar(i, j, k, RadSystem<SuOlsonProblemCgs>::radEnergy_index) = Erad;
    consVar(i, j, k, RadSystem<SuOlsonProblemCgs>::x1RadFlux_index) = 0;
    consVar(i, j, k, RadSystem<SuOlsonProblemCgs>::x2RadFlux_index) = 0;
    consVar(i, j, k, RadSystem<SuOlsonProblemCgs>::x3RadFlux_index) = 0;
  }

  // gas boundary conditions are the same on both sides
  const double Egas =
      RadSystem<SuOlsonProblemCgs>::ComputeEgasFromTgas(rho0, T_initial);
  consVar(i, j, k, RadSystem<SuOlsonProblemCgs>::gasEnergy_index) = Egas;
  consVar(i, j, k, RadSystem<SuOlsonProblemCgs>::gasDensity_index) = rho0;
  consVar(i, j, k, RadSystem<SuOlsonProblemCgs>::gasInternalEnergy_index) = 0.;
  consVar(i, j, k, RadSystem<SuOlsonProblemCgs>::x1GasMomentum_index) = 0.;
  consVar(i, j, k, RadSystem<SuOlsonProblemCgs>::x2GasMomentum_index) = 0.;
  consVar(i, j, k, RadSystem<SuOlsonProblemCgs>::x3GasMomentum_index) = 0.;
}

template <>
void RadhydroSimulation<SuOlsonProblemCgs>::setInitialConditionsAtLevel(
    int lev) {
  for (amrex::MFIter iter(state_new_[lev]); iter.isValid(); ++iter) {
    const amrex::Box &indexRange = iter.validbox();
    auto const &state = state_new_[lev].array(iter);

    amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
      const double Egas =
          RadSystem<SuOlsonProblemCgs>::ComputeEgasFromTgas(rho0, T_initial);
      const double Erad = a_rad * std::pow(T_initial, 4);

      state(i, j, k, RadSystem<SuOlsonProblemCgs>::radEnergy_index) = Erad;
      state(i, j, k, RadSystem<SuOlsonProblemCgs>::x1RadFlux_index) = 0;
      state(i, j, k, RadSystem<SuOlsonProblemCgs>::x2RadFlux_index) = 0;
      state(i, j, k, RadSystem<SuOlsonProblemCgs>::x3RadFlux_index) = 0;

      state(i, j, k, RadSystem<SuOlsonProblemCgs>::gasDensity_index) = rho0;
      state(i, j, k, RadSystem<SuOlsonProblemCgs>::gasEnergy_index) = Egas;
      state(i, j, k, RadSystem<SuOlsonProblemCgs>::gasInternalEnergy_index) = 0.;
      state(i, j, k, RadSystem<SuOlsonProblemCgs>::x1GasMomentum_index) = 0.;
      state(i, j, k, RadSystem<SuOlsonProblemCgs>::x2GasMomentum_index) = 0.;
      state(i, j, k, RadSystem<SuOlsonProblemCgs>::x3GasMomentum_index) = 0.;
    });
  }

  // set flag
  areInitialConditionsDefined_ = true;
}

auto problem_main() -> int {
  // This problem tests whether the numerical scheme is asymptotic preserving.
  // This requires both a spatial discretization *and* a temporal discretization
  // that have the asymptotic-preserving property. Operator splitting the
  // transport and source terms can give a splitting error that is arbitrarily
  // large in the asymptotic limit! A fully implicit method or a semi-implicit
  // predictor-corrector method [2] similar to SDC is REQUIRED for a correct solution.
  //
  // For discussion of the asymptotic-preserving property, see [1] and [2]. For
  // a discussion of the exact, self-similar solution to this problem, see [3].
  // Note that when used with an SDC time integrator, PLM (w/ asymptotic correction
  // in Riemann solver) does a good job, but not quite as good as linear DG on this
  // problem. There are some 'stair-stepping' artifacts that appear with PLM at low
  // resolution that do not appear when using DG. This is likely the "wide stencil"
  // issue discussed in [4].
  //
  // 1. R.G. McClarren, R.B. Lowrie, The effects of slope limiting on asymptotic-preserving
  //     numerical methods for hyperbolic conservation laws, Journal of
  //     Computational Physics 227 (2008) 9711–9726.
  // 2. R.G. McClarren, T.M. Evans, R.B. Lowrie, J.D. Densmore, Semi-implicit time integration
  //     for PN thermal radiative transfer, Journal of Computational Physics 227
  //     (2008) 7561-7586.
  // 3. Y. Zel'dovich, Y. Raizer, Physics of Shock Waves and High-Temperature Hydrodynamic
  //     Phenomena (1964), Ch. X.: Thermal Waves.
  // 4. Lowrie, R. B. and Morel, J. E., Issues with high-resolution Godunov methods for
  //     radiation hydrodynamics, Journal of Quantitative Spectroscopy and
  //     Radiative Transfer, 69, 475–489, 2001.

  // Problem parameters
  const int max_timesteps = 1e5;
  const double CFL_number = 0.9;
  const double initial_dt = 5.0e-12; // s
  const double max_dt = 5.0e-12;     // s
  const double max_time = 10.0e-9;   // s
  // const int nx = 60; // [18 == matches resolution of McClarren & Lowrie (2008)]
  // const double Lx = 0.66; // cm

  // Problem initialization
  std::cout << "radiation constant (code units) = "
            << RadSystem_Traits<SuOlsonProblemCgs>::radiation_constant << "\n";
  std::cout << "c_light (code units) = "
            << RadSystem_Traits<SuOlsonProblemCgs>::c_light << "\n";
  std::cout << "rho * c_v = " << rho0 * c_v << "\n";
  std::cout << "initial_dt = " << initial_dt << "\n";
  std::cout << "max_dt = " << max_dt << "\n";
  std::cout << "max_time = " << max_time << std::endl;

  constexpr int nvars = RadSystem<SuOlsonProblemCgs>::nvar_;
  amrex::Vector<amrex::BCRec> boundaryConditions(nvars);
  for (int n = 0; n < nvars; ++n) {
    boundaryConditions[n].setLo(0,
                                amrex::BCType::ext_dir); // custom (Marshak) x1
    boundaryConditions[n].setHi(0, amrex::BCType::foextrap); // extrapolate x1
    for (int i = 1; i < AMREX_SPACEDIM; ++i) {
      boundaryConditions[n].setLo(i, amrex::BCType::int_dir); // periodic
      boundaryConditions[n].setHi(i, amrex::BCType::int_dir);
    }
  }

  // Problem initialization
  RadhydroSimulation<SuOlsonProblemCgs> sim(boundaryConditions);
  sim.is_hydro_enabled_ = false;
  sim.is_radiation_enabled_ = true;
  sim.radiationReconstructionOrder_ = 3; // PPM
  sim.stopTime_ = max_time;
  sim.initDt_ = initial_dt;
  sim.maxDt_ = max_dt;
  sim.radiationCflNumber_ = CFL_number;
  sim.maxTimesteps_ = max_timesteps;
  sim.plotfileInterval_ = -1;

  // initialize
  sim.setInitialConditions();

  // evolve
  sim.evolve();

  // read output variables
  auto [position, values] = fextract(sim.state_new_[0], sim.Geom(0), 0, 0.0);
  const int nx = static_cast<int>(position.size());

  // compare against diffusion solution
  std::vector<double> xs(nx);
  std::vector<double> Trad_keV(nx);
  std::vector<double> Tgas_keV(nx);

  for (int i = 0; i < nx; ++i) {
    amrex::Real const x = position.at(i);
    xs.at(i) = x;

    const double Erad_t =
        values.at(RadSystem<SuOlsonProblemCgs>::radEnergy_index).at(i);
    const double Etot_t =
        values.at(RadSystem<SuOlsonProblemCgs>::gasEnergy_index).at(i);
    const double rho =
        values.at(RadSystem<SuOlsonProblemCgs>::gasDensity_index).at(i);
    const double x1GasMom =
        values.at(RadSystem<SuOlsonProblemCgs>::x1GasMomentum_index).at(i);

    const double Ekin = (x1GasMom * x1GasMom) / (2.0 * rho);
    const double Egas_t = (Etot_t - Ekin);

    Tgas_keV.at(i) = RadSystem<SuOlsonProblemCgs>::ComputeTgasFromEgas(rho, Egas_t) / T_hohlraum;
    Trad_keV.at(i) = std::pow(Erad_t / a_rad, 1. / 4.) / T_hohlraum;
  }

  // read in exact solution

  std::vector<double> xs_exact;
  std::vector<double> Tmat_exact;

  std::string filename = "../extern/marshak_similarity.csv";
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
    auto x_val = values.at(0);
    auto Tmat_val = values.at(1);

    xs_exact.push_back(x_val);
    Tmat_exact.push_back(Tmat_val);
  }

  // compute error norm

  // interpolate numerical solution onto exact tabulated solution
  std::vector<double> Tmat_interp(xs_exact.size());
  interpolate_arrays(xs_exact.data(), Tmat_interp.data(),
                     static_cast<int>(xs_exact.size()), xs.data(), Tgas_keV.data(),
                     static_cast<int>(xs.size()));

  double err_norm = 0.;
  double sol_norm = 0.;
  for (int i = 0; i < xs_exact.size(); ++i) {
    err_norm += std::abs(Tmat_interp[i] - Tmat_exact[i]);
    sol_norm += std::abs(Tmat_exact[i]);
  }

  const double error_tol = 0.05; // 5 per cent
  const double rel_error = err_norm / sol_norm;
  amrex::Print() << "Relative L1 error norm = " << rel_error << std::endl;

#ifdef HAVE_PYTHON
  // plot results
  matplotlibcpp::clf();
  std::map<std::string, std::string> Tgas_args;
  std::map<std::string, std::string> Tgas_exact_args;
  Tgas_args["label"] = "gas temperature";
  Tgas_args["marker"] = ".";
  Tgas_exact_args["label"] = "gas temperature (exact)";
  Tgas_exact_args["marker"] = "x";
  matplotlibcpp::plot(xs, Tgas_keV, Tgas_args);
  matplotlibcpp::plot(xs_exact, Tmat_exact, Tgas_exact_args);

  matplotlibcpp::ylim(0.0, 1.0);  // keV
  matplotlibcpp::xlim(0.0, 0.55); // cm
  matplotlibcpp::xlabel("length x (cm)");
  matplotlibcpp::ylabel("temperature (keV)");
  matplotlibcpp::legend();
  matplotlibcpp::title(fmt::format("time t = {:.4g}", sim.tNew_[0]));
  matplotlibcpp::save("./marshak_wave_asymptotic_gastemperature.pdf");
#endif // HAVE_PYTHON

  // Cleanup and exit
  std::cout << "Finished." << std::endl;

  int status = 0;
  if ((rel_error > error_tol) || std::isnan(rel_error)) {
    status = 1;
  }
  return status;
}
