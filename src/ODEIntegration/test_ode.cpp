//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file test_ode.cpp
/// \brief Defines a test problem for ODE integration.
///

#include "test_ode.hpp"
#include "AMReX_BLassert.H"
#include "Interpolate2D.hpp"
#include "hydro_system.hpp"
#include <algorithm>

using amrex::Real;

AMREX_GPU_HOST_DEVICE AMREX_INLINE auto cooling_function(Real const rho,
                                                         Real const T) -> Real {
  // use fitting function from Koyama & Inutsuka (2002)
  Real gamma_heat = 2.0e-26;
  Real lambda_cool = gamma_heat * (1.0e7 * std::exp(-114800. / (T + 1000.)) +
                                   14. * std::sqrt(T) * std::exp(-92. / T));
  Real rho_over_mh = rho / m_H;
  Real cooling_source_term =
      rho_over_mh * gamma_heat - (rho_over_mh * rho_over_mh) * lambda_cool;
  return cooling_source_term;
}

AMREX_GPU_HOST_DEVICE AMREX_INLINE auto
cloudy_cooling_function(Real const rho, Real const T, cloudy_tables *tables)
    -> Real {
  // interpolate cooling rates from Cloudy tables
  const Real nH = rho / m_H;
  const Real log_nH = std::log10(nH);
  const Real log_T = std::log10(T);

  auto primCoolTable = tables->primCooling->const_table();
  const double logPrimCool = interpolate2d(log_nH, log_T, *(tables->log_nH),
                                           *(tables->log_Tgas), primCoolTable);
  // amrex::Print() << "logPrimCool = " << logPrimCool << "\n";

  auto primHeatTable = tables->primHeating->const_table();
  const double logPrimHeat = interpolate2d(log_nH, log_T, *(tables->log_nH),
                                           *(tables->log_Tgas), primHeatTable);
  // amrex::Print() << "logPrimHeat = " << logPrimHeat << "\n";

  auto metalCoolTable = tables->metalCooling->const_table();
  const double logMetalCool = interpolate2d(
      log_nH, log_T, *(tables->log_nH), *(tables->log_Tgas), metalCoolTable);
  // amrex::Print() << "logMetalCool = " << logMetalCool << "\n";

  auto metalHeatTable = tables->metalHeating->const_table();
  const double logMetalHeat = interpolate2d(
      log_nH, log_T, *(tables->log_nH), *(tables->log_Tgas), metalHeatTable);
  // amrex::Print() << "logMetalCool = " << logMetalCool << "\n";

  const double netLambda_over_nsq_prim =
      std::pow(10., logPrimHeat) - std::pow(10., logPrimCool);
  const double netLambda_over_nsq_metals =
      std::pow(10., logMetalHeat) - std::pow(10., logMetalCool);
  const double netLambda_over_nsq =
      netLambda_over_nsq_prim + netLambda_over_nsq_metals;

  // multiply by the square of mass density (**NOT number density**)
  const double netLambda = (rho * rho) * netLambda_over_nsq;

  return netLambda;
}

AMREX_GPU_HOST_DEVICE AMREX_INLINE auto
ComputeTgasFromEgas(double rho, double Egas, cloudy_tables *tables) -> Real {
  // convert Egas (internal gas energy) to temperature
  auto table = tables->mean_mol_weight->const_table();

  // solve for temperature given Eint (at fixed adiabatic index)
  const Real gamma = HydroSystem<ODETest>::gamma_;
  const Real nH = rho / m_H;
  const Real C = (gamma - 1.) * Egas / (boltzmann_constant_cgs_ * nH);

  // solve for mu(T)*C == T.
  // (Grackle does this with a fixed-point iteration.)
  // (N.B. Using the HM2012_highdensity table leads to convergence failures.)
  Real mu_prev = NAN;
  Real mu_guess = 1.;
  Real Tgas = NAN;
  Real T_guess = NAN;
  const Real log_nH = std::log10(nH);
  const Real reltol = 1.0e-3;
  const int maxIter = 50;

  for (int n = 0; n < maxIter; ++n) {
    mu_prev = mu_guess; // save old mu

    // compute new guess for Tgas, bounded between 10 K and 1e9 K
    T_guess = std::clamp(C * mu_guess, 10., 1.0e9);

    // compute new mu from mu(T) table
    mu_guess = interpolate2d(log_nH, std::log10(T_guess), *(tables->log_nH),
                             *(tables->log_Tgas), table);

    // check if converged
    if (std::abs((C * mu_guess - T_guess) / T_guess) < reltol) {
      Tgas = T_guess;
      break;
    }
  }
  AMREX_ALWAYS_ASSERT_WITH_MESSAGE(!std::isnan(Tgas),
                                   "Tgas iteration failed to converge!");
  return Tgas;
}

AMREX_GPU_HOST_DEVICE AMREX_INLINE auto
ComputeEgasFromTgas(double rho, double Tgas, cloudy_tables *tables) -> Real {
  // convert Egas (internal gas energy) to temperature
  const Real gamma = HydroSystem<ODETest>::gamma_;
  const Real nH = rho / m_H;

  // compute mu from mu(T) table
  auto table = tables->mean_mol_weight->const_table();
  const Real mu = interpolate2d(std::log10(nH), std::log10(Tgas),
                                *(tables->log_nH), *(tables->log_Tgas), table);

  // compute thermal gas energy
  const Real Egas = (nH / mu) * boltzmann_constant_cgs_ * Tgas / (gamma - 1.);
  return Egas;
}

AMREX_GPU_HOST_DEVICE AMREX_INLINE auto
user_rhs(Real /*t*/, quokka::valarray<Real, 1> &y_data,
         quokka::valarray<Real, 1> &y_rhs, void *user_data) -> int {
  // unpack user_data
  auto *udata = static_cast<ODEUserData *>(user_data);
  Real rho = udata->rho;
  cloudy_tables *tables = udata->tables;

  // compute temperature (may depend on composition)
  Real Eint = y_data[0];
  Real T = ComputeTgasFromEgas(rho, Eint, tables);

  // compute cooling function
  y_rhs[0] = cloudy_cooling_function(rho, T, tables);
  return 0;
}

void readCloudyData(cloudy_tables &cloudyTables) {
  cloudy_data cloudy_primordial;
  cloudy_data cloudy_metals;
  code_units my_units; // cgs
  my_units.density_units = 1.0;
  my_units.length_units = 1.0;
  my_units.time_units = 1.0;
  my_units.velocity_units = 1.0;
  amrex::ParmParse pp;
  std::string grackle_hdf5_file;

  pp.query("grackle_data_file", grackle_hdf5_file);
  initialize_cloudy_data(cloudy_primordial, "Primordial", grackle_hdf5_file,
                         my_units);
  initialize_cloudy_data(cloudy_metals, "Metals", grackle_hdf5_file, my_units);

  cloudyTables.log_nH = std::make_unique<std::vector<double>>(
      cloudy_primordial.grid_parameters[0]);
  cloudyTables.log_Tgas = std::make_unique<std::vector<double>>(
      cloudy_primordial.grid_parameters[2]);

  int z_index = 0; // index along the redshift dimension

  cloudyTables.primCooling = std::make_unique<amrex::TableData<double, 2>>(
      extract_2d_table(cloudy_primordial.cooling_data, z_index));
  cloudyTables.primHeating = std::make_unique<amrex::TableData<double, 2>>(
      extract_2d_table(cloudy_primordial.heating_data, z_index));
  cloudyTables.mean_mol_weight = std::make_unique<amrex::TableData<double, 2>>(
      extract_2d_table(cloudy_primordial.mmw_data, z_index));

  cloudyTables.metalCooling = std::make_unique<amrex::TableData<double, 2>>(
      extract_2d_table(cloudy_metals.cooling_data, z_index));
  cloudyTables.metalHeating = std::make_unique<amrex::TableData<double, 2>>(
      extract_2d_table(cloudy_metals.heating_data, z_index));
}

auto problem_main() -> int {
  // read tables
  cloudy_tables cloudyTables;
  readCloudyData(cloudyTables);

  // run simulation
  const Real Eint0 = ComputeEgasFromTgas(rho0, Tgas0, &cloudyTables);
  const Real Edot0 = cloudy_cooling_function(rho0, Tgas0, &cloudyTables);
  const Real tcool = std::abs(Eint0 / Edot0);
  std::cout << "Initial temperature: " << Tgas0 << std::endl;
  std::cout << "Initial cooling time: " << tcool / seconds_in_year << std::endl;
  std::cout << "Initial edot = " << Edot0 << std::endl;

  double max_time = 10.0 * tcool;

  ODEUserData user_data{rho0, &cloudyTables};
  quokka::valarray<Real, 1> y = {Eint0};
  quokka::valarray<Real, 1> abstol = 1.0e-20 * y;
  // const Real rtol = 1.0e-15; // RK45 tol
  // const Real rtol = 1.0e-6; // RK23 tol
  const Real rtol = 1.0e-4; // RK12

  rk_adaptive_integrate(user_rhs, 0, y, max_time, &user_data, rtol, abstol);
  // adams_adaptive_integrate(user_rhs, 0, y, max_time, &user_data, rtol, abstol);

  const Real Tgas = ComputeTgasFromEgas(rho0, y[0], &cloudyTables);
  const Real Teq = 160.52611612610758; // for n_H = 0.01 cm^{-3}
  const Real Terr_rel = std::abs(Tgas - Teq) / Teq;
  const Real reltol = 1.0e-4; // relative error tolerance
  std::cout << "Final temperature: " << Tgas << std::endl;
  std::cout << "Relative error: " << Terr_rel << std::endl;

  // Cleanup and exit
  int status = 0;
  if ((Terr_rel > reltol) || (std::isnan(Terr_rel))) {
    status = 1;
  }
  return status;
}
