//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file test_ode.cpp
/// \brief Defines a test problem for ODE integration.
///

#include "test_ode.hpp"

using amrex::Real;

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

AMREX_GPU_HOST_DEVICE AMREX_INLINE auto
cloudy_cooling_function(Real const rho, Real const T, cloudy_tables *tables)
    -> Real {
  // interpolate cooling rates from Cloudy tables
  // TODO(benwibking): implement this using interpolate2d()
  return 0;
}

AMREX_GPU_HOST_DEVICE AMREX_INLINE auto
user_rhs(Real t, quokka::valarray<Real, 1> &y_data,
         quokka::valarray<Real, 1> &y_rhs, void *user_data) -> int {
  // unpack user_data
  auto *udata = static_cast<ODEUserData *>(user_data);
  Real rho = udata->rho;
  cloudy_tables *tables = udata->tables;

  // compute temperature (may depend on composition)
  Real Eint = y_data[0];
  Real T = RadSystem<ODETest>::ComputeTgasFromEgas(rho, Eint);

  // compute cooling function
  y_rhs[0] = cooling_function(rho, T);
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
  // Problem parameters
  const double max_time = 1.0e7 * seconds_in_year; // 10 Myr

  // read tables
  cloudy_tables cloudyTables;
  readCloudyData(cloudyTables);

  // run simulation
  const Real Eint0 = RadSystem<ODETest>::ComputeEgasFromTgas(rho0, Tgas0);
  const Real Edot0 = cooling_function(rho0, Tgas0);
  const Real tcool = std::abs(Eint0 / Edot0);
  std::cout << "Initial temperature: " << Tgas0 << std::endl;
  std::cout << "Initial cooling time: " << tcool << std::endl;

  ODEUserData user_data{rho0, &cloudyTables};
  quokka::valarray<Real, 1> y = {Eint0};
  quokka::valarray<Real, 1> abstol = 1.0e-20 * y;
  // const Real rtol = 1.0e-15; // RK45 tol
  // const Real rtol = 1.0e-6; // RK23 tol
  const Real rtol = 1.0e-4; // RK12

  rk_adaptive_integrate(user_rhs, 0, y, max_time, &user_data, rtol, abstol);

  const Real Tgas = RadSystem<ODETest>::ComputeTgasFromEgas(rho0, y[0]);
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
