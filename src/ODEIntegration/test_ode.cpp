//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file test_ode.cpp
/// \brief Defines a test problem for ODE integration.
///

#include "test_ode.hpp"
#include "CloudyCooling.hpp"
#include "radiation_system.hpp"

using amrex::Real;

constexpr double Tgas0 = 1.0e5;       // 6000.;    // K
constexpr double rho0 = 0.01 * hydrogen_mass_cgs_; // g cm^-3

struct ODEUserData {
  amrex::Real rho = NAN;
  cloudyGpuConstTables tables;
};

AMREX_GPU_HOST_DEVICE AMREX_INLINE auto
user_rhs(Real /*t*/, quokka::valarray<Real, 1> &y_data,
         quokka::valarray<Real, 1> &y_rhs, void *user_data) -> int {
  // unpack user_data
  auto *udata = static_cast<ODEUserData *>(user_data);
  Real rho = udata->rho;
  cloudyGpuConstTables &tables = udata->tables;

  // compute temperature (implicit solve, depends on composition)
  Real Eint = y_data[0];
  Real T = ComputeTgasFromEgas(rho, Eint, HydroSystem<ODETest>::gamma_, tables);

  // compute cooling function
  y_rhs[0] = cloudy_cooling_function(rho, T, tables);
  return 0;
}

auto problem_main() -> int {
  // read tables
  cloudy_tables cloudyTables;
  readCloudyData(cloudyTables);
  cloudyGpuConstTables tables = cloudyTables.const_tables();

  // run simulation
  const Real Eint0 = ComputeEgasFromTgas(
      rho0, Tgas0, HydroSystem<ODETest>::gamma_, tables);
  const Real Edot0 = cloudy_cooling_function(rho0, Tgas0, tables);
  const Real tcool = std::abs(Eint0 / Edot0);
  const Real max_time = 10.0 * tcool;

  std::cout << "Initial temperature: " << Tgas0 << std::endl;
  std::cout << "Initial cooling time: " << tcool / seconds_in_year << std::endl;
  std::cout << "Initial edot = " << Edot0 << std::endl;

  // solve cooling
  ODEUserData user_data{rho0, tables};
  quokka::valarray<Real, 1> y = {Eint0};
  quokka::valarray<Real, 1> abstol = 1.0e-20 * y;
  const Real rtol = 1.0e-4; // appropriate for RK12

  rk_adaptive_integrate(user_rhs, 0, y, max_time, &user_data, rtol, abstol);

  const Real Tgas = ComputeTgasFromEgas(
      rho0, y[0], HydroSystem<ODETest>::gamma_, tables);
  const Real Teq =
      160.52611612610758; // for n_H = 0.01 cm^{-3} (for IK cooling function)
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
