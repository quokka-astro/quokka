//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file cloud.cpp
/// \brief Implements a shock-cloud problem with radiative cooling.
///

// uncomment this to debug the root-finding code (does NOT work on GPU!)
#define BOOST_MATH_INSTRUMENT

#include <random>
#include <vector>

#include "AMReX.H"
#include "CloudyCooling.hpp"
#include "ODEIntegrate.hpp"
#include "cloud.hpp"
#include "hydro_system.hpp"
#include "radiation_system.hpp"

using amrex::Real;

struct ShockCloud {
}; // dummy type to allow compile-type polymorphism via template specialization

template <> struct EOS_Traits<ShockCloud> {
  static constexpr double gamma = 5. / 3.; // default value
  // if true, reconstruct e_int instead of pressure
  static constexpr bool reconstruct_eint = true;
};

auto problem_main() -> int {
  // Problem parameters
  const Real rho = 1.06569690624977e-24;   // g cm^-3;
  const Real Eint = 2.874790473172443e-12; // erg cm^-3

  // Read Cloudy tables
  cloudy_tables cloudyTables;
  readCloudyData(cloudyTables);
  auto tables = cloudyTables.const_tables();

  // int iter_count = 1;
  // amrex::launch(iter_count, [=] AMREX_GPU_DEVICE(int /*iter*/) {
  const Real T =
      ComputeTgasFromEgas(rho, Eint, HydroSystem<ShockCloud>::gamma_, tables);

  const Real rhoH = rho * cloudy_H_mass_fraction;
  const Real nH = rhoH / hydrogen_mass_cgs_;
  const Real log_nH = std::log10(nH);

  const Real C = (HydroSystem<ShockCloud>::gamma_ - 1.) * Eint /
                 (boltzmann_constant_cgs_ * (rho / hydrogen_mass_cgs_));
  const Real mu_sol = interpolate2d(log_nH, std::log10(T), tables.log_nH,
                                    tables.log_Tgas, tables.meanMolWeight);
  const Real relerr = std::abs((C * mu_sol - T) / T);

  printf("\nrho = %.17e, Eint = %.17e, mu = %f, Tgas = %f, relerr = %f\n", rho,
         Eint, mu_sol, T, relerr);
  //});

  // Cleanup and exit
  int status = 0;
  return status;
}
