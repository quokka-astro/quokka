//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file cloud.cpp
/// \brief Implements a shock-cloud problem with radiative cooling.
///

// uncomment this to debug the root-finding code (does NOT work on GPU!)
//#define BOOST_MATH_INSTRUMENT

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
  const Real rho = 2.2821435890860612e-30;   // g cm^-3;
  const Real Eint = 3.8522186325360341e-13; // erg cm^-3

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

  printf("\nrho = %.17e, Eint = %.17e, mu = %f, Tgas = %f, relerr = %e\n", rho,
         Eint, mu_sol, T, relerr);
  //});

#if 0
  // compute Field length
  auto lambda_F = [=] (Real nH0, Real T0) {
    const Real rho0 = nH0 * hydrogen_mass_cgs_ / cloudy_H_mass_fraction; // g cm^-3
    const Real Edot = cloudy_cooling_function(rho0, T0, tables);
    const Real ln_L = 29.7 + std::log(std::pow(nH0, -1./2.) * (T0 / 1.0e6));
    const Real conductivity = 1.84e-5 * std::pow(T0, 5./2.) / ln_L;
    const Real l = std::sqrt( conductivity * T0 / std::abs(Edot) );
    return std::make_pair(Edot, l);
  };

  auto [Edot0, l0] = lambda_F(1.0, 4.0e5);
  amrex::Print() << "Edot(nH = 1.0, T = 4e5) = " << Edot0 << "\n";
  amrex::Print() << "lambda_F = " << (l0 / 3.086e18) << " pc\n\n";

  auto [Edot1, l1] = lambda_F(0.1, 4.0e5);
  amrex::Print() << "Edot(nH = 0.1, T = 4e5) = " << Edot1 << "\n";
  amrex::Print() << "lambda_F = " << (l1 / 3.086e18) << " pc\n\n";
#endif

  // Cleanup and exit
  int status = 0;
  return status;
}
