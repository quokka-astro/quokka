#ifndef CLOUDYCOOLING_HPP_ // NOLINT
#define CLOUDYCOOLING_HPP_
//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file CloudyCooling.hpp
/// \brief Defines methods for interpolating cooling rates from Cloudy tables.
///

#include "AMReX.H"
#include "AMReX_BLassert.H"
#include "AMReX_Extension.H"
#include "AMReX_GpuQualifiers.H"

#include "GrackleDataReader.hpp"
#include "Interpolate2D.hpp"
#include "radiation_system.hpp"
#include "root_finding.hpp"
#include "FastMath.hpp"
#include <limits>

//   Set H mass fraction according to "abundances ism" in Cloudy,
//   which assumes n_He / n_H = 0.098. This gives a value of about 0.72.
//   Using the default value of 0.76 will result in negative electron
//   densities at low temperature.
//   Below, we set X = 1 / (1 + hydrogen_mass_cgs_e * n_He / n_H).

constexpr double cloudy_H_mass_fraction = 1. / (1. + 0.098 * 3.971);

struct cloudyGpuConstTables {
  // these are non-owning, so can make a copy of the whole struct
  amrex::Table1D<const Real> const log_nH;
  amrex::Table1D<const Real> const log_Tgas;

  amrex::Table2D<const Real> const cool;
  amrex::Table2D<const Real> const heat;
  amrex::Table2D<const Real> const meanMolWeight;
};

class cloudy_tables {
public:
  std::unique_ptr<amrex::TableData<double, 1>> log_nH;
  std::unique_ptr<amrex::TableData<double, 1>> log_Tgas;

  std::unique_ptr<amrex::TableData<double, 2>> cooling;
  std::unique_ptr<amrex::TableData<double, 2>> heating;
  std::unique_ptr<amrex::TableData<double, 2>> mean_mol_weight;

  [[nodiscard]] auto const_tables() const -> cloudyGpuConstTables;
};

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto
cloudy_cooling_function(Real const rho, Real const T,
                        cloudyGpuConstTables const &tables) -> Real {
  // interpolate cooling rates from Cloudy tables
  const Real rhoH = rho * cloudy_H_mass_fraction; // mass density of H species
  const Real nH = rhoH / hydrogen_mass_cgs_;
  const Real log_nH = std::log10(nH);
  const Real log_T = std::log10(T);

  const double logCool = interpolate2d(log_nH, log_T, tables.log_nH,
                                       tables.log_Tgas, tables.cool);

  const double logHeat = interpolate2d(log_nH, log_T, tables.log_nH,
                                       tables.log_Tgas, tables.heat);

  const double netLambda = FastMath::pow10(logHeat) - FastMath::pow10(logCool);

  // multiply by the square of H mass density (**NOT number density**)
  const double Edot = (rhoH * rhoH) * netLambda;

  return Edot;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto
ComputeEgasFromTgas(double rho, double Tgas, double gamma,
                    cloudyGpuConstTables const &tables) -> Real {
  // convert Egas (internal gas energy) to temperature
  const Real rhoH = rho * cloudy_H_mass_fraction;
  const Real nH = rhoH / hydrogen_mass_cgs_;

  // compute mu from mu(T) table
  const Real mu = interpolate2d(std::log10(nH), std::log10(Tgas), tables.log_nH,
                                tables.log_Tgas, tables.meanMolWeight);

  // compute thermal gas energy
  const Real n = rho / (hydrogen_mass_cgs_ * mu);
  const Real Pgas = n * boltzmann_constant_cgs_ * Tgas;
  const Real Egas = Pgas / (gamma - 1.);
  return Egas;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto
ComputeTgasFromEgas(double rho, double Egas, double gamma,
                    cloudyGpuConstTables const &tables) -> Real {
  // convert Egas (internal gas energy) to temperature

  // check whether temperature is out-of-bounds
  const Real Tmin_table = 10.;
  const Real Tmax_table = 1.0e9;
  const Real Eint_min = ComputeEgasFromTgas(rho, Tmin_table, gamma, tables);
  const Real Eint_max = ComputeEgasFromTgas(rho, Tmax_table, gamma, tables);

  if (Egas <= Eint_min) {
    return Tmin_table;
  } else if (Egas >= Eint_max) {
    return Tmax_table;
  }

  // solve for temperature given Eint (with fixed adiabatic index gamma)
  const Real rhoH = rho * cloudy_H_mass_fraction;
  const Real nH = rhoH / hydrogen_mass_cgs_;
  const Real log_nH = std::log10(nH);

  // mean molecular weight (in Grackle tables) is defined w/r/t
  // hydrogen_mass_cgs_
  const Real C = (gamma - 1.) * Egas /
                 (boltzmann_constant_cgs_ * (rho / hydrogen_mass_cgs_));

  // solve for mu(T)*C == T.
  // (Grackle does this with a fixed-point iteration. We use a more robust
  // method, similar to Brent's method, the TOMS748 method.)
  const Real reltol = 1.0e-5;
  const int maxIterLimit = 100;
  int maxIter = maxIterLimit;

  auto f = [log_nH, C, tables](const Real &T) noexcept {
    // compute new mu from mu(log10 T) table
    Real log_T = clamp(std::log10(T), 1., 9.);
    Real mu = interpolate2d(log_nH, log_T, tables.log_nH, tables.log_Tgas,
                            tables.meanMolWeight);
    Real fun = C * mu - T;
    return fun;
  };

  // compute temperature bounds using physics
  const Real mu_min = 0.60; // assuming fully ionized (mu ~ 0.6)
  const Real mu_max = 2.33; // assuming neutral fully molecular (mu ~ 2.33)
  const Real T_min = std::clamp(C * mu_min, Tmin_table, Tmax_table);
  const Real T_max = std::clamp(C * mu_max, Tmin_table, Tmax_table);

  // do root-finding
  quokka::math::eps_tolerance<Real> tol(reltol);
  auto bounds = quokka::math::toms748_solve(f, T_min, T_max, tol, maxIter);
  Real T_sol = 0.5 * (bounds.first + bounds.second);

  if ((maxIter >= maxIterLimit) || std::isnan(T_sol)) {
    printf("\nTgas iteration failed! rho = %.17g, Eint = %.17g, nH = %e, Tgas "
           "= %e, "
           "bounds.first = %e, bounds.second = %e, T_min = %e, T_max = %e, "
           "maxIter = %d\n",
           rho, Egas, nH, T_sol, bounds.first, bounds.second, T_min, T_max,
           maxIter);
    T_sol = NAN;
  }

  return T_sol;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto
ComputeCoolingLength(double rho, double Egas, double gamma,
                    cloudyGpuConstTables const &tables) -> Real {
  // convert (rho, Egas) to cooling length

  // 1. convert Egas (internal gas energy) to temperature
  const Real Tgas = ComputeTgasFromEgas(rho, Egas, gamma, tables);

  // 2. compute cooling time
  // interpolate cooling rates from Cloudy tables
  const Real rhoH = rho * cloudy_H_mass_fraction; // mass density of H species
  const Real nH = rhoH / hydrogen_mass_cgs_;
  const Real log_nH = std::log10(nH);
  const Real log_T = std::log10(Tgas);
  const double logCool = interpolate2d(log_nH, log_T, tables.log_nH,
                                       tables.log_Tgas, tables.cool);
  const double LambdaCool = FastMath::pow10(logCool);
  const double Edot = (rhoH * rhoH) * LambdaCool;
  // compute cooling time
  const Real t_cool = Egas / Edot;

  // 3. compute cooling length c_s t_cool
  // compute mu from mu(T) table
  const Real mu = interpolate2d(log_nH, log_T, tables.log_nH,
                                tables.log_Tgas, tables.meanMolWeight);
  const Real c_s = std::sqrt(gamma * boltzmann_constant_cgs_ * Tgas / (mu * hydrogen_mass_cgs_));

  // cooling length
  return c_s * t_cool;
}

void readCloudyData(cloudy_tables &cloudyTables);

#endif // CLOUDYCOOLING_HPP_
