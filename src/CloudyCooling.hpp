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

#include "AMReX_Extension.H"
#include "AMReX_GpuQualifiers.H"

#include "GrackleDataReader.hpp"
#include "Interpolate2D.hpp"
#include "radiation_system.hpp"

// From Grackle source code (initialize_chemistry_data.c, line 114):
//   In fully tabulated mode, set H mass fraction according to
//   the abundances in Cloudy, which assumes n_He / n_H = 0.1.
//   This gives a value of about 0.716. Using the default value
//   of 0.76 will result in negative electron densities at low
//   temperature. Below, we set X = 1 / (1 + hydrogen_mass_cgs_e * n_He / n_H).

constexpr Real cloudy_H_mass_fraction = 1. / (1. + 0.1 * 3.971);

struct cloudyGpuConstTables {
  // these are non-owning, so can make a copy of the whole struct
  amrex::Table1D<const Real> const log_nH;
  amrex::Table1D<const Real> const log_Tgas;

  amrex::Table2D<const Real> const primCool;
  amrex::Table2D<const Real> const primHeat;
  amrex::Table2D<const Real> const metalCool;
  amrex::Table2D<const Real> const metalHeat;
  amrex::Table2D<const Real> const meanMolWeight;
};

class cloudy_tables {
public:
  std::unique_ptr<amrex::TableData<double, 1>> log_nH;
  std::unique_ptr<amrex::TableData<double, 1>> log_Tgas;

  std::unique_ptr<amrex::TableData<double, 2>> primCooling;
  std::unique_ptr<amrex::TableData<double, 2>> primHeating;
  std::unique_ptr<amrex::TableData<double, 2>> metalCooling;
  std::unique_ptr<amrex::TableData<double, 2>> metalHeating;
  std::unique_ptr<amrex::TableData<double, 2>> mean_mol_weight;

  [[nodiscard]] auto const_tables() const -> cloudyGpuConstTables;
};

AMREX_GPU_HOST_DEVICE AMREX_INLINE auto
cloudy_cooling_function(Real const rho, Real const T,
                        cloudyGpuConstTables const &tables) -> Real {
  // interpolate cooling rates from Cloudy tables
  const Real rhoH = rho * cloudy_H_mass_fraction; // mass density of H species
  const Real nH = rhoH / hydrogen_mass_cgs_;
  const Real log_nH = std::log10(nH);
  const Real log_T = std::log10(T);

  const double logPrimCool = interpolate2d(log_nH, log_T, tables.log_nH,
                                           tables.log_Tgas, tables.primCool);

  const double logPrimHeat = interpolate2d(log_nH, log_T, tables.log_nH,
                                           tables.log_Tgas, tables.primHeat);

  const double logMetalCool = interpolate2d(log_nH, log_T, tables.log_nH,
                                            tables.log_Tgas, tables.metalCool);

  const double logMetalHeat = interpolate2d(log_nH, log_T, tables.log_nH,
                                            tables.log_Tgas, tables.metalHeat);

  const double netLambda_over_nsq_prim =
      std::pow(10., logPrimHeat) - std::pow(10., logPrimCool);
  const double netLambda_over_nsq_metals =
      std::pow(10., logMetalHeat) - std::pow(10., logMetalCool);
  const double netLambda_over_nsq =
      netLambda_over_nsq_prim + netLambda_over_nsq_metals;

  // multiply by the square of H mass density (**NOT number density**)
  const double netLambda = (rhoH * rhoH) * netLambda_over_nsq;

  return netLambda;
}

AMREX_GPU_HOST_DEVICE AMREX_INLINE auto
ComputeTgasFromEgas(double rho, double Egas, double gamma,
                    cloudyGpuConstTables const &tables) -> Real {
  // convert Egas (internal gas energy) to temperature

  // solve for temperature given Eint (with fixed adiabatic index gamma)
  const Real rhoH = rho * cloudy_H_mass_fraction;
  const Real nH = rhoH / hydrogen_mass_cgs_;
  const Real log_nH = std::log10(nH);

  // mean molecular weight (in Grackle tables) is defined w/r/t
  // hydrogen_mass_cgs_
  const Real C = (gamma - 1.) * Egas /
                 (boltzmann_constant_cgs_ * (rho / hydrogen_mass_cgs_));

  // solve for mu(T)*C == T.
  // (Grackle does this with a fixed-point iteration.)
  Real mu_prev = NAN;
  Real mu_guess = 1.;
  Real Tgas = NAN;
  Real T_guess = NAN;
  const Real reltol = 1.0e-3;
  const int maxIter = 50;

  for (int n = 0; n < maxIter; ++n) {
    mu_prev = mu_guess; // save old mu

    // compute new guess for Tgas, bounded between 10 K and 1e9 K
    T_guess = std::clamp(C * mu_guess, 10., 1.0e9);

    // compute new mu from mu(T) table
    mu_guess = interpolate2d(log_nH, std::log10(T_guess), tables.log_nH,
                             tables.log_Tgas, tables.meanMolWeight);

    // damp iteration
    mu_guess = 0.5 * (mu_guess + mu_prev);

    // check if converged
    if (std::abs((C * mu_guess - T_guess) / T_guess) < reltol) {
      Tgas = T_guess;
      break;
    }
  }
  AMREX_ALWAYS_ASSERT(!std::isnan(Tgas));
  //   "Tgas iteration failed to converge!"
  return Tgas;
}

AMREX_GPU_HOST_DEVICE AMREX_INLINE auto
ComputeEgasFromTgas(double rho, double Tgas, double gamma,
                    cloudyGpuConstTables const &tables) -> Real {
  // convert Egas (internal gas energy) to temperature
  const Real rhoH = rho * cloudy_H_mass_fraction;
  const Real nH = rhoH / hydrogen_mass_cgs_;

  // compute mu from mu(T) table
  const Real mu = interpolate2d(std::log10(nH), std::log10(Tgas), tables.log_nH,
                                tables.log_Tgas, tables.meanMolWeight);

  // compute thermal gas energy
  const Real Egas = (rho / (hydrogen_mass_cgs_ * mu)) *
                    boltzmann_constant_cgs_ * Tgas / (gamma - 1.);
  return Egas;
}

void readCloudyData(cloudy_tables &cloudyTables);

#endif // CLOUDYCOOLING_HPP_