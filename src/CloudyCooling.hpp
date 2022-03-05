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
#include <limits>

// From Grackle source code (initialize_chemistry_data.c, line 114):
//   In fully tabulated mode, set H mass fraction according to
//   the abundances in Cloudy, which assumes n_He / n_H = 0.1.
//   This gives a value of about 0.716. Using the default value
//   of 0.76 will result in negative electron densities at low
//   temperature. Below, we set X = 1 / (1 + hydrogen_mass_cgs_e * n_He / n_H).

constexpr double cloudy_H_mass_fraction = 1. / (1. + 0.1 * 3.971);
constexpr double sigma_T = 6.6524e-25; // Thomson cross section (cm^2)
constexpr double electron_mass_cgs = 9.1093897e-28; // electron mass (g)
constexpr double T_cmb = 2.725;                     // * (1 + z); // K
constexpr double E_cmb =
    radiation_constant_cgs_ * (T_cmb * T_cmb * T_cmb * T_cmb);

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

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto
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

  const double netLambda_prim =
      std::pow(10., logPrimHeat) - std::pow(10., logPrimCool);
  const double netLambda_metals =
      std::pow(10., logMetalHeat) - std::pow(10., logMetalCool);
  const double netLambda = netLambda_prim + netLambda_metals;

  // multiply by the square of H mass density (**NOT number density**)
  double Edot = (rhoH * rhoH) * netLambda;

  // compute dimensionless mean mol. weight mu from mu(T) table
  const double mu = interpolate2d(log_nH, log_T, tables.log_nH, tables.log_Tgas,
                                  tables.meanMolWeight);

  // compute electron density
  const double n_e =
      (rho / mu) * (1.0 - mu * (3.0 * cloudy_H_mass_fraction + 1.0) / 4.0);

  // photoelectric heating term
  const double Tsqrt = std::sqrt(T);
  constexpr double phi = 0.5; // phi_PAH from Wolfire et al. (2003)
  constexpr double G_0 = 1.7; // ISRF from Wolfire et al. (2003)
  const double epsilon =
      4.9e-2 / (1. + 4.0e-3 * std::pow(G_0 * Tsqrt / (n_e * phi), 0.73)) +
      3.7e-2 * std::pow(T / 1.0e4, 0.7) /
          (1. + 2.0e-4 * (G_0 * Tsqrt / (n_e * phi)));
  const double Gamma_pe = 1.3e-24 * nH * epsilon * G_0;
  Edot += Gamma_pe;

  // Compton term (CMB photons)
  // [e.g., Hirata 2018: doi:10.1093/mnras/stx2854]
  constexpr double Gamma_C =
      (8. * sigma_T * E_cmb) / (3. * electron_mass_cgs * c_light_cgs_);
  constexpr double C_n = Gamma_C * boltzmann_constant_cgs_ / (5. / 3. - 1.0);
  const double compton_CMB = -C_n * (T - T_cmb) * n_e;
  Edot += compton_CMB;

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
  const Real Egas = (rho / (hydrogen_mass_cgs_ * mu)) *
                    boltzmann_constant_cgs_ * Tgas / (gamma - 1.);
  return Egas;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto
ComputeTgasFromEgas(double rho, double Egas, double gamma,
                    cloudyGpuConstTables const &tables) -> Real {
  // convert Egas (internal gas energy) to temperature

  // check whether temperature is out-of-bounds
  const Real Tmin = 10.;
  const Real Tmax = 1.0e9;
  const Real Eint_min = ComputeEgasFromTgas(rho, Tmin, gamma, tables);
  const Real Eint_max = ComputeEgasFromTgas(rho, Tmax, gamma, tables);

  if (Egas <= Eint_min) {
    return Tmin;
  } else if (Egas >= Eint_max) {
    return Tmax;
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
  const Real T_min = C * 0.60; // assuming fully ionized (mu ~ 0.6)
  const Real T_max = C * 2.33; // assuming neutral fully molecular (mu ~ 2.33)

  // do root-finding
  quokka::math::eps_tolerance<Real> tol(reltol);
  auto bounds = quokka::math::toms748_solve(f, T_min, T_max, tol, maxIter);
  const Real T_sol = 0.5 * (bounds.first + bounds.second);

  if ((maxIter >= maxIterLimit) || std::isnan(T_sol)) {
    printf(
        "\nTgas iteration failed! rho = %.17g, Eint = %.17g, nH = %f, Tgas = %f, "
        "bounds.first = %f, bounds.second = %f, maxIter = %d\n",
        rho, Egas, nH, T_sol, bounds.first, bounds.second, maxIter);
  }
  AMREX_ALWAYS_ASSERT_WITH_MESSAGE(maxIter < maxIterLimit,
                                   "Temperature bisection failed!");

  return T_sol;
}

void readCloudyData(cloudy_tables &cloudyTables);

#endif // CLOUDYCOOLING_HPP_
