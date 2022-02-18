//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file test_ode.cpp
/// \brief Defines a test problem for ODE integration.
///

#include "AMReX_MultiFab.H"
#include "AMReX_REAL.H"

#include "hydro_system.hpp"
#include "radiation_system.hpp"
#include "rk4.hpp"
#include "test_ode.hpp"
#include "valarray.hpp"

using amrex::Real;

struct ODETest {};

constexpr double m_H = hydrogen_mass_cgs_;
constexpr double seconds_in_year = 3.154e7;

constexpr double Tgas0 = 6000.;     // K
constexpr double rho0 = 0.01 * m_H; // g cm^-3

struct ODEUserData {
  Real rho = NAN;
};

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

static auto user_rhs(Real t, quokka::valarray<Real, 1> &y_data,
                     quokka::valarray<Real, 1> &y_rhs, void *user_data) -> int {
  auto *udata = static_cast<ODEUserData *>(user_data);
  Real rho = udata->rho;
  Real Eint = y_data[0];
  Real T = RadSystem<ODETest>::ComputeTgasFromEgas(rho, Eint);
  y_rhs[0] = cooling_function(rho, T);
  return 0;
}

auto problem_main() -> int {
  // Problem parameters
  const double max_time = 1.0e7 * seconds_in_year; // 10 Myr

  // run simulation
  const Real Eint0 = RadSystem<ODETest>::ComputeEgasFromTgas(rho0, Tgas0);
  const Real Edot0 = cooling_function(rho0, Tgas0);
  const Real tcool = std::abs(Eint0 / Edot0);
  std::cout << "Initial temperature: " << Tgas0 << std::endl;
  std::cout << "Initial cooling time: " << tcool << std::endl;

  ODEUserData user_data{rho0};
  quokka::valarray<Real, 1> y = {Eint0};
  quokka::valarray<Real, 1> abstol = 1.0e-20 * y;
  const Real rtol = 1.0e-15;

  rk45_adaptive_integrate(user_rhs, 0, y, max_time, &user_data, rtol, abstol);

  const Real Tgas = RadSystem<ODETest>::ComputeTgasFromEgas(rho0, y[0]);
  const Real Teq = 160.52611612610758; // for n_H = 0.01 cm^{-3}
  const Real Terr_rel = std::abs(Tgas - Teq) / Teq;
  const Real reltol = 1.0e-15; // relative error tolerance
  std::cout << "Final temperature: " << Tgas << std::endl;
  std::cout << "Relative error: " << Terr_rel << std::endl;

  // Cleanup and exit
  int status = 0;
  if ((Terr_rel > reltol) || (std::isnan(Terr_rel))) {
    status = 1;
  }
  return status;
}
