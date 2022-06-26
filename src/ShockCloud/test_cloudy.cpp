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

template <> struct HydroSystem_Traits<ShockCloud> {
  static constexpr double gamma = 5. / 3.; // default value
  // if true, reconstruct e_int instead of pressure
  static constexpr bool reconstruct_eint = true;
};

struct ODEUserData {
  Real rho;
  cloudyGpuConstTables tables;
};

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto
user_rhs(Real /*t*/, quokka::valarray<Real, 1> &y_data,
         quokka::valarray<Real, 1> &y_rhs, void *user_data) -> int {
  // unpack user_data
  auto *udata = static_cast<ODEUserData *>(user_data);
  const Real rho = udata->rho;
  const Real gamma = HydroSystem<ShockCloud>::gamma_;
  cloudyGpuConstTables &tables = udata->tables;

  // check whether temperature is out-of-bounds
  const Real Tmin = 10.;
  const Real Tmax = 1.0e9;
  const Real Eint_min = ComputeEgasFromTgas(rho, Tmin, gamma, tables);
  const Real Eint_max = ComputeEgasFromTgas(rho, Tmax, gamma, tables);

  // compute temperature and cooling rate
  const Real Eint = y_data[0];

  if (Eint <= Eint_min) {
    // set cooling to value at Tmin
    y_rhs[0] = cloudy_cooling_function(rho, Tmin, tables);
  } else if (Eint >= Eint_max) {
    // set cooling to value at Tmax
    y_rhs[0] = cloudy_cooling_function(rho, Tmax, tables);
  } else {
    // ok, within tabulated cooling limits
    const Real T = ComputeTgasFromEgas(rho, Eint, gamma, tables);
    if (!std::isnan(T)) { // temp iteration succeeded
      y_rhs[0] = cloudy_cooling_function(rho, T, tables);
      AMREX_ASSERT(!std::isnan(y_rhs[0]));
    } else { // temp iteration failed
      y_rhs[0] = NAN;
      return 1; // failed
    }
  }

  return 0; // success
}

auto problem_main() -> int {
  // Problem parameters
  const Real rho = 2.27766918428822386e-22;  // g cm^-3;
  const Real Eint = 1.11777608454088878e-11; // erg cm^-3
  const Real dt = 1.92399749834457487e8;     // s

  // Read Cloudy tables
  cloudy_tables cloudyTables;
  readCloudyData(cloudyTables);
  auto tables = cloudyTables.const_tables();

  const Real T =
      ComputeTgasFromEgas(rho, Eint, HydroSystem<ShockCloud>::gamma_, tables);

  const Real rhoH = rho * cloudy_H_mass_fraction;
  const Real nH = rhoH / hydrogen_mass_cgs_;
  const Real log_nH = std::log10(nH);

  const Real C = (HydroSystem<ShockCloud>::gamma_ - 1.) * Eint /
                 (boltzmann_constant_cgs_ * (rho / hydrogen_mass_cgs_));
  const Real mu = interpolate2d(log_nH, std::log10(T), tables.log_nH,
                                    tables.log_Tgas, tables.meanMolWeight);
  const Real relerr = std::abs((C * mu - T) / T);

  const Real n_e = (rho / hydrogen_mass_cgs_) *
                     (1.0 - mu * (X + Y / 4. + Z / mean_metals_A)) /
                     (mu - (electron_mass_cgs / hydrogen_mass_cgs_));

  printf("\nrho = %.17e, Eint = %.17e, mu = %f, Tgas = %e, relerr = %e\n", rho,
         Eint, mu, T, relerr);
  printf("n_e = %e, n_e/n_H = %e\n", n_e, n_e/nH);

  const Real reltol_floor = 0.01;
  const Real rtol = 1.0e-4;       // not recommended to change this
  constexpr Real T_floor = 100.0; // K

  ODEUserData user_data{rho, tables};
  quokka::valarray<Real, 1> y = {Eint};
  quokka::valarray<Real, 1> abstol = {
      reltol_floor * ComputeEgasFromTgas(rho, T_floor,
                                         HydroSystem<ShockCloud>::gamma_,
                                         tables)};

  // do integration with RK2 (Heun's method)
  int nsteps = 0;
  rk_adaptive_integrate(user_rhs, 0, y, dt, &user_data, rtol, abstol, nsteps, true);

  // check if integration failed
  if (nsteps >= maxStepsODEIntegrate) {
    Real T =
        ComputeTgasFromEgas(rho, Eint, HydroSystem<ShockCloud>::gamma_, tables);
    Real Edot = cloudy_cooling_function(rho, T, tables);
    Real t_cool = Eint / Edot;
    printf("max substeps exceeded! rho = %g, Eint = %g, T = %g, cooling "
           "time = %g\n",
           rho, Eint, T, t_cool);
  } else {
    Real T =
        ComputeTgasFromEgas(rho, Eint, HydroSystem<ShockCloud>::gamma_, tables);
    Real Edot = cloudy_cooling_function(rho, T, tables);
    Real t_cool = Eint / Edot;
    printf("success! rho = %g, Eint = %g, T = %g, cooling time = %g\n", rho,
           Eint, T, t_cool);
  }

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
