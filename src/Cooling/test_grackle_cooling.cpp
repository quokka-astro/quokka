//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file test_grackle_cooling.cpp
/// \brief Testing the Grackle-like cooling implementation against a 'known good' solution.
///

// uncomment this to debug the root-finding code (does NOT work on GPU!)
// #define BOOST_MATH_INSTRUMENT

#include "AMReX_ParmParse.H"
#include "EOS.hpp"
#include "GrackleLikeCooling.hpp"
#include "ODEIntegrate.hpp"

using amrex::Real;

using quokka::GrackleLikeCooling::electron_mass_cgs;
using quokka::GrackleLikeCooling::mean_metals_A;
using quokka::GrackleLikeCooling::X;
using quokka::GrackleLikeCooling::Y;
using quokka::GrackleLikeCooling::Z;

static constexpr double gamma = 5. / 3.; // default value

auto problem_main() -> int
{
	// Problem parameters
	const Real rho = 2.27766918428822386e-22;  // g cm^-3;
	const Real Eint = 1.11777608454088878e-11; // erg cm^-3
	const Real dt = 1.92399749834457487e8;	   // s

	// Read Cloudy tables
	quokka::GrackleLikeCooling::grackle_tables cloudyTables;
	std::string filename;
	amrex::ParmParse const pp("cooling");
	pp.query("grackle_data_file", filename);
	quokka::GrackleLikeCooling::readGrackleData(filename, cloudyTables);
	auto tables = cloudyTables.const_tables();

	const Real T0 = quokka::GrackleLikeCooling::ComputeTgasFromEgas(rho, Eint, gamma, tables);

	const Real rhoH = rho * quokka::GrackleLikeCooling::cloudy_H_mass_fraction;
	const Real nH = rhoH / (C::m_p + C::m_e);
	const Real log_nH = std::log10(nH);

	const Real C = (gamma - 1.) * Eint / (C::k_B * (rho / (C::m_p + C::m_e)));
	const Real mu = interpolate2d(log_nH, std::log10(T0), tables.log_nH, tables.log_Tgas, tables.meanMolWeight);
	const Real relerr = std::abs((C * mu - T0) / T0);

	const Real n_e = (rho / (C::m_p + C::m_e)) * (1.0 - mu * (X + Y / 4. + Z / mean_metals_A)) / (mu - (electron_mass_cgs / (C::m_p + C::m_e)));

	printf("\nrho = %.17e, Eint = %.17e, mu = %f, Tgas = %e, relerr = %e\n", rho, Eint, mu, T0, relerr); // NOLINT
	printf("n_e = %e, n_e/n_H = %e\n", n_e, n_e / nH);						     // NOLINT

	const Real reltol_floor = 0.01;
	const Real rtol = 1.0e-4;	// not recommended to change this
	constexpr Real T_floor = 100.0; // K

	quokka::GrackleLikeCooling::ODEUserData user_data{rho, gamma, tables};
	quokka::valarray<Real, 1> y = {Eint};
	quokka::valarray<Real, 1> const abstol = {reltol_floor * quokka::GrackleLikeCooling::ComputeEgasFromTgas(rho, T_floor, gamma, tables)};

	// do integration with RK2 (Heun's method)
	int nsteps = 0;
	rk_adaptive_integrate(quokka::GrackleLikeCooling::user_rhs, 0, y, dt, &user_data, rtol, abstol, nsteps);

	// check if integration failed
	int status = -1;
	if (nsteps >= maxStepsODEIntegrate) {
		const Real T_final = quokka::GrackleLikeCooling::ComputeTgasFromEgas(rho, y[0], gamma, tables);
		const Real Edot = quokka::GrackleLikeCooling::cloudy_cooling_function(rho, T_final, tables);
		const Real t_cool = Eint / Edot;
		printf("max substeps exceeded! rho = %g, Eint = %g, T = %g, cooling " // NOLINT
		       "time = %g\n",
		       rho, Eint, T_final, t_cool);
		status = 1;
	} else {
		const Real T_final = quokka::GrackleLikeCooling::ComputeTgasFromEgas(rho, y[0], gamma, tables);
		const Real Edot = quokka::GrackleLikeCooling::cloudy_cooling_function(rho, T_final, tables);
		const Real t_cool = Eint / Edot;
		printf("success! rho = %g, Eint = %g, T = %g, cooling time = %g\n", rho, Eint, T_final, t_cool); // NOLINT
		status = 0;
	}

	// Cleanup and exit
	return status;
}
