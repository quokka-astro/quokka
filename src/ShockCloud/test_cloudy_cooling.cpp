//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file test_cloudy_cooling.cpp
/// \brief Testing the tabulated cooling implementation against a 'known good' solution.
///

// uncomment this to debug the root-finding code (does NOT work on GPU!)
// #define BOOST_MATH_INSTRUMENT

#include "AMReX_ParmParse.H"
#include "EOS.hpp"
#include "ODEIntegrate.hpp"
#include "TabulatedCooling.hpp"

using amrex::Real;

static constexpr double gamma = 5. / 3.; // default value

auto problem_main() -> int
{
	// Problem parameters
	const Real rho = 2.27766918428822386e-22;  // g cm^-3;
	const Real Eint = 1.11777608454088878e-11; // erg cm^-3
	const Real dt = 1.92399749834457487e8;	   // s

	// Read Cloudy tables
	quokka::TabulatedCooling::cloudy_tables cloudyTables;
	std::string filename;
	amrex::ParmParse const pp("cooling");
	pp.query("hdf5_data_file", filename);
	quokka::TabulatedCooling::readCloudyData(filename, cloudyTables);
	auto tables = cloudyTables.const_tables();

	const Real T0 = quokka::TabulatedCooling::ComputeTgasFromEgas(rho, Eint, gamma, tables);

	const Real rhoH = rho * quokka::TabulatedCooling::cloudy_H_mass_fraction;
	const Real nH = rhoH / (C::m_p + C::m_e);
	const Real log_nH = std::log10(nH);

	const Real C = (gamma - 1.) * Eint / (C::k_B * (rho / (C::m_p + C::m_e)));
	const Real mu = interpolate2d(log_nH, std::log10(T0), tables.log_nH, tables.log_Tgas, tables.meanMolWeight);
	const Real relerr = std::abs((C * mu - T0) / T0);

	printf("\nrho = %.17e, Eint = %.17e, mu = %f, Tgas = %e, relerr = %e\n", rho, Eint, mu, T0, relerr); // NOLINT

	const Real reltol_floor = 0.01;
	const Real rtol = 1.0e-4;	// not recommended to change this
	constexpr Real T_floor = 100.0; // K

	quokka::TabulatedCooling::ODEUserData user_data{rho, gamma, tables};
	quokka::valarray<Real, 1> y = {Eint};
	quokka::valarray<Real, 1> const abstol = {reltol_floor * quokka::TabulatedCooling::ComputeEgasFromTgas(rho, T_floor, gamma, tables)};

	// do integration with RK2 (Heun's method)
	int nsteps = 0;
	rk_adaptive_integrate(quokka::TabulatedCooling::user_rhs, 0, y, dt, &user_data, rtol, abstol, nsteps);

	// check if integration failed
	int status = -1;
	if (nsteps >= maxStepsODEIntegrate) {
		Real const T_final = quokka::TabulatedCooling::ComputeTgasFromEgas(rho, y[0], gamma, tables);
		Real const Edot = quokka::TabulatedCooling::cloudy_cooling_function(rho, T_final, tables);
		Real const t_cool = Eint / Edot;
		printf("max substeps exceeded! rho = %g, Eint = %g, T = %g, cooling " // NOLINT
		       "time = %g\n",
		       rho, Eint, T_final, t_cool);
		status = 1;
	} else {
		Real const T_final = quokka::TabulatedCooling::ComputeTgasFromEgas(rho, y[0], gamma, tables);
		Real const Edot = quokka::TabulatedCooling::cloudy_cooling_function(rho, T_final, tables);
		Real const t_cool = Eint / Edot;
		printf("success! rho = %g, Eint = %g, T = %g, cooling time = %g\n", rho, Eint, T_final, t_cool); // NOLINT
		status = 0;
	}

	// Cleanup and exit
	return status;
}
