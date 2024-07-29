//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file test_cloudy_cooling.cpp
/// \brief Testing the tabulated cooling implementation against a 'known good' solution.
///

#include "AMReX_ParmParse.H"
#include "EOS.hpp"
#include "ODEIntegrate.hpp"
#include "TabulatedCooling.hpp"
#include "fundamental_constants.H"

using amrex::Real;

static constexpr double gamma = 5. / 3.; // default value

auto problem_main() -> int
{
	// Problem parameters
	const Real rho = 100.0 * (C::m_p + C::m_e) / quokka::TabulatedCooling::cloudy_H_mass_fraction;
	const Real T0 = 1.0e4;	 // K
	const Real dt = 3.15e14; // s

	// Read Cloudy tables
	quokka::TabulatedCooling::cloudy_tables cloudyTables;
	std::string filename;
	amrex::ParmParse const pp("cooling");
	pp.query("hdf5_data_file", filename);
	quokka::TabulatedCooling::readCloudyData(filename, cloudyTables);
	auto tables = cloudyTables.const_tables();

	const Real Eint = quokka::TabulatedCooling::ComputeEgasFromTgas(rho, T0, gamma, tables);
	const Real rhoH = rho * quokka::TabulatedCooling::cloudy_H_mass_fraction;
	const Real nH = rhoH / (C::m_p + C::m_e);
	const Real log_nH = std::log10(nH);

	// check that the temperature is computed correctly
	const Real C = (gamma - 1.) * Eint / (C::k_B * (rho / (C::m_p + C::m_e)));
	const Real mu = interpolate2d(log_nH, std::log10(T0), tables.log_nH, tables.log_Tgas, tables.meanMolWeight);
	const Real relerr = std::abs((C * mu - T0) / T0);

	// print initial conditions
	printf("\nrho = %.17e, Eint = %.17e, nH = %e, mu = %f, Tgas = %e, relerr = %e\n", rho, Eint, nH, mu, T0, relerr); // NOLINT

	const Real reltol_floor = 0.01;
	const Real rtol = 1.0e-4;	// not recommended to change this
	constexpr Real T_floor = 100.0; // K

	quokka::TabulatedCooling::ODEUserData user_data{rho, gamma, tables};
	quokka::valarray<Real, 1> y = {Eint};
	quokka::valarray<Real, 1> const abstol = {reltol_floor * quokka::TabulatedCooling::ComputeEgasFromTgas(rho, T_floor, gamma, tables)};

	// do integration with RK2 (Heun's method)
	int nsteps = 0;
	rk_adaptive_integrate(quokka::TabulatedCooling::user_rhs, 0, y, dt, &user_data, rtol, abstol, nsteps);

	// check if integration succeeded
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
		Real const T_expected = 1947.82;
		Real const Edot = quokka::TabulatedCooling::cloudy_cooling_function(rho, T_final, tables);
		Real const t_cool = y[0] / Edot;
		Real const relerr = std::abs((T_final - T_expected) / T_expected);

		if (relerr < rtol) {
			printf("success!\n"); // NOLINT
			status = 0;
		} else {
			printf("failure: integrator obtained the wrong answer!\n"); // NOLINT
			printf("expected answer: T = %g\n", T_expected);	    // NOLINT
			status = 1;
		}
		printf("rho = %g, Eint = %g, T = %g, cooling time = %g\n", rho, y[0], T_final, t_cool); // NOLINT
	}

	// Cleanup and exit
	return status;
}
