//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file test_radiation_marshak.cpp
/// \brief Defines a test problem for radiation in the diffusion regime.
///

#include "test_radiation_marshak.hpp"

void testproblem_radiation_marshak()
{
	// For this problem, you must do reconstruction in the reduced
	// flux, *not* the flux. Otherwise, F exceeds cE at sharp temperature
	// gradients.

	// Problem parameters

	const int nx = 300;
	const double Lx = 30.0; // cm
	const double CFL_number = 0.4;
	const double constant_dt = 1.0e-12; // s
	const double max_time = 1.0e-9;	    // s
	const int max_timesteps = 1000;

	const double rho = 1.0;		      // g cm^-3
	const double initial_Tgas = 10.;      // K
	const double initial_Trad = 10.;      // K
	const double T_hohlraum = 3.481334e6; // K [= 300 eV]
	const double eps_SuOlson = 1.0;	      // Su & Olson (1997) test problem
	const double x0 = 5.0;		      // cm
	const double max_time_source_on = 1e-9; // s

	// Problem initialization

	RadSystem rad_system(RadSystem::Nx = nx, RadSystem::Lx = Lx,
			     RadSystem::CFL = CFL_number);

	auto nghost = rad_system.nghost();

	const auto a_rad = rad_system.radiation_constant();
	const double kelvin_to_eV = 8.617385e-5;
	const double alpha_SuOlson = 4.0 * a_rad / eps_SuOlson;

	const auto initial_Egas =
	    rho * alpha_SuOlson * std::pow(initial_Tgas, 4);
	const auto initial_Erad = a_rad * std::pow(initial_Trad, 4);

	for (int i = nghost; i < nx + nghost; ++i) {
		const auto idx_value = static_cast<double>(i - nghost);
		const double x =
		    Lx * ((idx_value + 0.5) / static_cast<double>(nx));

		rad_system.set_radEnergy(i) = initial_Erad;
		rad_system.set_x1RadFlux(i) = 0.0;
		rad_system.set_gasEnergy(i) = initial_Egas;
		rad_system.set_staticGasDensity(i) = rho;
		if (x < x0) {
			rad_system.set_radEnergySource(i) =
			    a_rad * std::pow(T_hohlraum, 4);
		}
	}

	const auto Erad0 = rad_system.ComputeRadEnergy();
	const auto Egas0 = rad_system.ComputeGasEnergy();
	const auto Etot0 = Erad0 + Egas0;

	const auto kappa = rad_system.ComputeOpacity(rho, initial_Tgas);
	const auto heating_time =
	    initial_Egas / (rho * kappa * rad_system.c_light_ *
			    (a_rad * std::pow(initial_Tgas, 4) - initial_Erad));

	std::cout << "Initial radiation temperature = " << initial_Trad << "\n";
	std::cout << "Initial gas temperature = " << initial_Tgas << "\n";
	std::cout << "Heating time = " << heating_time << "\n";

	// Main time loop

	for (int j = 0; j < max_timesteps; ++j) {
		if (rad_system.time() >= max_time) {
			break;
		}

		if (rad_system.time() >= max_time_source_on) {
			for (int i = nghost; i < nx + nghost; ++i) {
				rad_system.set_radEnergySource(i) = 0.0;
			}
		}

		rad_system.AdvanceTimestep(constant_dt);
		rad_system.AddSourceTerms(std::make_pair(nghost, nghost + nx));

		std::cout << "Timestep " << j << "; t = " << rad_system.time()
			  << "\n";

		const auto Erad = rad_system.ComputeRadEnergy();
		const auto Egas = rad_system.ComputeGasEnergy();
		const auto Etot = Erad + Egas;
		const auto Ediff = std::fabs(Etot - Etot0);

		std::cout << "radiation energy = " << Erad << "\n";
		std::cout << "gas energy = " << Egas << "\n";
		std::cout << "Total energy = " << Etot << "\n";
		std::cout << "(Energy nonconservation = " << Ediff << ")\n";
		std::cout << "\n";
	}

	std::vector<double> xs(nx);
	std::vector<double> Trad(nx);
	std::vector<double> Tgas(nx);

	for (int i = 0; i < nx; ++i) {
		const double x = Lx * ((i + 0.5) / static_cast<double>(nx));
		xs.at(i) = x;

		Trad.at(i) =
		    kelvin_to_eV *
		    std::pow(rad_system.radEnergy(i + nghost) / a_rad, 1. / 4.);

		const double this_Egas = rad_system.gasEnergy(i + nghost);
		Tgas.at(i) =
		    kelvin_to_eV *
		    std::pow(this_Egas / (rho * alpha_SuOlson), (1. / 4.));
	}

	matplotlibcpp::clf();

	std::map<std::string, std::string> Trad_args;
	Trad_args["label"] = "radiation temperature";
	matplotlibcpp::plot(xs, Trad, Trad_args);

	std::map<std::string, std::string> Tgas_args;
	Tgas_args["label"] = "gas temperature";
	matplotlibcpp::plot(xs, Tgas, Tgas_args);

	matplotlibcpp::legend();
	matplotlibcpp::xlabel("length x (cm)");
	matplotlibcpp::ylabel("temperature (eV)");
	matplotlibcpp::title(
	    fmt::format("time t = {:.4g}", constant_dt, rad_system.time()));
	matplotlibcpp::save(fmt::format("./marshak_wave.pdf"));

	// Cleanup and exit
	std::cout << "Finished." << std::endl;
}
