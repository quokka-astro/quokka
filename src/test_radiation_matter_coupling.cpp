//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file test_radiation_matter_coupling.cpp
/// \brief Defines a test problem for radiation-matter coupling.
///

#include "test_radiation_matter_coupling.hpp"

void testproblem_radiation_matter_coupling()
{
	// Problem parameters

	const int nx = 4;
	const double Lx = 1.0;
	const double CFL_number = 0.8;
	const double max_time = 1.0e3;
	const int max_timesteps = 5000;

	// Problem initialization

	RadSystem rad_system(RadSystem::Nx = nx, RadSystem::Lx = Lx,
			     RadSystem::CFL = CFL_number);

	auto nghost = rad_system.nghost();

	for (int i = nghost; i < nx + nghost; ++i) {
		const double Erad = 100.0;
		const double Egas = 1.0;
		const double rho = 1.0;

		rad_system.set_radEnergy(i) = Erad;
		rad_system.set_x1RadFlux(i) = 0.0;
		rad_system.set_gasEnergy(i) = Egas;
		rad_system.set_staticGasDensity(i) = rho;
	}

	std::vector<double> t;
	std::vector<double> Erad;
	std::vector<double> Egas;

	const auto initial_erad = rad_system.ComputeRadEnergy();

	// Main time loop

	for (int j = 0; j < max_timesteps; ++j) {
		if (rad_system.time() >= max_time) {
			break;
		}

		rad_system.AdvanceTimestep();
		rad_system.AddSourceTerms(std::make_pair(nghost, nghost + nx));

		std::cout << "Timestep " << j << "; t = " << rad_system.time()
			  << "\n";

		const auto current_erad = rad_system.ComputeRadEnergy();
		const auto current_egas = rad_system.ComputeGasEnergy();

		std::cout << "Total radiation energy = " << current_erad
			  << "\n";
		std::cout << "Total gas energy = " << current_egas << "\n";
		std::cout << "\n";

		t.push_back(rad_system.time());
		Erad.push_back(rad_system.radEnergy(0 + nghost));
		Egas.push_back(rad_system.gasEnergy(0 + nghost));
	}

	matplotlibcpp::clf();
	// matplotlibcpp::ylim(0.0, 1.0);

	std::map<std::string, std::string> erad_args;
	erad_args["label"] = "radiation energy density";
	// matplotlibcpp::plot(t, Erad, erad_args);

	std::map<std::string, std::string> egas_args;
	egas_args["label"] = "gas energy density";
	matplotlibcpp::plot(t, Egas, egas_args);

	matplotlibcpp::legend();
	matplotlibcpp::title(fmt::format("t = {:.4f}", rad_system.time()));
	matplotlibcpp::save(fmt::format("./radcoupling.png"));

	// Cleanup and exit
	std::cout << "Finished." << std::endl;
}
