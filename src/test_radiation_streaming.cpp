//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file test_radiation_streaming.cpp
/// \brief Defines a test problem for radiation in the free-streaming regime.
///

#include "test_radiation_streaming.hpp"

void testproblem_radiation_streaming()
{
	// Problem parameters

	const int nx = 500;
	const double Lx = 1.0;
	const double CFL_number = 0.4;
	const double max_time = 1.0;
	const int max_timesteps = 5000;

	// Problem initialization

	RadSystem rad_system(RadSystem::Nx = nx, RadSystem::Lx = Lx,
			     RadSystem::CFL = CFL_number);

	auto nghost = rad_system.nghost();

	for (int i = nghost; i < nx + nghost; ++i) {
		const auto idx_value = static_cast<double>(i - nghost);
		const double x =
		    Lx * ((idx_value + 0.5) / static_cast<double>(nx));

		double Erad = 1e-5;
		if (x < 0.1) {
			Erad = 1.0;
		}

		rad_system.set_radEnergy(i) = Erad;
		rad_system.set_x1RadFlux(i) = rad_system.c_light() * Erad;
	}

	std::vector<double> xs(nx);

	for (int i = 0; i < nx; ++i) {
		const double x = Lx * ((i + 0.5) / static_cast<double>(nx));
		xs.at(i) = x;
	}

	const auto initial_erad = rad_system.ComputeRadEnergy();

	// Main time loop

	for (int j = 0; j < max_timesteps; ++j) {
		if (rad_system.time() >= max_time) {
			break;
		}

		rad_system.AdvanceTimestep(max_time);

		std::cout << "Timestep " << j << "; t = " << rad_system.time()
			  << "\n";

		const auto current_erad = rad_system.ComputeRadEnergy();
		const auto erad_deficit = std::abs(current_erad - initial_erad);

		std::cout << "Total energy = " << current_erad << "\n";
		std::cout << "Energy nonconservation = " << erad_deficit
			  << "\n";
		std::cout << "\n";

		// Plot results every X timesteps
		std::vector<double> erad(nx);

		for (int i = 0; i < nx; ++i) {
			erad.at(i) = rad_system.radEnergy(i + nghost);
		}

		matplotlibcpp::clf();
		matplotlibcpp::ylim(0.0, 1.1);

		std::map<std::string, std::string> erad_args;
		erad_args["label"] = "energy density";
		matplotlibcpp::plot(xs, erad, erad_args);

		matplotlibcpp::legend();
		matplotlibcpp::title(
		    fmt::format("t = {:.4f}", rad_system.time()));
		matplotlibcpp::save(fmt::format("./rad_{:0>4d}.png", j));
	}

	// Cleanup and exit
	std::cout << "Finished." << std::endl;
}