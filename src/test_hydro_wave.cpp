//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file test_hydro_wave.cpp
/// \brief Defines a test problem for a linear hydro wave.
///

#include "test_hydro_wave.hpp"

void testproblem_hydro_wave()
{
	// Problem parameters

	const int nx = 500;
	const double Lx = 5.0;
	const double CFL_number = 0.4;
	const double max_time = 10.0;
	const int max_timesteps = 5000;
	const double gamma = 1.001; // ratio of specific heats
	const double amp = 0.00001; // wave amplitude

	const double rtol = 1e-6; //< absolute tolerance for conserved vars

	// Problem initialization

	HydroSystem<AthenaArray<double>> hydro_system(
	    {.nx = nx, .lx = Lx, .cflNumber = CFL_number, .gamma = gamma});

	auto nghost = hydro_system.nghost();

	for (int i = nghost; i < nx + nghost; ++i) {
		const auto idx_value = static_cast<double>(i - nghost);
		const double x =
		    Lx * ((idx_value + 0.5) / static_cast<double>(nx));

		const double rho = 1.0;
		const double vx = amp * std::cos(2.0 * M_PI * x);
		const double P = 1.0;

		hydro_system.set_density(i) = rho;
		hydro_system.set_x1Momentum(i) = rho * vx;
		hydro_system.set_energy(i) =
		    P / (gamma - 1.0) + 0.5 * rho * std::pow(vx, 2);
	}

	std::vector<double> xs(nx);
	std::vector<double> d_initial(nx + 2 * nghost);
	std::vector<double> v_initial(nx + 2 * nghost);
	std::vector<double> P_initial(nx + 2 * nghost);

	hydro_system.ConservedToPrimitive(hydro_system.consVar_,
					  std::make_pair(nghost, nx + nghost));

	for (int i = 0; i < nx; ++i) {
		const double x = Lx * ((i + 0.5) / static_cast<double>(nx));
		xs.at(i) = x;
	}

	for (int i = 0; i < nx + 2 * nghost; ++i) {
		d_initial.at(i) = hydro_system.density(i);
		v_initial.at(i) = hydro_system.x1Velocity(i);
		P_initial.at(i) = hydro_system.pressure(i);
	}

	const auto initial_mass = hydro_system.ComputeMass();

	// Main time loop

	for (int j = 0; j < max_timesteps; ++j) {
		if (hydro_system.time() >= max_time) {
			break;
		}

		hydro_system.AdvanceTimestep();

		std::cout << "Timestep " << j << "; t = " << hydro_system.time()
			  << "\n";

		const auto current_mass = hydro_system.ComputeMass();
		const auto mass_deficit = std::abs(current_mass - initial_mass);

		std::cout << "Total mass = " << current_mass << "\n";
		std::cout << "Mass nonconservation = " << mass_deficit << "\n";
		std::cout << "\n";

		// Plot results every X timesteps
		std::vector<double> d(nx);
		std::vector<double> vx(nx);
		std::vector<double> P(nx);

		hydro_system.ConservedToPrimitive(
		    hydro_system.consVar_, std::make_pair(nghost, nghost + nx));

		for (int i = 0; i < nx; ++i) {
			d.at(i) = hydro_system.primDensity(i + nghost);
			vx.at(i) = hydro_system.x1Velocity(i + nghost);
			P.at(i) = hydro_system.pressure(i + nghost);
		}

		matplotlibcpp::clf();
		matplotlibcpp::ylim(1.0 - amp, 1.0 + amp);

		std::map<std::string, std::string> d_args;
		d_args["label"] = "density";
		matplotlibcpp::plot(xs, d, d_args);

		matplotlibcpp::legend();
		matplotlibcpp::title(
		    fmt::format("t = {:.4f}", hydro_system.time()));
		matplotlibcpp::save(fmt::format("./hydro_{:0>4d}.png", j));

		assert((mass_deficit / initial_mass) < rtol); // NOLINT
	}

	// Cleanup and exit
	std::cout << "Finished." << std::endl;
}