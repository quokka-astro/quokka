//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file test_advection.cpp
/// \brief Defines a test problem for linear advection.
///

#include "test_advection.hpp"

template <typename array_t>
void write_density(LinearAdvectionSystem<array_t> &advection_system)
{
	std::cout << "density = ";

	auto nx = advection_system.nx();
	auto nghost = advection_system.nghost();

	for (int i = 0; i < nx + 2 * nghost; ++i) {
		std::cout << advection_system.density(i) << " ";
	}

	std::cout << "\n";
}

void testproblem_advection()
{
	// Problem parameters

	const int nx = 64;
	const double Lx = 1.0;
	const double advection_velocity = 1.0;
	const double CFL_number = 0.3;
	const double max_time = 1.0;
	const int max_timesteps = 1000;
	const int nvars = 1; // only density

	const double atol = 1e-10; //< absolute tolerance for mass conservation

	// Problem initialization

	LinearAdvectionSystem<AthenaArray<double>> advection_system(
	    {.nx = nx,
	     .lx = Lx,
	     .vx = advection_velocity,
	     .cflNumber = CFL_number,
	     .nvars = nvars});

	auto nghost = advection_system.nghost();

	for (int i = nghost; i < nx + nghost; ++i) {

		auto value = static_cast<double>((i - nghost + nx / 2) % nx);
		advection_system.set_density(i) = value;
	}

	std::vector<double> x(nx);
	std::vector<double> d_initial(nx);
	std::vector<double> d_final(nx);

	for (int i = 0; i < nx; ++i) {
		x.at(i) = static_cast<double>(i);
		d_initial.at(i) = advection_system.density(i + nghost);
	}

	std::cout << "Initial conditions:"
		  << "\n";
	write_density(advection_system);
	std::cout << "\n";

	const auto initial_mass = advection_system.ComputeMass();

	// Main time loop

	for (int j = 0; j < max_timesteps; ++j) {
		if (advection_system.time() >= max_time) {
			break;
		}

		advection_system.AdvanceTimestep();

		std::cout << "Timestep " << j
			  << "; t = " << advection_system.time() << "\n";

		write_density(advection_system);

		const auto current_mass = advection_system.ComputeMass();
		std::cout << "Total mass = " << current_mass << "\n";

		const auto mass_deficit = std::abs(current_mass - initial_mass);
		std::cout << "Mass nonconservation = " << mass_deficit << "\n";
		assert(mass_deficit < atol); // NOLINT

		std::cout << "\n";
	}

	// Plot results
	for (int i = 0; i < nx; ++i) {
		d_final.at(i) = advection_system.density(i + nghost);
	}

	matplotlibcpp::plot(x, d_initial, "--");
	matplotlibcpp::plot(x, d_final, ".-");
	matplotlibcpp::save(std::string("./advection.png"));

	// Cleanup and exit
	std::cout << "Finished." << std::endl;
}
