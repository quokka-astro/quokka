//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file main.cpp
/// \brief Driver for unigrid tests without hydro motion.
///
/// This file provides a driver for running unigrid test problems without hydro
/// motion (i.e. streaming and static diffusion including material-radiation
/// coupling).

// c++ headers
#include <cstring>
#include <iostream>

// external headers
#include "matplotlibcpp.h"

// internal headers
#include "athena_arrays.hpp"
#include "linear_advection.hpp"

// function definitions
void write_density(LinearAdvectionSystem &advection_system);

/// Entry function for test runner. To be written.
///
/// \param[in] argc Number of command-line arguments.
/// \param[in] argv Command-line arguments (separated by spaces).
///
/// \return Error code.
///
auto main() -> int
{
	// Initialization

	Kokkos::initialize();

	{ // objects cannot be created/destroyed in same scope as
	  // Kokkos::initialize or Kokkos::finalize

		// Problem parameters

		const int nx = 64;
		const double Lx = 1.0;
		const double advection_velocity = 1.0;
		const double CFL_number = 0.3;
		const double max_time = 1.0;
		const int max_timesteps = 1000;

		const double atol =
		    1e-10; //< absolute tolerance for mass conservation

		// Problem initialization
		// (We use named types in order to guarantee that we don't screw
		// up the order of the arguments to the constructor. Also it
		// makes C++ look like beautiful Python. With optimizations
		// enabled, there is no performance penalty whatsoever. See
		// https://github.com/joboccara/NamedType)

		LinearAdvectionSystem advection_system(
		    LinearAdvectionSystem::Nx = nx,
		    LinearAdvectionSystem::Lx = Lx,
		    LinearAdvectionSystem::Vx = advection_velocity,
		    LinearAdvectionSystem::CFL = CFL_number);

		auto nghost = advection_system.nghost();

		for (int i = nghost; i < nx + nghost; ++i) {

			auto value =
			    static_cast<double>((i - nghost + nx / 2) % nx);
			advection_system.density_(i) = value;

			// advection_system.density_(i) =
			//    std::sin(M_PI * (value + 0.5) /
			//    static_cast<double>(nx));
		}

		std::vector<double> x(nx), d_initial(nx), d_final(nx);
		for (int i = 0; i < nx; ++i) {
			x.at(i) = static_cast<double>(i);
			d_initial.at(i) = advection_system.density_(i + nghost);
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
				  << "; t = " << advection_system.time()
				  << "\n";

			write_density(advection_system);

			const auto current_mass =
			    advection_system.ComputeMass();
			std::cout << "Total mass = " << current_mass << "\n";

			const auto mass_deficit =
			    std::abs(current_mass - initial_mass);
			std::cout << "Mass nonconservation = " << mass_deficit
				  << "\n";
			assert(mass_deficit < atol); // NOLINT

			std::cout << "\n";
		}

		// Plot results
		for (int i = 0; i < nx; ++i) {
			d_final.at(i) = advection_system.density_(i + nghost);
		}

		matplotlibcpp::plot(x, d_initial, "--");
		matplotlibcpp::plot(x, d_final, ".-");
		matplotlibcpp::save(std::string("./advection.png"));

		// Cleanup and exit
		std::cout << "Finished." << std::endl;

	} // destructors must be called before Kokkos::finalize()
	Kokkos::finalize();

	return 0;
}

void write_density(LinearAdvectionSystem &advection_system)
{
	std::cout << "density = ";

	auto nx = advection_system.nx();
	auto nghost = advection_system.nghost();

	for (int i = 0; i < nx + 2 * nghost; ++i) {
		std::cout << advection_system.density_(i) << " ";
	}

	std::cout << "\n";
}
