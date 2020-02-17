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
	// Problem parameters

	const int nx = 32;
	const double Lx = 1.0;
	const double advection_velocity = 1.0;
	const double CFL_number = 0.8;

	const double atol = 1e-13; //< absolute tolerance for mass conservation

	// Problem initialization
	// (We use named types in order to guarantee that we don't screw up the
	// order of the arguments to the constructor. Also it makes C++ look
	// like beautiful Python. With optimizations enabled, there is no
	// performance penalty whatsoever. See
	// https://github.com/joboccara/NamedType)

	LinearAdvectionSystem advection_system(
	    LinearAdvectionSystem::Nx = nx, LinearAdvectionSystem::Lx = Lx,
	    LinearAdvectionSystem::Vx = advection_velocity,
	    LinearAdvectionSystem::CFL = CFL_number);

	auto nghost = advection_system.nghost();

	for (int i = nghost; i < nx + nghost; ++i) {
		auto value = static_cast<double>(i - nghost);
		// advection_system.density_(i) = value;

		advection_system.density_(i) =
		    1.0 +
		    std::sin(M_PI * (value + 0.5) / static_cast<double>(nx));
	}

	std::cout << "Initial conditions:"
		  << "\n";
	write_density(advection_system);
	std::cout << "\n";

	const auto initial_mass = advection_system.ComputeMass();

	// Main time loop

	const int max_timesteps = 2;

	for (int j = 0; j < max_timesteps; ++j) {
		std::cout << "Timestep " << j << "\n";

		advection_system.AdvanceTimestep();
		write_density(advection_system);

		const auto current_mass = advection_system.ComputeMass();
		std::cout << "Total mass = " << current_mass << "\n";

		const auto mass_deficit = std::abs(current_mass - initial_mass);
		std::cout << "Mass nonconservation = " << mass_deficit << "\n";
		assert(mass_deficit < atol); // NOLINT

		std::cout << "\n";
	}

	// Cleanup and exit
	std::cout << "Finished." << std::endl;

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
