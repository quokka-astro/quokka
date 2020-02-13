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
auto main(int argc, char *argv[]) -> int
{
	// Problem parameters

	const int nx = 32;
	const double advection_velocity = 1.0;
	const double Lx = 1.0;

	// Problem initialization

	LinearAdvectionSystem advection_system(nx, advection_velocity, Lx);

	for (int i = advection_system.NumGhostZones();
	     i < nx + advection_system.NumGhostZones(); ++i) {
		auto value =
		    static_cast<double>(i - advection_system.NumGhostZones());
		advection_system.density(i) = value;
	}

	write_density(advection_system);

	// Main time loop

	advection_system.AdvanceTimestep();

	write_density(advection_system);

	// Cleanup and exit

	return 0;
}

void write_density(LinearAdvectionSystem &advection_system)
{
	std::cout << "density = ";

	auto nx = advection_system.Nx();
	auto nghost = advection_system.NumGhostZones();

	for (int i = 0; i < nx + 2 * nghost; ++i) {
		std::cout << advection_system.density(i) << " ";
	}

	std::cout << "\n";
}
