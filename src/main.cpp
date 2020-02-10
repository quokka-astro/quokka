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

/// Entry function for test runner. To be written.
///
/// \param[in] argc Number of command-line arguments.
/// \param[in] argv Command-line arguments (separated by spaces).
///
/// \return Error code.
///
auto main(int argc, char *argv[]) -> int
{
	const int nx = 32;
	LinearAdvectionSystem advection_system(nx);

	const int k = 10;
	advection_system.density(k) = 1.0;

	std::cout << "density = ";

	for (int i = 0; i < nx; ++i) {
		std::cout << advection_system.density(i) << " ";
	}

	std::cout << "\n";

	return 0;
}
