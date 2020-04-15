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

// external headers

// internal headers
#include "test_advection.hpp"
#include "test_hydro_shocktube.hpp"
#include "test_hydro_wave.hpp"
#include "test_radiation_marshak.hpp"
#include "test_radiation_matter_coupling.hpp"
#include "test_radiation_streaming.hpp"

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

	{ // objects must be destroyed before Kokkos::finalize, so enter new
	  // scope here to do that automatically

		// testproblem_advection();
		// testproblem_hydro_shocktube();
		// testproblem_hydro_wave();
		// testproblem_radiation_streaming();
		testproblem_radiation_matter_coupling();
		//testproblem_radiation_marshak();

	} // destructors must be called before Kokkos::finalize()
	Kokkos::finalize();

	return 0;
}
