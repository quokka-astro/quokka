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
//#include <format> // when C++20 support arrives...
#include <iostream>

// external headers
#include "fmt/include/fmt/format.h"
#include "matplotlibcpp.h"

// internal headers
#include "athena_arrays.hpp"
#include "hydro_system.hpp"
#include "linear_advection.hpp"

// function definitions
void write_density(LinearAdvectionSystem &advection_system);
void testproblem_advection();
void testproblem_hydro();

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
		testproblem_hydro();

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

	LinearAdvectionSystem advection_system(
	    LinearAdvectionSystem::Nx = nx, LinearAdvectionSystem::Lx = Lx,
	    LinearAdvectionSystem::Vx = advection_velocity,
	    LinearAdvectionSystem::CFL = CFL_number,
	    LinearAdvectionSystem::Nvars = nvars);

	auto nghost = advection_system.nghost();

	for (int i = nghost; i < nx + nghost; ++i) {

		auto value = static_cast<double>((i - nghost + nx / 2) % nx);
		advection_system.set_density(i) = value;

		// advection_system.set_density(i) =
		//    std::sin(M_PI * (value + 0.5) /
		//    static_cast<double>(nx));
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

void testproblem_hydro()
{
	// Problem parameters

	const int nx = 32;
	const double Lx = 1.0;
	const double CFL_number = 0.3;
	const double max_time = 10.0;
	const int max_timesteps = 5000;
	const int nvars = 3; // density, x-momentum, energy

	const double atol = 1e-10; //< absolute tolerance for conserved vars

	// Problem initialization

	HydroSystem hydro_system(HydroSystem::Nx = nx, HydroSystem::Lx = Lx,
				 HydroSystem::CFL = CFL_number,
				 HydroSystem::Nvars = nvars);

	auto nghost = hydro_system.nghost();

	const double amp = 0.01;
	for (int i = nghost; i < nx + nghost; ++i) {
		const auto idx_value = static_cast<double>(i - nghost);
		const double x =
		    Lx * ((idx_value + 0.5) / static_cast<double>(nx));

		hydro_system.set_density(i) = 1.0;
		hydro_system.set_x1Momentum(i) =
		    hydro_system.density(i) * amp * std::cos(2.0 * M_PI * x);
		hydro_system.set_energy(i) = 1.0;
	}

	std::vector<double> x(nx);
	std::vector<double> d_initial(nx);
	std::vector<double> d_final(nx);
	for (int i = 0; i < nx; ++i) {
		x.at(i) = static_cast<double>(i);
		d_initial.at(i) = hydro_system.density(i + nghost);
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
		std::cout << "Total mass = " << current_mass << "\n";

		const auto mass_deficit = std::abs(current_mass - initial_mass);
		std::cout << "Mass nonconservation = " << mass_deficit << "\n";

		assert(mass_deficit < atol); // NOLINT

		std::cout << "\n";

		// Plot results every X timesteps
		for (int i = 0; i < nx; ++i) {
			d_final.at(i) = hydro_system.density(i + nghost);
		}
		matplotlibcpp::clf();
		matplotlibcpp::ylim(1.0 - amp, 1.0 + amp);
		matplotlibcpp::plot(x, d_final, ".-");
		// matplotlibcpp::plot(x, P_final, "--");
		matplotlibcpp::save(fmt::format("./hydro_{:0>4d}.png", j));
	}

	// Cleanup and exit
	std::cout << "Finished." << std::endl;
}
