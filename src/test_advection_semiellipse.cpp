//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file test_advection.cpp
/// \brief Defines a test problem for linear advection.
///

#include "test_advection_semiellipse.hpp"

auto main() -> int
{
	// Initialization

	Kokkos::initialize();

	int result = 0;

	{ // objects must be destroyed before Kokkos::finalize, so enter new
	  // scope here to do that automatically

		result = testproblem_advection();

	} // destructors must be called before Kokkos::finalize()
	Kokkos::finalize();

	return result;
}

struct SawtoothProblem {};

template <typename problem_t>
void write_density(LinearAdvectionSystem<problem_t> &advection_system)
{
	std::cout << "density = ";

	auto nx = advection_system.nx();
	auto nghost = advection_system.nghost();

	for (int i = 0; i < nx + 2 * nghost; ++i) {
		std::cout << advection_system.density(i) << " ";
	}

	std::cout << "\n";
}

auto testproblem_advection() -> int
{
	// Based on 
	// https://www.mathematik.uni-dortmund.de/~kuzmin/fcttvd.pdf
	// Section 6.2: Convection of a semi-ellipse

	// Problem parameters

	const int nx = 400;
	const double Lx = 1.0;
	const double advection_velocity = 1.0;
	const double CFL_number = 0.3;
	const double max_time = 1.0;
	const double max_dt = 1e-4;
	const int max_timesteps = 1e4;
	const int nvars = 1; // only density

	const double atol = 1e-10; //< absolute tolerance for mass conservation

	// Problem initialization

	LinearAdvectionSystem<SawtoothProblem> advection_system(
	    {.nx = nx,
	     .lx = Lx,
	     .vx = advection_velocity,
	     .cflNumber = CFL_number,
	     .nvars = nvars});

	auto nghost = advection_system.nghost();

	for (int i = nghost; i < nx + nghost; ++i) {

		auto x = (0.5 + static_cast<double>(i - nghost)) / nx;
		double dens = 0.0;
		if (std::abs(x - 0.2) <= 0.15) {
			dens = std::sqrt( 1.0 - std::pow((x-0.2)/0.15, 2) );
		}
		advection_system.set_density(i) = dens;
	}

	std::vector<double> x(nx);
	std::vector<double> d_initial(nx);
	std::vector<double> d_final(nx);

	for (int i = 0; i < nx; ++i) {
		x.at(i) = static_cast<double>(i);
		d_initial.at(i) = advection_system.density(i + nghost);
	}

	// Main time loop
	int j = 0;
	for (j = 0; j < max_timesteps; ++j) {
		if (advection_system.time() >= max_time) {
			break;
		}
		advection_system.AdvanceTimestepRK2(max_dt);
	}

	std::cout << "timestep " << j << "; t = " << advection_system.time() << "\n";


	// Compute error norm
	for (int i = 0; i < nx; ++i) {
		d_final.at(i) = advection_system.density(i + nghost);
	}
	
	double err_norm = 0.;
	double sol_norm = 0.;
	for (int i=0; i < nx; ++i) {
		err_norm += std::abs(d_final[i] - d_initial[i]);
		sol_norm += std::abs(d_initial[i]);
	}
	
	const double rel_error = err_norm / sol_norm;
	//std::cout << "Absolute error norm = " << err_norm << std::endl;
	//std::cout << "Reference solution norm = " << sol_norm << std::endl;
	std::cout << "Relative L1 error norm = " << rel_error << std::endl;

	const double err_tol = 0.015;
	int status = 0;
	if (rel_error > err_tol) {
		status = 1;
	}

	// Plot results
	std::map<std::string, std::string> d_initial_args;
	std::map<std::string, std::string> d_final_args;
	d_initial_args["label"] = "density (initial)";
	d_final_args["label"] = "density (final)";

	matplotlibcpp::clf();
	matplotlibcpp::plot(x, d_initial, d_initial_args);
	matplotlibcpp::plot(x, d_final, d_final_args);
	matplotlibcpp::legend();
	matplotlibcpp::save(std::string("./advection_semiellipse.pdf"));

	// Cleanup and exit
	std::cout << "Finished." << std::endl;

	return status;
}
