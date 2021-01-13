//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file test_advection.cpp
/// \brief Defines a test problem for linear advection.
///

#include "test_advection.hpp"

auto main(int argc, char** argv) -> int
{
	// Initialization

	amrex::Initialize(argc, argv);

	int result = 0;

	{ // objects must be destroyed before Kokkos::finalize, so enter new
	  // scope here to do that automatically

		result = testproblem_advection();

	} // destructors must be called before amrex::Finalize()
	amrex::Finalize();

	return result;
}

struct SawtoothProblem {};

template <typename problem_t>
void write_density(LinearAdvectionSystem<problem_t> &advection_system)
{
	amrex::Print() << "density = ";

	auto nx = advection_system.nx();
	auto nghost = advection_system.nghost();

	for (int i = 0; i < nx + 2 * nghost; ++i) {
		amrex::Print() << advection_system.density(i) << " ";
	}

	amrex::Print() << "\n";
}

auto testproblem_advection() -> int
{
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

		auto value = static_cast<double>((i - nghost + nx / 2) % nx) / nx;
		advection_system.set_density(i) = value;
	}

	std::vector<double> x(nx);
	std::vector<double> d_initial(nx);
	std::vector<double> d_final(nx);

	for (int i = 0; i < nx; ++i) {
		x.at(i) = static_cast<double>(i);
		d_initial.at(i) = advection_system.density(i + nghost);
	}

	//const auto initial_mass = advection_system.ComputeMass();

	// Main time loop
	int j = 0;
	for (j = 0; j < max_timesteps; ++j) {
		if (advection_system.time() >= max_time) {
			break;
		}
		advection_system.AdvanceTimestepRK2(max_dt);
	}

	amrex::Print() << "timestep " << j << "; t = " << advection_system.time() << "\n";
	//write_density(advection_system);

	//const auto current_mass = advection_system.ComputeMass();
	//amrex::Print() << "Total mass = " << current_mass << "\n";
	//const auto mass_deficit = std::abs(current_mass - initial_mass);
	//amrex::Print() << "Mass nonconservation = " << mass_deficit << "\n";
	//assert(mass_deficit < atol); // NOLINT

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
	//amrex::Print() << "Absolute error norm = " << err_norm << std::endl;
	//amrex::Print() << "Reference solution norm = " << sol_norm << std::endl;
	amrex::Print() << "Relative L1 error norm = " << rel_error << std::endl;

	const double err_tol = 0.01;
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
	matplotlibcpp::save(std::string("./advection.pdf"));

	// Cleanup and exit
	amrex::Print() << "Finished." << std::endl;

	return status;
}
