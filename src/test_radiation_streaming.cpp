//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file test_radiation_streaming.cpp
/// \brief Defines a test problem for radiation in the free-streaming regime.
///

#include "test_radiation_streaming.hpp"

auto main(int argc, char** argv) -> int
{
	// Initialization

	amrex::Initialize(argc, argv);

	int result = 0;

	{ // objects must be destroyed before Kokkos::finalize, so enter new
	  // scope here to do that automatically

		result = testproblem_radiation_streaming();

	} // destructors must be called before amrex::Finalize()
	amrex::Finalize();

	return result;
}

struct StreamingProblem {
};

const double erad_floor = 1.0e-5;
const double c = 1.0; // speed of light
const double chat = 0.2; // reduced speed of light

template <> void RadSystem<StreamingProblem>::FillGhostZones(array_t &cons)
{
	// x1 left side boundary
	for (int i = 0; i < nghost_; ++i) {
		const double Erad = 1.0;
		const double Frad = c_light_ * Erad;
		
		cons(radEnergy_index, i) = Erad;
		cons(x1RadFlux_index, i) = Frad;
	}

	// x1 right side boundary
	for (int i = nghost_ + nx_; i < nghost_ + nx_ + nghost_; ++i) {
		const double Erad = Erad_floor_;
		const double Frad = 0.;
		
		cons(radEnergy_index, i) = Erad;
		cons(x1RadFlux_index, i) = Frad;
	}
}

auto testproblem_radiation_streaming() -> int
{
	// Problem parameters

	const int nx = 1000;
	const double Lx = 1.0;
	const double CFL_number = 0.8;
	const double dt_max = 1e-2;
	const double tmax = 1.0;
	const int max_timesteps = 5000;

	// Problem initialization

	RadSystem<StreamingProblem> rad_system(
	    {.nx = nx, .lx = Lx, .cflNumber = CFL_number});

	rad_system.c_light_ = c;
	rad_system.c_hat_ = chat;
	rad_system.Erad_floor_ = erad_floor;

	auto nghost = rad_system.nghost();

	for (int i = nghost; i < nx + nghost; ++i) {
		double Erad = erad_floor;
		double Frad = 0.;

		rad_system.set_radEnergy(i) = Erad;
		rad_system.set_x1RadFlux(i) = Frad;
		rad_system.set_staticGasDensity(i) = 1e-10;
		rad_system.set_x1GasMomentum(i) = 0.0;
		rad_system.set_gasEnergy(i) = 1.0;
	}

	std::vector<double> xs(nx);
	std::vector<double> erad_exact(nx);
	for (int i = 0; i < nx; ++i) {
		const double x = Lx * ((i + 0.5) / static_cast<double>(nx));
		xs.at(i) = x;
		erad_exact.at(i) = (x <= chat*tmax) ? 1.0 : 0.0; 
	}

	const auto initial_erad = rad_system.ComputeRadEnergy();

	// Main time loop
	double dtPrev = NAN;
	int j = NAN;
	for (j = 0; j < max_timesteps; ++j) {
		if (rad_system.time() >= tmax) {
			break;
		}

		const double dtExpandFactor = 1.2;
		const double this_dt = rad_system.ComputeTimestep(std::min(dt_max, dtExpandFactor * dtPrev));
		rad_system.AdvanceTimestep(this_dt);
		dtPrev = this_dt;
	}

	amrex::Print() << "Timestep " << j << "; t = " << rad_system.time() << "\n";

	const auto current_erad = rad_system.ComputeRadEnergy();
	const auto erad_deficit = std::abs(current_erad - initial_erad);

	amrex::Print() << "Total energy = " << current_erad << "\n";
	amrex::Print() << "Energy nonconservation = " << erad_deficit << "\n";
	amrex::Print() << "\n";

	// compute error norm
	std::vector<double> erad(nx);

	for (int i = 0; i < nx; ++i) {
		erad.at(i) = rad_system.radEnergy(i + nghost);
	}

	double err_norm = 0.;
	double sol_norm = 0.;
	for (int i = 0; i < nx; ++i) {
		err_norm += std::abs(erad[i] - erad_exact[i]);
		sol_norm += std::abs(erad_exact[i]);
	}

	const double rel_err_norm = err_norm / sol_norm;
	const double rel_err_tol = 0.03;
	int status = NAN;
	if( rel_err_norm < rel_err_tol ) {
		status = 0;
	} else {
		status = 1;
	}

	amrex::Print() << "Relative L1 norm = " << rel_err_norm << std::endl;

	// Plot results every X timesteps

	matplotlibcpp::clf();
	matplotlibcpp::ylim(0.0, 1.1);

	std::map<std::string, std::string> erad_args;
	std::map<std::string, std::string> erad_exact_args;
	erad_args["label"] = "numerical solution";
	erad_exact_args["label"] = "exact solution";
	erad_exact_args["linestyle"] = "--";
	matplotlibcpp::plot(xs, erad, erad_args);
	matplotlibcpp::plot(xs, erad_exact, erad_exact_args);

	matplotlibcpp::legend();
	matplotlibcpp::title(fmt::format("t = {:.4f}", rad_system.time()));
	matplotlibcpp::save("./radiation_streaming.pdf");

	// Cleanup and exit
	amrex::Print() << "Finished." << std::endl;
	return status;
}