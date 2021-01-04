//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file test_hydro_wave.cpp
/// \brief Defines a test problem for a linear hydro wave.
///

#include "test_hydro_wave.hpp"

auto main() -> int
{
	// Initialization

	Kokkos::initialize();

	int result = 0;

	{ // objects must be destroyed before Kokkos::finalize, so enter new
	  // scope here to do that automatically

		result = testproblem_hydro_wave();

	} // destructors must be called before Kokkos::finalize()
	Kokkos::finalize();

	return result;
}

struct WaveProblem {};

template <>
void HyperbolicSystem<WaveProblem>::FillGhostZones(array_t &cons)
{
	// periodic boundary conditions
	// x1 right side boundary
	for (int n = 0; n < nvars_; ++n) {
		for (int i = nghost_ + nx_; i < nghost_ + nx_ + nghost_; ++i) {
			cons(n, i) = cons(n, i - nx_);
		}
	}

	// x1 left side boundary
	for (int n = 0; n < nvars_; ++n) {
		for (int i = 0; i < nghost_; ++i) {
			cons(n, i) = cons(n, i + nx_);
		}
	}
}

auto testproblem_hydro_wave() -> int
{
	// Based on the ATHENA test page:
	// https://www.astro.princeton.edu/~jstone/Athena/tests/linear-waves/linear-waves.html

	// Problem parameters

	const int nx = 100;
	const double Lx = 1.0;
	const double CFL_number = 0.8;
	const double max_time = 1.0;
	const double max_dt = 1e-3;
	const int max_timesteps = 1e3;

	const double gamma = 5./3.; // ratio of specific heats
	const double rho0 = 1.0;	// background density
	const double P0 = 1.0 / gamma; // background pressure
	const double v0 = 0.;		// background velocity
	const double A = 1.0e-6;	// perturbation amplitude

	const std::valarray<double> R = {1.0, -1.0, 1.5}; // right eigenvector of sound wave
	const std::valarray<double> U_0 = {rho0, rho0*v0, P0/(gamma - 1.0) + 0.5*rho0*std::pow(v0, 2)};

	const double rtol = 1e-10; //< absolute tolerance for conserved vars

	// Problem initialization

	HydroSystem<WaveProblem> hydro_system(
	    {.nx = nx, .lx = Lx, .cflNumber = CFL_number, .gamma = gamma});

	auto nghost = hydro_system.nghost();

	for (int i = nghost; i < nx + nghost; ++i) {
		const auto idx_value = static_cast<double>(i - nghost);
		//const double x = Lx * ((idx_value + 0.5) / static_cast<double>(nx));
		const double x_L = Lx * ((idx_value) / static_cast<double>(nx));
		const double x_R = Lx * ((idx_value + 1) / static_cast<double>(nx));
		const double dx = x_R - x_L;

		//const std::valarray<double> dU = A * R * std::sin(2.0 * M_PI * x);
		const std::valarray<double> dU = (A*R/(2.0*M_PI*dx))*(std::cos(2.0*M_PI*x_L) - std::cos(2.0*M_PI*x_R));

		hydro_system.set_density(i) 	= U_0[0] + dU[0];
		hydro_system.set_x1Momentum(i) 	= U_0[1] + dU[1];
		hydro_system.set_energy(i) 		= U_0[2] + dU[2];
	}

	std::vector<double> xs(nx);
	std::vector<double> d_initial(nx);
	std::vector<double> v_initial(nx);
	std::vector<double> P_initial(nx);

	hydro_system.ConservedToPrimitive(hydro_system.consVar_,
					  std::make_pair(nghost, nx + nghost));

	for (int i = 0; i < nx; ++i) {
		const double x = Lx * ((i + 0.5) / static_cast<double>(nx));
		xs.at(i) = x;
		d_initial.at(i) = (hydro_system.density(i+nghost) - rho0) / A;
		v_initial.at(i) = (hydro_system.x1Velocity(i+nghost) - v0) / A;
		P_initial.at(i) = (hydro_system.pressure(i+nghost) - P0) / A;
	}

	const auto initial_mass = hydro_system.ComputeMass();

	// Main time loop

	for (int j = 0; j < max_timesteps; ++j) {
		if (hydro_system.time() >= max_time) {
			break;
		}
		hydro_system.AdvanceTimestep(max_dt);
	}

	std::cout << "t = " << hydro_system.time() << "\n";

	const auto current_mass = hydro_system.ComputeMass();
	const auto mass_deficit = std::abs(current_mass - initial_mass);

	std::cout << "Total mass = " << current_mass << "\n";
	std::cout << "Mass nonconservation = " << mass_deficit << "\n";
	std::cout << "\n";

	// Plot results every X timesteps
	std::vector<double> d(nx);
	std::vector<double> v(nx);
	std::vector<double> P(nx);

	hydro_system.ConservedToPrimitive(hydro_system.consVar_,
					  std::make_pair(nghost, nghost + nx));

	for (int i = 0; i < nx; ++i) {
		d.at(i) = (hydro_system.primDensity(i + nghost) - rho0) / A;
		v.at(i) = (hydro_system.x1Velocity(i + nghost) - v0) / A;
		P.at(i) = (hydro_system.pressure(i + nghost) - P0) / A;
	}

	const double t = hydro_system.time();

	double rhoerr_norm = 0.;
	for(int i = 0; i < nx; ++i) {
		rhoerr_norm += std::abs(d[i] - d_initial[i]) / nx;
	}
	
	double vxerr_norm = 0.;
	for(int i = 0; i < nx; ++i) {
		vxerr_norm += std::abs(v[i] - v_initial[i]) / nx;
	}

	double Perr_norm = 0.;
	for(int i = 0; i < nx; ++i) {
		Perr_norm += std::abs(P[i] - P_initial[i]) / nx;
	}

	const double err_norm = std::sqrt(std::pow(rhoerr_norm, 2) +
									  std::pow(vxerr_norm,  2) +
									  std::pow(Perr_norm,   2) );

	const double err_tol = 0.003;
	int status = 0;
	if (err_norm > err_tol) {
		status = 1;
	}
	std::cout << "L1 error norm = " << err_norm << std::endl;

	// plot result
	std::map<std::string, std::string> d_args, dinit_args, dexact_args;
	d_args["label"] = "density";
	dinit_args["label"] = "density (initial)";

	matplotlibcpp::clf();
	matplotlibcpp::plot(xs, d, d_args);
	matplotlibcpp::plot(xs, d_initial, dinit_args);
	matplotlibcpp::legend();
	matplotlibcpp::title(fmt::format("t = {:.4f}", t));
	matplotlibcpp::save(fmt::format("./density_{:.4f}.pdf", t));

	std::map<std::string, std::string> P_args, Pinit_args, Pexact_args;
	P_args["label"] = "pressure";
	Pinit_args["label"] = "pressure (initial)";

	matplotlibcpp::clf();
	matplotlibcpp::plot(xs, P, P_args);
	matplotlibcpp::plot(xs, P_initial, Pinit_args);
	matplotlibcpp::legend();
	matplotlibcpp::title(fmt::format("t = {:.4f}", t));
	matplotlibcpp::save(fmt::format("./pressure_{:.4f}.pdf", t));

	std::map<std::string, std::string> v_args, vinit_args, vexact_args;
	v_args["label"] = "velocity";
	vinit_args["label"] = "velocity (initial)";

	matplotlibcpp::clf();
	matplotlibcpp::plot(xs, v, v_args);
	matplotlibcpp::plot(xs, v_initial, vinit_args);
	matplotlibcpp::legend();
	matplotlibcpp::title(fmt::format("t = {:.4f}", t));
	matplotlibcpp::save(fmt::format("./velocity_{:.4f}.pdf", t));


	assert((mass_deficit / initial_mass) < rtol); // NOLINT

	// Cleanup and exit
	std::cout << "Finished." << std::endl;
	return status;
}