//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file test_hydro_shocktube.cpp
/// \brief Defines a test problem for a shock tube.
///

#include "test_hydro_sms.hpp"

struct ShocktubeProblem {};

auto main() -> int
{
	// Initialization

	Kokkos::initialize();

	int result = 0;

	{ // objects must be destroyed before Kokkos::finalize, so enter new
	  // scope here to do that automatically

		result = testproblem_hydro_shocktube();

	} // destructors must be called before Kokkos::finalize()
	Kokkos::finalize();

	return result;
}

auto testproblem_hydro_shocktube() -> int
{
	// Problem parameters

	const int nx = 100;
	const double Lx = 1.0;
	const double CFL_number = 0.2;
	const double max_time = 1.0;
	const double fixed_dt = 1e-3;
	const int max_timesteps = 20000;
	const double gamma = 1.4; // ratio of specific heats

	const double rtol = 1e-6; //< absolute tolerance for conserved vars

	// Problem initialization

	HydroSystem<ShocktubeProblem> hydro_system(
	    {.nx = nx, .lx = Lx, .cflNumber = CFL_number, .gamma = gamma});

	auto nghost = hydro_system.nghost();

	for (int i = nghost; i < nx + nghost; ++i) {
		const auto idx_value = static_cast<double>(i - nghost);
		const double x = Lx * ((idx_value + 0.5) / static_cast<double>(nx));

		double m = NAN;
		double rho = NAN;
		double E = NAN;

		if (x < 0.5) {
			rho = 3.86;
			m = -3.1266;
			E = 27.0913;
		} else {
			rho = 1.0;
			m = -3.44;
			E = 8.4168;
		}

		hydro_system.set_density(i) = rho;
		hydro_system.set_x1Momentum(i) = m;
		hydro_system.set_energy(i) = E;
	}

	std::vector<double> xs(nx);
	std::vector<double> d_initial(nx + 2 * nghost);
	std::vector<double> v_initial(nx + 2 * nghost);
	std::vector<double> P_initial(nx + 2 * nghost);

	hydro_system.ConservedToPrimitive(hydro_system.consVar_,
					  std::make_pair(nghost, nx + nghost));

	for (int i = 0; i < nx; ++i) {
		const double x = Lx * ((i + 0.5) / static_cast<double>(nx));
		xs.at(i) = x;
	}

	for (int i = 0; i < nx + 2 * nghost; ++i) {
		d_initial.at(i) = hydro_system.density(i);
		v_initial.at(i) = hydro_system.x1Velocity(i);
		P_initial.at(i) = hydro_system.pressure(i);
	}

	// Main time loop

	for (int j = 0; j < max_timesteps; ++j) {
		if (hydro_system.time() >= max_time) {
			std::cout << "Stopping at t=" << hydro_system.time() << std::endl;
			break;
		}

		hydro_system.AdvanceTimestepRK2(fixed_dt);
	}

// read in exact solution
	int status = 0;

	std::vector<double> d(nx);
	std::vector<double> vx(nx);
	std::vector<double> P(nx);

	hydro_system.ConservedToPrimitive(hydro_system.consVar_,
					  std::make_pair(nghost, nghost + nx));

	for (int i = 0; i < nx; ++i) {
		d.at(i) = hydro_system.primDensity(i + nghost);
		vx.at(i) = hydro_system.x1Velocity(i + nghost);
		P.at(i) = hydro_system.pressure(i + nghost);
	}

	std::vector<double> xs_exact;
	std::vector<double> density_exact;
	std::vector<double> pressure_exact;
	std::vector<double> velocity_exact;

	for (int i = 0; i < nx; ++i) {
		const auto idx_value = static_cast<double>(i);
		const double x = Lx * ((idx_value + 0.5) / static_cast<double>(nx));

		double vx = NAN;
		double rho = NAN;
		double P = NAN;
		const double vshock = 0.1096;

		if (x < (0.5 + vshock*hydro_system.time())) {
			rho = 3.86;
			vx = -0.81;
			P = 10.3334;
		} else {
			rho = 1.0;
			vx = -3.44;
			P = 1.0;
		}

		xs_exact.push_back(x);
		density_exact.push_back(rho);
		pressure_exact.push_back(P);
		velocity_exact.push_back(vx);
	}

	// compute error norm

	std::vector<double> density_exact_interp(xs.size());
	interpolate_arrays(xs.data(), density_exact_interp.data(), xs.size(),
			   		   xs_exact.data(), density_exact.data(), xs_exact.size());

	double err_norm = 0.;
	double sol_norm = 0.;
	for (int i = 0; i < xs.size(); ++i) {
		err_norm += std::abs(d[i] - density_exact_interp[i]);
		sol_norm += std::abs(density_exact_interp[i]);
	}

	const double error_tol = 0.01;
	const double rel_error = err_norm / sol_norm;
	std::cout << "err_norm = " << err_norm << std::endl;
	std::cout << "sol_norm = " << sol_norm << std::endl;
	std::cout << "Relative L1 error norm = " << rel_error << std::endl;

	// Compute test success condition
	if(rel_error > error_tol) {
		status = 1;
	}

	// Plot results
	matplotlibcpp::clf();
	//matplotlibcpp::ylim(0.0, 5.0);

	std::map<std::string, std::string> d_args, dexact_args;
	d_args["label"] = "density";
	dexact_args["label"] = "density (exact solution)";
	matplotlibcpp::plot(xs, d, d_args);
	matplotlibcpp::plot(xs, density_exact_interp, dexact_args);

	matplotlibcpp::legend();
	matplotlibcpp::title(fmt::format("t = {:.4f}", hydro_system.time()));
	matplotlibcpp::save(fmt::format("./hydro_sms_{:.4f}.pdf", hydro_system.time()));

	// Cleanup and exit
	std::cout << "Finished." << std::endl;
	return status;
}