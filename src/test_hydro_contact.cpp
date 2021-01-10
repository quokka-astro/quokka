//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file test_hydro_shocktube.cpp
/// \brief Defines a test problem for a shock tube.
///

#include "test_hydro_contact.hpp"

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
	const double CFL_number = 0.8;
	const double max_time = 2.0;
	const double fixed_dt = 1e-3;
	const int max_timesteps = 2000;
	const double gamma = 1.4; // ratio of specific heats
	const double v_contact = 0.0; // contact wave velocity

	const double rtol = 1e-6; //< absolute tolerance for conserved vars

	// Problem initialization

	HydroSystem<ShocktubeProblem> hydro_system(
	    {.nx = nx, .lx = Lx, .cflNumber = CFL_number, .gamma = gamma});

	auto nghost = hydro_system.nghost();

	for (int i = nghost; i < nx + nghost; ++i) {
		const auto idx_value = static_cast<double>(i - nghost);
		const double x = Lx * ((idx_value + 0.5) / static_cast<double>(nx));

		double v = NAN;
		double rho = NAN;
		double P = NAN;

		if (x < 0.5) {
			rho = 1.4;
			v = v_contact;
			P = 1.0;
		} else {
			rho = 1.0;
			v = v_contact;
			P = 1.0;
		}

		hydro_system.set_density(i) = rho;
		hydro_system.set_x1Momentum(i) = rho*v;
		hydro_system.set_energy(i) = P/(gamma - 1.) + 0.5*rho*(v*v);
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
	int j = NAN;
	for (j = 0; j < max_timesteps; ++j) {
		if (hydro_system.time() >= max_time) {
			break;
		}

		hydro_system.AdvanceTimestepRK2(fixed_dt);
	}

	std::cout << "Stopping at timestep " << j << " t=" << hydro_system.time() << std::endl;

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
		const double vshock = v_contact;

		if (x < (0.5 + vshock*hydro_system.time())) {
			rho = 1.4;
			vx = 0.0;
			P = 1.0;
		} else {
			rho = 1.0;
			vx = 0.0;
			P = 1.0;
		}

		xs_exact.push_back(x);
		density_exact.push_back(rho);
		pressure_exact.push_back(P);
		velocity_exact.push_back(vx);
	}

	// compute error norm

	double err_norm = 0.;
	double sol_norm = 0.;
	for (int i = 0; i < xs.size(); ++i) {
		err_norm += std::abs(d[i] - density_exact[i]);
		sol_norm += std::abs(density_exact[i]);
	}

	// For a stationary isolated contact wave using the HLLC solver,
	// the error should be *exactly* (i.e., to *every* digit) zero.
	// [See Section 10.7 and Figure 10.20 of Toro (1998).]
	const double error_tol = 0.0; // this is not a typo
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

	std::unordered_map<std::string, std::string> d_args;
	std::map<std::string, std::string> dexact_args;
	d_args["label"] = "density";
	d_args["color"] = "black";
	dexact_args["label"] = "density (exact solution)";
	matplotlibcpp::scatter(xs, d, 10.0, d_args);
	matplotlibcpp::plot(xs, density_exact, dexact_args);

	matplotlibcpp::legend();
	matplotlibcpp::title(fmt::format("t = {:.4f}", hydro_system.time()));
	matplotlibcpp::save(fmt::format("./hydro_contact.pdf", hydro_system.time()));

	// Cleanup and exit
	std::cout << "Finished." << std::endl;
	return status;
}