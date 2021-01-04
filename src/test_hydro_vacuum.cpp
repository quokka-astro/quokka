//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file test_hydro_shocktube.cpp
/// \brief Defines a test problem for a shock tube.
///

#include "test_hydro_vacuum.hpp"

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
	const double max_time = 0.15;
	const double max_dt = 1e-3;
	const double initial_dt = 1e-3;
	const int max_timesteps = 5000;
	const double gamma = 1.4; // ratio of specific heats

	const double rtol = 1e-6; //< absolute tolerance for conserved vars

	// Problem initialization

	HydroSystem<ShocktubeProblem> hydro_system(
	    {.nx = nx, .lx = Lx, .cflNumber = CFL_number, .gamma = gamma});

	auto nghost = hydro_system.nghost();

	for (int i = nghost; i < nx + nghost; ++i) {
		const auto idx_value = static_cast<double>(i - nghost);
		const double x =
		    Lx * ((idx_value + 0.5) / static_cast<double>(nx));

		double vx = NAN;
		double rho = NAN;
		double P = NAN;

		if (x < 0.5) {
			rho = 1.0;
			vx = -2.0;
			P = 0.4;
		} else {
			rho = 1.0;
			vx = 2.0;
			P = 0.4;
		}

		hydro_system.set_density(i) = rho;
		hydro_system.set_x1Momentum(i) = rho * vx;
		hydro_system.set_energy(i) =
		    P / (gamma - 1.0) + 0.5 * rho * std::pow(vx, 2);
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

		const double this_dtMax = ((j == 0) ? initial_dt : max_dt);
		hydro_system.AdvanceTimestepRK2(this_dtMax);
	}

// read in exact solution
	int status = 0;

	std::vector<double> d(nx);
	std::vector<double> vx(nx);
	std::vector<double> P(nx);
	std::vector<double> e(nx);

	hydro_system.ConservedToPrimitive(hydro_system.consVar_,
					  std::make_pair(nghost, nghost + nx));

	for (int i = 0; i < nx; ++i) {
		d.at(i) = hydro_system.primDensity(i + nghost);
		vx.at(i) = hydro_system.x1Velocity(i + nghost);
		P.at(i) = hydro_system.pressure(i + nghost);
		e.at(i) = P.at(i) / ((gamma - 1.0)*d.at(i));
	}

	std::vector<double> xs_exact;
	std::vector<double> density_exact;
	std::vector<double> pressure_exact;
	std::vector<double> velocity_exact;
	std::vector<double> eint_exact;

	std::string filename = "../extern/Toro/e1rpex.out";
	std::ifstream fstream(filename, std::ios::in);
	assert(fstream.is_open());

	for (std::string line; std::getline(fstream, line);) {
		std::istringstream iss(line);
		std::vector<double> values;

		for (double value; iss >> value;) {
			values.push_back(value);
			//if(iss.peek() == ',') {
			//	iss.ignore();
			//}
		}
		auto x = values.at(0);
		auto density = values.at(1);
		auto velocity = values.at(2);
		auto pressure = values.at(3);
		auto eint = pressure / ( (gamma - 1.0) * density );

		//std::cout << x << " " << density << " " << velocity << " " << pressure << std::endl;

		xs_exact.push_back(x);
		density_exact.push_back(density);
		pressure_exact.push_back(pressure);
		velocity_exact.push_back(velocity);
		eint_exact.push_back(eint);
	}

	// compute error norm

	std::vector<double> density_exact_interp(xs.size());
	interpolate_arrays(xs.data(), density_exact_interp.data(), xs.size(),
			   		   xs_exact.data(), density_exact.data(), xs_exact.size());

	std::vector<double> eint_exact_interp(xs.size());
	interpolate_arrays(xs.data(), eint_exact_interp.data(), xs.size(),
			   		   xs_exact.data(), eint_exact.data(), xs_exact.size());

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
	//matplotlibcpp::ylim(0.0, 11.0);

	std::map<std::string, std::string> d_args, dexact_args;
	d_args["label"] = "density";
	dexact_args["label"] = "density (exact solution)";
	matplotlibcpp::plot(xs, d, d_args);
	matplotlibcpp::plot(xs, density_exact_interp, dexact_args);

	std::map<std::string, std::string> vx_args;
	vx_args["label"] = "velocity";
	//matplotlibcpp::plot(xs, vx, vx_args);

	std::map<std::string, std::string> P_args;
	P_args["label"] = "pressure";
	//matplotlibcpp::plot(xs, P, P_args);

	matplotlibcpp::legend();
	matplotlibcpp::title(fmt::format("t = {:.4f}", hydro_system.time()));
	matplotlibcpp::save(fmt::format("./hydro_vacuum_{:.4f}.pdf", hydro_system.time()));

	// internal energy plot
	matplotlibcpp::clf();

	std::map<std::string, std::string> e_args, eexact_args;
	e_args["label"] = "specific internal energy";
	eexact_args["label"] = "exact solution";
	matplotlibcpp::plot(xs, e, e_args);
	matplotlibcpp::plot(xs, eint_exact_interp, eexact_args);

	matplotlibcpp::legend();
	matplotlibcpp::title(fmt::format("t = {:.4f}", hydro_system.time()));
	matplotlibcpp::save(fmt::format("./hydro_vacuum_eint_{:.4f}.pdf", hydro_system.time()));


	// Cleanup and exit
	std::cout << "Finished." << std::endl;
	return status;
}