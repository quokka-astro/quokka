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

int testproblem_hydro_wave()
{
	// Based on the ATHENA test page:
	// https://www.astro.princeton.edu/~jstone/Athena/tests/linear-waves/linear-waves.html

	// Problem parameters

	const int nx = 256;
	const double Lx = 5.0;
	const double CFL_number = 0.4;
	const double max_time = 0.1;
	const double max_dt = 1e-2;
	const int max_timesteps = 5000;

	const double gamma = 5./3.; // ratio of specific heats
	const double rho0 = 1.0;	// background density
	const double c_s0 = 1.0;	// background sound speed
	const double P0 = rho0 * std::pow(c_s0, 2) / gamma; // background pressure
	const double v0 = 0.0;		// background velocity
	const double amp = 1.0e-6;	// perturbation amplitude

	const double rtol = 1e-6; //< absolute tolerance for conserved vars

	// Problem initialization

	HydroSystem<WaveProblem> hydro_system(
	    {.nx = nx, .lx = Lx, .cflNumber = CFL_number, .gamma = gamma});

	auto nghost = hydro_system.nghost();

	for (int i = nghost; i < nx + nghost; ++i) {
		const auto idx_value = static_cast<double>(i - nghost);
		const double x =
		    Lx * ((idx_value + 0.5) / static_cast<double>(nx));

		const double vx = v0 + amp * std::cos(2.0 * M_PI * x);

		hydro_system.set_density(i) = rho0;
		hydro_system.set_x1Momentum(i) = rho0 * vx;
		hydro_system.set_energy(i) =
		    P0 / (gamma - 1.0) + 0.5 * rho0 * std::pow(vx, 2);
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
		d_initial.at(i) = (hydro_system.density(i+nghost) - rho0) / amp;
		v_initial.at(i) = (hydro_system.x1Velocity(i+nghost) - v0) / amp;
		P_initial.at(i) = (hydro_system.pressure(i+nghost) - P0) / amp;
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
		d.at(i) = (hydro_system.primDensity(i + nghost) - rho0) / amp;
		v.at(i) = (hydro_system.x1Velocity(i + nghost) - v0) / amp;
		P.at(i) = (hydro_system.pressure(i + nghost) - P0) / amp;
	}

	// compute error norm
	std::vector<double> d_exact(nx);
	std::vector<double> v_exact(nx);
	std::vector<double> P_exact(nx);

	const double t = hydro_system.time();
	const double phi_d = -M_PI/2.0;
	const double phi_v = 0.;
	const double phi_P = -M_PI/2.0;

	for(int i = 0; i < nx; ++i) {
		const double x = xs[i];
		// there are *two* waves in the initial conditions!
		const double amp_d = (0.5*(gamma - 1.)/gamma);
		const double amp_P = (0.5*(gamma - 1.));

		// wrong amplitude
		v_exact[i] = (1./std::sqrt(2.))*std::cos(2.0*M_PI*(x + c_s0*t) + phi_v) +
						(1./std::sqrt(2.))*std::cos(2.0*M_PI*(x - c_s0*t) + phi_v);

		d_exact[i] = (1./std::sqrt(2.))*std::cos(2.0*M_PI*(x + c_s0*t) + phi_d) +
						(1./std::sqrt(2.))*std::cos(2.0*M_PI*(x + c_s0*t) + phi_d); // wrong?
						
		P_exact[i] = std::cos(2.0*M_PI*(x + c_s0*t) + phi_P); // wrong?
	}

	double err_norm = 0.;
	double sol_norm = 0.;
	for(int i = 0; i < nx; ++i) {
		err_norm += std::pow(P[i] - P_exact[i], 2);
		sol_norm += std::pow(P_exact[i], 2);
	}
	
	const double err_tol = 1e-5;
	const double rel_err = err_norm / sol_norm;
	int status = 0;
	if (rel_err > err_tol) {
		status = 1;
	}
	std::cout << "Relative error norm = " << rel_err << std::endl;

	// plot result
	std::map<std::string, std::string> d_args, dinit_args, dexact_args;
	d_args["label"] = "density";
	dinit_args["label"] = "density (initial)";
	dexact_args["label"] = "density (exact)";

	matplotlibcpp::clf();
	matplotlibcpp::plot(xs, d, d_args);
	matplotlibcpp::plot(xs, d_initial, dinit_args);
	//matplotlibcpp::plot(xs, v);
	matplotlibcpp::plot(xs, d_exact, dexact_args);
	matplotlibcpp::legend();
	matplotlibcpp::title(fmt::format("t = {:.4f}", t));
	matplotlibcpp::save(fmt::format("./density_{:.4f}.pdf", t));

	std::map<std::string, std::string> P_args, Pinit_args, Pexact_args;
	P_args["label"] = "pressure";
	Pinit_args["label"] = "pressure (initial)";
	Pexact_args["label"] = "pressure (exact)";

	matplotlibcpp::clf();
	matplotlibcpp::plot(xs, P, P_args);
	matplotlibcpp::plot(xs, P_initial, Pinit_args);
	matplotlibcpp::plot(xs, P_exact, Pexact_args);
	matplotlibcpp::legend();
	matplotlibcpp::title(fmt::format("t = {:.4f}", t));
	matplotlibcpp::save(fmt::format("./pressure_{:.4f}.pdf", t));

	std::map<std::string, std::string> v_args, vinit_args, vexact_args;
	v_args["label"] = "velocity";
	vinit_args["label"] = "velocity (initial)";
	vexact_args["label"] = "velocity (exact)";

	matplotlibcpp::clf();
	matplotlibcpp::plot(xs, v, v_args);
	matplotlibcpp::plot(xs, v_initial, vinit_args);
	matplotlibcpp::plot(xs, v_exact, vexact_args);
	matplotlibcpp::legend();
	matplotlibcpp::title(fmt::format("t = {:.4f}", t));
	matplotlibcpp::save(fmt::format("./velocity_{:.4f}.pdf", t));


	assert((mass_deficit / initial_mass) < rtol); // NOLINT

	// Cleanup and exit
	std::cout << "Finished." << std::endl;
	return status;
}