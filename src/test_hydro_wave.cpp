//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file test_hydro_wave.cpp
/// \brief Defines a test problem for a linear hydro wave.
///

#include "test_hydro_wave.hpp"
#include "AMReX_ParmParse.H"
#include "HydroSimulation.hpp"
#include "hydro_system.hpp"

auto main(int argc, char **argv) -> int
{
	// Initialization (copied from ExaWind)

	amrex::Initialize(argc, argv, true, MPI_COMM_WORLD, []() {
		amrex::ParmParse pp("amrex");
		// Set the defaults so that we throw an exception instead of attempting
		// to generate backtrace files. However, if the user has explicitly set
		// these options in their input files respect those settings.
		if (!pp.contains("throw_exception")) {
			pp.add("throw_exception", 1);
		}
		if (!pp.contains("signal_handling")) {
			pp.add("signal_handling", 0);
		}
	});

	int result = 0;

	{ // objects must be destroyed before amrex::finalize, so enter new
	  // scope here to do that automatically

		result = testproblem_hydro_wave();

	} // destructors must be called before amrex::Finalize()
	amrex::Finalize();

	return result;
}

struct WaveProblem {
};

template <> double HydroSystem<WaveProblem>::gamma_ = 5. / 3.;

template <> void HydroSimulation<WaveProblem>::setInitialConditions()
{
	amrex::GpuArray<Real, AMREX_SPACEDIM> dx = simGeometry_.CellSizeArray();
	amrex::GpuArray<Real, AMREX_SPACEDIM> prob_lo = simGeometry_.ProbLoArray();

	for (amrex::MFIter iter(state_old_); iter.isValid(); ++iter) {
		const amrex::Box &indexRange = iter.validbox(); // excludes ghost zones
		auto const &state = state_new_.array(iter);

		amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
			const amrex::Real x_L = prob_lo[0] + (i + Real(0.0)) * dx[0];
			const amrex::Real x_R = prob_lo[0] + (i + Real(1.0)) * dx[0];
			const amrex::Real dx = x_R - x_L;

			const double gamma = HydroSystem<WaveProblem>::gamma_;
			const double rho0 = 1.0;       // background density
			const double P0 = 1.0 / gamma; // background pressure
			const double v0 = 0.;	       // background velocity
			const double A = 1.0e-6;       // perturbation amplitude

			const std::valarray<double> R = {1.0, -1.0,
							 1.5}; // right eigenvector of sound wave
			const std::valarray<double> U_0 = {
			    rho0, rho0 * v0, P0 / (gamma - 1.0) + 0.5 * rho0 * std::pow(v0, 2)};
			const std::valarray<double> dU =
			    (A * R / (2.0 * M_PI * dx)) *
			    (std::cos(2.0 * M_PI * x_L) - std::cos(2.0 * M_PI * x_R));

			state(i, j, k, HydroSystem<WaveProblem>::density_index) = U_0[0] + dU[0];
			state(i, j, k, HydroSystem<WaveProblem>::x1Momentum_index) = U_0[1] + dU[1];
			state(i, j, k, HydroSystem<WaveProblem>::energy_index) = U_0[2] + dU[2];
		});
	}

	// set flag
	areInitialConditionsDefined_ = true;
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

	// Problem initialization



	// Main time loop



	// Plot results every X timesteps

	std::vector<double> xs(nx);
	std::vector<double> d_initial(nx);
	std::vector<double> v_initial(nx);
	std::vector<double> P_initial(nx);

	for (int i = 0; i < nx; ++i) {
		const double x = Lx * ((i + 0.5) / static_cast<double>(nx));
		xs.at(i) = x;
		d_initial.at(i) = (hydro_system.density(i + nghost) - rho0) / A;
		v_initial.at(i) = (hydro_system.x1Velocity(i + nghost) - v0) / A;
		P_initial.at(i) = (hydro_system.pressure(i + nghost) - P0) / A;
	}

	std::vector<double> d(nx);
	std::vector<double> v(nx);
	std::vector<double> P(nx);

	for (int i = 0; i < nx; ++i) {
		d.at(i) = (hydro_system.primDensity(i + nghost) - rho0) / A;
		v.at(i) = (hydro_system.x1Velocity(i + nghost) - v0) / A;
		P.at(i) = (hydro_system.pressure(i + nghost) - P0) / A;
	}

	double rhoerr_norm = 0.;
	for (int i = 0; i < nx; ++i) {
		rhoerr_norm += std::abs(d[i] - d_initial[i]) / nx;
	}

	double vxerr_norm = 0.;
	for (int i = 0; i < nx; ++i) {
		vxerr_norm += std::abs(v[i] - v_initial[i]) / nx;
	}

	double Perr_norm = 0.;
	for (int i = 0; i < nx; ++i) {
		Perr_norm += std::abs(P[i] - P_initial[i]) / nx;
	}

	const double err_norm =
	    std::sqrt(std::pow(rhoerr_norm, 2) + std::pow(vxerr_norm, 2) + std::pow(Perr_norm, 2));

	const double err_tol = 0.003;
	int status = 0;
	if (err_norm > err_tol) {
		status = 1;
	}

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
	amrex::Print() << "Finished." << std::endl;
	return status;
}