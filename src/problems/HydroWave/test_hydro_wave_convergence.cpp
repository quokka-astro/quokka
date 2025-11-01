//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file test_hydro_wave_convergence.cpp
/// \brief Defines a Richardson convergence test for the linear hydro wave.
///

#include "hydro/hydro_system.hpp"
#include <fmt/format.h>
#include <limits>
#include <valarray>

#include "AMReX_Array.H"
#include "AMReX_Array4.H"
#include "AMReX_ParmParse.H"
#include "AMReX_REAL.H"

#include "QuokkaSimulation.hpp"
#include "util/fextract.hpp"

struct WaveProblem {
};

template <> struct quokka::EOS_Traits<WaveProblem> {
	static constexpr double gamma = 5. / 3.;
	static constexpr double mean_molecular_weight = C::m_u;
};

template <> struct Physics_Traits<WaveProblem> {
	static constexpr bool is_self_gravity_enabled = false;
	// cell-centred
	static constexpr bool is_hydro_enabled = true;
	static constexpr int numMassScalars = 0;		     // number of mass scalars
	static constexpr int numPassiveScalars = numMassScalars + 0; // number of passive scalars
	static constexpr bool is_radiation_enabled = false;
	// face-centred
	static constexpr bool is_mhd_enabled = false;
	static constexpr int nGroups = 1; // number of radiation groups
	static constexpr UnitSystem unit_system = UnitSystem::CGS;
};

constexpr double rho0 = 1.0;					    // background density
constexpr double P0 = 1.0 / quokka::EOS_Traits<WaveProblem>::gamma; // background pressure
constexpr double v0 = 0.;					    // background velocity
constexpr double amp = 1.0e-6;					    // perturbation amplitude

AMREX_GPU_DEVICE void computeWaveSolution(int i, int j, int k, amrex::Array4<amrex::Real> const &state, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx,
					  amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_lo)
{
	const amrex::Real x_L = prob_lo[0] + (i + static_cast<amrex::Real>(0.0)) * dx[0];
	const amrex::Real x_R = prob_lo[0] + (i + static_cast<amrex::Real>(1.0)) * dx[0];
	const amrex::Real A = amp;

	const quokka::valarray<double, 3> R = {1.0, -1.0, 1.5}; // right eigenvector of sound wave
	const quokka::valarray<double, 3> U_0 = {rho0, rho0 * v0, P0 / (quokka::EOS_Traits<WaveProblem>::gamma - 1.0) + 0.5 * rho0 * std::pow(v0, 2)};
	const quokka::valarray<double, 3> dU = (A * R / (2.0 * M_PI * dx[0])) * (std::cos(2.0 * M_PI * x_L) - std::cos(2.0 * M_PI * x_R));

	double const rho = U_0[0] + dU[0];
	double const xmom = U_0[1] + dU[1];
	double const Etot = U_0[2] + dU[2];
	double const Eint = Etot - 0.5 * (xmom * xmom) / rho;

	state(i, j, k, HydroSystem<WaveProblem>::density_index) = rho;
	state(i, j, k, HydroSystem<WaveProblem>::x1Momentum_index) = xmom;
	state(i, j, k, HydroSystem<WaveProblem>::x2Momentum_index) = 0;
	state(i, j, k, HydroSystem<WaveProblem>::x3Momentum_index) = 0;
	state(i, j, k, HydroSystem<WaveProblem>::energy_index) = Etot;
	state(i, j, k, HydroSystem<WaveProblem>::internalEnergy_index) = Eint;
}

template <> void QuokkaSimulation<WaveProblem>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	// extract variables required from the geom object
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const dx = grid_elem.dx_;
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const prob_lo = grid_elem.prob_lo_;
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const amrex::Array4<double> &state_cc = grid_elem.array_;
	const int ncomp_cc = Physics_Indices<WaveProblem>::nvarTotal_cc;
	// loop over the grid and set the initial condition
	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		for (int n = 0; n < ncomp_cc; ++n) {
			state_cc(i, j, k, n) = 0; // fill unused components with zeros
		}
		computeWaveSolution(i, j, k, state_cc, dx, prob_lo);
	});
}

auto runWaveTest(int nx) -> double
{
	// Problem parameters
	const double CFL_number = 0.1;
	const double max_time = 1.0;
	const int max_timesteps = std::max(20000, nx * 100);

	// Problem initialization
	const int ncomp_cc = Physics_Indices<WaveProblem>::nvarTotal_cc;
	amrex::Vector<amrex::BCRec> BCs_cc(ncomp_cc);
	for (int n = 0; n < ncomp_cc; ++n) {
		for (int i = 0; i < AMREX_SPACEDIM; ++i) {
			BCs_cc[n].setLo(i, amrex::BCType::int_dir); // periodic
			BCs_cc[n].setHi(i, amrex::BCType::int_dir);
		}
	}

	// Set grid dimensions using AMReX parameter system
	amrex::ParmParse pp("amr");
	amrex::Vector<int> const ncells = {nx, 8, 8};
	pp.add("max_level", 0);
	pp.add("blocking_factor", 8);
	pp.add("max_grid_size", nx);
	pp.addarr("n_cell", ncells);

	// Set domain bounds using AMReX parameter system
	amrex::ParmParse pp_geom("geometry");
	amrex::Vector<double> const prob_lo = {0.0, 0.0, 0.0};
	amrex::Vector<double> const prob_hi = {1.0, 1.0, 1.0};
	amrex::Vector<int> const is_periodic = {1, 1, 1};
	pp_geom.addarr("prob_lo", prob_lo);
	pp_geom.addarr("prob_hi", prob_hi);
	pp_geom.addarr("is_periodic", is_periodic);

	QuokkaSimulation<WaveProblem> sim(BCs_cc);

	sim.cflNumber_ = CFL_number;
	sim.stopTime_ = max_time;
	sim.maxTimesteps_ = max_timesteps;

	// set initial conditions
	sim.setInitialConditions();
	auto [pos_exact, val_exact] = fextract(sim.state_new_cc_[0], sim.geom[0], 0, 0.5);

	// Main time loop
	sim.evolve();

	auto [position, values] = fextract(sim.state_new_cc_[0], sim.geom[0], 0, 0.5);
	int const nx_final = static_cast<int>(position.size());

	// compute error norm
	amrex::Real err_sq = 0.;
	for (int n = 0; n < QuokkaSimulation<WaveProblem>::ncompHydro_; ++n) {
		if (n == HydroSystem<WaveProblem>::internalEnergy_index) {
			continue;
		}
		amrex::Real dU_k = 0.;
		for (int i = 0; i < nx_final; ++i) {
			// Δ Uk = ∑i |Uk,in - Uk,i0| / Nx
			const amrex::Real U_k0 = val_exact.at(n)[i];
			const amrex::Real U_k1 = values.at(n)[i];
			dU_k += std::abs(U_k1 - U_k0) / static_cast<double>(nx_final);
		}
		// ε = || Δ U || = [&sum_k (Δ Uk)2]^{1/2}
		err_sq += dU_k * dU_k;
	}
	const amrex::Real epsilon = std::sqrt(err_sq);

	return epsilon;
}

auto problem_main() -> int
{
	// Richardson convergence test: run at increasing resolution until machine precision is reached
	const double machine_precision_target = 1.0e3 * std::numeric_limits<double>::epsilon();
	const int nx_initial = 32;
	const int nx_max = 4096;
	bool reached_target = false;

	// Silence TinyProfiler so convergence logs stay readable
	{
		amrex::ParmParse pp_tp("tiny_profiler");
		if (!pp_tp.contains("output_file")) {
			pp_tp.add("output_file", "/dev/null");
		}
	}

	// Suppress per-step logging from the coarse timestep loop
	{
		amrex::ParmParse pp_general;
		if (!pp_general.contains("suppress_output")) {
			pp_general.add("suppress_output", 1);
		}
	}

	amrex::Vector<int> resolutions;
	amrex::Vector<double> errors;
	amrex::Vector<double> dx_values;

	amrex::Print() << "Running Richardson convergence test for HydroWave:\n";
	amrex::Print() << "Resolution\tError Norm\n";
	amrex::Print() << "----------\t----------\n";

	for (int nx = nx_initial; nx <= nx_max; nx *= 2) {
		double const error = runWaveTest(nx);

		resolutions.push_back(nx);
		errors.push_back(error);
		dx_values.push_back(1.0 / static_cast<double>(nx)); // dx = L / nx for unit domain

		amrex::Print() << fmt::format("{:10d}\t{:.6e}\n", nx, error);

		if (error <= machine_precision_target) {
			reached_target = true;
			break;
		}

		if (nx == nx_max) {
			amrex::Print() << fmt::format("\nReached maximum resolution (nx = {}) without achieving the target error {:.3e}\n", nx_max, machine_precision_target);
			break;
		}
	}

	// Calculate convergence rates using Richardson extrapolation
	amrex::Print() << "\nConvergence Rate Analysis:\n";
	amrex::Print() << "Resolution Pair\tObserved Rate\tExpected Rate\n";
	amrex::Print() << "---------------\t-------------\t-------------\n";

	bool convergence_passed = true;
	const double expected_rate = 2.0; // PPM should give ~2nd order for smooth problems
	const double tolerance = 0.3;	  // Allow 30% deviation from expected rate

	for (int i = 1; i < resolutions.size(); ++i) {
		// Calculate convergence rate: p = log(E(2h)/E(h)) / log(2)
		double const log_error_ratio = std::log(errors[i - 1] / errors[i]);
		double const log_dx_ratio = std::log(dx_values[i - 1] / dx_values[i]);
		double const observed_rate = log_error_ratio / log_dx_ratio;

		amrex::Print() << fmt::format("{:4d} -> {:4d}\t{:13.2f}\t{:13.1f}\n", resolutions[i - 1], resolutions[i], observed_rate, expected_rate);

		// Check if convergence rate is within acceptable range
		if (std::abs(observed_rate - expected_rate) > tolerance) {
			convergence_passed = false;
		}
	}

	// Calculate overall convergence rate from first to last resolution
	if (resolutions.size() >= 2) {
		double const overall_log_error_ratio = std::log(errors[0] / errors.back());
		double const overall_log_dx_ratio = std::log(dx_values[0] / dx_values.back());
		double const overall_rate = overall_log_error_ratio / overall_log_dx_ratio;

		amrex::Print() << fmt::format("\nOverall convergence rate: {:.2f}\n", overall_rate);
		amrex::Print() << fmt::format("Expected rate: {:.1f}\n", expected_rate);

		if (std::abs(overall_rate - expected_rate) > tolerance) {
			convergence_passed = false;
		}
	}

	// Output results for analysis
	if (amrex::ParallelDescriptor::IOProcessor()) {
		std::ofstream file("hydro_wave_convergence.csv");
		file << "nx,dx,error\n";
		for (int i = 0; i < resolutions.size(); ++i) {
			file << fmt::format("{},{:.6e},{:.6e}\n", resolutions[i], dx_values[i], errors[i]);
		}
		file.close();
		amrex::Print() << "\nConvergence data written to hydro_wave_convergence.csv\n";
	}

	// Test status
	if (convergence_passed) {
		if (reached_target) {
			amrex::Print() << fmt::format("\n✓ Richardson convergence test PASSED (target error {:.3e} reached)\n", machine_precision_target);
		} else {
			amrex::Print() << "\n✓ Richardson convergence test PASSED\n";
		}
		return 0;
	} else {
		amrex::Print() << "\n✗ Richardson convergence test FAILED\n";
		amrex::Print() << "Observed convergence rate deviates from expected rate by more than " << tolerance << "\n";
		return 1;
	}
}
