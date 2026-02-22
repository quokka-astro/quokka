//==============================================================================
// Shared Richardson convergence driver
//==============================================================================

#ifndef QUOKKA_RICHARDSON_HPP
#define QUOKKA_RICHARDSON_HPP

#include <cmath>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

#include <format>

#include "AMReX.H"
#include "AMReX_ParallelDescriptor.H"
#include "AMReX_ParmParse.H"
#include "AMReX_Print.H"
#include "AMReX_Vector.H"

namespace quokka::richardson
{

struct Parameters {
	double machine_precision_target = 0.0;
	int nx_initial = 0;
	int nx_max = 0;
	double expected_rate = 2.0;
	double tolerance = 0.3;
	std::string test_name;
	std::string csv_filename;
};

inline void applyQuietDefaults()
{
	{
		amrex::ParmParse pp_tp("tiny_profiler");
		if (!pp_tp.contains("output_file")) {
			pp_tp.add("output_file", std::string("/dev/null"));
		}
	}

	{
		amrex::ParmParse pp_general;
		if (!pp_general.contains("suppress_output")) {
			pp_general.add("suppress_output", 1);
		}
	}
}

template <typename Callable> auto run(const Parameters &params, Callable &&runTest) -> int
{
	amrex::Vector<int> resolutions;
	amrex::Vector<double> errors;
	amrex::Vector<double> dx_values;

	bool reached_target = false;

	amrex::Print() << std::format("Running Richardson convergence test for {}:\n", params.test_name);
	amrex::Print() << "Resolution\tError Norm\n";
	amrex::Print() << "----------\t----------\n";

	for (int nx = params.nx_initial; nx <= params.nx_max; nx *= 2) {
		double error = std::forward<Callable>(runTest)(nx);
		amrex::ParallelDescriptor::Bcast(&error, 1, amrex::ParallelDescriptor::IOProcessorNumber());

		resolutions.push_back(nx);
		errors.push_back(error);
		dx_values.push_back(1.0 / static_cast<double>(nx)); // dx = L / nx for unit domain

		amrex::Print() << std::format("{:10d}\t{:.6e}\n", nx, error);

		if (error <= params.machine_precision_target) {
			reached_target = true;
			break;
		}

		if (nx == params.nx_max) {
			amrex::Print() << std::format("\nReached maximum resolution (nx = {}) without achieving the target error {:.3e}\n", params.nx_max,
						      params.machine_precision_target);
			break;
		}
	}

	amrex::Print() << "\nConvergence Rate Analysis:\n";
	amrex::Print() << "Resolution Pair\tObserved Rate\tExpected Rate\n";
	amrex::Print() << "---------------\t-------------\t-------------\n";

	bool convergence_passed = true;

	for (int i = 1; i < resolutions.size(); ++i) {
		double const log_error_ratio = std::log(errors[i - 1] / errors[i]);
		double const log_dx_ratio = std::log(dx_values[i - 1] / dx_values[i]);
		double const observed_rate = log_error_ratio / log_dx_ratio;

		amrex::Print() << std::format("{:4d} -> {:4d}\t{:13.2f}\t{:13.1f}\n", resolutions[i - 1], resolutions[i], observed_rate, params.expected_rate);

		if (observed_rate + params.tolerance < params.expected_rate) {
			convergence_passed = false;
		}
	}

	if (resolutions.size() >= 2) {
		double const overall_log_error_ratio = std::log(errors[0] / errors.back());
		double const overall_log_dx_ratio = std::log(dx_values[0] / dx_values.back());
		double const overall_rate = overall_log_error_ratio / overall_log_dx_ratio;

		amrex::Print() << std::format("\nOverall convergence rate: {:.2f}\n", overall_rate);
		amrex::Print() << std::format("Expected rate: {:.1f}\n", params.expected_rate);

		if (overall_rate + params.tolerance < params.expected_rate) {
			convergence_passed = false;
		}
	}

	if (!params.csv_filename.empty() && amrex::ParallelDescriptor::IOProcessor()) {
		std::ofstream file(params.csv_filename);
		file << "nx,dx,error\n";
		for (int i = 0; i < resolutions.size(); ++i) {
			file << std::format("{},{:.6e},{:.6e}\n", resolutions[i], dx_values[i], errors[i]);
		}
		file.close();
		amrex::Print() << std::format("\nConvergence data written to {}\n", params.csv_filename);
	}

	if (convergence_passed) {
		if (reached_target) {
			amrex::Print() << std::format("\n✓ Richardson convergence test PASSED (target error {:.3e} reached)\n",
						      params.machine_precision_target);
		} else {
			amrex::Print() << "\n✓ Richardson convergence test PASSED\n";
		}
		return 0;
	}

	amrex::Print() << "\n✗ Richardson convergence test FAILED\n";
	amrex::Print() << "Observed convergence rate deviates from expected rate by more than " << params.tolerance << "\n";
	return 1;
}

} // namespace quokka::richardson

#endif // QUOKKA_RICHARDSON_HPP
