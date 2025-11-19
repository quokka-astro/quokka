// ABOUTME: Test problem for cooling integrator accuracy using isochoric cooling
// ABOUTME: Uses the ResampledCooling module via runtime parameter
//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file test_resampled_cooling.cpp
/// \brief Defines a test problem for cooling integrator accuracy using the ResampledCooling module.
///

#ifdef HAVE_PYTHON
#include "util/matplotlibcpp.h"
#endif
#include "cooling/ResampledCooling.hpp"
#include "math/interpolate.hpp"
#include "util/BC.hpp"
#include <fmt/format.h>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <vector>

#include "AMReX_BC_TYPES.H"
#include "AMReX_ParmParse.H"

#include "QuokkaSimulation.hpp"
#include "hydro/hydro_system.hpp"
#include "util/fextract.hpp"

struct ResampledCoolingTest {
}; // dummy type to allow compile-time polymorphism via template specialization

// Function to read CSV reference solution
auto readReferenceCSV(const std::string &filename) -> std::pair<std::vector<double>, std::vector<double>>
{
	std::vector<double> t_ref;
	std::vector<double> T_ref;
	std::ifstream file(filename);

	if (!file.is_open()) {
		amrex::Abort("Could not open reference solution file: " + filename);
	}

	std::string line;
	// Skip header line if present
	if (std::getline(file, line) && (line.find("time") != std::string::npos || line.find("Time") != std::string::npos)) {
		// Header found, continue reading data
	} else {
		// No header, reset to beginning
		file.clear();
		file.seekg(0);
	}

	while (std::getline(file, line)) {
		std::stringstream ss(line);
		std::string time_str;
		std::string temp_str;

		// Expect CSV format: time, temperature
		if (std::getline(ss, time_str, ',') && std::getline(ss, temp_str)) {
			try {
				double const t = std::stod(time_str);
				double const T = std::stod(temp_str);
				t_ref.push_back(t);
				T_ref.push_back(T);
			} catch (const std::exception &e) {
				// Skip invalid lines - this is expected for malformed CSV entries
				(void)e; // suppress unused variable warning
				continue;
			}
		}
	}

	if (t_ref.empty()) {
		amrex::Abort("No valid data found in reference solution file: " + filename);
	}

	return {t_ref, T_ref};
}

template <> struct SimulationData<ResampledCoolingTest> {
	std::vector<double> t_vec_;
	std::vector<double> T_vec_;
};

template <> struct quokka::EOS_Traits<ResampledCoolingTest> {
	static constexpr double mean_molecular_weight = C::m_u;
	static constexpr double gamma = 5. / 3.;
};

template <> struct Physics_Traits<ResampledCoolingTest> {
	static constexpr bool is_self_gravity_enabled = false;
	static constexpr bool is_hydro_enabled = true;
	static constexpr int numMassScalars = 0;		     // number of mass scalars
	static constexpr int numPassiveScalars = numMassScalars + 0; // number of passive scalars
	static constexpr bool is_radiation_enabled = false;
	static constexpr bool is_mhd_enabled = false;
	static constexpr int nGroups = 1; // number of radiation groups
	static constexpr UnitSystem unit_system = UnitSystem::CGS;
};

// Initial conditions: hot gas that will cool down
constexpr double T_initial = 1.0e7;	// K
constexpr double rho_initial = 1.0e-26; // g cm^-3 (constant density for isochoric)

template <> void QuokkaSimulation<ResampledCoolingTest>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const amrex::Array4<double> &state_cc = grid_elem.array_;

	// Compute initial internal energy from temperature
	const double k_B = C::k_B;
	const double gamma = quokka::EOS_Traits<ResampledCoolingTest>::gamma;
	const double mu = 0.6 * quokka::EOS_Traits<ResampledCoolingTest>::mean_molecular_weight;

	// For ideal gas: P = (gamma - 1) * rho * e_int
	// and P = rho * k_B * T / (mu * m_u)
	// Therefore: e_int = k_B * T / ((gamma - 1) * mu * m_u)
	const double e_int_initial = k_B * T_initial / ((gamma - 1.0) * mu);
	const double Eint_initial = rho_initial * e_int_initial;

	// loop over the grid and set the initial condition
	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		state_cc(i, j, k, HydroSystem<ResampledCoolingTest>::density_index) = rho_initial;
		state_cc(i, j, k, HydroSystem<ResampledCoolingTest>::x1Momentum_index) = 0.;
		state_cc(i, j, k, HydroSystem<ResampledCoolingTest>::x2Momentum_index) = 0.;
		state_cc(i, j, k, HydroSystem<ResampledCoolingTest>::x3Momentum_index) = 0.;
		state_cc(i, j, k, HydroSystem<ResampledCoolingTest>::energy_index) = Eint_initial;
		state_cc(i, j, k, HydroSystem<ResampledCoolingTest>::internalEnergy_index) = Eint_initial;
	});
}

template <> void QuokkaSimulation<ResampledCoolingTest>::computeAfterTimestep()
{
	// Extract solution at center of domain
	auto [position, values] = fextract(state_new_cc_[0], Geom(0), 0, 0.5);

	if (amrex::ParallelDescriptor::IOProcessor()) {
		userData_.t_vec_.push_back(tNew_[0]);

		const amrex::Real Etot = values.at(HydroSystem<ResampledCoolingTest>::energy_index)[0];
		const amrex::Real rho = values.at(HydroSystem<ResampledCoolingTest>::density_index)[0];
		// For isochoric cooling with no kinetic energy, Eint = Etot
		const amrex::Real Eint = Etot;

		// Get temperature from tables
		amrex::Real T = NAN;
		if (coolingTableType_.empty()) {
			coolingTableType_ = "resampled";
		}
		if (coolingTableType_ == "resampled") {
			T = quokka::ResampledCooling::ComputeTgasFromEgas(rho, Eint, resampledTables_.const_tables());
		} else {
			amrex::Abort("Unsupported cooling table type: " + coolingTableType_);
		}

		userData_.T_vec_.push_back(T);
	}
}

auto problem_main() -> int
{
	// Read runtime parameters
	amrex::ParmParse const pp("cooling_test");
	std::string reference_file;
	pp.query("reference_solution_file", reference_file);

	std::string output_csv_file;
	pp.query("output_csv_file", output_csv_file);

	// Set boundary conditions - extrapolate
	auto BCs_cc = quokka::BC<ResampledCoolingTest>(quokka::BCType::foextrap);

	QuokkaSimulation<ResampledCoolingTest> sim(BCs_cc);

	sim.plotfileInterval_ = -1;

	// initialize
	sim.setInitialConditions();

	// Add initial condition to data vectors
	if (amrex::ParallelDescriptor::IOProcessor()) {
		sim.userData_.t_vec_.push_back(0.0);
		sim.userData_.T_vec_.push_back(T_initial);
	}

	// evolve
	sim.evolve();

	// Analyze results
	int status = 0;

	if (amrex::ParallelDescriptor::IOProcessor()) {
		// Check that gas has cooled significantly
		const double T_final = sim.userData_.T_vec_.back();
		const double T_ratio = T_final / T_initial;
		amrex::Print() << "Temperature ratio (final/initial): " << T_ratio << '\n';

		if (T_ratio > 0.1) {
			amrex::Print() << "ERROR: Gas did not cool sufficiently! T_final/T_initial = " << T_ratio << '\n';
			status = 1;
		}

		// Compare with reference solution if provided
		if (!reference_file.empty()) {
			amrex::Print() << "Comparing with reference solution: " << reference_file << '\n';

			try {
				auto [t_ref, T_ref] = readReferenceCSV(reference_file);

				// Interpolate reference solution onto simulation timesteps
				std::vector<double> T_ref_interp(sim.userData_.t_vec_.size());
				interpolate_arrays(sim.userData_.t_vec_.data(), T_ref_interp.data(), static_cast<int>(sim.userData_.t_vec_.size()),
						   t_ref.data(), T_ref.data(), static_cast<int>(t_ref.size()));

				// Compute L1 error norm
				double err_norm = 0.; // NOLINT(misc-const-correctness)
				double sol_norm = 0.; // NOLINT(misc-const-correctness)
				for (size_t i = 0; i < sim.userData_.t_vec_.size(); ++i) {
					err_norm += std::abs(sim.userData_.T_vec_[i] - T_ref_interp[i]);
					sol_norm += std::abs(T_ref_interp[i]);
				}
				const double rel_error = err_norm / sol_norm;
				const double error_tol = 0.02; // should be about 1% between different cooling modules

				amrex::Print() << "Reference solution comparison:" << '\n';
				amrex::Print() << "  Data points in reference: " << t_ref.size() << '\n';
				amrex::Print() << "  Simulation timesteps: " << sim.userData_.t_vec_.size() << '\n';
				amrex::Print() << "  Relative L1 error norm: " << rel_error << '\n';
				amrex::Print() << "  Error tolerance: " << error_tol << '\n';

				if (rel_error > error_tol) {
					amrex::Print() << "ERROR: Solution differs from reference by more than tolerance!" << '\n';
					status = 1;
				} else {
					amrex::Print() << "SUCCESS: Solution matches reference within tolerance." << '\n';
				}

			} catch (const std::exception &e) {
				amrex::Print() << "ERROR: Failed to compare with reference solution: " << e.what() << '\n';
				status = 1;
			}
		} else {
			amrex::Print() << "No reference solution file provided for comparison." << '\n';
		}

		// Output cooling solution to CSV file if requested
		if (!output_csv_file.empty()) {
			amrex::Print() << "Writing cooling solution to CSV file: " << output_csv_file << '\n';

			std::ofstream csv_file(output_csv_file);
			if (!csv_file.is_open()) {
				amrex::Print() << "ERROR: Could not open output CSV file: " << output_csv_file << '\n';
				status = 1;
			} else {
				// Write CSV header
				csv_file << "time,temperature\n";

				// Write data
				const std::vector<double> &t = sim.userData_.t_vec_;
				const std::vector<double> &T = sim.userData_.T_vec_;
				for (size_t i = 0; i < t.size(); ++i) {
					csv_file << std::scientific << std::setprecision(10) << t[i] << "," << T[i] << "\n";
				}

				csv_file.close();
				amrex::Print() << "Successfully wrote " << t.size() << " data points to " << output_csv_file << '\n';
			}
		}

#ifdef HAVE_PYTHON
		// Plot results
		const std::vector<double> &t = sim.userData_.t_vec_;
		const std::vector<double> &T = sim.userData_.T_vec_;

		// Convert time to Myr
		std::vector<double> t_Myr(t.size());
		const double s_to_Myr = 1.0 / (3.15576e13); // 1 Myr in seconds
		for (size_t i = 0; i < t.size(); ++i) {
			t_Myr[i] = t[i] * s_to_Myr;
		}

		// Plot temperature evolution
		matplotlibcpp::clf();
		std::map<std::string, std::string> sim_args;
		sim_args["label"] = "Simulation";
		sim_args["marker"] = "o";
		sim_args["markersize"] = "3";
		matplotlibcpp::plot(t_Myr, T, sim_args);

		// Plot reference solution if available
		if (!reference_file.empty()) {
			try {
				auto [t_ref, T_ref] = readReferenceCSV(reference_file);
				std::vector<double> t_ref_Myr(t_ref.size());
				const double s_to_Myr = 1.0 / (3.15576e13);
				for (size_t i = 0; i < t_ref.size(); ++i) {
					t_ref_Myr[i] = t_ref[i] * s_to_Myr;
				}

				std::map<std::string, std::string> ref_args;
				ref_args["label"] = "Reference";
				ref_args["linestyle"] = "--";
				ref_args["color"] = "red";
				matplotlibcpp::plot(t_ref_Myr, T_ref, ref_args);
				matplotlibcpp::legend();
			} catch (const std::exception &e) {
				// Continue without reference plot - this is expected if file is malformed
				(void)e; // suppress unused variable warning
			}
		}
		matplotlibcpp::xscale("log");
		matplotlibcpp::yscale("log");
		matplotlibcpp::xlabel("time (Myr)");
		matplotlibcpp::ylabel("Temperature (K)");
		matplotlibcpp::title("Isochoric Cooling Test");
		matplotlibcpp::save("./cooling_temperature.pdf");
#endif
	}

	// Cleanup and exit
	amrex::Print() << "Finished." << '\n';
	return status;
}
