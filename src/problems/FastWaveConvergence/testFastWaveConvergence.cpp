//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file testFastWaveConvergence.cpp
/// \brief Defines a Richardson convergence test for the fast MHD wave.
///

#include <bitset>
#include <cassert>
#include <cmath>
#include <gcem.hpp>
#include <iostream>

#include "AMReX_Array.H"
#include "AMReX_Array4.H"
#include "AMReX_REAL.H"

#include "QuokkaSimulation.hpp"
#include "grid.hpp"
#include "physics_info.hpp"
#include "util/BC.hpp"
#include "util/fextract.hpp"

struct FastWaveConvergence {
};

template <> struct quokka::EOS_Traits<FastWaveConvergence> {
	static constexpr double gamma = 5. / 3.;
	static constexpr double mean_molecular_weight = C::m_u;
	static constexpr double boltzmann_constant = C::k_B;
};

template <> struct Physics_Traits<FastWaveConvergence> {
	static constexpr bool is_hydro_enabled = true;
	static constexpr int numMassScalars = 0;		     // number of mass scalars
	static constexpr int numPassiveScalars = numMassScalars + 0; // number of passive scalars
	static constexpr bool is_self_gravity_enabled = false;
	static constexpr bool is_radiation_enabled = false;
	static constexpr bool is_mhd_enabled = true;
	static constexpr int nGroups = 1; // number of radiation groups
	static constexpr UnitSystem unit_system = UnitSystem::CGS;
};

// constants
constexpr double sound_speed = 1.0;
constexpr double gamma_gas = quokka::EOS_Traits<FastWaveConvergence>::gamma;

// background states
constexpr double bg_density = 1.0;
constexpr double bg_pressure = sound_speed * sound_speed * bg_density / gamma_gas;
constexpr double bg_mag_amplitude = 1.;

// theta is the angle between k and background magnetic field bg_mag
constexpr double theta_degrees = 90.0; // degrees
constexpr double cos_theta = gcem::cos(theta_degrees * M_PI / 180.0);

// k = 2 pi / wave length
// box length = 1, so |k| in [1, inf)
constexpr double num_modes = 1;
constexpr double k_amplitude = 2 * M_PI * num_modes;

// input perturbation: choose to do this via the relative density field in [0, 1]. remember, the linear regime is valid when this perturbation is small
constexpr double delta_b = 1e-4;

constexpr double alfven_speed = bg_mag_amplitude / gcem::sqrt(bg_density);
constexpr double magnetosonic_speed = gcem::sqrt(alfven_speed * alfven_speed + sound_speed * sound_speed);
constexpr double bg_mag_x3 = bg_mag_amplitude;

constexpr double omega =
    gcem::sqrt(gcem::pow(k_amplitude, 2) / 2.0 *
	       (gcem::pow(magnetosonic_speed, 2) + gcem::sqrt(gcem::pow(magnetosonic_speed, 4) - 4.0 * gcem::pow(alfven_speed, 2) * gcem::pow(sound_speed, 2) *
												     gcem::pow(cos_theta, 2)))); // NOLINT(cert-err58-cpp)

AMREX_GPU_DEVICE auto computeMagneticVectorPotential_x(double x1, double x2, double /*x3*/, double time)
{
	return -x2 / 2.0 * (bg_mag_amplitude + delta_b * std::cos(omega * time - k_amplitude * x1));
}
AMREX_GPU_DEVICE auto computeMagneticVectorPotential_y(double x1, double /*x2*/, double /*x3*/, double time) -> double
{
	return bg_mag_amplitude * x1 / 2.0 + ((delta_b)*std::sin(omega * time - k_amplitude * x1) / (-2.0 * k_amplitude));
}
AMREX_GPU_DEVICE auto computeMagneticVectorPotential_z(double /*x1*/, double /*x2*/, double /*x3*/, double /*time*/) -> double { return 0.0; }

////////////////////////////////////
///
///
///

AMREX_GPU_DEVICE void computeWaveSolution(int i, int j, int k, amrex::Array4<amrex::Real> const &state, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx,
					  amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_lo, quokka::centering cen, quokka::direction dir,
					  double time)
{
	const amrex::Real x1_L = prob_lo[0] + i * dx[0];
	const amrex::Real x2_L = prob_lo[1] + j * dx[1];
	const amrex::Real x3_L = prob_lo[2] + k * dx[2];

	const amrex::Real x1_C = x1_L + static_cast<amrex::Real>(0.5) * dx[0];

	if (cen == quokka::centering::cc) {
		const double cos_wave_C = std::cos(omega * time - k_amplitude * x1_C);

		// magnetic field at the center of the cell
		const double x1mag = 0.0;
		const double x2mag = 0.0;
		const double x3mag = bg_mag_amplitude + delta_b * cos_wave_C;

		const double density = bg_density + bg_density * delta_b / bg_mag_amplitude * cos_wave_C;
		const double pressure = bg_pressure + bg_pressure * gamma_gas * delta_b / bg_mag_amplitude * cos_wave_C;
		const double x1vel = magnetosonic_speed * delta_b / bg_mag_amplitude * cos_wave_C;
		const double x2vel = 0.0;
		const double x3vel = 0.0;

		const double velocity_magnitude = std::sqrt(std::pow(x1vel, 2) + std::pow(x2vel, 2) + std::pow(x3vel, 2));
		const double momentum = density * velocity_magnitude;
		const double Ekin = 0.5 * std::pow(momentum, 2) / density;
		const double Emag = 0.5 * (x1mag * x1mag + x2mag * x2mag + x3mag * x3mag);
		const double Eint = pressure / (gamma_gas - 1);
		const double Etot = Ekin + Emag + Eint;

		state(i, j, k, HydroSystem<FastWaveConvergence>::density_index) = density;
		state(i, j, k, HydroSystem<FastWaveConvergence>::x1Momentum_index) = x1vel * density;
		state(i, j, k, HydroSystem<FastWaveConvergence>::x2Momentum_index) = x2vel * density;
		state(i, j, k, HydroSystem<FastWaveConvergence>::x3Momentum_index) = x3vel * density;
		state(i, j, k, HydroSystem<FastWaveConvergence>::energy_index) = Etot;
		state(i, j, k, HydroSystem<FastWaveConvergence>::internalEnergy_index) = Eint;
	} else if (cen == quokka::centering::fc) {
		const double x1mag = (computeMagneticVectorPotential_z(x1_L, x2_L + dx[1], x3_L + dx[2] / 2, time) -
				      computeMagneticVectorPotential_z(x1_L, x2_L, x3_L + dx[2] / 2, time)) /
					 dx[1] -
				     (computeMagneticVectorPotential_y(x1_L, x2_L + dx[1] / 2, x3_L + dx[2], time) -
				      computeMagneticVectorPotential_y(x1_L, x2_L + dx[1] / 2, x3_L, time)) /
					 dx[2];
		const double x2mag = (computeMagneticVectorPotential_x(x1_L + dx[0] / 2, x2_L, x3_L + dx[2], time) -
				      computeMagneticVectorPotential_x(x1_L + dx[0] / 2, x2_L, x3_L, time)) /
					 dx[2] -
				     (computeMagneticVectorPotential_z(x1_L + dx[0], x2_L, x3_L + dx[2] / 2, time) -
				      computeMagneticVectorPotential_z(x1_L, x2_L, x3_L + dx[2] / 2, time)) /
					 dx[0];

		const double x3mag = ((computeMagneticVectorPotential_y(x1_L + dx[0], x2_L + (dx[1] / 2), x3_L, time) -
				       computeMagneticVectorPotential_y(x1_L, x2_L + (dx[1] / 2), x3_L, time)) /
				      dx[0]) -
				     ((computeMagneticVectorPotential_x(x1_L + (dx[0] / 2), x2_L + dx[1], x3_L, time) -
				       computeMagneticVectorPotential_x(x1_L + (dx[0] / 2), x2_L, x3_L, time)) /
				      dx[1]);

		if (dir == quokka::direction::x) {
			state(i, j, k, MHDSystem<FastWaveConvergence>::bfield_index) = x1mag;
		} else if (dir == quokka::direction::y) {
			state(i, j, k, MHDSystem<FastWaveConvergence>::bfield_index) = x2mag;
		} else if (dir == quokka::direction::z) {
			state(i, j, k, MHDSystem<FastWaveConvergence>::bfield_index) = x3mag;
		}
	}
}

template <> void QuokkaSimulation<FastWaveConvergence>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	// extract grid information
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = grid_elem.dx_;
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = grid_elem.prob_lo_;
	const amrex::Array4<double> &state_cc = grid_elem.array_;
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const quokka::centering cen = grid_elem.cen_;
	const quokka::direction dir = grid_elem.dir_;

	const int ncomp_cc = Physics_Indices<FastWaveConvergence>::nvarTotal_cc;
	// loop over the grid and set the initial condition
	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		for (int n = 0; n < ncomp_cc; ++n) {
			state_cc(i, j, k, n) = 0; // fill unused quantities with zeros
		}
		computeWaveSolution(i, j, k, state_cc, dx, prob_lo, cen, dir, 0);
	});
}

template <> void QuokkaSimulation<FastWaveConvergence>::setInitialConditionsOnGridFaceVars(quokka::grid const &grid_elem)
{
	// extract grid information
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = grid_elem.dx_;
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = grid_elem.prob_lo_;
	const amrex::Array4<double> &state_fc = grid_elem.array_;
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const quokka::centering cen = grid_elem.cen_;
	const quokka::direction dir = grid_elem.dir_;

	const int ncomp_fc = Physics_Indices<FastWaveConvergence>::nvarPerDim_fc;
	// loop over the grid and set the initial condition
	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		for (int n = 0; n < ncomp_fc; ++n) {
			state_fc(i, j, k, n) = 0; // fill unused quantities with zeros
		}
		computeWaveSolution(i, j, k, state_fc, dx, prob_lo, cen, dir, 0);
	});
}

template <>
void QuokkaSimulation<FastWaveConvergence>::computeReferenceSolution(amrex::MultiFab &ref, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx,
								     amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_lo)
{
	for (amrex::MFIter iter(ref); iter.isValid(); ++iter) {
		const amrex::Box &indexRange = iter.validbox();
		auto const &stateExact = ref.array(iter);
		auto const ncomp = ref.nComp();
        const amrex::Real time = tNew_[0];

		amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
			for (int n = 0; n < ncomp; ++n) {
				stateExact(i, j, k, n) = 0.0; // fill unused quantities with zeros
			}
			computeWaveSolution(i, j, k, stateExact, dx, prob_lo, quokka::centering::cc, quokka::direction::na, time);
		});
	}
}

template <>
void QuokkaSimulation<FastWaveConvergence>::computeReferenceSolution_fc(amrex::MultiFab &ref, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx,
									amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_lo,
									quokka::direction const dir)
{
	for (amrex::MFIter iter(ref); iter.isValid(); ++iter) {
		const amrex::Box &indexRange = iter.validbox();
		auto const &stateExact = ref.array(iter);
		auto const ncomp = ref.nComp();
               const amrex::Real time = tNew_[0];


		amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
			for (int n = 0; n < ncomp; ++n) {
				stateExact(i, j, k, n) = 0.0; // fill unused quantities with zeros
			}
			computeWaveSolution(i, j, k, stateExact, dx, prob_lo, quokka::centering::fc, dir, time);
		});
	}
}

auto runWaveTest(int nx) -> double
{
	// Problem parameters
	const double CFL_number = 0.3;
	const double max_time = 1.0;
	const int max_timesteps = std::max(20000, nx * 100);

	// Problem initialization
	const int ncomp_cc = Physics_Indices<FastWaveConvergence>::nvarTotal_cc;
	amrex::Vector<amrex::BCRec> BCs_cc(ncomp_cc);
	for (int n = 0; n < ncomp_cc; ++n) {
		for (int i = 0; i < AMREX_SPACEDIM; ++i) {
			BCs_cc[n].setLo(i, amrex::BCType::int_dir); // periodic
			BCs_cc[n].setHi(i, amrex::BCType::int_dir);
		}
	}

	const int nvars_fc = Physics_Indices<FastWaveConvergence>::nvarTotal_fc;
	amrex::Vector<amrex::BCRec> BCs_fc(nvars_fc);
	for (int icomp = 0; icomp < nvars_fc; ++icomp) {
		for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
			BCs_fc[icomp].setLo(idim, amrex::BCType::int_dir); // periodic
			BCs_fc[icomp].setHi(idim, amrex::BCType::int_dir);
		}
	}

	// Set grid dimensions using AMReX parameter system
	amrex::ParmParse pp("amr");
	amrex::Vector<int> const ncells = {nx, 8, 8};
	pp.add("max_level", 0);
	pp.add("blocking_factor_x", nx);
	pp.add("blocking_factor_y", 8);
	pp.add("blocking_factor_z", 8);
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

	QuokkaSimulation<FastWaveConvergence> sim(BCs_cc, BCs_fc);

    sim.computeReferenceSolution_ = true;
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
	for (int n = 0; n < QuokkaSimulation<FastWaveConvergence>::ncompHydro_; ++n) {
		if (n == HydroSystem<FastWaveConvergence>::internalEnergy_index) {
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
	const double machine_precision_target = 2.0e-11;
	const int nx_initial = 128;
	const int nx_max = 2048;
	bool reached_target = false;

	// Silence TinyProfiler so convergence logs stay readable
	{
		amrex::ParmParse pp_tp("tiny_profiler");
		if (!pp_tp.contains("output_file")) {
			pp_tp.add("output_file", std::string("/dev/null"));
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

	amrex::Print() << "Running Richardson convergence test for FastWave:\n";
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
			amrex::Print() << fmt::format("\nReached maximum resolution (nx = {}) without achieving the target error {:.3e}\n", nx_max,
						      machine_precision_target);
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
		if (observed_rate + tolerance < expected_rate) {
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

		if (overall_rate + tolerance < expected_rate) {
			convergence_passed = false;
		}
	}

	// Output results for analysis
	if (amrex::ParallelDescriptor::IOProcessor()) {
		std::ofstream file("fast_wave_convergence.csv");
		file << "nx,dx,error\n";
		for (int i = 0; i < resolutions.size(); ++i) {
			file << fmt::format("{},{:.6e},{:.6e}\n", resolutions[i], dx_values[i], errors[i]);
		}
		file.close();
		amrex::Print() << "\nConvergence data written to fast_wave_convergence.csv\n";
	}

	// Test status
	if (convergence_passed) {
		if (reached_target) {
			amrex::Print() << fmt::format("\n✓ Richardson convergence test PASSED (target error {:.3e} reached)\n", machine_precision_target);
		} else {
			amrex::Print() << "\n✓ Richardson convergence test PASSED\n";
		}
		return 0;
	}

	amrex::Print() << "\n✗ Richardson convergence test FAILED\n";
	amrex::Print() << "Observed convergence rate deviates from expected rate by more than " << tolerance << "\n";
	return 1;
}
