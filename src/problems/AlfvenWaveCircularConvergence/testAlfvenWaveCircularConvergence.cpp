//==============================================================================
// Copyright 2022 Neco Kriel.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file testAlfvenWaveCircularConvergence.cpp
/// \brief Defines a Richardson convergence test for the circularly-polarized Alfven wave, and
///        makes sure face-centred quantities are created correctly.
///

#include <algorithm>
#include <cassert>
#include <cmath>
#include <format>
#include <gcem.hpp>

#include "AMReX_Array.H"
#include "AMReX_Array4.H"
#include "AMReX_ParmParse.H"
#include "AMReX_Print.H"
#include "AMReX_REAL.H"

#include "QuokkaSimulation.hpp"
#include "grid.hpp"
#include "physics_info.hpp"
#include "util/BC.hpp"
#include "util/richardson.hpp"

struct AlfvenWaveCircular {
};

template <> struct quokka::EOS_Traits<AlfvenWaveCircular> {
	static constexpr double gamma = 5. / 3.;
	static constexpr double mean_molecular_weight = C::m_u;
};

template <> struct Physics_Traits<AlfvenWaveCircular> : DefaultPhysicsTraits {
	static constexpr bool is_hydro_enabled = true;
	static constexpr bool is_mhd_enabled = true;
};

// constants
constexpr double sound_speed = 1.0;
constexpr double gamma_gas = quokka::EOS_Traits<AlfvenWaveCircular>::gamma;

// we have set up the problem so that the direction of wave propagation, vec(k), is aligned with the x1-direction, and the background magnetic field sits in the
// x1-x2 plane

// k = 2 pi / wave_length. note: wave_length should be an integer, because of periodic BCs + the requirement that the magnetic field be continuous. also, the
// box length = 1, so |k| in [1, inf)
constexpr double num_modes = 1;
constexpr double k_amplitude = 2.0 * M_PI * num_modes;

// background states
constexpr double bg_density = 1.0;
constexpr double bg_pressure = sound_speed * sound_speed * bg_density / gamma_gas;
constexpr double bg_mag_amplitude = 1.0;

// magnetic field properties
constexpr double delta_b = 1e-6;
constexpr double alfven_speed = bg_mag_amplitude / gcem::sqrt(bg_density);

constexpr double omega = gcem::sqrt(gcem::pow(alfven_speed, 2.0) * gcem::pow(k_amplitude, 2.0));

AMREX_GPU_DEVICE auto computeMagneticVectorPotential_x(double x1, double /*x2*/, double x3, double time) -> double
{
	return bg_mag_amplitude * delta_b * x3 * std::sin(omega * time - k_amplitude * x1);
}

AMREX_GPU_DEVICE auto computeMagneticVectorPotential_y(double x1, double /*x2*/, double /*x3*/, double time) -> double
{
	return -bg_mag_amplitude * delta_b / k_amplitude * std::sin(omega * time - k_amplitude * x1);
}

AMREX_GPU_DEVICE auto computeMagneticVectorPotential_z(double /*x1*/, double x2, double /*x3*/, double /*time*/) -> double { return bg_mag_amplitude * x2; }

AMREX_GPU_DEVICE void computeWaveSolution(int i, int j, int k, amrex::Array4<amrex::Real> const &state, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx,
					  amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_lo, quokka::centering cen, quokka::direction dir,
					  double time)
{
	const amrex::Real x1_L = prob_lo[0] + i * dx[0];
	const amrex::Real x2_L = prob_lo[1] + j * dx[1];
	const amrex::Real x3_L = prob_lo[2] + k * dx[2];

	const amrex::Real x1_C = x1_L + static_cast<amrex::Real>(0.5) * dx[0];

	if (cen == quokka::centering::cc) {
		const double sin_wave_C = std::sin(omega * time - k_amplitude * x1_C);
		const double cos_wave_C = std::cos(omega * time - k_amplitude * x1_C);

		const double density = bg_density;
		const double pressure = bg_pressure;
		const double x1vel = 0.0;
		const double x2vel = -omega * delta_b / (sound_speed * k_amplitude) * sin_wave_C;
		const double x3vel = -omega * delta_b / (sound_speed * k_amplitude) * cos_wave_C;

		// magnetic field at the center of the cell
		const double x1mag = bg_mag_amplitude;
		const double x2mag = bg_mag_amplitude * delta_b * sin_wave_C;
		const double x3mag = bg_mag_amplitude * delta_b * cos_wave_C;

		const double velocity_magnitude = std::sqrt(std::pow(x1vel, 2) + std::pow(x2vel, 2) + std::pow(x3vel, 2));
		const double momentum = density * velocity_magnitude;
		const double Ekin = 0.5 * std::pow(momentum, 2.0) / density;
		const double Emag = 0.5 * (x1mag * x1mag + x2mag * x2mag + x3mag * x3mag);
		const double Eint = pressure / (gamma_gas - 1);
		const double Etot = Ekin + Emag + Eint;

		state(i, j, k, HydroSystem<AlfvenWaveCircular>::density_index) = density;
		state(i, j, k, HydroSystem<AlfvenWaveCircular>::x1Momentum_index) = x1vel * density;
		state(i, j, k, HydroSystem<AlfvenWaveCircular>::x2Momentum_index) = x2vel * density;
		state(i, j, k, HydroSystem<AlfvenWaveCircular>::x3Momentum_index) = x3vel * density;
		state(i, j, k, HydroSystem<AlfvenWaveCircular>::energy_index) = Etot;
		state(i, j, k, HydroSystem<AlfvenWaveCircular>::internalEnergy_index) = Eint;
	} else if (cen == quokka::centering::fc) {
		const double x1mag = (computeMagneticVectorPotential_z(x1_L, x2_L + dx[1], x3_L + dx[2] / 2.0, time) -
				      computeMagneticVectorPotential_z(x1_L, x2_L, x3_L + dx[2] / 2.0, time)) /
					 dx[1] -
				     (computeMagneticVectorPotential_y(x1_L, x2_L + dx[1] / 2.0, x3_L + dx[2], time) -
				      computeMagneticVectorPotential_y(x1_L, x2_L + dx[1] / 2.0, x3_L, time)) /
					 dx[2];
		const double x2mag = (computeMagneticVectorPotential_x(x1_L + dx[0] / 2.0, x2_L, x3_L + dx[2], time) -
				      computeMagneticVectorPotential_x(x1_L + dx[0] / 2.0, x2_L, x3_L, time)) /
					 dx[2] -
				     (computeMagneticVectorPotential_z(x1_L + dx[0], x2_L, x3_L + dx[2] / 2.0, time) -
				      computeMagneticVectorPotential_z(x1_L, x2_L, x3_L + dx[2] / 2.0, time)) /
					 dx[0];
		const double x3mag = (computeMagneticVectorPotential_y(x1_L + dx[0], x2_L + dx[1] / 2.0, x3_L, time) -
				      computeMagneticVectorPotential_y(x1_L, x2_L + dx[1] / 2.0, x3_L, time)) /
					 dx[0] -
				     (computeMagneticVectorPotential_x(x1_L + dx[0] / 2.0, x2_L + dx[1], x3_L, time) -
				      computeMagneticVectorPotential_x(x1_L + dx[0] / 2.0, x2_L, x3_L, time)) /
					 dx[1];
		if (dir == quokka::direction::x) {
			state(i, j, k, MHDSystem<AlfvenWaveCircular>::bfield_index) = x1mag;
		} else if (dir == quokka::direction::y) {
			state(i, j, k, MHDSystem<AlfvenWaveCircular>::bfield_index) = x2mag;
		} else if (dir == quokka::direction::z) {
			state(i, j, k, MHDSystem<AlfvenWaveCircular>::bfield_index) = x3mag;
		}
	}
}

template <> void QuokkaSimulation<AlfvenWaveCircular>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	// extract grid information
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = grid_elem.dx_;
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = grid_elem.prob_lo_;
	const amrex::Array4<double> &state_cc = grid_elem.array_;
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const quokka::centering cen = grid_elem.cen_;
	const quokka::direction dir = grid_elem.dir_;

	const int ncomp_cc = Physics_Indices<AlfvenWaveCircular>::nvarTotal_cc;
	// loop over the grid and set the initial condition
	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		for (int n = 0; n < ncomp_cc; ++n) {
			state_cc(i, j, k, n) = 0; // fill unused quantities with zeros
		}
		computeWaveSolution(i, j, k, state_cc, dx, prob_lo, cen, dir, 0);
	});
}

template <> void QuokkaSimulation<AlfvenWaveCircular>::setInitialConditionsOnGridFaceVars(quokka::grid const &grid_elem)
{
	// extract grid information
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = grid_elem.dx_;
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = grid_elem.prob_lo_;
	const amrex::Array4<double> &state_fc = grid_elem.array_;
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const quokka::centering cen = grid_elem.cen_;
	const quokka::direction dir = grid_elem.dir_;

	const int ncomp_fc = Physics_Indices<AlfvenWaveCircular>::nvarPerDim_fc;
	// loop over the grid and set the initial condition
	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		for (int n = 0; n < ncomp_fc; ++n) {
			state_fc(i, j, k, n) = 0; // fill unused quantities with zeros
		}
		computeWaveSolution(i, j, k, state_fc, dx, prob_lo, cen, dir, 0);
	});
}

template <>
void QuokkaSimulation<AlfvenWaveCircular>::computeReferenceSolution(amrex::MultiFab &ref, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx,
								    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_lo)
{
	const amrex::Real time = tNew_[0];
	for (amrex::MFIter iter(ref); iter.isValid(); ++iter) {
		const amrex::Box &indexRange = iter.validbox();
		auto const &stateExact = ref.array(iter);
		auto const ncomp = ref.nComp();

		amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
			for (int n = 0; n < ncomp; ++n) {
				stateExact(i, j, k, n) = 0.0; // fill unused quantities with zeros
			}
			computeWaveSolution(i, j, k, stateExact, dx, prob_lo, quokka::centering::cc, quokka::direction::na, time);
		});
	}
}

template <>
void QuokkaSimulation<AlfvenWaveCircular>::computeReferenceSolution_fc(amrex::MultiFab &ref, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx,
								       amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_lo, quokka::direction const dir)
{
	const amrex::Real time = tNew_[0];
	for (amrex::MFIter iter(ref); iter.isValid(); ++iter) {
		const amrex::Box &indexRange = iter.validbox();
		auto const &stateExact = ref.array(iter);
		auto const ncomp = ref.nComp();

		amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
			for (int n = 0; n < ncomp; ++n) {
				stateExact(i, j, k, n) = 0.0; // fill unused quantities with zeros
			}
			computeWaveSolution(i, j, k, stateExact, dx, prob_lo, quokka::centering::fc, dir, time);
		});
	}
}

auto runWaveTest(int nx, int ny, int nz) -> double
{
	const double wavelength = 1.0 / num_modes;
	const double max_time = wavelength / alfven_speed;
	const int max_timesteps = std::max(20000, nx * 100);

	// Set grid dimensions using AMReX parameter system
	amrex::ParmParse pp("amr");
	amrex::Vector<int> const ncells = {nx, ny, nz};

	if (!pp.contains("blocking_factor")) {
		pp.add("blocking_factor", 8);
	}

	if (!pp.contains("max_grid_size")) {
		pp.add("max_grid_size", 128);
	}

	pp.add("max_level", 0);
	pp.addarr("n_cell", ncells);

	// Set domain bounds using AMReX parameter system
	amrex::ParmParse pp_geom("geometry");
	amrex::Vector<double> const prob_lo = {0.0, 0.0, 0.0};
	amrex::Vector<double> const prob_hi = {1.0, 1.0, 1.0};
	amrex::Vector<int> const is_periodic = {1, 1, 1};
	pp_geom.addarr("prob_lo", prob_lo);
	pp_geom.addarr("prob_hi", prob_hi);
	pp_geom.addarr("is_periodic", is_periodic);

	// Setup boundary conditions
	auto BCs_cc = quokka::BC<AlfvenWaveCircular>(quokka::BCType::int_dir);

	const int nvars_fc = Physics_Indices<AlfvenWaveCircular>::nvarTotal_fc;
	amrex::Vector<amrex::BCRec> BCs_fc(nvars_fc);
	for (int icomp = 0; icomp < nvars_fc; ++icomp) {
		for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
			BCs_fc[icomp].setLo(idim, amrex::BCType::int_dir);
			BCs_fc[icomp].setHi(idim, amrex::BCType::int_dir);
		}
	}

	// Run simulation
	QuokkaSimulation<AlfvenWaveCircular> sim(BCs_cc, BCs_fc);

	sim.stopTime_ = max_time;
	sim.maxTimesteps_ = max_timesteps;
	sim.setInitialConditions();

	// Main time loop
	sim.evolve();

	return sim.computeErrorNorm();
}

auto problem_main() -> int
{
	bool run_convergence = true;
	bool run_sim = false;
	double error_tol = 0.003;
	{
		amrex::ParmParse const pp("setup");
		pp.query("run_convergence", run_convergence);
		pp.query("run_sim", run_sim);
		pp.query("error_tol", error_tol);
	}

	// AlfvenWaveCircularConvergence does not model resistivity; abort early rather than silently
	// producing a wrong reference solution if mhd.resistivity is set (applies to both modes).
	{
		double eta = 0.0;
		amrex::ParmParse const mhd_pp("mhd");
		mhd_pp.query("resistivity", eta);
		if (eta != 0.0) {
			amrex::Abort("AlfvenWaveCircularConvergence does not support mhd.resistivity != 0; use AlfvenWaveLinearConvergence "
				     "for resistivity validation.");
		}
	}

	int status = 0;

	if (run_sim) {
		auto BCs_cc = quokka::BC<AlfvenWaveCircular>(quokka::BCType::int_dir);
		const int nvars_fc = Physics_Indices<AlfvenWaveCircular>::nvarTotal_fc;
		amrex::Vector<amrex::BCRec> BCs_fc(nvars_fc);
		for (int icomp = 0; icomp < nvars_fc; ++icomp) {
			for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
				BCs_fc[icomp].setLo(idim, amrex::BCType::int_dir);
				BCs_fc[icomp].setHi(idim, amrex::BCType::int_dir);
			}
		}

		QuokkaSimulation<AlfvenWaveCircular> sim(BCs_cc, BCs_fc);
		sim.setInitialConditions();
		sim.evolve();

		const double error_norm = sim.computeErrorNorm();
		amrex::Print() << std::format("\nrun_sim error norm = {:.6e}  (tol = {:.6e})\n", error_norm, error_tol);
		if (error_norm > error_tol) {
			status = 1;
		}
	}

	if (run_convergence) {
		quokka::richardson::applyQuietDefaults();

		quokka::richardson::Parameters params{};
		params.machine_precision_target = 2.0e-10;
		params.nx_initial = 16;
		params.nx_max = 128;
		{
			amrex::ParmParse const pp("setup");
			pp.query("nx_start", params.nx_initial);
			pp.query("nx_max", params.nx_max);
			pp.query("machine_precision_target", params.machine_precision_target);
		}
		params.expected_rate = 2.0;
		params.tolerance = 0.3;
		params.test_name = "Alfven Wave Circular";
		params.csv_filename = "alfven_wave_circular_convergence.csv";

		if (quokka::richardson::run(params, [](int nx, int ny, int nz) { return runWaveTest(nx, ny, nz); }) != 0) {
			status = 1;
		}
	}

	return status;
}
