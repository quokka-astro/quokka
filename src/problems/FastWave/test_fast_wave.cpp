//==============================================================================
// Copyright 2025 Neco Elizabeth Cole-Kodikara.
// Credit to Nico Kriel for creating the MHD module and Alfven wave test
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file test_fast_wave.cpp
/// \brief Defines a test problem for magnetosonic waves of the fast type.
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
#include "test_fast_wave.hpp"
#include "util/BC.hpp"
#include "util/fextract.hpp"

struct FastWave {
};

template <> struct quokka::EOS_Traits<FastWave> {
	static constexpr double gamma = 5. / 3.;
	static constexpr double mean_molecular_weight = C::m_u;
	static constexpr double boltzmann_constant = C::k_B;
};

template <> struct Physics_Traits<FastWave> {
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
constexpr double gamma_gas = quokka::EOS_Traits<FastWave>::gamma;

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

		state(i, j, k, HydroSystem<FastWave>::density_index) = density;
		state(i, j, k, HydroSystem<FastWave>::x1Momentum_index) = x1vel * density;
		state(i, j, k, HydroSystem<FastWave>::x2Momentum_index) = x2vel * density;
		state(i, j, k, HydroSystem<FastWave>::x3Momentum_index) = x3vel * density;
		state(i, j, k, HydroSystem<FastWave>::energy_index) = Etot;
		state(i, j, k, HydroSystem<FastWave>::internalEnergy_index) = Eint;
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
			state(i, j, k, MHDSystem<FastWave>::bfield_index) = x1mag;
		} else if (dir == quokka::direction::y) {
			state(i, j, k, MHDSystem<FastWave>::bfield_index) = x2mag;
		} else if (dir == quokka::direction::z) {
			state(i, j, k, MHDSystem<FastWave>::bfield_index) = x3mag;
		}
	}
}

template <> void QuokkaSimulation<FastWave>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	// extract grid information
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = grid_elem.dx_;
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = grid_elem.prob_lo_;
	const amrex::Array4<double> &state_cc = grid_elem.array_;
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const quokka::centering cen = grid_elem.cen_;
	const quokka::direction dir = grid_elem.dir_;

	const int ncomp_cc = Physics_Indices<FastWave>::nvarTotal_cc;
	// loop over the grid and set the initial condition
	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		for (int n = 0; n < ncomp_cc; ++n) {
			state_cc(i, j, k, n) = 0; // fill unused quantities with zeros
		}
		computeWaveSolution(i, j, k, state_cc, dx, prob_lo, cen, dir, 0);
	});
}

template <> void QuokkaSimulation<FastWave>::setInitialConditionsOnGridFaceVars(quokka::grid const &grid_elem)
{
	// extract grid information
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = grid_elem.dx_;
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = grid_elem.prob_lo_;
	const amrex::Array4<double> &state_fc = grid_elem.array_;
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const quokka::centering cen = grid_elem.cen_;
	const quokka::direction dir = grid_elem.dir_;

	const int ncomp_fc = Physics_Indices<FastWave>::nvarPerDim_fc;
	// loop over the grid and set the initial condition
	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		for (int n = 0; n < ncomp_fc; ++n) {
			state_fc(i, j, k, n) = 0; // fill unused quantities with zeros
		}
		computeWaveSolution(i, j, k, state_fc, dx, prob_lo, cen, dir, 0);
	});
}

template <>
void QuokkaSimulation<FastWave>::computeReferenceSolution(amrex::MultiFab &ref, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx,
							  amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_lo)
{
	for (amrex::MFIter iter(ref); iter.isValid(); ++iter) {
		const amrex::Box &indexRange = iter.validbox();
		auto const &stateExact = ref.array(iter);
		auto const ncomp = ref.nComp();

		amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
			for (int n = 0; n < ncomp; ++n) {
				stateExact(i, j, k, n) = 0.0; // fill unused quantities with zeros
			}
			computeWaveSolution(i, j, k, stateExact, dx, prob_lo, quokka::centering::cc, quokka::direction::na, 0);
		});
	}
}

template <>
void QuokkaSimulation<FastWave>::computeReferenceSolution_fc(amrex::MultiFab &ref, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx,
							     amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_lo, quokka::direction const dir)
{
	for (amrex::MFIter iter(ref); iter.isValid(); ++iter) {
		const amrex::Box &indexRange = iter.validbox();
		auto const &stateExact = ref.array(iter);
		auto const ncomp = ref.nComp();

		amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
			for (int n = 0; n < ncomp; ++n) {
				stateExact(i, j, k, n) = 0.0; // fill unused quantities with zeros
			}
			computeWaveSolution(i, j, k, stateExact, dx, prob_lo, quokka::centering::fc, dir, 0);
		});
	}
}

auto problem_main() -> int
{
	auto BCs_cc = quokka::BC<FastWave>(quokka::BCType::int_dir); // periodic

	const int nvars_fc = Physics_Indices<FastWave>::nvarTotal_fc;
	amrex::Vector<amrex::BCRec> BCs_fc(nvars_fc);
	for (int icomp = 0; icomp < nvars_fc; ++icomp) {
		for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
			BCs_fc[icomp].setLo(idim, quokka::BCType::int_dir); // periodic
			BCs_fc[icomp].setHi(idim, quokka::BCType::int_dir);
		}
	}

	QuokkaSimulation<FastWave> sim(BCs_cc, BCs_fc);
	sim.computeReferenceSolution_ = true;
	sim.setInitialConditions();
	sim.evolve();

	// Compute test success condition
	int status = 0;
	const double error_tol = 0.002;
	if (sim.errorNorm_ > error_tol) {
		status = 1;
	}

	return status;
}