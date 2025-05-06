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
#include <iostream>
#include <stdexcept>
#include <valarray>

#include "AMReX_Array.H"
#include "AMReX_Array4.H"
#include "AMReX_Print.H"
#include "AMReX_REAL.H"

#include "QuokkaSimulation.hpp"
#include "grid.hpp"
#include "physics_info.hpp"
#include "test_fast_wave.hpp"
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
const double cos_theta = std::cos(theta_degrees * M_PI / 180.0);
const double sin_theta = std::sin(theta_degrees * M_PI / 180.0);

// k = 2 pi / wave length
// box length = 1, so |k| in [1, inf)
constexpr double num_modes = 1;
constexpr double k_amplitude = 2 * M_PI * num_modes;

// input perturbation: choose to do this via the relative denisty field in [0, 1]. remember, the linear regime is valid when this perturbation is small
constexpr double delta_b = 1e-6;

const double alfven_speed = bg_mag_amplitude / std::sqrt(bg_density);
const double magnetosonic_speed = std::sqrt(alfven_speed*alfven_speed + sound_speed*sound_speed);
const double bg_mag_x1 = 0.0;
const double bg_mag_x2 = 0.0;
const double bg_mag_x3 = bg_mag_amplitude;

double compute_omega(double k_amplitude, double magnetosonic_speed, double alfven_speed, double sound_speed, double cos_theta) {
  return std::sqrt(std::pow(k_amplitude, 2) / 2.0 * (std::pow(magnetosonic_speed, 2) + std::sqrt(std::pow(magnetosonic_speed, 4) - 4.0 * std::pow(alfven_speed, 2) * std::pow(sound_speed, 2) * std::pow(cos_theta, 2))));
}

const double omega = compute_omega(k_amplitude, magnetosonic_speed, alfven_speed, sound_speed, cos_theta);

AMREX_GPU_DEVICE double computeMagneticVectorPotential_x(double x1, double x2, double x3, double time)
{
	// return -bg_mag_x3 * x2;
	return -x2 / 2.0 * (bg_mag_amplitude + delta_b * std::cos(omega * time - k_amplitude * x1));
}
AMREX_GPU_DEVICE double computeMagneticVectorPotential_y(double x1, double x2, double x3, double time)
{
	// return delta_b / k_amplitude * std::sin(omega * time - k_amplitude * x1);
	return bg_mag_amplitude * x1 / 2.0 + ((delta_b)*std::sin(omega * time - k_amplitude * x1) / (-2.0 * k_amplitude));
}
AMREX_GPU_DEVICE double computeMagneticVectorPotential_z(double x1, double x2, double x3, double time) { return 0.0; }

////////////////////////////////////
AMREX_GPU_DEVICE void computeWaveSolution(int i, int j, int k, amrex::Array4<amrex::Real> const &state, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx,
					  amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_lo, quokka::centering cen, quokka::direction dir,
					  double time)
{
	const amrex::Real x1_L = prob_lo[0] + i * dx[0];
	const amrex::Real x2_L = prob_lo[1] + j * dx[1];
	const amrex::Real x3_L = prob_lo[2] + k * dx[2];

	const amrex::Real x1_C = x1_L + static_cast<amrex::Real>(0.5) * dx[0];
	const amrex::Real x2_C = x2_L + static_cast<amrex::Real>(0.5) * dx[1];
	const amrex::Real x3_C = x3_L + static_cast<amrex::Real>(0.5) * dx[2];

	if (cen == quokka::centering::cc) {
		const double cos_wave_C = std::cos(omega * time - k_amplitude * x1_C);

		// magnetic field at the center of the cell
		const double x1mag = 0.0;
		const double x2mag = 0.0;
		const double x3mag = bg_mag_amplitude + delta_b * cos_wave_C;
		// std::cout << std::fixed;
		// std::cout << std::setprecision(54);
		// std::cout << "Bmag: " << x3mag << '\n';

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

		// std::cout << std::fixed;
		// std::cout << std::setprecision(54);
		// std::cout << "Bmag: " << x3mag << '\n';
		if (i == 0) {
			std::cout << i << ' ' << j << ' ' << k;
			std::cout << std::fixed;
			std::cout << std::setprecision(4);
			std::cout << ": x1_L, x2_L, x3_L: " << x1_L << '\t' << x2_L << '\t' << x3_L << '\n';
			std::cout << std::fixed;
			std::cout << std::setprecision(52);
			std::cout << "Bmag: " << x3mag << "\n\n";
		}

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
	const int nvars_cc = Physics_Indices<FastWave>::nvarTotal_cc;
	amrex::Vector<amrex::BCRec> BCs_cc(nvars_cc);
	for (int icomp = 0; icomp < nvars_cc; ++icomp) {
		for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
			BCs_cc[icomp].setLo(idim, amrex::BCType::int_dir); // periodic
			BCs_cc[icomp].setHi(idim, amrex::BCType::int_dir);
		}
	}

	const int nvars_fc = Physics_Indices<FastWave>::nvarTotal_fc;
	amrex::Vector<amrex::BCRec> BCs_fc(nvars_fc);
	for (int icomp = 0; icomp < nvars_fc; ++icomp) {
		for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
			BCs_fc[icomp].setLo(idim, amrex::BCType::int_dir); // periodic
			BCs_fc[icomp].setHi(idim, amrex::BCType::int_dir);
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