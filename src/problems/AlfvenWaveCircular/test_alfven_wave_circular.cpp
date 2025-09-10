//==============================================================================
// Copyright 2022 Neco Kriel.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file test_fc_quantities.cpp
/// \brief Defines a test problem to make sure face-centred quantities are created correctly.
///

#include <cassert>
#include <cmath>
#include <gcem.hpp>

#include "AMReX_Array.H"
#include "AMReX_Array4.H"
#include "AMReX_REAL.H"

#include "QuokkaSimulation.hpp"
#include "util/BC.hpp"
#include "grid.hpp"
#include "physics_info.hpp"

struct AlfvenWaveCircular {
};

template <> struct quokka::EOS_Traits<AlfvenWaveCircular> {
	static constexpr double gamma = 5. / 3.;
	static constexpr double mean_molecular_weight = C::m_u;
	static constexpr double boltzmann_constant = C::k_B;
};

template <> struct Physics_Traits<AlfvenWaveCircular> {
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
void QuokkaSimulation<AlfvenWaveCircular>::computeReferenceSolution_fc(amrex::MultiFab &ref, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx,
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
	auto BCs_cc = quokka::BC<AlfvenWaveCircular>(amrex::BCType::int_dir); // periodic

	const int nvars_fc = Physics_Indices<AlfvenWaveCircular>::nvarTotal_fc;
	amrex::Vector<amrex::BCRec> BCs_fc(nvars_fc);
	for (int icomp = 0; icomp < nvars_fc; ++icomp) {
		for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
			BCs_fc[icomp].setLo(idim, amrex::BCType::int_dir); // periodic
			BCs_fc[icomp].setHi(idim, amrex::BCType::int_dir);
		}
	}

	QuokkaSimulation<AlfvenWaveCircular> sim(BCs_cc, BCs_fc);
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
