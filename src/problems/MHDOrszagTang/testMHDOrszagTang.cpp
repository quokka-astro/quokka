//==============================================================================
// Copyright 2025 Ben Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file testMHDMHDOrszagTang.cpp
/// \brief Setup a test problem for the Orszag-Tang MHD vortex.
///   This problem is based on the implementation here:
///   https://github.com/PrincetonUniversity/athena/blob/master/src/pgen/orszag_tang.cpp.
///	  (Phil Hopkins made several typos on this page, do not use:
///	  https://www.astro.princeton.edu/~jstone/Athena/tests/orszag-tang/pagesource.html)
///

#include <cmath>

#include "AMReX_Array.H"
#include "AMReX_Array4.H"
#include "AMReX_GpuQualifiers.H"
#include "AMReX_REAL.H"

#include "QuokkaSimulation.hpp"
#include "grid.hpp"
#include "hydro/EOS.hpp"
#include "hydro/hydro_system.hpp"
#include "physics_info.hpp"
#include "util/BC.hpp"

struct MHDOrszagTang {
};

template <> struct quokka::EOS_Traits<MHDOrszagTang> {
	static constexpr double gamma = 5. / 3.;
	static constexpr double mean_molecular_weight = C::m_u;
	static constexpr double boltzmann_constant = C::k_B;
};

template <> struct Physics_Traits<MHDOrszagTang> {
	static constexpr bool is_hydro_enabled = true;
	static constexpr int numMassScalars = 0;		     // number of mass scalars
	static constexpr int numPassiveScalars = numMassScalars + 0; // number of passive scalars
	static constexpr bool is_self_gravity_enabled = false;
	static constexpr bool is_radiation_enabled = false;
	static constexpr bool is_dust_enabled = false;
	static constexpr int nDustGroups = 1; // number of dust groups
	static constexpr bool is_mhd_enabled = true;
	static constexpr int nGroups = 1; // number of radiation groups
	static constexpr UnitSystem unit_system = UnitSystem::CGS;
};

constexpr double B0 = 1.0 / gcem::sqrt(4.0 * PI);

AMREX_GPU_DEVICE AMREX_FORCE_INLINE auto A_z(double x, double y) -> double
{
	return B0 / (4.0 * M_PI) * (std::cos(4.0 * M_PI * x) - 2.0 * std::cos(2.0 * M_PI * y));
};

AMREX_GPU_DEVICE AMREX_FORCE_INLINE auto B_x(double xL, double yL, const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> &dx) -> double
{
	return (A_z(xL, yL + dx[1]) - A_z(xL, yL)) / dx[1];
};

AMREX_GPU_DEVICE AMREX_FORCE_INLINE auto B_y(double xL, double yL, const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> &dx) -> double
{
	return -(A_z(xL + dx[0], yL) - A_z(xL, yL)) / dx[0];
};

template <> void QuokkaSimulation<MHDOrszagTang>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	// extract grid information
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = grid_elem.dx_;
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = grid_elem.prob_lo_;
	const amrex::Array4<double> &state_cc = grid_elem.array_;
	const amrex::Box &indexRange = grid_elem.indexRange_;

	constexpr double gamma_gas = quokka::EOS_Traits<MHDOrszagTang>::gamma;
	constexpr double rho0 = 25. / (36. * M_PI);
	constexpr double P0 = 5. / (12. * M_PI);

	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		const double x = prob_lo[0] + ((i + 0.5) * dx[0]);
		const double y = prob_lo[1] + ((j + 0.5) * dx[1]);

		const double vx = std::sin(2 * M_PI * y);
		const double vy = -std::sin(2 * M_PI * x);

		const double Bx = 0.5 * (B_x(x - 0.5 * dx[0], y - 0.5 * dx[1], dx) + B_x(x + 0.5 * dx[0], y - 0.5 * dx[1], dx));
		const double By = 0.5 * (B_y(x - 0.5 * dx[0], y - 0.5 * dx[1], dx) + B_y(x - 0.5 * dx[0], y + 0.5 * dx[1], dx));

		const double Ekin = 0.5 * rho0 * (vx * vx + vy * vy);
		const double Eint = P0 / (gamma_gas - 1.0);
		const double Emag = 0.5 * (Bx * Bx + By * By);

		state_cc(i, j, k, HydroSystem<MHDOrszagTang>::density_index) = rho0;
		state_cc(i, j, k, HydroSystem<MHDOrszagTang>::x1Momentum_index) = rho0 * vx;
		state_cc(i, j, k, HydroSystem<MHDOrszagTang>::x2Momentum_index) = rho0 * vy;
		state_cc(i, j, k, HydroSystem<MHDOrszagTang>::x3Momentum_index) = 0;
		state_cc(i, j, k, HydroSystem<MHDOrszagTang>::internalEnergy_index) = Eint;
		state_cc(i, j, k, HydroSystem<MHDOrszagTang>::energy_index) = Eint + Ekin + Emag;
	});
}

template <> void QuokkaSimulation<MHDOrszagTang>::setInitialConditionsOnGridFaceVars(quokka::grid const &grid_elem)
{
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = grid_elem.dx_;
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = grid_elem.prob_lo_;
	const amrex::Array4<double> &state_fc = grid_elem.array_;
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const quokka::direction dir = grid_elem.dir_;

	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		const double xL = prob_lo[0] + (i * dx[0]);
		const double yL = prob_lo[1] + (j * dx[1]);

		if (dir == quokka::direction::x) {
			state_fc(i, j, k, Physics_Indices<MHDOrszagTang>::mhdFirstIndex) = B_x(xL, yL, dx);
		} else if (dir == quokka::direction::y) {
			state_fc(i, j, k, Physics_Indices<MHDOrszagTang>::mhdFirstIndex) = B_y(xL, yL, dx);
		} else if (dir == quokka::direction::z) {
			state_fc(i, j, k, Physics_Indices<MHDOrszagTang>::mhdFirstIndex) = 0;
		}
	});
}

auto problem_main() -> int
{
	QuokkaSimulation<MHDOrszagTang> sim;
	sim.setInitialConditions();
	sim.evolve();
	return 0;
}
