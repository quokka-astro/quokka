//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file testMHDRyuJones2aShockTube.cpp
/// \brief Defines the RJ2a MHD Riemann problem (Figure 2a of Ryu & Jones 1995, ApJ 442, 228).
///

#ifdef HAVE_PYTHON
#include "util/matplotlibcpp.h"
#endif
#include "hydro/hydro_system.hpp"
#include "math/interpolate.hpp"
#include <cmath>
#include <format>
#include <fstream>
#include <numbers>
#include <string>
#include <unordered_map>

#include "AMReX_BC_TYPES.H"

#include "QuokkaSimulation.hpp"
#include "physics_info.hpp"
#include "radiation/radiation_system.hpp"
#include "util/ArrayUtil.hpp"
#include "util/BC.hpp"
#include "util/fextract.hpp"

struct RyuJones2aShockTubeProblem {
};

template <> struct quokka::EOS_Traits<RyuJones2aShockTubeProblem> {
	static constexpr double gamma = 5. / 3.;
	static constexpr double mean_molecular_weight = C::m_u;
};

template <> struct Physics_Traits<RyuJones2aShockTubeProblem> : DefaultPhysicsTraits {
	static constexpr UnitSystem unit_system = UnitSystem::CONSTANTS;
	static constexpr bool is_hydro_enabled = true;
	static constexpr bool is_mhd_enabled = true;
};

// left- and right- side shock states (Ryu & Jones 1995, Figure 2a)

constexpr amrex::Real rho_L = 1.08;
constexpr amrex::Real P_L = 0.95;
constexpr amrex::Real vx_L = 1.2;
constexpr amrex::Real vy_L = 0.01;
constexpr amrex::Real vz_L = 0.5;

constexpr amrex::Real rho_R = 1.0;
constexpr amrex::Real P_R = 1.0;
constexpr amrex::Real vx_R = 0.0;
constexpr amrex::Real vy_R = 0.0;
constexpr amrex::Real vz_R = 0.0;

constexpr amrex::Real Bx = std::numbers::inv_sqrtpi; // 2 / sqrt(4 pi); constant
constexpr amrex::Real By_L = 1.0155412503859613;     // 3.6 / sqrt(4 pi)
constexpr amrex::Real By_R = 1.1283791670955125;     // 4 / sqrt(4 pi)
constexpr amrex::Real Bz = std::numbers::inv_sqrtpi; // 2 / sqrt(4 pi); constant

template <> void QuokkaSimulation<RyuJones2aShockTubeProblem>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = grid_elem.dx_;
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = grid_elem.prob_lo_;
	const amrex::Array4<double> &state_cc = grid_elem.array_;
	const amrex::Box &indexRange = grid_elem.indexRange_;

	const int ncomp_cc = Physics_Indices<RyuJones2aShockTubeProblem>::nvarTotal_cc;

	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		const double x = prob_lo[0] + ((i + 0.5) * dx[0]);
		const auto gamma = quokka::EOS_Traits<RyuJones2aShockTubeProblem>::gamma;
		double rho = NAN;
		double P = NAN;
		double vx = NAN;
		double vy = NAN;
		double vz = NAN;
		double x2mag = NAN;

		if (x < 0.5) {
			rho = rho_L;
			P = P_L;
			vx = vx_L;
			vy = vy_L;
			vz = vz_L;
			x2mag = By_L;
		} else {
			rho = rho_R;
			P = P_R;
			vx = vx_R;
			vy = vy_R;
			vz = vz_R;
			x2mag = By_R;
		}

		const double x1mag = Bx;
		const double x3mag = Bz;
		const double Emag = 0.5 * (x1mag * x1mag + x2mag * x2mag + x3mag * x3mag);
		const double Ekin = 0.5 * rho * (vx * vx + vy * vy + vz * vz);
		AMREX_ASSERT(!std::isnan(vx));
		AMREX_ASSERT(!std::isnan(rho));
		AMREX_ASSERT(!std::isnan(P));
		for (int n = 0; n < ncomp_cc; ++n) {
			state_cc(i, j, k, n) = 0.;
		}
		state_cc(i, j, k, HydroSystem<RyuJones2aShockTubeProblem>::density_index) = rho;
		state_cc(i, j, k, HydroSystem<RyuJones2aShockTubeProblem>::x1Momentum_index) = vx * rho;
		state_cc(i, j, k, HydroSystem<RyuJones2aShockTubeProblem>::x2Momentum_index) = vy * rho;
		state_cc(i, j, k, HydroSystem<RyuJones2aShockTubeProblem>::x3Momentum_index) = vz * rho;
		state_cc(i, j, k, HydroSystem<RyuJones2aShockTubeProblem>::energy_index) = P / (gamma - 1.) + Ekin + Emag;
		state_cc(i, j, k, HydroSystem<RyuJones2aShockTubeProblem>::internalEnergy_index) = P / (gamma - 1.);
	});
}

template <> void QuokkaSimulation<RyuJones2aShockTubeProblem>::setInitialConditionsOnGridFaceVars(quokka::grid const &grid_elem)
{
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = grid_elem.dx_;
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = grid_elem.prob_lo_;
	const amrex::Array4<double> &state_fc = grid_elem.array_;
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const quokka::direction dir = grid_elem.dir_;

	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		const amrex::Real x1_L = prob_lo[0] + i * dx[0];

		const double x1mag = Bx;
		const double x3mag = Bz;
		double x2mag = NAN;
		if (x1_L < 0.5) {
			x2mag = By_L;
		} else {
			x2mag = By_R;
		}

		if (dir == quokka::direction::x) {
			state_fc(i, j, k, Physics_Indices<RyuJones2aShockTubeProblem>::mhdFirstIndex) = x1mag;
		} else if (dir == quokka::direction::y) {
			state_fc(i, j, k, Physics_Indices<RyuJones2aShockTubeProblem>::mhdFirstIndex) = x2mag;
		} else if (dir == quokka::direction::z) {
			state_fc(i, j, k, Physics_Indices<RyuJones2aShockTubeProblem>::mhdFirstIndex) = x3mag;
		}
	});
}

template <>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE void
AMRSimulation<RyuJones2aShockTubeProblem>::setCustomBoundaryConditions(const amrex::IntVect &iv, amrex::Array4<amrex::Real> const &consVar, int /*dcomp*/,
								       int /*numcomp*/, amrex::GeometryData const &geom, const amrex::Real /*time*/,
								       const amrex::BCRec * /*bcr*/, int /*bcomp*/, int /*orig_comp*/)
{
	// Number of variables (use Physics_Indices which correctly accounts for enabled physics)
	constexpr int nvar = Physics_Indices<RyuJones2aShockTubeProblem>::nvarTotal_cc;
	const auto gamma = quokka::EOS_Traits<RyuJones2aShockTubeProblem>::gamma;

	const double Emag_L = 0.5 * (Bx * Bx + By_L * By_L + Bz * Bz);
	const double Emag_R = 0.5 * (Bx * Bx + By_R * By_R + Bz * Bz);
	const double Ekin_L = 0.5 * rho_L * (vx_L * vx_L + vy_L * vy_L + vz_L * vz_L);
	const double Ekin_R = 0.5 * rho_R * (vx_R * vx_R + vy_R * vy_R + vz_R * vz_R);

	// Prepare left boundary values (left state)
	amrex::GpuArray<amrex::Real, nvar> low_bdr_cells{};
	low_bdr_cells[RadSystem<RyuJones2aShockTubeProblem>::gasEnergy_index] = P_L / (gamma - 1.) + Ekin_L + Emag_L;
	low_bdr_cells[RadSystem<RyuJones2aShockTubeProblem>::gasInternalEnergy_index] = P_L / (gamma - 1.);
	low_bdr_cells[RadSystem<RyuJones2aShockTubeProblem>::gasDensity_index] = rho_L;
	low_bdr_cells[RadSystem<RyuJones2aShockTubeProblem>::x1GasMomentum_index] = vx_L * rho_L;
	low_bdr_cells[RadSystem<RyuJones2aShockTubeProblem>::x2GasMomentum_index] = vy_L * rho_L;
	low_bdr_cells[RadSystem<RyuJones2aShockTubeProblem>::x3GasMomentum_index] = vz_L * rho_L;

	// Prepare right boundary values (right state)
	amrex::GpuArray<amrex::Real, nvar> high_bdr_cells{};
	for (int n = 0; n < nvar; ++n) {
		high_bdr_cells[n] = 0;
	}
	high_bdr_cells[RadSystem<RyuJones2aShockTubeProblem>::gasEnergy_index] = P_R / (gamma - 1.) + Ekin_R + Emag_R;
	high_bdr_cells[RadSystem<RyuJones2aShockTubeProblem>::gasInternalEnergy_index] = P_R / (gamma - 1.);
	high_bdr_cells[RadSystem<RyuJones2aShockTubeProblem>::gasDensity_index] = rho_R;
	high_bdr_cells[RadSystem<RyuJones2aShockTubeProblem>::x1GasMomentum_index] = vx_R * rho_R;
	high_bdr_cells[RadSystem<RyuJones2aShockTubeProblem>::x2GasMomentum_index] = vy_R * rho_R;
	high_bdr_cells[RadSystem<RyuJones2aShockTubeProblem>::x3GasMomentum_index] = vz_R * rho_R;

	// Apply boundary conditions using helper functions (direction 0 = x-axis)
	setConstantDirichletBCLo<0>(iv, consVar, geom, low_bdr_cells);
	setConstantDirichletBCHi<0>(iv, consVar, geom, high_bdr_cells);
}

template <>
template <quokka::direction dir>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE void AMRSimulation<RyuJones2aShockTubeProblem>::setCustomBoundaryConditionsFaceVar(
    const amrex::IntVect &iv, amrex::Array4<amrex::Real> const &consVar_fc, int /*dcomp*/, int /*numcomp*/, amrex::GeometryData const &geom,
    const amrex::Real /*time*/, const amrex::BCRec * /*bcr*/, int /*bcomp*/, int /*orig_comp*/)
{
	// Prepare boundary values for each face direction: {x-face, y-face, z-face}
	const amrex::GpuArray<amrex::Real, 3> low_bdr_values = {Bx, By_L, Bz};
	const amrex::GpuArray<amrex::Real, 3> high_bdr_values = {Bx, By_R, Bz};

	setConstantDirichletBCFaceVarLo<0, dir, 3>(iv, consVar_fc, geom, low_bdr_values);
	setConstantDirichletBCFaceVarHi<0, dir, 3>(iv, consVar_fc, geom, high_bdr_values);
}

auto problem_main() -> int
{
	QuokkaSimulation<RyuJones2aShockTubeProblem> sim;

	// Main time loop
	sim.setInitialConditions();
	sim.evolve();

	return 0;
}
