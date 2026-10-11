//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2026 Neco Kriel.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file testIsothermalMHDShockTube.cpp
/// \brief Isothermal-EOS analogue of the Brio & Wu MHD shock tube; regresses the divide-by-zero
/// in the HLLD and LLF_MHD solvers' total-energy formula when gamma == 1 (issue #2225).
///

#include "hydro/hydro_system.hpp"
#include <cmath>

#include "AMReX_BC_TYPES.H"

#include "QuokkaSimulation.hpp"
#include "physics_info.hpp"
#include "util/BC.hpp"

struct IsothermalMHDShockTube {};

// isothermal EOS: pressure = cs^2 * rho (no thermal energy equation)
template <> struct quokka::EOS_Traits<IsothermalMHDShockTube> {
	static constexpr double gamma = 1.0;
	static constexpr double cs_isothermal = 1.0; // dimensionless sound speed
	static constexpr double mean_molecular_weight = C::m_u;
};

template <> struct Physics_Traits<IsothermalMHDShockTube> : DefaultPhysicsTraits {
	static constexpr UnitSystem unit_system = UnitSystem::CONSTANTS;
	static constexpr bool is_hydro_enabled = true;
	static constexpr bool is_mhd_enabled = true;
};

// isothermal EOS has no internal energy to reconstruct; pressure computed directly from rho
template <> struct HydroSystem_Traits<IsothermalMHDShockTube> {
	static constexpr bool reconstruct_eint = false;
};

// left- and right- side shock states (density/field jump only; isothermal EOS carries no
// independent pressure state, so the discontinuity is driven by rho and B alone)

constexpr amrex::Real rho_L = 1.0;
constexpr amrex::Real rho_R = 0.125;

constexpr amrex::Real Bx = 0.75; // constant
constexpr amrex::Real By_L = 1.0;
constexpr amrex::Real By_R = -1.0;
constexpr amrex::Real Bz = 0.0; // constant

template <> void QuokkaSimulation<IsothermalMHDShockTube>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = grid_elem.dx_;
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = grid_elem.prob_lo_;
	const amrex::Array4<double> &state_cc = grid_elem.array_;
	const amrex::Box &indexRange = grid_elem.indexRange_;

	const int ncomp_cc = Physics_Indices<IsothermalMHDShockTube>::nvarTotal_cc;

	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		const double x = prob_lo[0] + ((i + 0.5) * dx[0]);
		const double rho = (x < 0.5) ? rho_L : rho_R;

		for (int n = 0; n < ncomp_cc; ++n) {
			state_cc(i, j, k, n) = 0.;
		}
		state_cc(i, j, k, HydroSystem<IsothermalMHDShockTube>::density_index) = rho;
		state_cc(i, j, k, HydroSystem<IsothermalMHDShockTube>::x1Momentum_index) = 0.;
		state_cc(i, j, k, HydroSystem<IsothermalMHDShockTube>::x2Momentum_index) = 0.;
		state_cc(i, j, k, HydroSystem<IsothermalMHDShockTube>::x3Momentum_index) = 0.;
		state_cc(i, j, k, HydroSystem<IsothermalMHDShockTube>::energy_index) = 0.;
		state_cc(i, j, k, HydroSystem<IsothermalMHDShockTube>::internalEnergy_index) = 0.;
	});
}

template <> void QuokkaSimulation<IsothermalMHDShockTube>::setInitialConditionsOnGridFaceVars(quokka::grid const &grid_elem)
{
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = grid_elem.dx_;
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = grid_elem.prob_lo_;
	const amrex::Array4<double> &state_fc = grid_elem.array_;
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const quokka::direction dir = grid_elem.dir_;

	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		const amrex::Real x1_L = prob_lo[0] + i * dx[0];
		const double x2mag = (x1_L < 0.5) ? By_L : By_R;

		if (dir == quokka::direction::x) {
			state_fc(i, j, k, Physics_Indices<IsothermalMHDShockTube>::mhdFirstIndex) = Bx;
		} else if (dir == quokka::direction::y) {
			state_fc(i, j, k, Physics_Indices<IsothermalMHDShockTube>::mhdFirstIndex) = x2mag;
		} else if (dir == quokka::direction::z) {
			state_fc(i, j, k, Physics_Indices<IsothermalMHDShockTube>::mhdFirstIndex) = Bz;
		}
	});
}

template <>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE void
AMRSimulation<IsothermalMHDShockTube>::setCustomBoundaryConditions(const amrex::IntVect &iv, amrex::Array4<amrex::Real> const &consVar, int /*dcomp*/,
								   int /*numcomp*/, amrex::GeometryData const &geom, const amrex::Real /*time*/,
								   const amrex::BCRec * /*bcr*/, int /*bcomp*/, int /*orig_comp*/)
{
	constexpr int nvar = Physics_Indices<IsothermalMHDShockTube>::nvarTotal_cc;

	amrex::GpuArray<amrex::Real, nvar> low_bdr_cells{};
	low_bdr_cells[HydroSystem<IsothermalMHDShockTube>::density_index] = rho_L;

	amrex::GpuArray<amrex::Real, nvar> high_bdr_cells{};
	high_bdr_cells[HydroSystem<IsothermalMHDShockTube>::density_index] = rho_R;

	setConstantDirichletBCLo<0>(iv, consVar, geom, low_bdr_cells);
	setConstantDirichletBCHi<0>(iv, consVar, geom, high_bdr_cells);
}

template <>
template <quokka::direction dir>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE void
AMRSimulation<IsothermalMHDShockTube>::setCustomBoundaryConditionsFaceVar(const amrex::IntVect &iv, amrex::Array4<amrex::Real> const &consVar_fc, int /*dcomp*/,
									  int /*numcomp*/, amrex::GeometryData const &geom, const amrex::Real /*time*/,
									  const amrex::BCRec * /*bcr*/, int /*bcomp*/, int /*orig_comp*/)
{
	const amrex::GpuArray<amrex::Real, 3> low_bdr_values = {Bx, By_L, Bz};
	const amrex::GpuArray<amrex::Real, 3> high_bdr_values = {Bx, By_R, Bz};

	setConstantDirichletBCFaceVarLo<0, dir, 3>(iv, consVar_fc, geom, low_bdr_values);
	setConstantDirichletBCFaceVarHi<0, dir, 3>(iv, consVar_fc, geom, high_bdr_values);
}

auto problem_main() -> int
{
	QuokkaSimulation<IsothermalMHDShockTube> sim;

	sim.setInitialConditions();
	sim.evolve();

	return 0;
}
