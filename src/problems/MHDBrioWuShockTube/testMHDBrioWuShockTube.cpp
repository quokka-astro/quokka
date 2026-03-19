//==============================================================================
// Copyright 2025 Elizabeth Cole-Kodikara.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file testMHDBrioWuShockTube.cpp
/// \brief Setup a test problem for the Brio-Wu MHD shock tube.
///

#ifdef HAVE_PYTHON
#include "util/matplotlibcpp.h"
#endif
#include "hydro/hydro_system.hpp"
#include "math/interpolate.hpp"
#include <cmath>
#include <format>
#include <fstream>
#include <string>
#include <unordered_map>

#include "AMReX_BC_TYPES.H"

#include "QuokkaSimulation.hpp"
#include "physics_info.hpp"
#include "radiation/radiation_system.hpp"
#include "util/ArrayUtil.hpp"
#include "util/BC.hpp"
#include "util/fextract.hpp"

struct MHDBrioWuShockTube {
};

template <> struct quokka::EOS_Traits<MHDBrioWuShockTube> {
	static constexpr double gamma = 2.0;
	static constexpr double mean_molecular_weight = C::m_u;
};

template <> struct Physics_Traits<MHDBrioWuShockTube> {
	static constexpr bool is_self_gravity_enabled = false;
	// cell-centred
	static constexpr bool is_hydro_enabled = true;
	static constexpr int numMassScalars = 0;		     // number of mass scalars
	static constexpr int numPassiveScalars = numMassScalars + 0; // number of passive scalars
	static constexpr bool is_radiation_enabled = false;
	static constexpr bool is_dust_enabled = false;
	static constexpr int nDustGroups = 1; // number of dust groups
	// face-centred
	static constexpr bool is_mhd_enabled = true;
	static constexpr int nGroups = 1; // number of radiation groups
	static constexpr UnitSystem unit_system = UnitSystem::CGS;
};

// left- and right- side shock states

constexpr amrex::Real rho_L = 1.0;
constexpr amrex::Real P_L = 1.0;
constexpr amrex::Real rho_R = 0.125;
constexpr amrex::Real P_R = 0.1;

constexpr amrex::Real Bx = 0.75; // constant
constexpr amrex::Real By_L = 1.0;
constexpr amrex::Real By_R = -1.0;
constexpr amrex::Real Bz = 0.0; // constant

template <> void QuokkaSimulation<MHDBrioWuShockTube>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = grid_elem.dx_;
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = grid_elem.prob_lo_;
	const amrex::Array4<double> &state_cc = grid_elem.array_;
	const amrex::Box &indexRange = grid_elem.indexRange_;

	const int ncomp_cc = Physics_Indices<MHDBrioWuShockTube>::nvarTotal_cc;

	// magnetic field at center of cell
	const double x1mag = 0.75; // constant
	const double x3mag = 0.0;

	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		const double x = prob_lo[0] + ((i + 0.5) * dx[0]);
		const auto gamma = quokka::EOS_Traits<MHDBrioWuShockTube>::gamma;
		double rho = NAN;
		double P = NAN;
		double x2mag = NAN;

		if (x < 0.5) {
			rho = rho_L;
			P = P_L;
			x2mag = By_L;
		} else {
			rho = rho_R;
			P = P_R;
			x2mag = By_R;
		}

		const double vx = 0.0;
		const double vy = 0.0;
		const double vz = 0.0;
		const double Emag = 0.5 * (x1mag * x1mag + x2mag * x2mag + x3mag * x3mag);
		AMREX_ASSERT(!std::isnan(vx));
		AMREX_ASSERT(!std::isnan(rho));
		AMREX_ASSERT(!std::isnan(P));
		for (int n = 0; n < ncomp_cc; ++n) {
			state_cc(i, j, k, n) = 0.;
		}
		state_cc(i, j, k, HydroSystem<MHDBrioWuShockTube>::density_index) = rho;
		state_cc(i, j, k, HydroSystem<MHDBrioWuShockTube>::x1Momentum_index) = vx * rho;
		state_cc(i, j, k, HydroSystem<MHDBrioWuShockTube>::x2Momentum_index) = vy * rho;
		state_cc(i, j, k, HydroSystem<MHDBrioWuShockTube>::x3Momentum_index) = vz * rho;
		state_cc(i, j, k, HydroSystem<MHDBrioWuShockTube>::energy_index) = P / (gamma - 1.) + 0.5 * rho * (vx * vx) + Emag;
		state_cc(i, j, k, HydroSystem<MHDBrioWuShockTube>::internalEnergy_index) = P / (gamma - 1.);
	});
}

template <> void QuokkaSimulation<MHDBrioWuShockTube>::setInitialConditionsOnGridFaceVars(quokka::grid const &grid_elem)
{
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = grid_elem.dx_;
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = grid_elem.prob_lo_;
	const amrex::Array4<double> &state_fc = grid_elem.array_;
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const quokka::direction dir = grid_elem.dir_;

	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		const amrex::Real x1_L = prob_lo[0] + i * dx[0];

		const double x1mag = 0.75; // constant
		const double x3mag = 0.0;
		double x2mag = NAN;
		if (x1_L < 0.5) {
			x2mag = By_L;
		} else {
			x2mag = By_R;
		}

		if (dir == quokka::direction::x) {
			state_fc(i, j, k, Physics_Indices<MHDBrioWuShockTube>::mhdFirstIndex) = x1mag;
		} else if (dir == quokka::direction::y) {
			state_fc(i, j, k, Physics_Indices<MHDBrioWuShockTube>::mhdFirstIndex) = x2mag;
		} else if (dir == quokka::direction::z) {
			state_fc(i, j, k, Physics_Indices<MHDBrioWuShockTube>::mhdFirstIndex) = x3mag;
		}
	});
}

template <>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE void
AMRSimulation<MHDBrioWuShockTube>::setCustomBoundaryConditions(const amrex::IntVect &iv, amrex::Array4<amrex::Real> const &consVar, int /*dcomp*/,
								int /*numcomp*/, amrex::GeometryData const &geom, const amrex::Real /*time*/,
								const amrex::BCRec * /*bcr*/, int /*bcomp*/, int /*orig_comp*/)
{
	// Number of variables (use Physics_Indices which correctly accounts for enabled physics)
	constexpr int nvar = Physics_Indices<MHDBrioWuShockTube>::nvarTotal_cc;
	const auto gamma = quokka::EOS_Traits<MHDBrioWuShockTube>::gamma;

	const double Emag_L = 0.5 * (Bx * Bx + By_L * By_L + Bz * Bz);
	const double Emag_R = 0.5 * (Bx * Bx + By_R * By_R + Bz * Bz);

	// Prepare left boundary values (left state)
	amrex::GpuArray<amrex::Real, nvar> low_bdr_cells{};
	// Initialize all to 0 first

	// Set specific values
	low_bdr_cells[RadSystem<MHDBrioWuShockTube>::gasEnergy_index] = P_L / (gamma - 1.) + Emag_L;
	low_bdr_cells[RadSystem<MHDBrioWuShockTube>::gasInternalEnergy_index] = P_L / (gamma - 1.);
	low_bdr_cells[RadSystem<MHDBrioWuShockTube>::gasDensity_index] = rho_L;
	low_bdr_cells[RadSystem<MHDBrioWuShockTube>::x1GasMomentum_index] = 0.;
	low_bdr_cells[RadSystem<MHDBrioWuShockTube>::x2GasMomentum_index] = 0.;
	low_bdr_cells[RadSystem<MHDBrioWuShockTube>::x3GasMomentum_index] = 0.;

	// Prepare right boundary values (right state)
	amrex::GpuArray<amrex::Real, nvar> high_bdr_cells{};
	// Initialize all to 0 first
	for (int n = 0; n < nvar; ++n) {
		high_bdr_cells[n] = 0;
	}
	// Set specific values
	high_bdr_cells[RadSystem<MHDBrioWuShockTube>::gasEnergy_index] = P_R / (gamma - 1.) + Emag_R;
	high_bdr_cells[RadSystem<MHDBrioWuShockTube>::gasInternalEnergy_index] = P_R / (gamma - 1.);
	high_bdr_cells[RadSystem<MHDBrioWuShockTube>::gasDensity_index] = rho_R;
	high_bdr_cells[RadSystem<MHDBrioWuShockTube>::x1GasMomentum_index] = 0.;
	high_bdr_cells[RadSystem<MHDBrioWuShockTube>::x2GasMomentum_index] = 0.;
	high_bdr_cells[RadSystem<MHDBrioWuShockTube>::x3GasMomentum_index] = 0.;

	// Apply boundary conditions using helper functions (direction 0 = x-axis)
	setConstantDirichletBCLo<0>(iv, consVar, geom, low_bdr_cells);
	setConstantDirichletBCHi<0>(iv, consVar, geom, high_bdr_cells);
}

template <>
template <quokka::direction dir>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE void
AMRSimulation<MHDBrioWuShockTube>::setCustomBoundaryConditionsFaceVar(const amrex::IntVect &iv, amrex::Array4<amrex::Real> const &consVar_fc, int /*dcomp*/,
								       int /*numcomp*/, amrex::GeometryData const &geom, const amrex::Real /*time*/,
								       const amrex::BCRec * /*bcr*/, int /*bcomp*/, int /*orig_comp*/)
{
	// Prepare boundary values for each face direction: {x-face, y-face, z-face}
	// For low boundary (left side)
	const amrex::GpuArray<amrex::Real, 3> low_bdr_values = {Bx, By_L, Bz};

	// For high boundary (right side)
	const amrex::GpuArray<amrex::Real, 3> high_bdr_values = {Bx, By_R, Bz};

	// Apply boundary conditions using helper functions (boundary_dim 0 = x-axis)
	// The helper functions will internally select the appropriate value based on face_dir
	setConstantDirichletBCFaceVarLo<0, dir, 3>(iv, consVar_fc, geom, low_bdr_values);
	setConstantDirichletBCFaceVarHi<0, dir, 3>(iv, consVar_fc, geom, high_bdr_values);
}

template <> void QuokkaSimulation<MHDBrioWuShockTube>::refineGrid(int lev, amrex::TagBoxArray &tags, Real /*time*/, int /*ngrow*/)
{
	// tag cells for refinement

	const Real eta_threshold = 0.1; // gradient refinement threshold
	const Real rho_min = 0.01;	// minimum rho for refinement
	auto const &dx = geom[lev].CellSizeArray();

	for (amrex::MFIter mfi(state_new_cc_[lev]); mfi.isValid(); ++mfi) {
		const amrex::Box &box = mfi.validbox();
		const auto state = state_new_cc_[lev].const_array(mfi);
		const auto tag = tags.array(mfi);

		amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
			int const n = 0;
			Real const rho = state(i, j, k, n);
			Real const del_x = (state(i + 1, j, k, n) - state(i - 1, j, k, n)) / (2.0 * dx[0]);
			Real const gradient_indicator = std::sqrt(del_x * del_x) / rho;

			if (gradient_indicator > eta_threshold && rho >= rho_min) {
				tag(i, j, k) = amrex::TagBox::SET;
			}
		});
	}
}

auto problem_main() -> int
{
	QuokkaSimulation<MHDBrioWuShockTube> sim;

	// Main time loop
	sim.setInitialConditions();
	sim.evolve();

	return 0;
}
