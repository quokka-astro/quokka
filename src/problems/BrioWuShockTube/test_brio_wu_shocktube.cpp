//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file test_hydro_shocktube_CMA.cpp
/// \brief Defines a test problem for a shock tube with passive scalars using consistent multi-fluid advection (CMA).
/// Implementing shock tube proglem from Plewa and Muller 1999, A&A 342, 179
///

#ifdef HAVE_PYTHON
#include "util/matplotlibcpp.h"
#endif
#include "hydro/hydro_system.hpp"
#include "math/interpolate.hpp"
#include <cmath>
#include <fmt/format.h>
#include <fstream>
#include <string>
#include <unordered_map>

#include "AMReX_BC_TYPES.H"

#include "QuokkaSimulation.hpp"
#include "radiation/radiation_system.hpp"
#include "util/ArrayUtil.hpp"
#include "util/fextract.hpp"

struct MHDShocktubeProblem {
};

template <> struct quokka::EOS_Traits<MHDShocktubeProblem> {
	static constexpr double gamma = 2.0;
	static constexpr double mean_molecular_weight = C::m_u;
};

template <> struct Physics_Traits<MHDShocktubeProblem> {
	static constexpr bool is_self_gravity_enabled = false;
	// cell-centred
	static constexpr bool is_hydro_enabled = true;
	static constexpr int numMassScalars = 0;		     // number of mass scalars
	static constexpr int numPassiveScalars = numMassScalars + 0; // number of passive scalars
	static constexpr bool is_radiation_enabled = false;
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

template <> void QuokkaSimulation<MHDShocktubeProblem>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = grid_elem.dx_;
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = grid_elem.prob_lo_;
	const amrex::Array4<double> &state_cc = grid_elem.array_;
	const amrex::Box &indexRange = grid_elem.indexRange_;

	const int ncomp_cc = Physics_Indices<MHDShocktubeProblem>::nvarTotal_cc;

	// magnetic field at center of cell
	const double x1mag = 0.75; // constant
	const double x3mag = 0.0;

	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		const double x = prob_lo[0] + ((i + 0.5) * dx[0]);
		const auto gamma = quokka::EOS_Traits<MHDShocktubeProblem>::gamma;
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
		state_cc(i, j, k, HydroSystem<MHDShocktubeProblem>::density_index) = rho;
		state_cc(i, j, k, HydroSystem<MHDShocktubeProblem>::x1Momentum_index) = vx * rho;
		state_cc(i, j, k, HydroSystem<MHDShocktubeProblem>::x2Momentum_index) = vy * rho;
		state_cc(i, j, k, HydroSystem<MHDShocktubeProblem>::x3Momentum_index) = vz * rho;
		state_cc(i, j, k, HydroSystem<MHDShocktubeProblem>::energy_index) = P / (gamma - 1.) + 0.5 * rho * (vx * vx) + Emag;
		state_cc(i, j, k, HydroSystem<MHDShocktubeProblem>::internalEnergy_index) = P / (gamma - 1.);
	});
}

template <> void QuokkaSimulation<MHDShocktubeProblem>::setInitialConditionsOnGridFaceVars(quokka::grid const &grid_elem)
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
			state_fc(i, j, k, Physics_Indices<MHDShocktubeProblem>::mhdFirstIndex) = x1mag;
		} else if (dir == quokka::direction::y) {
			state_fc(i, j, k, Physics_Indices<MHDShocktubeProblem>::mhdFirstIndex) = x2mag;
		} else if (dir == quokka::direction::z) {
			state_fc(i, j, k, Physics_Indices<MHDShocktubeProblem>::mhdFirstIndex) = x3mag;
		}
	});
}

template <>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE void
AMRSimulation<MHDShocktubeProblem>::setCustomBoundaryConditions(const amrex::IntVect &iv, amrex::Array4<amrex::Real> const &consVar, int /*dcomp*/, int numcomp,
								amrex::GeometryData const &geom, const amrex::Real /*time*/, const amrex::BCRec * /*bcr*/,
								int /*bcomp*/, int /*orig_comp*/)
{
#if (AMREX_SPACEDIM == 1)
	auto i = iv.toArray()[0];
	int j = 0;
	int k = 0;
#endif
#if (AMREX_SPACEDIM == 2)
	auto [i, j] = iv.toArray();
	int k = 0;
#endif
#if (AMREX_SPACEDIM == 3)
	auto [i, j, k] = iv.toArray();
#endif

	amrex::Box const &box = geom.Domain();
	amrex::GpuArray<int, 3> lo = box.loVect3d();
	amrex::GpuArray<int, 3> hi = box.hiVect3d();
	const auto gamma = quokka::EOS_Traits<MHDShocktubeProblem>::gamma;

	const double Emag_L = 0.5 * (Bx * Bx + By_L * By_L + Bz * Bz);
	const double Emag_R = 0.5 * (Bx * Bx + By_R * By_R + Bz * Bz);

	if (i < lo[0]) {
		// x1 left side boundary -- constant
		for (int n = 0; n < numcomp; ++n) {
			consVar(i, j, k, n) = 0;
		}

		consVar(i, j, k, RadSystem<MHDShocktubeProblem>::gasEnergy_index) = P_L / (gamma - 1.) + Emag_L;
		consVar(i, j, k, RadSystem<MHDShocktubeProblem>::gasInternalEnergy_index) = P_L / (gamma - 1.);
		consVar(i, j, k, RadSystem<MHDShocktubeProblem>::gasDensity_index) = rho_L;
		consVar(i, j, k, RadSystem<MHDShocktubeProblem>::x1GasMomentum_index) = 0.;
		consVar(i, j, k, RadSystem<MHDShocktubeProblem>::x2GasMomentum_index) = 0.;
		consVar(i, j, k, RadSystem<MHDShocktubeProblem>::x3GasMomentum_index) = 0.;

	} else if (i >= hi[0]) {
		// x1 right-side boundary -- constant
		for (int n = 0; n < numcomp; ++n) {
			consVar(i, j, k, n) = 0;
		}

		consVar(i, j, k, RadSystem<MHDShocktubeProblem>::gasEnergy_index) = P_R / (gamma - 1.) + Emag_R;
		consVar(i, j, k, RadSystem<MHDShocktubeProblem>::gasInternalEnergy_index) = P_R / (gamma - 1.);
		consVar(i, j, k, RadSystem<MHDShocktubeProblem>::gasDensity_index) = rho_R;
		consVar(i, j, k, RadSystem<MHDShocktubeProblem>::x1GasMomentum_index) = 0.;
		consVar(i, j, k, RadSystem<MHDShocktubeProblem>::x2GasMomentum_index) = 0.;
		consVar(i, j, k, RadSystem<MHDShocktubeProblem>::x3GasMomentum_index) = 0.;
	}
}

template <>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE void
AMRSimulation<MHDShocktubeProblem>::setCustomBoundaryConditionsFaceVar(const amrex::IntVect &iv, amrex::Array4<amrex::Real> const &consVar_fc, int /*dcomp*/,
								       int /*numcomp*/, amrex::GeometryData const &geom, const amrex::Real /*time*/,
								       const amrex::BCRec * /*bcr*/, int /*bcomp*/, int /*orig_comp*/, quokka::direction dir)
{
#if (AMREX_SPACEDIM == 1)
	auto i = iv.toArray()[0];
	int const j = 0;
	int const k = 0;
#endif
#if (AMREX_SPACEDIM == 2)
	auto [i, j] = iv.toArray();
	int const k = 0;
#endif
#if (AMREX_SPACEDIM == 3)
	auto [i, j, k] = iv.toArray();
#endif
	amrex::Box const &box = geom.Domain();
	amrex::GpuArray<int, 3> lo = box.loVect3d();
	amrex::GpuArray<int, 3> hi = box.hiVect3d();

	// Use direction to determine which boundaries to check
	switch (dir) {
		case quokka::direction::x:
			if (i <= lo[0]) {
				consVar_fc(i, j, k, Physics_Indices<MHDShocktubeProblem>::mhdFirstIndex) = Bx;
			} else if (i > hi[0]) {
				consVar_fc(i, j, k, Physics_Indices<MHDShocktubeProblem>::mhdFirstIndex) = Bx;
			}
			break;
		case quokka::direction::y:
			if (i < lo[0]) {
				// Set y-direction left boundary values
				consVar_fc(i, j, k, Physics_Indices<MHDShocktubeProblem>::mhdFirstIndex) = By_L;
			} else if (i > hi[0]) {
				// Set y-direction right boundary values
				consVar_fc(i, j, k, Physics_Indices<MHDShocktubeProblem>::mhdFirstIndex) = By_R;
			}
			break;
		case quokka::direction::z:
			if (i < lo[0]) {
				// Set z-direction left boundary values
				consVar_fc(i, j, k, Physics_Indices<MHDShocktubeProblem>::mhdFirstIndex) = Bz;
			} else if (i >= hi[0]) {
				// Set z-direction right boundary values
				consVar_fc(i, j, k, Physics_Indices<MHDShocktubeProblem>::mhdFirstIndex) = Bz;
			}
			break;
		case quokka::direction::na:
			break; // do nothing
	}
}

template <> void QuokkaSimulation<MHDShocktubeProblem>::refineGrid(int lev, amrex::TagBoxArray &tags, Real /*time*/, int /*ngrow*/)
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
	const int ncomp_cc = Physics_Indices<MHDShocktubeProblem>::nvarTotal_cc;
	amrex::Vector<amrex::BCRec> BCs_cc(ncomp_cc);
	for (int n = 0; n < ncomp_cc; ++n) {
		BCs_cc[n].setLo(0, amrex::BCType::ext_dir); // Dirichlet
		BCs_cc[n].setHi(0, amrex::BCType::ext_dir);
		for (int i = 1; i < AMREX_SPACEDIM; ++i) {
			BCs_cc[n].setLo(i, amrex::BCType::int_dir); // periodic
			BCs_cc[n].setHi(i, amrex::BCType::int_dir);
		}
	}

	const int nvars_fc = Physics_Indices<MHDShocktubeProblem>::nvarTotal_fc;
	amrex::Vector<amrex::BCRec> BCs_fc(nvars_fc);
	for (int icomp = 0; icomp < nvars_fc; ++icomp) {
		BCs_fc[icomp].setLo(0, amrex::BCType::ext_dir); // Dirichlet
		BCs_fc[icomp].setHi(0, amrex::BCType::ext_dir);
		for (int i = 1; i < AMREX_SPACEDIM; ++i) {
			BCs_fc[icomp].setLo(i, amrex::BCType::int_dir); // periodic
			BCs_fc[icomp].setHi(i, amrex::BCType::int_dir);
		}
	}

	QuokkaSimulation<MHDShocktubeProblem> sim(BCs_cc, BCs_fc);

	// Main time loop
	sim.setInitialConditions();
	sim.evolve();

	return 0;
}
