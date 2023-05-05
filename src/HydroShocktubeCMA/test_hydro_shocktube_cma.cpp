//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file test_hydro_shocktube_CMA.cpp
/// \brief Defines a test problem for a shock tube with passive scalars using consistent molecular advection (CMA).
///

#include <cmath>
#include <string>
#include <unordered_map>

#include "AMReX_BC_TYPES.H"

#include "ArrayUtil.hpp"
#include "RadhydroSimulation.hpp"
#include "fextract.hpp"
#include "hydro_system.hpp"
#include "radiation_system.hpp"
#include "test_hydro_shocktube_cma.hpp"
#ifdef HAVE_PYTHON
#include "matplotlibcpp.h"
#endif

struct ShocktubeProblem {
};

template <> struct quokka::EOS_Traits<ShocktubeProblem> {
	static constexpr double gamma = 1.4;
	static constexpr double mean_molecular_weight = NAN;
	static constexpr double boltzmann_constant = quokka::boltzmann_constant_cgs;
};

template <> struct Physics_Traits<ShocktubeProblem> {
	// cell-centred
	static constexpr bool is_hydro_enabled = true;
	static constexpr bool is_chemistry_enabled = false;
	static constexpr int numMassScalars = 3;		     // number of mass scalars
	static constexpr int numPassiveScalars = numMassScalars + 0; // number of passive scalars
	static constexpr bool is_radiation_enabled = false;
	// face-centred
	static constexpr bool is_mhd_enabled = false;
};

// left- and right- side shock states
constexpr amrex::Real rho_L = 1.0;
constexpr amrex::Real P_L = 1.0;
constexpr amrex::Real rho_R = 0.125;
constexpr amrex::Real P_R = 0.1;

template <> void RadhydroSimulation<ShocktubeProblem>::setInitialConditionsOnGrid(quokka::grid grid_elem)
{
	// extract variables required from the geom object
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const dx = grid_elem.dx_;
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const prob_lo = grid_elem.prob_lo_;
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const amrex::Array4<double> &state_cc = grid_elem.array_;

	const int ncomp_cc = Physics_Indices<ShocktubeProblem>::nvarTotal_cc;
	const int nmscalars = Physics_Traits<ShocktubeProblem>::numMassScalars;
	// loop over the grid and set the initial condition
	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		amrex::Real const x = prob_lo[0] + (i + static_cast<amrex::Real>(0.5)) * dx[0];

		const double vx = 0.0;
		double rho = NAN;
		double P = NAN;
		std::array<Real, nmscalars> specie = {-1.0}; // why can't this loop read in numPassiveScalars

		// from Plewa and Muller 1999
		if (x <= 0.5) {
			rho = rho_L;
			P = P_L;
		} else {
			rho = rho_R;
			P = P_R;
		}

		// initialize mass fractions of species
		if (x <= 0.5) {
			specie[0] = 0.8;
		} else if (x > 0.5 && x <= 0.75) {
			specie[0] = 0.3;
		} else {
			specie[0] = 0.1;
		}

		specie[1] = 0.3 * pow(sin(20 * 3.14 * x), 2);
		specie[2] = 1 - specie[0] - specie[1];

		AMREX_ASSERT(!std::isnan(vx));
		AMREX_ASSERT(!std::isnan(rho));
		AMREX_ASSERT(!std::isnan(P));

		const auto gamma = quokka::EOS_Traits<ShocktubeProblem>::gamma;
		for (int n = 0; n < ncomp_cc; ++n) {
			state_cc(i, j, k, n) = 0.;
		}
		state_cc(i, j, k, HydroSystem<ShocktubeProblem>::density_index) = rho;
		state_cc(i, j, k, HydroSystem<ShocktubeProblem>::x1Momentum_index) = rho * vx;
		state_cc(i, j, k, HydroSystem<ShocktubeProblem>::x2Momentum_index) = 0.;
		state_cc(i, j, k, HydroSystem<ShocktubeProblem>::x3Momentum_index) = 0.;
		state_cc(i, j, k, HydroSystem<ShocktubeProblem>::energy_index) = P / (gamma - 1.) + 0.5 * rho * (vx * vx);
		state_cc(i, j, k, HydroSystem<ShocktubeProblem>::internalEnergy_index) = P / (gamma - 1.);

		for (int nn = 0; nn < nmscalars; ++nn) {
			state_cc(i, j, k, HydroSystem<ShocktubeProblem>::scalar0_index + nn) = specie[nn];
		}
	});
}

template <>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE void
AMRSimulation<ShocktubeProblem>::setCustomBoundaryConditions(const amrex::IntVect &iv, amrex::Array4<amrex::Real> const &consVar, int /*dcomp*/, int numcomp,
							     amrex::GeometryData const &geom, const amrex::Real /*time*/, const amrex::BCRec * /*bcr*/,
							     int /*bcomp*/, int /*orig_comp*/)
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
	const auto gamma = quokka::EOS_Traits<ShocktubeProblem>::gamma;

	if (i < lo[0]) {
		// x1 left side boundary -- constant
		for (int n = 0; n < numcomp; ++n) {
			consVar(i, j, k, n) = 0;
		}

		consVar(i, j, k, RadSystem<ShocktubeProblem>::gasEnergy_index) = P_L / (gamma - 1.);
		consVar(i, j, k, RadSystem<ShocktubeProblem>::gasInternalEnergy_index) = P_L / (gamma - 1.);
		consVar(i, j, k, RadSystem<ShocktubeProblem>::gasDensity_index) = rho_L;
		consVar(i, j, k, RadSystem<ShocktubeProblem>::x1GasMomentum_index) = 0.;
		consVar(i, j, k, RadSystem<ShocktubeProblem>::x2GasMomentum_index) = 0.;
		consVar(i, j, k, RadSystem<ShocktubeProblem>::x3GasMomentum_index) = 0.;

		consVar(i, j, k, HydroSystem<ShocktubeProblem>::scalar0_index + 0) = 0.8;
		consVar(i, j, k, HydroSystem<ShocktubeProblem>::scalar0_index + 1) = 0.3 * pow(sin(20 * 3.14 * 0), 2);
		consVar(i, j, k, HydroSystem<ShocktubeProblem>::scalar0_index + 2) = 1 - 0.8 - 0.3 * pow(sin(20 * 3.14 * 0), 2);

	} else if (i >= hi[0]) {
		// x1 right-side boundary -- constant
		for (int n = 0; n < numcomp; ++n) {
			consVar(i, j, k, n) = 0;
		}

		consVar(i, j, k, RadSystem<ShocktubeProblem>::gasEnergy_index) = P_R / (gamma - 1.);
		consVar(i, j, k, RadSystem<ShocktubeProblem>::gasInternalEnergy_index) = P_R / (gamma - 1.);
		consVar(i, j, k, RadSystem<ShocktubeProblem>::gasDensity_index) = rho_R;
		consVar(i, j, k, RadSystem<ShocktubeProblem>::x1GasMomentum_index) = 0.;
		consVar(i, j, k, RadSystem<ShocktubeProblem>::x2GasMomentum_index) = 0.;
		consVar(i, j, k, RadSystem<ShocktubeProblem>::x3GasMomentum_index) = 0.;

		consVar(i, j, k, HydroSystem<ShocktubeProblem>::scalar0_index + 0) = 0.1;
		consVar(i, j, k, HydroSystem<ShocktubeProblem>::scalar0_index + 1) = 0.3 * pow(sin(20 * 3.14 * 1), 2);
		consVar(i, j, k, HydroSystem<ShocktubeProblem>::scalar0_index + 2) = 1 - 0.1 - 0.3 * pow(sin(20 * 3.14 * 1), 2);
	}
}

template <> void RadhydroSimulation<ShocktubeProblem>::ErrorEst(int lev, amrex::TagBoxArray &tags, Real /*time*/, int /*ngrow*/)
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
	// Problem parameters
	const double max_time = 0.4;
	const int max_timesteps = 1;

	// Problem initialization
	const int ncomp_cc = Physics_Indices<ShocktubeProblem>::nvarTotal_cc;
	amrex::Vector<amrex::BCRec> BCs_cc(ncomp_cc);
	for (int n = 0; n < ncomp_cc; ++n) {
		BCs_cc[0].setLo(0, amrex::BCType::ext_dir); // Dirichlet
		BCs_cc[0].setHi(0, amrex::BCType::ext_dir);
		for (int i = 1; i < AMREX_SPACEDIM; ++i) {
			BCs_cc[n].setLo(i, amrex::BCType::int_dir); // periodic
			BCs_cc[n].setHi(i, amrex::BCType::int_dir);
		}
	}

	RadhydroSimulation<ShocktubeProblem> sim(BCs_cc);

	// sim.cflNumber_ = CFL_number;
	// sim.maxDt_ = max_dt;
	// sim.initDt_ = initial_dt;
	sim.stopTime_ = max_time;
	sim.maxTimesteps_ = max_timesteps;

	// Main time loop
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
