//==============================================================================
// Copyright 2022 Neco Kriel and Ben Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file testCurrentSheet.cpp
/// \brief Defines a test problem to test magnetic reconnection in a current sheet.
///   This problem is based on the description here:
///	  https://www.astro.princeton.edu/~jstone/Athena/tests/current-sheet/current-sheet.html
///

#include <cmath>

#include "AMReX_Array.H"
#include "AMReX_Array4.H"
#include "AMReX_REAL.H"

#include "QuokkaSimulation.hpp"
#include "grid.hpp"
#include "hydro/EOS.hpp"
#include "hydro/hydro_system.hpp"
#include "physics_info.hpp"
#include "util/BC.hpp"

struct CurrentSheet {
};

template <> struct quokka::EOS_Traits<CurrentSheet> {
	static constexpr double gamma = 5. / 3.;
	static constexpr double mean_molecular_weight = C::m_u;
};

template <> struct Physics_Traits<CurrentSheet> : DefaultPhysicsTraits {
	static constexpr bool is_hydro_enabled = true;
	static constexpr bool is_mhd_enabled = true;
};

// constants
constexpr double gamma_gas = quokka::EOS_Traits<CurrentSheet>::gamma;
constexpr double beta = 0.1;
constexpr double A = 0.1;

// background states
constexpr double rho0 = 1.0;
constexpr double P0 = 0.5 * beta;

template <> void QuokkaSimulation<CurrentSheet>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	// extract grid information
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = grid_elem.dx_;
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = grid_elem.prob_lo_;
	const amrex::Array4<double> &state_cc = grid_elem.array_;
	const amrex::Box &indexRange = grid_elem.indexRange_;

	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		const double y = prob_lo[1] + ((j + 0.5) * dx[1]);
		const double vx = A * std::sin(2.0 * M_PI * y);

		const double Ekin = 0.5 * rho0 * (vx * vx);
		const double Eint = P0 / (gamma_gas - 1.0);
		const double Emag = 0.5;

		state_cc(i, j, k, HydroSystem<CurrentSheet>::density_index) = rho0;
		state_cc(i, j, k, HydroSystem<CurrentSheet>::x1Momentum_index) = rho0 * vx;
		state_cc(i, j, k, HydroSystem<CurrentSheet>::x2Momentum_index) = 0;
		state_cc(i, j, k, HydroSystem<CurrentSheet>::x3Momentum_index) = 0;
		state_cc(i, j, k, HydroSystem<CurrentSheet>::internalEnergy_index) = Eint;
		state_cc(i, j, k, HydroSystem<CurrentSheet>::energy_index) = Eint + Ekin + Emag;
	});
}

template <> void QuokkaSimulation<CurrentSheet>::setInitialConditionsOnGridFaceVars(quokka::grid const &grid_elem)
{
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = grid_elem.dx_;
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = grid_elem.prob_lo_;
	const amrex::Array4<double> &state_fc = grid_elem.array_;
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const quokka::direction dir = grid_elem.dir_;

	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		const double x = prob_lo[0] + ((i + 0.5) * dx[0]);

		double by = NAN;
		if (std::abs(x) > 0.25) {
			by = 1.;
		} else {
			by = -1.;
		}

		// somehow this does not work -- the outputs show that x-BField is nonzero...
		if (dir == quokka::direction::x) {
			state_fc(i, j, k, Physics_Indices<CurrentSheet>::mhdFirstIndex) = 0;
		} else if (dir == quokka::direction::y) {
			state_fc(i, j, k, Physics_Indices<CurrentSheet>::mhdFirstIndex) = by;
		} else if (dir == quokka::direction::z) {
			state_fc(i, j, k, Physics_Indices<CurrentSheet>::mhdFirstIndex) = 0;
		}
	});
}

auto problem_main() -> int
{
	const int nvars_fc = Physics_Indices<CurrentSheet>::nvarTotal_fc;
	amrex::Vector<amrex::BCRec> BCs_fc(nvars_fc);
	for (int icomp = 0; icomp < nvars_fc; ++icomp) {
		for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
			BCs_fc[icomp].setLo(idim, amrex::BCType::int_dir); // periodic
			BCs_fc[icomp].setHi(idim, amrex::BCType::int_dir);
		}
	}

	QuokkaSimulation<CurrentSheet> sim;
	sim.setInitialConditions();
	sim.evolve();
	return 0;
}
