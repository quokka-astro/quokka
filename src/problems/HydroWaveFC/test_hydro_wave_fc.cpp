//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file test_hydro_wave.cpp
/// \brief Defines a test problem for a linear hydro wave.
///

#ifdef HAVE_PYTHON
#include "util/matplotlibcpp.h"
#endif
#include "hydro/hydro_system.hpp"
#include <fmt/format.h>
#include <valarray>

#include "AMReX_Array.H"
#include "AMReX_Array4.H"
#include "AMReX_REAL.H"

#include "QuokkaSimulation.hpp"
#include "hydro/hydro_system.hpp"
#include "util/fextract.hpp"

struct WaveProblemFC {
};

template <> struct quokka::EOS_Traits<WaveProblemFC> {
	static constexpr double gamma = 5. / 3.;
	static constexpr double mean_molecular_weight = C::m_u;
};

template <> struct Physics_Traits<WaveProblemFC> {
	// cell-centred
	static constexpr bool is_hydro_enabled = true;
	static constexpr int numMassScalars = 0;		     // number of mass scalars
	static constexpr int numPassiveScalars = numMassScalars + 0; // number of passive scalars
	static constexpr bool is_radiation_enabled = false;
	// face-centred
	static constexpr bool is_mhd_enabled = false;
	static constexpr int nGroups = 1; // number of radiation groups
	static constexpr UnitSystem unit_system = UnitSystem::CGS;
};

constexpr double rho0 = 1.0;					      // background density
constexpr double P0 = 1.0 / quokka::EOS_Traits<WaveProblemFC>::gamma; // background pressure
constexpr double v0 = 0.;					      // background velocity
constexpr double amp = 1.0e-6;					      // perturbation amplitude

AMREX_GPU_DEVICE void computeWaveSolution(int i, int j, int k, amrex::Array4<amrex::Real> const &state, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx,
					  amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_lo)
{
	const amrex::Real x_L = prob_lo[0] + (i + static_cast<amrex::Real>(0.0)) * dx[0];
	const amrex::Real x_R = prob_lo[0] + (i + static_cast<amrex::Real>(1.0)) * dx[0];
	const amrex::Real A = amp;

	const quokka::valarray<double, 3> R = {1.0, -1.0, 1.5}; // right eigenvector of sound wave
	const quokka::valarray<double, 3> U_0 = {rho0, rho0 * v0, P0 / (quokka::EOS_Traits<WaveProblemFC>::gamma - 1.0) + 0.5 * rho0 * std::pow(v0, 2)};
	const quokka::valarray<double, 3> dU = (A * R / (2.0 * M_PI * dx[0])) * (std::cos(2.0 * M_PI * x_L) - std::cos(2.0 * M_PI * x_R));

	double const rho = U_0[0] + dU[0];
	double const xmom = U_0[1] + dU[1];
	double const Etot = U_0[2] + dU[2];
	double const Eint = Etot - 0.5 * (xmom * xmom) / rho;

	state(i, j, k, HydroSystem<WaveProblemFC>::density_index) = rho;
	state(i, j, k, HydroSystem<WaveProblemFC>::x1Momentum_index) = xmom;
	state(i, j, k, HydroSystem<WaveProblemFC>::x2Momentum_index) = 0;
	state(i, j, k, HydroSystem<WaveProblemFC>::x3Momentum_index) = 0;
	state(i, j, k, HydroSystem<WaveProblemFC>::energy_index) = Etot;
	state(i, j, k, HydroSystem<WaveProblemFC>::internalEnergy_index) = Eint;
}

template <> void QuokkaSimulation<WaveProblemFC>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	// extract variables required from the geom object
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const dx = grid_elem.dx_;
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const prob_lo = grid_elem.prob_lo_;
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const amrex::Array4<double> &state_cc = grid_elem.array_;
	const int ncomp_cc = Physics_Indices<WaveProblemFC>::nvarTotal_cc;
	// loop over the grid and set the initial condition
	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		for (int n = 0; n < ncomp_cc; ++n) {
			state_cc(i, j, k, n) = 0; // fill unused components with zeros
		}
		computeWaveSolution(i, j, k, state_cc, dx, prob_lo);
	});
}

auto problem_main() -> int
{
	// Based on the ATHENA test page:
	// https://www.astro.princeton.edu/~jstone/Athena/tests/linear-waves/linear-waves.html

	// Problem parameters
	const double CFL_number = 0.1;
	const double max_time = 1.0;
	const int max_timesteps = 1;

	// Problem initialization
	const int ncomp_cc = Physics_Indices<WaveProblemFC>::nvarTotal_cc;
	amrex::Vector<amrex::BCRec> BCs_cc(ncomp_cc);
	for (int n = 0; n < ncomp_cc; ++n) {
		for (int i = 0; i < AMREX_SPACEDIM; ++i) {
			BCs_cc[n].setLo(i, amrex::BCType::int_dir); // periodic
			BCs_cc[n].setHi(i, amrex::BCType::int_dir);
		}
	}

	QuokkaSimulation<WaveProblemFC> sim(BCs_cc);

	sim.do_tracers = 1;

	// set initial conditions
	sim.setInitialConditions();
	auto [pos_exact, val_exact] = fextract(sim.state_new_cc_[0], sim.geom[0], 0, 0.5);

	// Main time loop
	sim.evolve();

	auto [position, values] = fextract(sim.state_new_cc_[0], sim.geom[0], 0, 0.5);
	auto const nx = static_cast<int>(position.size());
	std::vector<double> const xs = position;

	// compute error norm
	amrex::Real err_sq = 0.;
	for (int n = 0; n < QuokkaSimulation<WaveProblemFC>::ncompHydro_; ++n) {
		if (n == HydroSystem<WaveProblemFC>::internalEnergy_index) {
			continue;
		}
		amrex::Real dU_k = 0.;
		for (int i = 0; i < nx; ++i) {
			// Δ Uk = ∑i |Uk,in - Uk,i0| / Nx
			const amrex::Real U_k0 = val_exact.at(n)[i];
			const amrex::Real U_k1 = values.at(n)[i];
			dU_k += std::abs(U_k1 - U_k0) / static_cast<double>(nx);
		}
		// ε = || Δ U || = [&sum_k (Δ Uk)2]^{1/2}
		err_sq += dU_k * dU_k;
	}
	const amrex::Real epsilon = std::sqrt(err_sq);
	amrex::Print() << "rms of component-wise L1 error norms = " << epsilon << '\n';

	const double err_tol = 1.0e-8; // for Nx = 100
	int status = 0;
	if (epsilon > err_tol) {
		status = 1;
	}

	return status;
}
