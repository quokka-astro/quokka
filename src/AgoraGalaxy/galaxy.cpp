//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2024 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file galaxy.cpp
/// \brief Defines a simulation using the AGORA isolated galaxy initial conditions.
///

#include <cmath>

#include "AMReX_BC_TYPES.H"
#include "AMReX_DistributionMapping.H"
#include "AMReX_Geometry.H"
#include "AMReX_MultiFab.H"
#include "AMReX_ParmParse.H"
#include "AMReX_REAL.H"

#include "EOS.hpp"
#include "RadhydroSimulation.hpp"
#include "fundamental_constants.H"
#include "galaxy.hpp"
#include "hydro_system.hpp"

struct AgoraGalaxy {
};

template <> struct quokka::EOS_Traits<AgoraGalaxy> {
	static constexpr double gamma = 5. / 3.;
	static constexpr double mean_molecular_weight = C::m_u;
	static constexpr double boltzmann_constant = C::k_B;
};

template <> struct HydroSystem_Traits<AgoraGalaxy> {
	static constexpr bool reconstruct_eint = true;
};

template <> struct Physics_Traits<AgoraGalaxy> {
	static constexpr bool is_hydro_enabled = true;
	static constexpr bool is_radiation_enabled = false;
	static constexpr bool is_mhd_enabled = false;
	static constexpr int numMassScalars = 0;		     // number of mass scalars
	static constexpr int numPassiveScalars = numMassScalars + 0; // number of passive scalars
	static constexpr int nGroups = 1;			     // number of radiation groups
};

template <> struct SimulationData<AgoraGalaxy> {
	std::vector<amrex::Real> radius{};
	std::vector<amrex::Real> vcirc{};
};

template <> void RadhydroSimulation<AgoraGalaxy>::preCalculateInitialConditions()
{
	// 1. read in circular velocity table
	// 2. copy to GPU
	// 3. save interpolator object
	//
	// TODO(bwibking): implement.
}

template <> void RadhydroSimulation<AgoraGalaxy>::setInitialConditionsOnGrid(quokka::grid grid_elem)
{
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = grid_elem.dx_;
	const amrex::Array4<double> &state_cc = grid_elem.array_;
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = grid_elem.prob_lo_;
	const amrex::Real gamma = quokka::EOS_Traits<AgoraGalaxy>::gamma;

	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		// Cartesian coordinates
		amrex::Real const x = prob_lo[0] + (i + static_cast<amrex::Real>(0.5)) * dx[0];
		amrex::Real const y = prob_lo[1] + (j + static_cast<amrex::Real>(0.5)) * dx[1];

		// cylindrical coordinates
		amrex::Real const R = std::sqrt(std::pow(x, 2) + std::pow(y, 2));
		amrex::Real const theta = std::atan2(x, y);

		// compute double exponential density profile
		double const rho = 1.0e-22; // g cm^{-3}

		// interpolate circular velocity based on radius of cell center
		double const vcirc = 0;
		double const vx = vcirc * std::cos(theta);
		double const vy = vcirc * std::sin(theta);
		double const vz = 0;
		double const vsq = vx * vx + vy * vy + vz * vz;

		// compute temperature
		double T = NAN;
		if (R < 20.0e3 * C::parsec) {
			T = 1.0e4; // K
		} else {
			T = 1.0e6; // K
		}
		const double Eint = quokka::EOS<AgoraGalaxy>::ComputeEintFromTgas(rho, T);

		state_cc(i, j, k, HydroSystem<AgoraGalaxy>::density_index) = rho;
		state_cc(i, j, k, HydroSystem<AgoraGalaxy>::x1Momentum_index) = rho * vx;
		state_cc(i, j, k, HydroSystem<AgoraGalaxy>::x2Momentum_index) = rho * vy;
		state_cc(i, j, k, HydroSystem<AgoraGalaxy>::x3Momentum_index) = rho * vz;
		state_cc(i, j, k, HydroSystem<AgoraGalaxy>::energy_index) = Eint + 0.5 * rho * vsq;
		state_cc(i, j, k, HydroSystem<AgoraGalaxy>::internalEnergy_index) = Eint;
	});
}

template <> void RadhydroSimulation<AgoraGalaxy>::createInitialParticles()
{
	// read particles from ASCII file
	const int nreal_extra = 4; // mass vx vy vz
	CICParticles->SetVerbose(1);
	CICParticles->InitFromAsciiFile("AgoraGalaxy_particles.txt", nreal_extra, nullptr);
}

template <> void RadhydroSimulation<AgoraGalaxy>::ComputeDerivedVar(int lev, std::string const &dname, amrex::MultiFab &mf, const int ncomp_cc_in) const
{
	// compute derived variables and save in 'mf'
	if (dname == "gpot") {
		const int ncomp = ncomp_cc_in;
		auto const &phi_arr = phi[lev].const_arrays();
		auto output = mf.arrays();
		amrex::ParallelFor(mf, [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept { output[bx](i, j, k, ncomp) = phi_arr[bx](i, j, k); });
	}
}

auto problem_main() -> int
{
	auto isNormalComp = [=](int n, int dim) {
		if ((n == HydroSystem<AgoraGalaxy>::x1Momentum_index) && (dim == 0)) {
			return true;
		}
		if ((n == HydroSystem<AgoraGalaxy>::x2Momentum_index) && (dim == 1)) {
			return true;
		}
		if ((n == HydroSystem<AgoraGalaxy>::x3Momentum_index) && (dim == 2)) {
			return true;
		}
		return false;
	};

	const int ncomp_cc = Physics_Indices<AgoraGalaxy>::nvarTotal_cc;
	amrex::Vector<amrex::BCRec> BCs_cc(ncomp_cc);
	for (int n = 0; n < ncomp_cc; ++n) {
		for (int i = 0; i < AMREX_SPACEDIM; ++i) {
			if (isNormalComp(n, i)) {
				BCs_cc[n].setLo(i, amrex::BCType::reflect_odd);
				BCs_cc[n].setHi(i, amrex::BCType::reflect_odd);
			} else {
				BCs_cc[n].setLo(i, amrex::BCType::reflect_even);
				BCs_cc[n].setHi(i, amrex::BCType::reflect_even);
			}
		}
	}

	// Problem initialization
	RadhydroSimulation<AgoraGalaxy> sim(BCs_cc);
	sim.doPoissonSolve_ = 1; // enable self-gravity

	// initialize
	sim.setInitialConditions();

	// evolve
	sim.evolve();

	const int status = 0;
	return status;
}
