//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file channel.cpp
/// \brief Implements a subsonic channel flow problem with Navier-Stokes
///        Characteristic Boundary Conditions (NSCBC).
///
#include <random>
#include <vector>

#include "AMReX.H"
#include "AMReX_BC_TYPES.H"
#include "AMReX_BLProfiler.H"
#include "AMReX_BLassert.H"
#include "AMReX_FabArray.H"
#include "AMReX_Geometry.H"
#include "AMReX_GpuDevice.H"
#include "AMReX_IntVect.H"
#include "AMReX_MultiFab.H"
#include "AMReX_ParallelContext.H"
#include "AMReX_ParallelDescriptor.H"
#include "AMReX_REAL.H"
#include "AMReX_SPACE.H"
#include "AMReX_TableData.H"
#include "AMReX_iMultiFab.H"

#include "RadhydroSimulation.hpp"
#include "channel.hpp"
#include "hydro_system.hpp"
#include "radiation_system.hpp"

using amrex::Real;

struct Channel {
}; // dummy type to allow compile-type polymorphism via template specialization

template <> struct quokka::EOS_Traits<Channel> {
	static constexpr double gamma = 5. / 3.; // default value
	static constexpr double mean_molecular_weight = 1.0;
	static constexpr double boltzmann_constant = C::k_B;
};

template <> struct Physics_Traits<Channel> {
	static constexpr bool is_hydro_enabled = true;
	static constexpr bool is_chemistry_enabled = false;
	static constexpr bool is_mhd_enabled = false;
	static constexpr int numMassScalars = 0;		     // number of mass scalars
	static constexpr int numPassiveScalars = numMassScalars + 0; // number of passive scalars
	static constexpr bool is_radiation_enabled = false;
};

constexpr Real Tgas0 = 1.0e7; // K
constexpr Real nH0 = 1.0e-4;  // cm^-3
constexpr Real M0 = 2.0;      // Mach number of shock

constexpr Real P0 = nH0 * Tgas0 * C::k_B;      // erg cm^-3
constexpr Real rho0 = nH0 * (C::m_p + C::m_e); // g cm^-3

// cloud-tracking variables needed for Dirichlet boundary condition
AMREX_GPU_MANAGED static Real rho_wind = 0;
AMREX_GPU_MANAGED static Real v_wind = 0;
AMREX_GPU_MANAGED static Real P_wind = 0;
AMREX_GPU_MANAGED static Real delta_vx = 0;

template <> void RadhydroSimulation<Channel>::setInitialConditionsOnGrid(quokka::grid grid_elem)
{
	// set initial conditions
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const dx = grid_elem.dx_;
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = grid_elem.prob_lo_;
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_hi = grid_elem.prob_hi_;
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const amrex::Array4<double> &state_cc = grid_elem.array_;

	Real const Lx = (prob_hi[0] - prob_lo[0]);
	Real const Ly = (prob_hi[1] - prob_lo[1]);
	Real const Lz = (prob_hi[2] - prob_lo[2]);

	Real const x0 = prob_lo[0] + 0.5 * (prob_hi[0] - prob_lo[0]);
	Real const y0 = prob_lo[1] + 0.8 * (prob_hi[1] - prob_lo[1]);
	Real const z0 = prob_lo[2] + 0.5 * (prob_hi[2] - prob_lo[2]);

	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		Real const rho = rho0;
		Real const xmom = 0;
		Real const ymom = 0;
		Real const zmom = 0;
		Real const Eint = (quokka::EOS_Traits<Channel>::gamma - 1.) * P0;
		Real const Egas = RadSystem<Channel>::ComputeEgasFromEint(rho, xmom, ymom, zmom, Eint);

		state_cc(i, j, k, HydroSystem<Channel>::density_index) = rho;
		state_cc(i, j, k, HydroSystem<Channel>::x1Momentum_index) = xmom;
		state_cc(i, j, k, HydroSystem<Channel>::x2Momentum_index) = ymom;
		state_cc(i, j, k, HydroSystem<Channel>::x3Momentum_index) = zmom;
		state_cc(i, j, k, HydroSystem<Channel>::energy_index) = Egas;
		state_cc(i, j, k, HydroSystem<Channel>::internalEnergy_index) = Eint;
	});
}

template <>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE void AMRSimulation<Channel>::setCustomBoundaryConditions(const amrex::IntVect &iv, amrex::Array4<Real> const &consVar,
											     int /*dcomp*/, int /*numcomp*/, amrex::GeometryData const &geom,
											     const Real /*time*/, const amrex::BCRec * /*bcr*/, int /*bcomp*/,
											     int /*orig_comp*/)
{
	auto [i, j, k] = iv.dim3();

	amrex::Box const &box = geom.Domain();
	const auto &domain_lo = box.loVect3d();
	const auto &domain_hi = box.hiVect3d();
	const int ilo = domain_lo[0];

	if (i < ilo) {
		// x1 lower boundary -- constant
		// compute downstream shock conditions from rho0, P0, and M0
		constexpr Real gamma = quokka::EOS_Traits<Channel>::gamma;
		constexpr Real rho2 = rho0 * (gamma + 1.) * M0 * M0 / ((gamma - 1.) * M0 * M0 + 2.);
		constexpr Real P2 = P0 * (2. * gamma * M0 * M0 - (gamma - 1.)) / (gamma + 1.);
		Real const v2 = M0 * std::sqrt(gamma * P2 / rho2);

		Real const rho = rho2;
		Real const xmom = rho2 * v2;
		Real const ymom = 0;
		Real const zmom = 0;
		Real const Eint = (gamma - 1.) * P2;
		Real const Egas = RadSystem<Channel>::ComputeEgasFromEint(rho, xmom, ymom, zmom, Eint);

		consVar(i, j, k, HydroSystem<Channel>::density_index) = rho;
		consVar(i, j, k, HydroSystem<Channel>::x1Momentum_index) = xmom;
		consVar(i, j, k, HydroSystem<Channel>::x2Momentum_index) = ymom;
		consVar(i, j, k, HydroSystem<Channel>::x3Momentum_index) = zmom;
		consVar(i, j, k, HydroSystem<Channel>::energy_index) = Egas;
		consVar(i, j, k, HydroSystem<Channel>::internalEnergy_index) = Eint;
	}
}

auto problem_main() -> int
{
	// Problem parameters
	constexpr double seconds_in_year = 3.154e7;
	const double max_time = 2.0e6 * seconds_in_year; // 2 Myr

	// Problem initialization
	constexpr int ncomp_cc = Physics_Indices<Channel>::nvarTotal_cc;
	amrex::Vector<amrex::BCRec> BCs_cc(ncomp_cc);
	for (int n = 0; n < ncomp_cc; ++n) {
		BCs_cc[n].setLo(0, amrex::BCType::ext_dir);  // Dirichlet
		BCs_cc[n].setHi(0, amrex::BCType::foextrap); // extrapolate

		BCs_cc[n].setLo(1, amrex::BCType::int_dir); // periodic
		BCs_cc[n].setHi(1, amrex::BCType::int_dir);

		BCs_cc[n].setLo(2, amrex::BCType::int_dir);
		BCs_cc[n].setHi(2, amrex::BCType::int_dir);
	}

	RadhydroSimulation<Channel> sim(BCs_cc);
	sim.stopTime_ = max_time;

	// Set initial conditions
	sim.setInitialConditions();

	// run simulation
	sim.evolve();

	// Cleanup and exit
	int const status = 0;
	return status;
}
