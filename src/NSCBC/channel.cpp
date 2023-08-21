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

#include "EOS.hpp"
#include "RadhydroSimulation.hpp"
#include "channel.hpp"
#include "hydro_system.hpp"
#include "physics_info.hpp"
#include "physics_numVars.hpp"
#include "radiation_system.hpp"
#include "valarray.hpp"

using amrex::Real;

struct Channel {
}; // dummy type to allow compile-type polymorphism via template specialization

template <> struct quokka::EOS_Traits<Channel> {
	static constexpr double gamma = 5. / 3.; // default value
	static constexpr double mean_molecular_weight = C::m_p + C::m_e;
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

constexpr Real Tgas0 = 300;		       // K
constexpr Real nH0 = 1.0;		       // cm^-3
constexpr Real M0 = 2.0;		       // Mach number of shock
constexpr Real P0 = nH0 * Tgas0 * C::k_B;      // erg cm^-3
constexpr Real rho0 = nH0 * (C::m_p + C::m_e); // g cm^-3

// cloud-tracking variables needed for Dirichlet boundary condition
namespace
{
AMREX_GPU_MANAGED Real rho_wind = 0; // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
AMREX_GPU_MANAGED Real v_wind = 0;   // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
AMREX_GPU_MANAGED Real P_wind = 0;   // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
AMREX_GPU_MANAGED Real delta_vx = 0; // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
};				     // namespace

template <> void RadhydroSimulation<Channel>::setInitialConditionsOnGrid(quokka::grid grid_elem)
{
	// set initial conditions
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const dx = grid_elem.dx_;
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = grid_elem.prob_lo_;
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_hi = grid_elem.prob_hi_;
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const amrex::Array4<double> &state_cc = grid_elem.array_;

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

AMREX_GPU_DEVICE AMREX_FORCE_INLINE auto dQ_dx_outflow_x1_upper(quokka::valarray<Real, HydroSystem<Channel>::nvar_> const &Q, const Real p_t)
    -> quokka::valarray<Real, HydroSystem<Channel>::nvar_>
{
	// return dQ/dx
	const Real rho = Q[0];
	const Real u = Q[1];
	const Real v = Q[2];
	const Real w = Q[3];
	const Real p = Q[4];

	const Real drho_dx = dQ_dx_data[0];
	const Real du_dx = dQ_dx_data[1];
	const Real dv_dx = dQ_dx_data[2];
	const Real dp_dx = dQ_dx_data[4];

	// compute sub-expressions
	const Real c = quokka::EOS<Channel>::ComputeSoundSpeed(rho, p);
	const Real M = std::sqrt(u * u + v * v + w * w) / c;
	amrex::Real const K = 0.25 * c * (1 - M * M) / L_x;

	quokka::valarray<Real, HydroSystem<Channel>::nvar_> dQ_dx{};
	dQ_dx[0] = (1.0 / 2.0) * (-K * (p - p_t) + (c - u) * (2 * std::pow(c, 2) * drho_dx + c * du_dx * rho - dp_dx)) / (std::pow(c, 2) * (c - u));
	dQ_dx[1] = (1.0 / 2.0) * (K * (p - p_t) + (c - u) * (c * du_dx * rho + dp_dx)) / (c * rho * (c - u));
	dQ_dx[2] = dv_dx;
	dQ_dx[3] = NAN;
	dQ_dx[4] = 0.5 * (-K * (p - p_t) + (c - u) * (c * du_dx * rho + dp_dx)) / (c - u);

	return dQ_dx;
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
	constexpr int nvar = HydroSystem<Channel>::nvar_;
	constexpr Real gamma = quokka::EOS_Traits<Channel>::gamma;

	if (i < ilo) {
		// x1 lower boundary -- subsonic inflow
		const Real dx = geom.CellSize(0);

		// compute dQ/dx
		quokka::valarray<amrex::Real, nvar> dQ_dx = Compute_dQ_dx(Q);

		const Real rho_ip1 = consVar(ilo + 1, j, k, HydroSystem<Channel>::density_index);
		const Real x1mom_ip1 = consVar(ilo + 1, j, k, HydroSystem<Channel>::x1Momentum_index);
		const Real x2mom_ip1 = consVar(ilo + 1, j, k, HydroSystem<Channel>::x2Momentum_index);
		const Real x3mom_ip1 = consVar(ilo + 1, j, k, HydroSystem<Channel>::x3Momentum_index);
		const Real E_ip1 = consVar(ilo + 1, j, k, HydroSystem<Channel>::energy_index);
		const Real Eint_ip1 = E_ip1 - 0.5 * (x1mom_ip1 * x1mom_ip1 + x2mom_ip1 * x2mom_ip1 + x3mom_ip1 * x3mom_ip1) / rho_ip1;

		quokka::valarray<amrex::Real, nvar> Q_ip1{rho_ip1, x1mom_ip1 / rho_ip1, x2mom_ip1 / rho_ip1, x3mom_ip1 / rho_ip1, Eint_ip1 / (gamma - 1.)};
		quokka::valarray<amrex::Real, nvar> Q_im1 = Q_ip1 - 2.0 * dx * dQ_dx;

		Real const rho = Q_im1[0];
		Real const xmom = rho * Q_im1[1];
		Real const ymom = rho * Q_im1[2];
		Real const zmom = rho * Q_im1[3];
		Real const Eint = (gamma - 1.) * Q_im1[4];
		Real const Egas = Eint + 0.5 * (xmom * xmom + ymom * ymom + zmom * zmom) / rho;

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
	// Problem initialization
	constexpr int ncomp_cc = Physics_Indices<Channel>::nvarTotal_cc;
	amrex::Vector<amrex::BCRec> BCs_cc(ncomp_cc);
	for (int n = 0; n < ncomp_cc; ++n) {
		BCs_cc[n].setLo(0, amrex::BCType::ext_dir);  // Dirichlet
		BCs_cc[n].setHi(0, amrex::BCType::foextrap); // extrapolate

		if constexpr (AMREX_SPACEDIM >= 2) {
			BCs_cc[n].setLo(1, amrex::BCType::int_dir); // periodic
			BCs_cc[n].setHi(1, amrex::BCType::int_dir);
		} else if (AMREX_SPACEDIM == 3) {
			BCs_cc[n].setLo(2, amrex::BCType::int_dir);
			BCs_cc[n].setHi(2, amrex::BCType::int_dir);
		}
	}

	RadhydroSimulation<Channel> sim(BCs_cc);

	// Set initial conditions
	sim.setInitialConditions();

	// run simulation
	sim.evolve();

	// Cleanup and exit
	int const status = 0;
	return status;
}
