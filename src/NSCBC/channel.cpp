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
constexpr Real P0 = nH0 * Tgas0 * C::k_B;      // erg cm^-3
constexpr Real rho0 = nH0 * (C::m_p + C::m_e); // g cm^-3

// global variables needed for Dirichlet boundary condition
namespace
{
AMREX_GPU_MANAGED Real u_inflow = 0; // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
AMREX_GPU_MANAGED Real v_inflow = 0; // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
AMREX_GPU_MANAGED Real w_inflow = 0; // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
AMREX_GPU_MANAGED Real P_inflow = 0; // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
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

AMREX_GPU_DEVICE AMREX_FORCE_INLINE auto dQ_dx_outflow_x1_upper(quokka::valarray<Real, 5> const &Q, quokka::valarray<Real, 5> const &dQ_dx_data, const Real P_t,
								const Real L_x) -> quokka::valarray<Real, 5>
{
	// return dQ/dx corresponding to subsonic outflow on the x1 upper boundary
	const Real rho = Q[0];
	const Real u = Q[1];
	const Real v = Q[2];
	const Real w = Q[3];
	const Real P = Q[4];

	const Real drho_dx = dQ_dx_data[0];
	const Real du_dx = dQ_dx_data[1];
	const Real dv_dx = dQ_dx_data[2];
	const Real dw_dx = dQ_dx_data[3];
	const Real dP_dx = dQ_dx_data[4];

	const Real c = quokka::EOS<Channel>::ComputeSoundSpeed(rho, P);
	const Real M = std::sqrt(u * u + v * v + w * w) / c;
	amrex::Real const K = 0.25 * c * (1 - M * M) / L_x;

	// see SymPy notebook for derivation
	quokka::valarray<Real, 5> dQ_dx{};
	dQ_dx[0] = (1.0 / 2.0) * (-K * (P - P_t) + (c - u) * (2.0 * std::pow(c, 2) * drho_dx + c * du_dx * rho - 1.0 * dP_dx)) / (std::pow(c, 2) * (c - u));
	dQ_dx[1] = (1.0 / 2.0) * (K * (P - P_t) + (c - u) * (c * du_dx * rho + dP_dx)) / (c * rho * (c - u));
	dQ_dx[2] = dv_dx;
	dQ_dx[3] = dw_dx;
	dQ_dx[4] = 0.5 * (-K * (P - P_t) + (c - u) * (c * du_dx * rho + dP_dx)) / (c - u);

	return dQ_dx;
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE auto dQ_dx_inflow_x1_lower(quokka::valarray<Real, 5> const &Q, quokka::valarray<Real, 5> const &dQ_dx_data, const Real P_t,
							       const Real u_t, const Real v_t, const Real w_t, const Real L_x) -> quokka::valarray<Real, 5>
{
	// return dQ/dx corresponding to subsonic inflow on the x1 lower boundary
	const Real rho = Q[0];
	const Real u = Q[1];
	const Real v = Q[2];
	const Real w = Q[3];
	const Real P = Q[4];

	const Real drho_dx = dQ_dx_data[0];
	const Real du_dx = dQ_dx_data[1];
	const Real dv_dx = dQ_dx_data[2];
	const Real dw_dx = dQ_dx_data[3];
	const Real dP_dx = dQ_dx_data[4];

	const Real c = quokka::EOS<Channel>::ComputeSoundSpeed(rho, P);
	const Real M = std::sqrt(u * u + v * v + w * w) / c;

	const Real eta_2 = -0.278;
	const Real eta_3 = 1.0;
	const Real eta_4 = 1.0;
	const Real eta_5 = 0.278;

	// see SymPy notebook for derivation
	quokka::valarray<Real, 5> dQ_dx{};
	dQ_dx[0] = (1.0 / 2.0) *
		   (L_x * u * (c + u) * (-1.0 * c * du_dx * rho + dP_dx) - std::pow(c, 2) * eta_5 * rho * u * (std::pow(M, 2) - 1) * (u - u_t) +
		    2 * c * eta_2 * (P - P_t) * (c + u)) /
		   (L_x * std::pow(c, 2) * u * (c + u));
	dQ_dx[1] = (1.0 / 2.0) * (L_x * (c + u) * (c * du_dx * rho - 1.0 * dP_dx) - std::pow(c, 2) * eta_5 * rho * (std::pow(M, 2) - 1) * (u - u_t)) /
		   (L_x * c * rho * (c + u));
	dQ_dx[2] = c * eta_3 * (v - v_t) / (L_x * u);
	dQ_dx[3] = c * eta_4 * (w - w_t) / (L_x * u);
	dQ_dx[4] = 0.5 * (L_x * (c + u) * (-c * du_dx * rho + dP_dx) - std::pow(c, 2) * eta_5 * rho * (std::pow(M, 2) - 1) * (u - u_t)) / (L_x * (c + u));

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
	const Real Lx = box.length(0);
	const auto &domain_lo = box.loVect3d();
	const auto &domain_hi = box.hiVect3d();
	const int ilo = domain_lo[0];

	constexpr Real gamma = quokka::EOS_Traits<Channel>::gamma;
	const Real P_inflow = ::P_inflow;
	const Real u_inflow = ::u_inflow;
	const Real v_inflow = ::v_inflow;
	const Real w_inflow = ::w_inflow;

	if (i < ilo) {
		// x1 lower boundary -- subsonic inflow
		const Real dx = geom.CellSize(0);

		// read in primitive vars
		const Real rho_ip1 = consVar(ilo + 1, j, k, HydroSystem<Channel>::density_index);
		const Real x1mom_ip1 = consVar(ilo + 1, j, k, HydroSystem<Channel>::x1Momentum_index);
		const Real x2mom_ip1 = consVar(ilo + 1, j, k, HydroSystem<Channel>::x2Momentum_index);
		const Real x3mom_ip1 = consVar(ilo + 1, j, k, HydroSystem<Channel>::x3Momentum_index);
		const Real E_ip1 = consVar(ilo + 1, j, k, HydroSystem<Channel>::energy_index);
		const Real Eint_ip1 = E_ip1 - 0.5 * (x1mom_ip1 * x1mom_ip1 + x2mom_ip1 * x2mom_ip1 + x3mom_ip1 * x3mom_ip1) / rho_ip1;

		// compute one-sided dQ/dx from the data
		quokka::valarray<amrex::Real, 5> Q_i{};
		quokka::valarray<amrex::Real, 5> Q_ip1{rho_ip1, x1mom_ip1 / rho_ip1, x2mom_ip1 / rho_ip1, x3mom_ip1 / rho_ip1, Eint_ip1 / (gamma - 1.)};
		quokka::valarray<amrex::Real, 5> Q_ip2{};
		quokka::valarray<amrex::Real, 5> dQ_dx_data = -3. * Q_i + 4. * Q_ip1 - Q_ip2 / (2. * dx);

		// compute dQ/dx with modified characteristics
		quokka::valarray<amrex::Real, 5> dQ_dx = dQ_dx_inflow_x1_lower(Q_i, dQ_dx_data, P_inflow, u_inflow, v_inflow, w_inflow, Lx);

		// compute centered ghost values
		quokka::valarray<amrex::Real, 5> Q_im1 = Q_ip1 - 2.0 * dx * dQ_dx;

		// convert to conserved vars
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
