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
#include "AMReX_ParmParse.H"
#include "AMReX_REAL.H"
#include "AMReX_SPACE.H"
#include "AMReX_TableData.H"
#include "AMReX_iMultiFab.H"

#include "EOS.hpp"
#include "HydroState.hpp"
#include "RadhydroSimulation.hpp"
#include "channel.hpp"
#include "fundamental_constants.H"
#include "hydro_system.hpp"
#include "physics_info.hpp"
#include "physics_numVars.hpp"
#include "radiation_system.hpp"
#include "valarray.hpp"

using amrex::Real;

struct Channel {
}; // dummy type to allow compile-type polymorphism via template specialization

template <> struct quokka::EOS_Traits<Channel> {
	static constexpr double gamma = 1.1;
	static constexpr double mean_molecular_weight = 28.96 * C::m_u; // air
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

// global variables needed for Dirichlet boundary condition and initial conditions
namespace
{
Real rho0 = NAN;			// NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
Real u0 = NAN;				// NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
AMREX_GPU_MANAGED Real Tgas0 = NAN;	// NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
AMREX_GPU_MANAGED Real P_outflow = NAN; // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
AMREX_GPU_MANAGED Real u_inflow = NAN;	// NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
AMREX_GPU_MANAGED Real v_inflow = NAN;	// NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
AMREX_GPU_MANAGED Real w_inflow = NAN;	// NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
};					// namespace

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
		Real const xmom = rho0 * u0;
		Real const ymom = 0;
		Real const zmom = 0;
		Real const Eint = quokka::EOS<Channel>::ComputeEintFromTgas(rho0, Tgas0);
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
	amrex::Real const K = 0.25 * c * (1 - M * M) / L_x; // must be non-zero for well-posed Euler equations

	// see SymPy notebook for derivation
	quokka::valarray<Real, 5> dQ_dx{};
	dQ_dx[0] = 0.5 * (-K * (P - P_t) + (c - u) * (2.0 * c * c * drho_dx + c * du_dx * rho - dP_dx)) / (c * c * (c - u));
	dQ_dx[1] = 0.5 * (K * (P - P_t) + (c - u) * (c * du_dx * rho + dP_dx)) / (c * rho * (c - u));
	dQ_dx[2] = dv_dx;
	dQ_dx[3] = dw_dx;
	dQ_dx[4] = 0.5 * (-K * (P - P_t) + (c - u) * (c * du_dx * rho + dP_dx)) / (c - u);

	return dQ_dx;
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE auto dQ_dx_inflow_x1_lower(quokka::valarray<Real, 5> const &Q, quokka::valarray<Real, 5> const &dQ_dx_data, const Real T_t,
							       const Real u_t, const Real v_t, const Real w_t, const Real L_x) -> quokka::valarray<Real, 5>
{
	// return dQ/dx corresponding to subsonic inflow on the x1 lower boundary
	// (This is only necessary for continuous inflows, i.e., where as shock or contact discontinuity is not desired.)
	// NOTE: This boundary condition is only defined for an ideal gas (with constant k_B/mu).

	const Real rho = Q[0];
	const Real u = Q[1];
	const Real v = Q[2];
	const Real w = Q[3];
	const Real P = Q[4];
	const Real Eint = quokka::EOS<Channel>::ComputeEintFromPres(rho, P);
	const Real T = quokka::EOS<Channel>::ComputeTgasFromEint(rho, Eint);

	const Real drho_dx = dQ_dx_data[0];
	const Real du_dx = dQ_dx_data[1];
	const Real dv_dx = dQ_dx_data[2];
	const Real dw_dx = dQ_dx_data[3];
	const Real dP_dx = dQ_dx_data[4];

	const Real c = quokka::EOS<Channel>::ComputeSoundSpeed(rho, P);
	const Real M = std::sqrt(u * u + v * v + w * w) / c;

	const Real eta_2 = 2.;
	const Real eta_3 = 2.;
	const Real eta_4 = 2.;
	const Real eta_5 = 2.;

	const Real R = quokka::EOS_Traits<Channel>::boltzmann_constant / quokka::EOS_Traits<Channel>::mean_molecular_weight;

	// see SymPy notebook for derivation
	quokka::valarray<Real, 5> dQ_dx{};
	if (u != 0.) {
		dQ_dx[0] = 0.5 *
			   (L_x * u * (c + u) * (-c * du_dx * rho + dP_dx) - (c * c) * eta_5 * rho * u * ((M * M) - 1) * (u - u_t) -
			    2 * c * eta_2 * rho * R * (T - T_t) * (c + u)) /
			   (L_x * (c * c) * u * (c + u));
		dQ_dx[1] = 0.5 * (L_x * (c + u) * (c * du_dx * rho - dP_dx) - (c * c) * eta_5 * rho * ((M * M) - 1) * (u - u_t)) / (L_x * c * rho * (c + u));
		dQ_dx[2] = c * eta_3 * (v - v_t) / (L_x * u);
		dQ_dx[3] = c * eta_4 * (w - w_t) / (L_x * u);
		dQ_dx[4] = 0.5 * (L_x * (c + u) * (-c * du_dx * rho + dP_dx) - (c * c) * eta_5 * rho * ((M * M) - 1) * (u - u_t)) / (L_x * (c + u));
	} else { // u == 0
		dQ_dx[0] = 0.5 * (L_x * c * (-c * du_dx * rho + dP_dx) + (c * c) * eta_5 * rho * u_t * ((M * M) - 1)) / (L_x * std::pow(c, 3));
		dQ_dx[1] = 0.5 * (L_x * c * (c * du_dx * rho - dP_dx) + (c * c) * eta_5 * rho * u_t * ((M * M) - 1)) / (L_x * (c * c) * rho);
		dQ_dx[2] = 0;
		dQ_dx[3] = 0;
		dQ_dx[4] = 0.5 * (L_x * c * (-c * du_dx * rho + dP_dx) + (c * c) * eta_5 * rho * u_t * ((M * M) - 1)) / (L_x * c);
	}

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
	const int ihi = domain_hi[0];
	const Real dx = geom.CellSize(0);
	const Real Lx = geom.prob_domain.length(0);

	const Real T_inflow = ::Tgas0;
	const Real u_inflow = ::u_inflow;
	const Real v_inflow = ::v_inflow;
	const Real w_inflow = ::w_inflow;

	const Real P_outflow = ::P_outflow;

	if (i < ilo) {
		// x1 lower boundary -- subsonic inflow

		// compute one-sided dQ/dx from the data
		quokka::valarray<amrex::Real, 5> const Q_i = HydroSystem<Channel>::ComputePrimVars(consVar, ilo, j, k);
		quokka::valarray<amrex::Real, 5> const Q_ip1 = HydroSystem<Channel>::ComputePrimVars(consVar, ilo + 1, j, k);
		quokka::valarray<amrex::Real, 5> const Q_ip2 = HydroSystem<Channel>::ComputePrimVars(consVar, ilo + 2, j, k);
		quokka::valarray<amrex::Real, 5> const dQ_dx_data = (-3. * Q_i + 4. * Q_ip1 - Q_ip2) / (2. * dx);

		// compute dQ/dx with modified characteristics
		quokka::valarray<amrex::Real, 5> const dQ_dx = dQ_dx_inflow_x1_lower(Q_i, dQ_dx_data, T_inflow, u_inflow, v_inflow, w_inflow, Lx);

		// compute centered ghost values
		quokka::valarray<amrex::Real, 5> const Q_im1 = Q_ip1 - 2.0 * dx * dQ_dx;
		quokka::valarray<amrex::Real, 5> const Q_im2 = -2.0 * Q_ip1 - 3.0 * Q_i + 6.0 * Q_im1 + 6.0 * dx * dQ_dx;
		quokka::valarray<amrex::Real, 5> const Q_im3 = 3.0 * Q_ip1 + 10.0 * Q_i - 18.0 * Q_im1 + 6.0 * Q_im2 - 12.0 * dx * dQ_dx;
		quokka::valarray<amrex::Real, 5> const Q_im4 = -2.0 * Q_ip1 - 13.0 * Q_i + 24.0 * Q_im1 - 12.0 * Q_im2 + 4.0 * Q_im3 + 12.0 * dx * dQ_dx;

		// set cell values
		quokka::valarray<amrex::Real, 5> consCell{};
		if (i == ilo - 1) {
			consCell = HydroSystem<Channel>::ComputeConsVars(Q_im1);
		} else if (i == ilo - 2) {
			consCell = HydroSystem<Channel>::ComputeConsVars(Q_im2);
		} else if (i == ilo - 3) {
			consCell = HydroSystem<Channel>::ComputeConsVars(Q_im3);
		} else if (i == ilo - 4) {
			consCell = HydroSystem<Channel>::ComputeConsVars(Q_im4);
		}
		consVar(i, j, k, HydroSystem<Channel>::density_index) = consCell[0];
		consVar(i, j, k, HydroSystem<Channel>::x1Momentum_index) = consCell[1];
		consVar(i, j, k, HydroSystem<Channel>::x2Momentum_index) = consCell[2];
		consVar(i, j, k, HydroSystem<Channel>::x3Momentum_index) = consCell[3];
		consVar(i, j, k, HydroSystem<Channel>::energy_index) = consCell[4];

	} else if (i > ihi) {
		// x1 upper boundary -- subsonic outflow

		// compute one-sided dQ/dx from the data
		quokka::valarray<amrex::Real, 5> const Q_i = HydroSystem<Channel>::ComputePrimVars(consVar, ihi, j, k);
		quokka::valarray<amrex::Real, 5> const Q_im1 = HydroSystem<Channel>::ComputePrimVars(consVar, ihi - 1, j, k);
		quokka::valarray<amrex::Real, 5> const Q_im2 = HydroSystem<Channel>::ComputePrimVars(consVar, ihi - 2, j, k);
		quokka::valarray<amrex::Real, 5> const dQ_dx_data = (Q_im2 - 4.0 * Q_im1 + 3.0 * Q_i) / (2.0 * dx);

		// compute dQ/dx with modified characteristics
		quokka::valarray<amrex::Real, 5> const dQ_dx = dQ_dx_outflow_x1_upper(Q_i, dQ_dx_data, P_outflow, Lx);

		// compute centered ghost values
		quokka::valarray<amrex::Real, 5> const Q_ip1 = Q_im1 + 2.0 * dx * dQ_dx;
		quokka::valarray<amrex::Real, 5> const Q_ip2 = -2.0 * Q_im1 - 3.0 * Q_i + 6.0 * Q_ip1 - 6.0 * dx * dQ_dx;
		quokka::valarray<amrex::Real, 5> const Q_ip3 = 3.0 * Q_im1 + 10.0 * Q_i - 18.0 * Q_ip1 + 6.0 * Q_ip2 + 12.0 * dx * dQ_dx;
		quokka::valarray<amrex::Real, 5> const Q_ip4 = -2.0 * Q_im1 - 13.0 * Q_i + 24.0 * Q_ip1 - 12.0 * Q_ip2 + 4.0 * Q_ip3 - 12.0 * dx * dQ_dx;

		// set cell values
		quokka::valarray<amrex::Real, 5> consCell{};
		if (i == ihi + 1) {
			consCell = HydroSystem<Channel>::ComputeConsVars(Q_ip1);
		} else if (i == ihi + 2) {
			consCell = HydroSystem<Channel>::ComputeConsVars(Q_ip2);
		} else if (i == ihi + 3) {
			consCell = HydroSystem<Channel>::ComputeConsVars(Q_ip3);
		} else if (i == ihi + 4) {
			consCell = HydroSystem<Channel>::ComputeConsVars(Q_ip4);
		}
		consVar(i, j, k, HydroSystem<Channel>::density_index) = consCell[0];
		consVar(i, j, k, HydroSystem<Channel>::x1Momentum_index) = consCell[1];
		consVar(i, j, k, HydroSystem<Channel>::x2Momentum_index) = consCell[2];
		consVar(i, j, k, HydroSystem<Channel>::x3Momentum_index) = consCell[3];
		consVar(i, j, k, HydroSystem<Channel>::energy_index) = consCell[4];
	}
}

auto problem_main() -> int
{
	// Problem initialization
	constexpr int ncomp_cc = Physics_Indices<Channel>::nvarTotal_cc;
	amrex::Vector<amrex::BCRec> BCs_cc(ncomp_cc);
	for (int n = 0; n < ncomp_cc; ++n) {
		BCs_cc[n].setLo(0, amrex::BCType::ext_dir); // NSCBC inflow
		BCs_cc[n].setHi(0, amrex::BCType::ext_dir); // NSCBC outflow

		if constexpr (AMREX_SPACEDIM >= 2) {
			BCs_cc[n].setLo(1, amrex::BCType::int_dir); // periodic
			BCs_cc[n].setHi(1, amrex::BCType::int_dir);
		} else if (AMREX_SPACEDIM == 3) {
			BCs_cc[n].setLo(2, amrex::BCType::int_dir);
			BCs_cc[n].setHi(2, amrex::BCType::int_dir);
		}
	}

	RadhydroSimulation<Channel> sim(BCs_cc);

	amrex::ParmParse pp("channel");
	// initial condition parameters
	pp.query("rho0", ::rho0);   // initial density [g/cc]
	pp.query("Tgas0", ::Tgas0); // initial temperature [K]
	pp.query("u0", ::u0);	    // initial velocity [cm/s]
	// boundary condition parameters
	pp.query("u_inflow", ::u_inflow); // inflow velocity along x-axis [cm/s]
	pp.query("v_inflow", ::v_inflow); // transverse inflow velocity (v_y) [cm/s]
	pp.query("w_inflow", ::w_inflow); // transverse inflow velocity (v_z) [cm/s]

	// compute derived parameters
	const Real Eint0 = quokka::EOS<Channel>::ComputeEintFromTgas(rho0, Tgas0);
	::P_outflow = quokka::EOS<Channel>::ComputePressure(rho0, Eint0);
	amrex::Print() << "Derived outflow pressure is " << ::P_outflow << " erg/cc.\n";

	// Set initial conditions
	sim.setInitialConditions();

	// run simulation
	sim.evolve();

	// Cleanup and exit
	int const status = 0;
	return status;
}
