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
#include <tuple>
#include <vector>

#include "AMReX.H"
#include "AMReX_Array.H"
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
#include "AMReX_iMultiFab.H"

#include "ArrayUtil.hpp"
#include "EOS.hpp"
#include "HydroState.hpp"
#include "RadhydroSimulation.hpp"
#include "channel.hpp"
#include "fextract.hpp"
#include "fundamental_constants.H"
#include "hydro_system.hpp"
#include "physics_info.hpp"
#include "physics_numVars.hpp"
#include "radiation_system.hpp"
#include "valarray.hpp"
#ifdef HAVE_PYTHON
#include "matplotlibcpp.h"
#endif

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
	static constexpr int numPassiveScalars = numMassScalars + 1; // number of passive scalars
	static constexpr bool is_radiation_enabled = false;
};

// global variables needed for Dirichlet boundary condition and initial conditions
namespace
{
Real rho0 = NAN;						       // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
Real u0 = NAN;							       // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
Real s0 = NAN;							       // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
AMREX_GPU_MANAGED Real Tgas0 = NAN;				       // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
AMREX_GPU_MANAGED Real P_outflow = NAN;				       // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
AMREX_GPU_MANAGED Real u_inflow = NAN;				       // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
AMREX_GPU_MANAGED Real v_inflow = NAN;				       // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
AMREX_GPU_MANAGED Real w_inflow = NAN;				       // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
GpuArray<Real, Physics_Traits<Channel>::numPassiveScalars> s_inflow{}; // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
};								       // namespace

template <> void RadhydroSimulation<Channel>::setInitialConditionsOnGrid(quokka::grid grid_elem)
{
	// set initial conditions
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const dx = grid_elem.dx_;
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const prob_lo = grid_elem.prob_lo_;
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const prob_hi = grid_elem.prob_hi_;
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const amrex::Array4<double> &state_cc = grid_elem.array_;

	Real const rho = rho0;
	Real const xmom = rho0 * u0;
	Real const ymom = 0;
	Real const zmom = 0;
	Real const Eint = quokka::EOS<Channel>::ComputeEintFromTgas(rho0, Tgas0);
	Real const Egas = RadSystem<Channel>::ComputeEgasFromEint(rho, xmom, ymom, zmom, Eint);
	Real const scalar = s0;

	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		state_cc(i, j, k, HydroSystem<Channel>::density_index) = rho;
		state_cc(i, j, k, HydroSystem<Channel>::x1Momentum_index) = xmom;
		state_cc(i, j, k, HydroSystem<Channel>::x2Momentum_index) = ymom;
		state_cc(i, j, k, HydroSystem<Channel>::x3Momentum_index) = zmom;
		state_cc(i, j, k, HydroSystem<Channel>::energy_index) = Egas;
		state_cc(i, j, k, HydroSystem<Channel>::internalEnergy_index) = Eint;
		state_cc(i, j, k, HydroSystem<Channel>::scalar0_index) = scalar;
	});
}

template <int Nscalars>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE auto dQ_dx_outflow_x1_upper(quokka::valarray<Real, Physics_NumVars::numHydroVars + Nscalars> const &Q,
								quokka::valarray<Real, Physics_NumVars::numHydroVars + Nscalars> const &dQ_dx_data,
								const Real P_t, const Real L_x)
    -> quokka::valarray<Real, Physics_NumVars::numHydroVars + Nscalars>
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
	const Real dEint_aux_dx = dQ_dx_data[5];

	const Real c = quokka::EOS<Channel>::ComputeSoundSpeed(rho, P);
	const Real M = std::sqrt(u * u + v * v + w * w) / c;
	amrex::Real const K = 0.25 * c * (1 - M * M) / L_x; // must be non-zero for well-posed Euler equations

	// see SymPy notebook for derivation
	quokka::valarray<Real, Physics_NumVars::numHydroVars + Nscalars> dQ_dx{};
	dQ_dx[0] = 0.5 * (-K * (P - P_t) + (c - u) * (2.0 * c * c * drho_dx + c * du_dx * rho - dP_dx)) / (c * c * (c - u));
	dQ_dx[1] = 0.5 * (K * (P - P_t) + (c - u) * (c * du_dx * rho + dP_dx)) / (c * rho * (c - u));
	dQ_dx[2] = dv_dx;
	dQ_dx[3] = dw_dx;
	dQ_dx[4] = 0.5 * (-K * (P - P_t) + (c - u) * (c * du_dx * rho + dP_dx)) / (c - u);
	dQ_dx[5] = dEint_aux_dx;
	for (int i = 0; i < Nscalars; ++i) {
		dQ_dx[6 + i] = dQ_dx_data[6 + i];
	}

	return dQ_dx;
}

template <int Nscalars>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE auto dQ_dx_inflow_x1_lower(quokka::valarray<Real, Physics_NumVars::numHydroVars + Nscalars> const &Q,
							       quokka::valarray<Real, Physics_NumVars::numHydroVars + Nscalars> const &dQ_dx_data,
							       const Real T_t, const Real u_t, const Real v_t, const Real w_t,
							       amrex::GpuArray<Real, Nscalars> const &s_t, const Real L_x)
    -> quokka::valarray<Real, Physics_NumVars::numHydroVars + Nscalars>
{
	// return dQ/dx corresponding to subsonic inflow on the x1 lower boundary
	// (This is only necessary for continuous inflows, i.e., where as shock or contact discontinuity is not desired.)
	// NOTE: This boundary condition is only defined for an ideal gas (with constant k_B/mu).

	const Real rho = Q[0];
	const Real u = Q[1];
	const Real v = Q[2];
	const Real w = Q[3];
	const Real P = Q[4];
	const Real Eint_aux = Q[5];
	amrex::GpuArray<Real, Nscalars> s{};
	for (int i = 0; i < Nscalars; ++i) {
		s[i] = Q[6 + i];
	}

	const Real T = quokka::EOS<Channel>::ComputeTgasFromEint(rho, quokka::EOS<Channel>::ComputeEintFromPres(rho, P));
	const Real Eint_aux_t = quokka::EOS<Channel>::ComputeEintFromTgas(rho, T_t);

	const Real du_dx = dQ_dx_data[1];
	const Real dP_dx = dQ_dx_data[4];

	const Real c = quokka::EOS<Channel>::ComputeSoundSpeed(rho, P);
	const Real M = std::sqrt(u * u + v * v + w * w) / c;

	const Real eta_2 = 2.;
	const Real eta_3 = 2.;
	const Real eta_4 = 2.;
	const Real eta_5 = 2.;
	const Real eta_6 = 2.;

	const Real R = quokka::EOS_Traits<Channel>::boltzmann_constant / quokka::EOS_Traits<Channel>::mean_molecular_weight;

	// see SymPy notebook for derivation
	quokka::valarray<Real, Physics_NumVars::numHydroVars + Nscalars> dQ_dx{};
	if (u != 0.) {
		dQ_dx[0] = 0.5 *
			   (L_x * u * (c + u) * (-c * du_dx * rho + dP_dx) - 2 * R * c * eta_2 * rho * (c + u) * (T - T_t) -
			    std::pow(c, 2) * eta_5 * rho * u * (std::pow(M, 2) - 1) * (u - u_t)) /
			   (L_x * std::pow(c, 2) * u * (c + u));
		dQ_dx[1] = 0.5 * (L_x * (c + u) * (c * du_dx * rho - dP_dx) - std::pow(c, 2) * eta_5 * rho * (std::pow(M, 2) - 1) * (u - u_t)) /
			   (L_x * c * rho * (c + u));
		dQ_dx[2] = c * eta_3 * (v - v_t) / (L_x * u);
		dQ_dx[3] = c * eta_4 * (w - w_t) / (L_x * u);
		dQ_dx[4] =
		    0.5 * (L_x * (c + u) * (-c * du_dx * rho + dP_dx) - std::pow(c, 2) * eta_5 * rho * (std::pow(M, 2) - 1) * (u - u_t)) / (L_x * (c + u));
		dQ_dx[5] = c * eta_6 * (Eint_aux - Eint_aux_t) / (L_x * u);
		for (int i = 0; i < Nscalars; ++i) {
			dQ_dx[6 + i] = c * eta_6 * (s[i] - s_t[i]) / (L_x * u);
		}
	} else { // u == 0
		dQ_dx[0] = 0.5 * (L_x * c * (-c * du_dx * rho + dP_dx) + (c * c) * eta_5 * rho * u_t * ((M * M) - 1)) / (L_x * std::pow(c, 3));
		dQ_dx[1] = 0.5 * (L_x * c * (c * du_dx * rho - dP_dx) + (c * c) * eta_5 * rho * u_t * ((M * M) - 1)) / (L_x * (c * c) * rho);
		dQ_dx[2] = 0;
		dQ_dx[3] = 0;
		dQ_dx[4] = 0.5 * (L_x * c * (-c * du_dx * rho + dP_dx) + (c * c) * eta_5 * rho * u_t * ((M * M) - 1)) / (L_x * c);
		dQ_dx[5] = 0;
		for (int i = 0; i < Nscalars; ++i) {
			dQ_dx[6 + i] = 0;
		}
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
	const GpuArray<amrex::Real, HydroSystem<Channel>::nscalars_> s_inflow = ::s_inflow;

	constexpr int N = HydroSystem<Channel>::nvar_;

	if (i < ilo) {
		// x1 lower boundary -- subsonic inflow

		// compute one-sided dQ/dx from the data
		quokka::valarray<amrex::Real, N> const Q_i = HydroSystem<Channel>::ComputePrimVars(consVar, ilo, j, k);
		quokka::valarray<amrex::Real, N> const Q_ip1 = HydroSystem<Channel>::ComputePrimVars(consVar, ilo + 1, j, k);
		quokka::valarray<amrex::Real, N> const Q_ip2 = HydroSystem<Channel>::ComputePrimVars(consVar, ilo + 2, j, k);
		quokka::valarray<amrex::Real, N> const dQ_dx_data = (-3. * Q_i + 4. * Q_ip1 - Q_ip2) / (2. * dx);

		// compute dQ/dx with modified characteristics
		quokka::valarray<amrex::Real, N> const dQ_dx =
		    dQ_dx_inflow_x1_lower<HydroSystem<Channel>::nscalars_>(Q_i, dQ_dx_data, T_inflow, u_inflow, v_inflow, w_inflow, s_inflow, Lx);

		// compute centered ghost values
		quokka::valarray<amrex::Real, N> const Q_im1 = Q_ip1 - 2.0 * dx * dQ_dx;
		quokka::valarray<amrex::Real, N> const Q_im2 = -2.0 * Q_ip1 - 3.0 * Q_i + 6.0 * Q_im1 + 6.0 * dx * dQ_dx;
		quokka::valarray<amrex::Real, N> const Q_im3 = 3.0 * Q_ip1 + 10.0 * Q_i - 18.0 * Q_im1 + 6.0 * Q_im2 - 12.0 * dx * dQ_dx;
		quokka::valarray<amrex::Real, N> const Q_im4 = -2.0 * Q_ip1 - 13.0 * Q_i + 24.0 * Q_im1 - 12.0 * Q_im2 + 4.0 * Q_im3 + 12.0 * dx * dQ_dx;

		// set cell values
		quokka::valarray<amrex::Real, N> consCell{};
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
		consVar(i, j, k, HydroSystem<Channel>::internalEnergy_index) = consCell[5];
		for (int i = 0; i < HydroSystem<Channel>::nscalars_; ++i) {
			consVar(i, j, k, HydroSystem<Channel>::scalar0_index + i) = consCell[6 + i];
		}

	} else if (i > ihi) {
		// x1 upper boundary -- subsonic outflow

		// compute one-sided dQ/dx from the data
		quokka::valarray<amrex::Real, N> const Q_i = HydroSystem<Channel>::ComputePrimVars(consVar, ihi, j, k);
		quokka::valarray<amrex::Real, N> const Q_im1 = HydroSystem<Channel>::ComputePrimVars(consVar, ihi - 1, j, k);
		quokka::valarray<amrex::Real, N> const Q_im2 = HydroSystem<Channel>::ComputePrimVars(consVar, ihi - 2, j, k);
		quokka::valarray<amrex::Real, N> const dQ_dx_data = (Q_im2 - 4.0 * Q_im1 + 3.0 * Q_i) / (2.0 * dx);

		// compute dQ/dx with modified characteristics
		quokka::valarray<amrex::Real, N> const dQ_dx = dQ_dx_outflow_x1_upper<HydroSystem<Channel>::nscalars_>(Q_i, dQ_dx_data, P_outflow, Lx);

		// compute centered ghost values
		quokka::valarray<amrex::Real, N> const Q_ip1 = Q_im1 + 2.0 * dx * dQ_dx;
		quokka::valarray<amrex::Real, N> const Q_ip2 = -2.0 * Q_im1 - 3.0 * Q_i + 6.0 * Q_ip1 - 6.0 * dx * dQ_dx;
		quokka::valarray<amrex::Real, N> const Q_ip3 = 3.0 * Q_im1 + 10.0 * Q_i - 18.0 * Q_ip1 + 6.0 * Q_ip2 + 12.0 * dx * dQ_dx;
		quokka::valarray<amrex::Real, N> const Q_ip4 = -2.0 * Q_im1 - 13.0 * Q_i + 24.0 * Q_ip1 - 12.0 * Q_ip2 + 4.0 * Q_ip3 - 12.0 * dx * dQ_dx;

		// set cell values
		quokka::valarray<amrex::Real, N> consCell{};
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
		consVar(i, j, k, HydroSystem<Channel>::internalEnergy_index) = consCell[5];
		for (int i = 0; i < HydroSystem<Channel>::nscalars_; ++i) {
			consVar(i, j, k, HydroSystem<Channel>::scalar0_index + i) = consCell[6 + i];
		}
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

	amrex::ParmParse const pp("channel");
	// initial condition parameters
	pp.query("rho0", ::rho0);   // initial density [g/cc]
	pp.query("Tgas0", ::Tgas0); // initial temperature [K]
	pp.query("u0", ::u0);	    // initial velocity [cm/s]
	pp.query("s0", ::s0);	    // initial passive scalar [dimensionless]
	// boundary condition parameters
	pp.query("u_inflow", ::u_inflow);    // inflow velocity along x-axis [cm/s]
	pp.query("v_inflow", ::v_inflow);    // transverse inflow velocity (v_y) [cm/s]
	pp.query("w_inflow", ::w_inflow);    // transverse inflow velocity (v_z) [cm/s]
	pp.query("s_inflow", ::s_inflow[0]); // inflow passive scalar [dimensionless]

	// compute derived parameters
	const Real Eint0 = quokka::EOS<Channel>::ComputeEintFromTgas(rho0, Tgas0);
	::P_outflow = quokka::EOS<Channel>::ComputePressure(rho0, Eint0);
	amrex::Print() << "Derived outflow pressure is " << ::P_outflow << " erg/cc.\n";

	// Set initial conditions
	sim.setInitialConditions();

	// run simulation
	sim.evolve();

	// extract slice
	auto [position, values] = fextract(sim.state_new_cc_[0], sim.geom[0], 0, 0., true);
	int const nx = static_cast<int>(position.size());
	std::vector<double> const xs = position;
	std::vector<double> xs_exact = position;

	// extract solution
	std::vector<double> d(nx);
	std::vector<double> vx(nx);
	std::vector<double> P(nx);
	std::vector<double> s(nx);
	std::vector<double> density_exact(nx);
	std::vector<double> velocity_exact(nx);
	std::vector<double> Pexact(nx);
	std::vector<double> sexact(nx);

	for (int i = 0; i < nx; ++i) {
		{
			amrex::Real const rho = values.at(HydroSystem<Channel>::density_index)[i];
			amrex::Real const xmom = values.at(HydroSystem<Channel>::x1Momentum_index)[i];
			amrex::Real const Egas = values.at(HydroSystem<Channel>::energy_index)[i];
			amrex::Real const scalar = values.at(HydroSystem<Channel>::scalar0_index)[i];
			amrex::Real const Eint = Egas - (xmom * xmom) / (2.0 * rho);
			amrex::Real const gamma = quokka::EOS_Traits<Channel>::gamma;
			d.at(i) = rho;
			vx.at(i) = xmom / rho;
			P.at(i) = ((gamma - 1.0) * Eint);
			s.at(i) = scalar;
		}
		{
			density_exact.at(i) = rho0;
			velocity_exact.at(i) = u_inflow;
			Pexact.at(i) = P_outflow;
			sexact.at(i) = s_inflow[0];
		}
	}
	std::vector<std::vector<double>> const sol{d, vx, P, s};
	std::vector<std::vector<double>> const sol_exact{density_exact, velocity_exact, Pexact, sexact};

	// compute error norm
	amrex::Real err_sq = 0.;
	for (int n = 0; n < sol.size(); ++n) {
		amrex::Real dU_k = 0.;
		amrex::Real U_k = 0;
		for (int i = 0; i < nx; ++i) {
			// Δ Uk = ∑i |Uk,in - Uk,i0| / Nx
			const amrex::Real U_k0 = sol_exact.at(n)[i];
			const amrex::Real U_k1 = sol.at(n)[i];
			dU_k += std::abs(U_k1 - U_k0) / static_cast<double>(nx);
			U_k += std::abs(U_k0) / static_cast<double>(nx);
		}
		amrex::Print() << "dU_" << n << " = " << dU_k << " U_k = " << U_k << "\n";
		// ε = || Δ U / U || = [&sum_k (ΔU_k/U_k)^2]^{1/2}
		err_sq += std::pow(dU_k / U_k, 2);
	}
	const amrex::Real epsilon = std::sqrt(err_sq);
	amrex::Print() << "rms of component-wise relative L1 error norms = " << epsilon << "\n\n";

#ifdef HAVE_PYTHON
	if (amrex::ParallelDescriptor::IOProcessor()) {
		// Plot results
		const int skip = 4;	    // only plot every 8 elements of exact solution
		const double msize = 5.0; // marker size
		using mpl_arg = std::map<std::string, std::string>;
		using mpl_sarg = std::unordered_map<std::string, std::string>;

		matplotlibcpp::clf();
		mpl_arg d_args;
		mpl_sarg dexact_args;
		d_args["label"] = "density";
		d_args["color"] = "C0";
		dexact_args["marker"] = "o";
		dexact_args["color"] = "C0";
		matplotlibcpp::plot(xs, d, d_args);
		matplotlibcpp::scatter(strided_vector_from(xs_exact, skip), strided_vector_from(density_exact, skip), msize, dexact_args);
		matplotlibcpp::legend();
		matplotlibcpp::xlabel("length x");
		matplotlibcpp::tight_layout();
		matplotlibcpp::save("./channel_flow_density.pdf");

		matplotlibcpp::clf();
		std::map<std::string, std::string> vx_args;
		vx_args["label"] = "velocity";
		vx_args["color"] = "C3";
		matplotlibcpp::plot(xs, vx, vx_args);
		mpl_sarg vexact_args;
		vexact_args["marker"] = "o";
		vexact_args["color"] = "C3";
		matplotlibcpp::scatter(strided_vector_from(xs_exact, skip), strided_vector_from(velocity_exact, skip), msize, vexact_args);
		matplotlibcpp::legend();
		matplotlibcpp::xlabel("length x");
		matplotlibcpp::tight_layout();
		matplotlibcpp::save("./channel_flow_velocity.pdf");

		matplotlibcpp::clf();
		std::map<std::string, std::string> P_args;
		P_args["label"] = "pressure";
		P_args["color"] = "C4";
		matplotlibcpp::plot(xs, P, P_args);
		mpl_sarg Pexact_args;
		Pexact_args["marker"] = "o";
		Pexact_args["color"] = "C4";
		matplotlibcpp::scatter(strided_vector_from(xs_exact, skip), strided_vector_from(Pexact, skip), msize, Pexact_args);
		matplotlibcpp::legend();
		matplotlibcpp::xlabel("length x");
		matplotlibcpp::tight_layout();
		matplotlibcpp::save("./channel_flow_pressure.pdf");

		matplotlibcpp::clf();
		std::map<std::string, std::string> s_args;
		s_args["label"] = "passive scalar";
		s_args["color"] = "C4";
		matplotlibcpp::plot(xs, s, s_args);
		mpl_sarg sexact_args;
		sexact_args["marker"] = "o";
		sexact_args["color"] = "C4";
		matplotlibcpp::scatter(strided_vector_from(xs_exact, skip), strided_vector_from(sexact, skip), msize, sexact_args);
		matplotlibcpp::legend();
		matplotlibcpp::xlabel("length x");
		matplotlibcpp::tight_layout();
		matplotlibcpp::save("./channel_flow_scalar.pdf");
	}
#endif

	// Compute test success condition
	int status = 0;
	const double error_tol = 0.0007;
	if (epsilon > error_tol) {
		status = 1;
	}
	return status;
}
