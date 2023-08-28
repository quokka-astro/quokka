#ifndef NSCBC_OUTFLOW_HPP_ // NOLINT
#define NSCBC_OUTFLOW_HPP_
//==============================================================================
// Quokka -- two-moment radiation hydrodynamics on GPUs for astrophysics
// Copyright 2023 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file NSCBC_outflow.hpp
/// \brief Implements the Navier-Stokes Characteristic Boundary Condition for
/// subsonic outflows. (This also works trivially for _super_sonic outflows.)

#include "AMReX_GpuQualifiers.H"
#include "AMReX_REAL.H"
#include "EOS.hpp"
#include "hydro_system.hpp"
#include "physics_numVars.hpp"
#include "valarray.hpp"

namespace NSCBC
{
namespace detail
{
template <typename problem_t>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE auto dQ_dx_outflow_x1_upper(quokka::valarray<amrex::Real, HydroSystem<problem_t>::nvar_> const &Q,
								quokka::valarray<amrex::Real, HydroSystem<problem_t>::nvar_> const &dQ_dx_data,
								quokka::valarray<amrex::Real, HydroSystem<problem_t>::nvar_> const &dQ_dy_data,
								quokka::valarray<amrex::Real, HydroSystem<problem_t>::nvar_> const &dQ_dz_data,
								const amrex::Real P_t, const amrex::Real L_x)
    -> quokka::valarray<amrex::Real, HydroSystem<problem_t>::nvar_>
{
	// return dQ/dx corresponding to subsonic outflow on the x1 upper boundary

	const amrex::Real rho = Q[0];
	const amrex::Real u = Q[1];
	const amrex::Real v = Q[2];
	const amrex::Real w = Q[3];
	const amrex::Real P = Q[4];

	// normal derivatives
	const amrex::Real drho_dx = dQ_dx_data[0];
	const amrex::Real du_dx = dQ_dx_data[1];
	const amrex::Real dv_dx = dQ_dx_data[2];
	const amrex::Real dw_dx = dQ_dx_data[3];
	const amrex::Real dP_dx = dQ_dx_data[4];
	const amrex::Real dEint_aux_dx = dQ_dx_data[5];

	// x2 transverse derivatives
	const amrex::Real du_dy = dQ_dy_data[1];
	const amrex::Real dv_dy = dQ_dy_data[2];
	const amrex::Real dP_dy = dQ_dy_data[4];

	// x3 transverse derivatives
	const amrex::Real du_dz = dQ_dz_data[1];
	const amrex::Real dw_dz = dQ_dz_data[3];
	const amrex::Real dP_dz = dQ_dz_data[4];

	const amrex::Real c = quokka::EOS<problem_t>::ComputeSoundSpeed(rho, P);
	const amrex::Real M = std::sqrt(u * u + v * v + w * w) / c;
	const amrex::Real beta = M;
	const amrex::Real K = 0.25 * c * (1 - M * M) / L_x; // must be non-zero for well-posedness

	// see SymPy notebook for derivation of dQ_dx
	// (common subexpressions)
	const amrex::Real x0 = std::pow(c, 2);
	const amrex::Real x1 = c - u;
	const amrex::Real x2 = c * rho;
	const amrex::Real x3 = du_dx * x2;
	const amrex::Real x4 = rho * x0;
	const amrex::Real x5 = K * (P - P_t) + (beta - 1) * (dP_dy * v + dP_dz * w - du_dy * v * x2 - du_dz * w * x2 + dv_dy * x4 + dw_dz * x4);
	const amrex::Real x6 = (1.0 / 2.0) / x1;
	const amrex::Real x7 = dP_dx + x3;

	quokka::valarray<amrex::Real, HydroSystem<problem_t>::nvar_> dQ_dx{};
	dQ_dx[0] = x6 * (x1 * (-dP_dx + 2 * drho_dx * x0 + x3) - x5) / x0;
	dQ_dx[1] = x6 * (x1 * x7 + x5) / (c * rho);
	dQ_dx[2] = dv_dx;
	dQ_dx[3] = dw_dx;
	dQ_dx[4] = x6 * (x1 * x7 - x5);
	dQ_dx[5] = dEint_aux_dx;
	for (int i = 0; i < HydroSystem<problem_t>::nscalars_; ++i) {
		dQ_dx[6 + i] = dQ_dx_data[6 + i];
	}

	return dQ_dx;
}
} // namespace detail

template <typename problem_t>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE void setOutflowX1Upper(const amrex::IntVect &iv, amrex::Array4<amrex::Real> const &consVar, amrex::GeometryData const &geom,
							   const amrex::Real P_outflow)
{
	// x1 upper boundary -- subsonic outflow
	auto [i, j, k] = iv.dim3();
	amrex::Box const &box = geom.Domain();
	const auto &domain_hi = box.hiVect3d();
	const int ihi = domain_hi[0];
	const Real dx = geom.CellSize(0);
	const Real Lx = geom.prob_domain.length(0);
	constexpr int N = HydroSystem<problem_t>::nvar_;

	// compute one-sided dQ/dx from the data
	quokka::valarray<amrex::Real, N> const Q_i = HydroSystem<problem_t>::ComputePrimVars(consVar, ihi, j, k);
	quokka::valarray<amrex::Real, N> const Q_im1 = HydroSystem<problem_t>::ComputePrimVars(consVar, ihi - 1, j, k);
	quokka::valarray<amrex::Real, N> const Q_im2 = HydroSystem<problem_t>::ComputePrimVars(consVar, ihi - 2, j, k);
	quokka::valarray<amrex::Real, N> const dQ_dx_data = (Q_im2 - 4.0 * Q_im1 + 3.0 * Q_i) / (2.0 * dx);

	// compute two-sided dQ/dy from the data
	quokka::valarray<amrex::Real, N> Q_jp1{};
	quokka::valarray<amrex::Real, N> Q_jm1{};
	quokka::valarray<amrex::Real, N> dQ_dy_data{};

	if constexpr (AMREX_SPACEDIM >= 2) {
		const Real dy = geom.CellSize(1);
		Q_jp1 = HydroSystem<problem_t>::ComputePrimVars(consVar, ihi, j + 1, k);
		Q_jm1 = HydroSystem<problem_t>::ComputePrimVars(consVar, ihi, j - 1, k);
		dQ_dy_data = (Q_jp1 - Q_jm1) / (2.0 * dy);
	}

	// compute two-sided dQ/dz from the data
	quokka::valarray<amrex::Real, N> Q_kp1{};
	quokka::valarray<amrex::Real, N> Q_km1{};
	quokka::valarray<amrex::Real, N> dQ_dz_data{};

	if constexpr (AMREX_SPACEDIM == 3) {
		const Real dz = geom.CellSize(2);
		Q_kp1 = HydroSystem<problem_t>::ComputePrimVars(consVar, ihi, j, k + 1);
		Q_km1 = HydroSystem<problem_t>::ComputePrimVars(consVar, ihi, j, k - 1);
		dQ_dz_data = (Q_kp1 - Q_km1) / (2.0 * dz);
	}

	// compute dQ/dx with modified characteristics
	quokka::valarray<amrex::Real, N> const dQ_dx = detail::dQ_dx_outflow_x1_upper<problem_t>(Q_i, dQ_dx_data, dQ_dy_data, dQ_dz_data, P_outflow, Lx);

	// compute centered ghost values
	quokka::valarray<amrex::Real, N> const Q_ip1 = Q_im1 + 2.0 * dx * dQ_dx;
	quokka::valarray<amrex::Real, N> const Q_ip2 = -2.0 * Q_im1 - 3.0 * Q_i + 6.0 * Q_ip1 - 6.0 * dx * dQ_dx;
	quokka::valarray<amrex::Real, N> const Q_ip3 = 3.0 * Q_im1 + 10.0 * Q_i - 18.0 * Q_ip1 + 6.0 * Q_ip2 + 12.0 * dx * dQ_dx;
	quokka::valarray<amrex::Real, N> const Q_ip4 = -2.0 * Q_im1 - 13.0 * Q_i + 24.0 * Q_ip1 - 12.0 * Q_ip2 + 4.0 * Q_ip3 - 12.0 * dx * dQ_dx;

	// set cell values
	quokka::valarray<amrex::Real, N> consCell{};
	if (i == ihi + 1) {
		consCell = HydroSystem<problem_t>::ComputeConsVars(Q_ip1);
	} else if (i == ihi + 2) {
		consCell = HydroSystem<problem_t>::ComputeConsVars(Q_ip2);
	} else if (i == ihi + 3) {
		consCell = HydroSystem<problem_t>::ComputeConsVars(Q_ip3);
	} else if (i == ihi + 4) {
		consCell = HydroSystem<problem_t>::ComputeConsVars(Q_ip4);
	}

	consVar(i, j, k, HydroSystem<problem_t>::density_index) = consCell[0];
	consVar(i, j, k, HydroSystem<problem_t>::x1Momentum_index) = consCell[1];
	consVar(i, j, k, HydroSystem<problem_t>::x2Momentum_index) = consCell[2];
	consVar(i, j, k, HydroSystem<problem_t>::x3Momentum_index) = consCell[3];
	consVar(i, j, k, HydroSystem<problem_t>::energy_index) = consCell[4];
	consVar(i, j, k, HydroSystem<problem_t>::internalEnergy_index) = consCell[5];
	for (int i = 0; i < HydroSystem<problem_t>::nscalars_; ++i) {
		consVar(i, j, k, HydroSystem<problem_t>::scalar0_index + i) = consCell[6 + i];
	}
}
} // namespace NSCBC

#endif // NSCBC_OUTFLOW_HPP_