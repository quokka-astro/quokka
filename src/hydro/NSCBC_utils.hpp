#ifndef NSCBC_UTILS_HPP_ // NOLINT
#define NSCBC_UTILS_HPP_
//==============================================================================
// Quokka -- two-moment radiation hydrodynamics on GPUs for astrophysics
// Copyright 2024.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file NSCBC_utils.hpp
/// \brief Shared helpers for NSCBC boundary condition implementations.

#include "hydro/hydro_system.hpp"

namespace NSCBC
{
namespace detail
{
template <typename problem_t, SlopeLimiter limiter = SlopeLimiter::minmod>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE auto limit_normal_gradient(quokka::valarray<amrex::Real, HydroSystem<problem_t>::nvar_> const &high_order_grad,
							       quokka::valarray<amrex::Real, HydroSystem<problem_t>::nvar_> const &QL,
							       quokka::valarray<amrex::Real, HydroSystem<problem_t>::nvar_> const &QC,
							       quokka::valarray<amrex::Real, HydroSystem<problem_t>::nvar_> const &QR, const amrex::Real dx)
    -> quokka::valarray<amrex::Real, HydroSystem<problem_t>::nvar_>
{
	quokka::valarray<amrex::Real, HydroSystem<problem_t>::nvar_> grad{};
	for (int n = 0; n < HydroSystem<problem_t>::nvar_; ++n) {
		const amrex::Real slopeL = (QC[n] - QL[n]) / dx;
		const amrex::Real slopeR = (QR[n] - QC[n]) / dx;
		const amrex::Real plm_slope = HyperbolicSystem<problem_t>::template SlopeFunc<limiter>(slopeL, slopeR);
		grad[n] = HyperbolicSystem<problem_t>::template SlopeFunc<limiter>(high_order_grad[n], plm_slope);
	}
	return grad;
}

template <typename problem_t, SlopeLimiter limiter = SlopeLimiter::minmod>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE auto centered_limited_gradient(quokka::valarray<amrex::Real, HydroSystem<problem_t>::nvar_> const &Qm,
								   quokka::valarray<amrex::Real, HydroSystem<problem_t>::nvar_> const &Qc,
								   quokka::valarray<amrex::Real, HydroSystem<problem_t>::nvar_> const &Qp, const amrex::Real dx)
    -> quokka::valarray<amrex::Real, HydroSystem<problem_t>::nvar_>
{
	quokka::valarray<amrex::Real, HydroSystem<problem_t>::nvar_> grad{};
	for (int n = 0; n < HydroSystem<problem_t>::nvar_; ++n) {
		const amrex::Real slopeL = (Qc[n] - Qm[n]) / dx;
		const amrex::Real slopeR = (Qp[n] - Qc[n]) / dx;
		grad[n] = HyperbolicSystem<problem_t>::template SlopeFunc<limiter>(slopeL, slopeR);
	}
	return grad;
}
} // namespace detail
} // namespace NSCBC

#endif // NSCBC_UTILS_HPP_
