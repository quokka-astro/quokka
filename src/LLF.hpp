#ifndef LLF_HPP_ // NOLINT
#define LLF_HPP_

#include "AMReX_Extension.H"
#include "AMReX_GpuQualifiers.H"
#include <AMReX.H>
#include <AMReX_REAL.H>

#include "ArrayView.hpp"
#include "HydroState.hpp"
#include "valarray.hpp"

namespace quokka::Riemann
{
// LLF (Local Lax-Friedrichs) / Rusanov solver.
//
template <int N_scalars, int fluxdim>
AMREX_FORCE_INLINE AMREX_GPU_DEVICE auto LLF(quokka::HydroState<N_scalars> const &sL, quokka::HydroState<N_scalars> const &sR, const double /*gamma*/,
					     const double /*du*/, const double /*dw*/) -> quokka::valarray<double, fluxdim>
{

	// compute wave speed
	const double S_max = std::max(std::abs(sL.u) + sL.cs, std::abs(sR.u) + sR.cs);

	quokka::valarray<double, fluxdim> U_L = {sL.rho, sL.rho * sL.u, sL.rho * sL.v, sL.rho * sL.w, sL.E, sL.Eint};
	quokka::valarray<double, fluxdim> U_R = {sR.rho, sR.rho * sR.u, sR.rho * sR.v, sR.rho * sR.w, sR.E, sR.Eint};

	// The remaining components are passive scalars, so just copy them from
	// x1LeftState and x1RightState into the (left, right) state vectors U_L and
	// U_R
	for (int n = 0; n < N_scalars; ++n) {
		const int nstart = fluxdim - N_scalars;
		U_L[nstart + n] = sL.scalar[n];
		U_R[nstart + n] = sR.scalar[n];
	}

	const quokka::valarray<double, fluxdim> D_L = {0., 1., 0., 0., sL.u, 0.};
	const quokka::valarray<double, fluxdim> D_R = {0., 1., 0., 0., sR.u, 0.};

	const quokka::valarray<double, fluxdim> F_L = sL.u * U_L + sL.P * D_L;
	const quokka::valarray<double, fluxdim> F_R = sR.u * U_R + sR.P * D_R;

	quokka::valarray<double, fluxdim> F = 0.5 * (F_L + F_R - S_max * (U_R - U_L));

	return F;
}
} // namespace quokka::Riemann

#endif // HLLC_HPP_