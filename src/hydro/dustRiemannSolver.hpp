#ifndef DUSTRIEMANNSOLVER_HPP_ // NOLINT
#define DUSTRIEMANNSOLVER_HPP_

#include "AMReX_Extension.H"
#include "AMReX_GpuQualifiers.H"
#include <AMReX.H>
#include <AMReX_REAL.H>

#include "hydro/EOS.hpp"
#include "hydro/HydroState.hpp"
#include "util/ArrayView.hpp"
#include "util/valarray.hpp"

namespace quokka::Riemann
{
// dust Riemann solver following Huang & Bai (2022).
template <typename problem_t, int fluxdim>
AMREX_FORCE_INLINE AMREX_GPU_DEVICE auto dustRiemannSolver(quokka::DustState const &sL, quokka::DustState const &sR) -> quokka::valarray<double, fluxdim>
{
	quokka::valarray<double, fluxdim> F{};

	if (sL.u > 0.0 && sR.u > 0.0) {
		F[0] = sL.rho * sL.u;
		F[1] = sL.rho * sL.u * sL.u;
		F[2] = sL.rho * sL.u * sL.v;
		F[3] = sL.rho * sL.u * sL.w;
	} else if (sL.u < 0.0 && sR.u < 0.0) {
		F[0] = sR.rho * sR.u;
		F[1] = sR.rho * sR.u * sR.u;
		F[2] = sR.rho * sR.u * sR.v;
		F[3] = sR.rho * sR.u * sR.w;
	} else if (sL.u > 0.0 && sR.u < 0.0) {
		F[0] = sL.rho * sL.u + sR.rho * sR.u;
		F[1] = sL.rho * sL.u * sL.u + sR.rho * sR.u * sR.u;
		F[2] = sL.rho * sL.u * sL.v + sR.rho * sR.u * sR.v;
		F[3] = sL.rho * sL.u * sL.w + sR.rho * sR.u * sR.w;
	} else {
	}

	return F;
}
} // namespace quokka::Riemann

#endif // DUSTRIEMANNSOLVER_HPP_