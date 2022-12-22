#ifndef HLLC_HPP_ // NOLINT
#define HLLC_HPP_

#include "AMReX_Extension.H"
#include "AMReX_GpuQualifiers.H"
#include <AMReX.H>
#include <AMReX_REAL.H>

#include "ArrayView.hpp"
#include "HydroState.hpp"
#include "valarray.hpp"

namespace quokka::Riemann
{
// HLLC solver following Toro (1998) and Balsara (2017).
// [Carbuncle correction:
//  Minoshima & Miyoshi, "A low-dissipation HLLD approximate Riemann solver
//  	for a very wide range of Mach numbers," JCP (2021).]
//
template <int N_scalars, int fluxdim>
AMREX_FORCE_INLINE AMREX_GPU_DEVICE auto HLLC(quokka::HydroState<N_scalars> const &sL, quokka::HydroState<N_scalars> const &sR, const double gamma,
					      const double du, const double dw) -> quokka::valarray<double, fluxdim>
{
	// compute Roe averages
	double wl = std::sqrt(sL.rho);
	double wr = std::sqrt(sR.rho);
	double norm = 1. / (wl + wr);
	double u_tilde = (wl * sL.u + wr * sR.u) * norm;
	double v_tilde = (wl * sL.v + wr * sR.v) * norm;
	double w_tilde = (wl * sL.w + wr * sR.w) * norm;
	double vsq_tilde = u_tilde * u_tilde + v_tilde * v_tilde + w_tilde * w_tilde;
	double H_L = (sL.E + sL.P) / sL.rho; // sL specific enthalpy
	double H_R = (sR.E + sR.P) / sR.rho; // sR specific enthalpy
	double H_tilde = (wl * H_L + wr * H_R) * norm;
	double cs_tilde = NAN;
	if (gamma != 1.0) {
		cs_tilde = (gamma - 1.) * (H_tilde - 0.5 * vsq_tilde);
	} else {
		cs_tilde = 0.5 * (sL.cs + sR.cs);
	}

	// compute wave speeds following Batten et al. (1997)

	const double S_L = std::min(sL.u - sL.cs, u_tilde - cs_tilde);
	const double S_R = std::max(sR.u + sR.cs, u_tilde + cs_tilde);

	// carbuncle correction [Eq. 10 of Minoshima & Miyoshi (2021)]

	const double cs_max = std::max(sL.cs, sR.cs);
	const double tp = std::min(1., (cs_max - std::min(du, 0.)) / (cs_max - std::min(dw, 0.)));
	const double theta = tp * tp * tp * tp;

	// compute speed of the 'star' state

	const double S_star =
	    (theta * (sR.P - sL.P) + (sL.rho * sL.u * (S_L - sL.u) - sR.rho * sR.u * (S_R - sR.u))) / (sL.rho * (S_L - sL.u) - sR.rho * (S_R - sR.u));

	// Low-dissipation pressure correction 'phi' [Eq. 23 of Minoshima & Miyoshi]

	const double vmag_L = std::sqrt(sL.u * sL.u + sL.v * sL.v + sL.w * sL.w);
	const double vmag_R = std::sqrt(sR.u * sR.u + sR.v * sR.v + sR.w * sR.w);
	const double chi = std::min(1., std::max(vmag_L, vmag_R) / cs_max);
	const double phi = chi * (2. - chi);

	const double P_LR = 0.5 * (sL.P + sR.P) + 0.5 * phi * (sL.rho * (S_L - sL.u) * (S_star - sL.u) + sR.rho * (S_R - sR.u) * (S_star - sR.u));

	/// compute fluxes

	// N.B.: quokka::valarray is written to allow assigning <= fluxdim
	// components, so this works even if there are more components than
	// enumerated in the initializer list. The remaining components are
	// assigned a default value of zero.

	const quokka::valarray<double, fluxdim> D_L = {0., 1., 0., 0., sL.u, 0.};
	const quokka::valarray<double, fluxdim> D_R = {0., 1., 0., 0., sR.u, 0.};
	const quokka::valarray<double, fluxdim> D_star = {0., 1., 0., 0., S_star, 0.};

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

	const quokka::valarray<double, fluxdim> F_L = sL.u * U_L + sL.P * D_L;
	const quokka::valarray<double, fluxdim> F_R = sR.u * U_R + sR.P * D_R;

	const quokka::valarray<double, fluxdim> F_starL = (S_star * (S_L * U_L - F_L) + S_L * P_LR * D_star) / (S_L - S_star);
	const quokka::valarray<double, fluxdim> F_starR = (S_star * (S_R * U_R - F_R) + S_R * P_LR * D_star) / (S_R - S_star);

	// open the Riemann fan
	quokka::valarray<double, fluxdim> F{};

	// HLLC flux
	if (S_L > 0.0) {
		F = F_L;
	} else if ((S_star > 0.0) && (S_L <= 0.0)) {
		F = F_starL;
	} else if ((S_star <= 0.0) && (S_R >= 0.0)) {
		F = F_starR;
	} else { // S_R < 0.0
		F = F_R;
	}

	return F;
}
} // namespace quokka::Riemann

#endif // HLLC_HPP_