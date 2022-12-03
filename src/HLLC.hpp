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
	// compute PVRS states (Toro 10.5.2)

	const double rho_bar = 0.5 * (sL.rho + sR.rho);
	const double cs_bar = 0.5 * (sL.cs + sR.cs);
	const double P_PVRS = 0.5 * (sL.P + sR.P) - 0.5 * (sR.u - sL.u) * (rho_bar * cs_bar);

	// choose P_star adaptively (Fig. 9.4 of Toro)

	const double P_min = std::min(sL.P, sR.P);
	const double P_max = std::max(sL.P, sR.P);
	const double Q = P_max / P_min;
	constexpr double Q_crit = 2.;
	double P_star = NAN;

	if (gamma == 1.0) {
		P_star = std::max(P_PVRS, 0.0); // only estimate compatible with gamma = 1
	} else {				// gamma > 1
		if ((Q < Q_crit) && (P_min < P_PVRS) && (P_PVRS < P_max)) {
			// use PVRS estimate
			P_star = P_PVRS;
		} else if (P_PVRS < P_min) {
			// compute two-rarefaction TRRS states (Toro 10.5.2, eq. 10.63)
			const double z = (gamma - 1.) / (2. * gamma);
			const double P_TRRS =
			    std::pow((sL.cs + sR.cs - 0.5 * (gamma - 1.) * (sR.u - sL.u)) / (sL.cs / std::pow(sL.P, z) + sR.cs / std::pow(sR.P, z)), (1. / z));
			P_star = P_TRRS;
		} else {
			// compute two-shock TSRS states (Toro 10.5.2, eq. 10.65)
			const double A_L = 2. / ((gamma + 1.) * sL.rho);
			const double A_R = 2. / ((gamma + 1.) * sR.rho);
			const double B_L = ((gamma - 1.) / (gamma + 1.)) * sL.P;
			const double B_R = ((gamma - 1.) / (gamma + 1.)) * sR.P;
			const double P_0 = std::max(P_PVRS, 0.0);
			const double G_L = std::sqrt(A_L / (P_0 + B_L));
			const double G_R = std::sqrt(A_R / (P_0 + B_R));
			const double delta_u = sR.u - sL.u;
			const double P_TSRS = (G_L * sL.P + G_R * sR.P - delta_u) / (G_L + G_R);
			P_star = P_TSRS;
		}
	}

	// compute wave speeds

	const double q_L = (P_star <= sL.P) ? 1.0 : std::sqrt(1.0 + ((gamma + 1.0) / (2.0 * gamma)) * ((P_star / sL.P) - 1.0));
	const double q_R = (P_star <= sR.P) ? 1.0 : std::sqrt(1.0 + ((gamma + 1.0) / (2.0 * gamma)) * ((P_star / sR.P) - 1.0));

	double S_L = sL.u - q_L * sL.cs;
	double S_R = sR.u + q_R * sR.cs;

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