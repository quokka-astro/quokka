#ifndef HLLC_HPP_ // NOLINT
#define HLLC_HPP_

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
// HLLC solver following Toro (1998) and Balsara (2017).
// [Carbuncle correction:
//  Minoshima & Miyoshi, "A low-dissipation HLLD approximate Riemann solver
//  	for a very wide range of Mach numbers," JCP (2021).]
//
template <typename problem_t, int N_scalars, int N_mscalars, int fluxdim>
AMREX_FORCE_INLINE AMREX_GPU_DEVICE auto HLLC(quokka::HydroState<N_scalars, N_mscalars> const &sL, quokka::HydroState<N_scalars, N_mscalars> const &sR,
					      const double gamma, const double du, const double dw) -> quokka::valarray<double, fluxdim>
{
	// compute Roe averages

	const double wl = std::sqrt(sL.rho);
	const double wr = std::sqrt(sR.rho);
	const double norm = 1. / (wl + wr);
	const double u_tilde = (wl * sL.u + wr * sR.u) * norm;
	const double v_tilde = (wl * sL.v + wr * sR.v) * norm;
	const double w_tilde = (wl * sL.w + wr * sR.w) * norm;
	const double vsq_tilde = u_tilde * u_tilde + v_tilde * v_tilde + w_tilde * w_tilde;
	const double H_L = (sL.E + sL.P) / sL.rho; // sL specific enthalpy
	const double H_R = (sR.E + sR.P) / sR.rho; // sR specific enthalpy
	const double H_tilde = (wl * H_L + wr * H_R) * norm;
	double cs_tilde = NAN;

	// compute nonlinear wavespeed correction [Rider, Computers & Fluids 28 (1999) 741-777]
	// (Only applicable in compressions. Significantly reduces slow-moving shock oscillations.)

	const double dU = sL.u - sR.u;
	double S_L = NAN;
	double S_R = NAN;
	if (gamma != 1.0) {

		auto [dedr_L, dedp_L, drdp_L, dpdr_s_L, G_L] = ::quokka::EOS<problem_t>::ComputeOtherDerivatives(sL.rho, sL.P, sL.massScalar);
		auto [dedr_R, dedp_R, drdp_R, dpdr_s_R, G_R] = ::quokka::EOS<problem_t>::ComputeOtherDerivatives(sR.rho, sR.P, sR.massScalar);

		// equation A.5a of Kershaw+1998
		// need specific internal energy here
		const double C_tilde_rho = 0.5 * ((sL.Eint / sL.rho) + (sR.Eint / sR.rho) + sL.rho * dedr_L + sR.rho * dedr_R);

		// equation A.5b of Kershaw+1998
		const double C_tilde_P = 0.5 * ((sL.Eint / sL.rho) * drdp_L + (sR.Eint / sR.rho) * drdp_R + sL.rho * dedp_L + sR.rho * dedp_R);

		// avoid NAN cs_tilde
		const double cs_exp = H_tilde - 0.5 * vsq_tilde - C_tilde_rho;
		if (cs_exp <= 0) {
			// take the average of the left and right states
			cs_tilde = 0.5 * (sL.cs + sR.cs);
		} else {
			// equation 4.12 of Kershaw+1998
			cs_tilde = std::sqrt(cs_exp / C_tilde_P);
		}

		const double s_NL = 0.5 * G_L * std::max(dU, 0.); // second-order wavespeed correction
		const double s_NR = 0.5 * G_R * std::max(dU, 0.);

		// compute wave speeds following Batten et al. (1997)
		S_L = std::min(sL.u - (sL.cs + s_NL), u_tilde - (cs_tilde + s_NL));
		S_R = std::max(sR.u + (sR.cs + s_NR), u_tilde + (cs_tilde + s_NR));

	} else {
		cs_tilde = 0.5 * (sL.cs + sR.cs);
		const double G_gamma_L = 1.0;
		const double G_gamma_R = 1.0;

		const double G_L = 0.5 * (G_gamma_L + 1.);
		const double G_R = 0.5 * (G_gamma_R + 1.);

		const double s_NL = 0.5 * G_L * std::max(dU, 0.);
		const double s_NR = 0.5 * G_R * std::max(dU, 0.);

		S_L = std::min(sL.u - (sL.cs + s_NL), u_tilde - (cs_tilde + s_NL));
		S_R = std::max(sR.u + (sR.cs + s_NR), u_tilde + (cs_tilde + s_NR));
	}

	// Use the Minoshima & Miyoshi (2021), Eq. 10 shock sensor to blend
	// toward the more diffusive two-wave HLL flux below. Do not suppress the
	// normal pressure jump in S_star: transverse compression can also occur
	// next to a genuine pressure-driven flow.
	const double cs_max = std::max(sL.cs, sR.cs);
	const double tp = std::min(1., (cs_max - std::min(du, 0.)) / (cs_max - std::min(dw, 0.)));
	const double theta = tp * tp * tp * tp;

	// compute speed of the 'star' state with the full pressure jump
	const double S_star = ((sR.P - sL.P) + (sL.rho * sL.u * (S_L - sL.u) - sR.rho * sR.u * (S_R - sR.u))) / (sL.rho * (S_L - sL.u) - sR.rho * (S_R - sR.u));

	// Low-dissipation pressure correction 'phi' [Eq. 23 of Minoshima & Miyoshi]

	const double vmag_L = std::sqrt(sL.u * sL.u + sL.v * sL.v + sL.w * sL.w);
	const double vmag_R = std::sqrt(sR.u * sR.u + sR.v * sR.v + sR.w * sR.w);
	const double chi = std::min(1., std::max(vmag_L, vmag_R) / cs_max);
	const double phi = chi * (2. - chi);

	const double P_LR = 0.5 * (sL.P + sR.P) + 0.5 * phi * (sL.rho * (S_L - sL.u) * (S_star - sL.u) + sR.rho * (S_R - sR.u) * (S_star - sR.u));

	/// compute fluxes
	//
	// Every flux below (F_L, F_R, F_starL, F_starR, F_HLL and the theta blend)
	// is a linear combination
	//     F = wL * U_L + wR * U_R + pMom * e_x1mom + pEn * e_energy,
	// where U = (rho, rho u, rho v, rho w, E, Eint, scalars...). Compute the
	// scalar weights for the selected region only, with one reciprocal per
	// wave-speed difference, instead of building full flux vectors and dividing
	// them component by component. This is algebraically identical to
	//   F_L     = u_L U_L + P_L D_L
	//   F_starL = (S_star (S_L U_L - F_L) + S_L P_LR D_star) / (S_L - S_star)
	//   F_HLL   = (S_R F_L - S_L F_R + S_L S_R (U_R - U_L)) / (S_R - S_L)
	// with D_K = (0, 1, 0, 0, u_K, 0, ...) and D_star = (0, 1, 0, 0, S_star, 0, ...).

	double wL = 0.;	  // weight of U_L
	double wR = 0.;	  // weight of U_R
	double pMom = 0.; // additional x1-momentum flux
	double pEn = 0.;  // additional energy flux

	// open the Riemann fan (HLLC flux)
	if (S_L > 0.0) {
		// F = F_L
		wL = sL.u;
		pMom = sL.P;
		pEn = sL.P * sL.u;
	} else if ((S_star > 0.0) && (S_L <= 0.0)) {
		// F = F_starL
		const double inv = 1.0 / (S_L - S_star);
		wL = S_star * (S_L - sL.u) * inv;
		pMom = (S_L * P_LR - S_star * sL.P) * inv;
		pEn = S_star * (S_L * P_LR - sL.P * sL.u) * inv;
	} else if ((S_star <= 0.0) && (S_R >= 0.0)) {
		// F = F_starR
		const double inv = 1.0 / (S_R - S_star);
		wR = S_star * (S_R - sR.u) * inv;
		pMom = (S_R * P_LR - S_star * sR.P) * inv;
		pEn = S_star * (S_R * P_LR - sR.P * sR.u) * inv;
	} else { // S_R < 0.0
		// F = F_R
		wR = sR.u;
		pMom = sR.P;
		pEn = sR.P * sR.u;
	}

	// Both solvers reduce to the same upwind flux outside the Riemann fan.
	// Within the fan, blend the entire flux vector, including auxiliary
	// internal energy and passive scalars, using the same face weight.
	// No pressure-jump threshold switches off this correction.
	if ((theta < 1.0) && (S_L < 0.0) && (S_R > 0.0)) {
		// F_HLL weights
		const double invLR = 1.0 / (S_R - S_L);
		const double hL = S_R * (sL.u - S_L) * invLR;
		const double hR = S_L * (S_R - sR.u) * invLR;
		const double hMom = (S_R * sL.P - S_L * sR.P) * invLR;
		const double hEn = (S_R * sL.P * sL.u - S_L * sR.P * sR.u) * invLR;

		// F = theta * F + (1 - theta) * F_HLL
		const double omt = 1.0 - theta;
		wL = theta * wL + omt * hL;
		wR = theta * wR + omt * hR;
		pMom = theta * pMom + omt * hMom;
		pEn = theta * pEn + omt * hEn;
	}

	// N.B.: quokka::valarray is value-initialized to zero, so any components
	// beyond the hydro variables and passive scalars stay zero.
	quokka::valarray<double, fluxdim> F{};
	F[0] = wL * sL.rho + wR * sR.rho;
	F[1] = wL * (sL.rho * sL.u) + wR * (sR.rho * sR.u) + pMom;
	F[2] = wL * (sL.rho * sL.v) + wR * (sR.rho * sR.v);
	F[3] = wL * (sL.rho * sL.w) + wR * (sR.rho * sR.w);
	F[4] = wL * sL.E + wR * sR.E + pEn;
	F[5] = wL * sL.Eint + wR * sR.Eint;

	// The remaining components are passive scalars (U = scalar), fluxed with
	// the same weights.
	for (int n = 0; n < N_scalars; ++n) {
		const int nstart = fluxdim - N_scalars;
		F[nstart + n] = wL * sL.scalar[n] + wR * sR.scalar[n];
	}

	return F;
}
} // namespace quokka::Riemann

#endif // HLLC_HPP_
