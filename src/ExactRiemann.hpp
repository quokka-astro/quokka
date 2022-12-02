#ifndef EXACTRIEMANN_HPP_ // NOLINT
#define EXACTRIEMANN_HPP_

#include "AMReX_BLassert.H"
#include "AMReX_Extension.H"
#include "AMReX_GpuQualifiers.H"
#include <AMReX.H>
#include <AMReX_REAL.H>

#include "ArrayView.hpp"
#include "HydroState.hpp"
#include "valarray.hpp"

namespace quokka::Riemann
{
namespace detail
{
// pressure function f_{L,R}(p)
// 	the solution of f_L(p) + f_R(p) + du == 0 is the star region pressure p.
//
template <int N_scalars> AMREX_FORCE_INLINE AMREX_GPU_DEVICE auto f_p(double P, quokka::HydroState<N_scalars> const &s, const double gamma)
{
	double f = NAN;
	if (P > s.P) { // shock
		const double A = 2. / ((gamma + 1.) * s.rho);
		const double B = ((gamma - 1.) / (gamma + 1.)) * s.P;
		f = (P - s.P) * std::sqrt(A / (P + B));
	} else { // rarefaction
		f = (2.0 * s.cs) / (gamma - 1.) * (std::pow(P / s.P, (gamma - 1.) / (2. * gamma)) - 1.);
	}
	return f;
}

// derivative of the pressure function d(f_{L,R}(p)) / dp.
//
template <int N_scalars> AMREX_FORCE_INLINE AMREX_GPU_DEVICE auto fprime_p(double P, quokka::HydroState<N_scalars> const &s, const double gamma)
{
	double fprime = NAN;
	if (P > s.P) { // shock
		const double A = 2. / ((gamma + 1.) * s.rho);
		const double B = ((gamma - 1.) / (gamma + 1.)) * s.P;
		fprime = std::sqrt(A / (B + P)) * (1. - (P - s.P) / (2. * (B + P)));
	} else { // rarefaction
		fprime = std::pow(P / s.P, -(gamma + 1.) / (2. * gamma)) / (s.rho * s.cs);
	}
	return fprime;
}

// Iteratively solve for the exact pressure in the star region.
// following Chapter 4 of Toro (1998).
//
template <int N_scalars>
AMREX_FORCE_INLINE AMREX_GPU_DEVICE auto ComputePstar(quokka::HydroState<N_scalars> const &sL, quokka::HydroState<N_scalars> const &sR, const double gamma)
{
	// compute PVRS states (Toro 10.5.2)

	const double rho_bar = 0.5 * (sL.rho + sR.rho);
	const double cs_bar = 0.5 * (sL.cs + sR.cs);
	const double P_PVRS = 0.5 * (sL.P + sR.P) - 0.5 * (sR.u - sL.u) * (rho_bar * cs_bar);

	// choose an initial guess for P_star adaptively (Fig. 9.4 of Toro)

	const double P_min = std::min(sL.P, sR.P);
	const double P_max = std::max(sL.P, sR.P);
	double P_star = NAN;

	if ((P_min < P_PVRS) && (P_PVRS < P_max)) {
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

	if (P_star <= 0.) {
		P_star = 0.5 * (sL.P + sR.P);
	}

	auto f = [=] AMREX_GPU_DEVICE(double P) {
		const double du = sR.u - sL.u;
		return f_p(P, sL, gamma) + f_p(P, sR, gamma) + du;
	};

	auto fprime = [=] AMREX_GPU_DEVICE(double P) { return fprime_p(P, sL, gamma) + fprime_p(P, sR, gamma); };

	// solve p(P_star) == 0 to get P_star
	const int max_iter = 20;
	const double reltol = 1.0e-6;
	bool riemann_success = false;
	double P_prev = NAN;
	double rel_diff = NAN;
	for (int n = 0; n < max_iter; ++n) {
		P_prev = P_star;
		const double fp = f(P_prev);
		const double fp_prime = fprime(P_prev);
		double P_new = P_prev - fp / fp_prime;
		if (P_new <= 0) {
			// use damping factor to avoid negative pressures
			double damping = 0.5 * P_prev * (fp_prime / fp);
			P_new = P_prev - damping * (fp / fp_prime);
		}
		P_star = P_new;
		rel_diff = std::abs(P_star - P_prev) / (0.5 * (P_star + P_prev));
		if (rel_diff < reltol) {
			riemann_success = true;
			break;
		}
	}
#if 0
	if (!riemann_success) {
		printf("[ExactRiemann] pressure iteration failed to converge! P_guess = %g P_prev = %g rel_diff = %g\n", P_star, P_prev, rel_diff);
		amrex::Abort("pressure iteration failed to converge in exact Riemann solver!");
	}
#endif
	return P_star;
}
} // namespace detail

// Exact Riemann solver following Chapter 4 of Toro (1998).
//
template <int N_scalars, int fluxdim>
AMREX_FORCE_INLINE AMREX_GPU_DEVICE auto Exact(quokka::HydroState<N_scalars> const &sL, quokka::HydroState<N_scalars> const &sR, const double gamma,
					       const double /*du*/, const double /*dw*/) -> quokka::valarray<double, fluxdim>
{
	// check if vacuum is created
	const double du_crit = 2.0 * (sL.cs + sR.cs) / (gamma - 1.);
	const double du = sR.u - sL.u;
	// AMREX_ALWAYS_ASSERT(du_crit > du); // pressure positivity condition

	// compute pressure in the star region
	const double P_star = detail::ComputePstar(sL, sR, gamma);

	// compute particle velocity in the star region
	const double u_star = 0.5 * (sL.u + sR.u) + 0.5 * (detail::f_p(P_star, sR, gamma) - detail::f_p(P_star, sL, gamma));

	/// sample RP solution at S = x/t = 0, save into HydroState s.

	quokka::HydroState<N_scalars> s{}; // sampled exact state

	// left side of contact: u_star > 0
	if (u_star > 0.) {
		s = sL; // copy Eint, transvere velocities, passive scalars

		if (P_star > sL.P) { // left shock wave
			const double S_L = sL.u - sL.cs * std::sqrt((gamma + 1.) / (2. * gamma) * (P_star / sL.P) + (gamma - 1.) / (2. * gamma));
			if (S_L > 0.) { // left input state
				s = sL;
			} else { // left star state
				const double rho_star =
				    sL.rho * (P_star / sL.P + (gamma - 1.) / (gamma + 1.)) / ((gamma - 1.) / (gamma + 1.) * (P_star / sL.P) + 1.);
				s.rho = rho_star;
				s.u = u_star;
				s.P = P_star;
			}

		} else { // left rarefaction wave
			const double cs_star = sL.cs * std::pow(P_star / sL.P, (gamma - 1.) / (2. * gamma));
			const double S_H = sL.u - sL.cs;
			const double S_T = u_star - cs_star;
			if (S_H > 0.) { // left input state
				s = sL;
			} else if ((S_H <= 0.) && (S_T > 0)) { // rarefaction fan state
				const double rho_fan = sL.rho * std::pow(2. / (gamma + 1.) + sL.u * (gamma - 1.) / ((gamma + 1.) * sL.cs), 2. / (gamma - 1.));
				const double u_fan = 2. / (gamma + 1.) * (sL.cs + 0.5 * (gamma - 1.) * sL.u);
				const double P_fan =
				    sL.P * std::pow(2. / (gamma + 1.) + sL.u * (gamma - 1.) / ((gamma + 1.) * sL.cs), 2. * gamma / (gamma - 1.));
				s.rho = rho_fan;
				s.u = u_fan;
				s.P = P_fan;
			} else { // left star state
				const double rho_starFan = sL.rho * std::pow(P_star / sL.P, 1. / gamma);
				s.rho = rho_starFan;
				s.u = u_star;
				s.P = P_star;
			}
		}
		// right side of contact: u_star <= 0
	} else {
		s = sR; // copy Eint, transverse velocities, passive scalars

		if (P_star > sR.P) { // right shock wave
			const double S_R = sR.u + sR.cs * std::sqrt((gamma + 1.) / (2. * gamma) * (P_star / sR.P) + (gamma - 1.) / (2. * gamma));
			if (S_R > 0.) { // right star state
				const double rho_star =
				    sR.rho * (P_star / sR.P + (gamma - 1.) / (gamma + 1.)) / ((gamma - 1.) / (gamma + 1.) * (P_star / sR.P) + 1.);
				s.rho = rho_star;
				s.u = u_star;
				s.P = P_star;
			} else { // right input state
				s = sR;
			}

		} else { // right rarefaction wave
			const double cs_star = sR.cs * std::pow(P_star / sR.P, (gamma - 1.) / (2. * gamma));
			const double S_H = sR.u + sR.cs;
			const double S_T = u_star + cs_star;
			if (S_T > 0.) { // right star state
				const double rho_starFan = sR.rho * std::pow(P_star / sR.P, 1. / gamma);
				s.rho = rho_starFan;
				s.u = u_star;
				s.P = P_star;
			} else if ((S_T <= 0.) && (S_H > 0.)) { // rarefaction fan state
				const double rho_fan = sR.rho * std::pow(2. / (gamma + 1.) - sR.u * (gamma - 1.) / ((gamma + 1.) * sR.cs), 2. / (gamma - 1.));
				const double u_fan = 2. / (gamma + 1.) * (-sR.cs + 0.5 * (gamma - 1.) * sR.u);
				const double P_fan =
				    sR.P * std::pow(2. / (gamma + 1.) - sR.u * (gamma - 1.) / ((gamma + 1.) * sR.cs), 2. * gamma / (gamma - 1.));
				s.rho = rho_fan;
				s.u = u_fan;
				s.P = P_fan;
			} else { // right input state
				s = sR;
			}
		}
	}

	// recompute the total energy
	s.E = s.P / (gamma - 1.) + 0.5 * s.rho * (s.u * s.u + s.v * s.v + s.w + s.w);

	/// compute fluxes

	// N.B.: quokka::valarray is written to allow assigning <= fluxdim
	// components, so this works even if there are more components than
	// enumerated in the initializer list. The remaining components are
	// assigned a default value of zero.
	const quokka::valarray<double, fluxdim> D_m = {0., 1., 0., 0., s.u, 0.};

	// compute exact state at interface
	quokka::valarray<double, fluxdim> U_m = {s.rho, s.rho * s.u, s.rho * s.v, s.rho * s.w, s.E, s.Eint};

	// The remaining components are passive scalars, so just copy them from
	// x1LeftState and x1RightState into the state vector U_m
	for (int n = 0; n < N_scalars; ++n) {
		const int nstart = fluxdim - N_scalars;
		U_m[nstart + n] = s.scalar[n];
	}

	// compute F(U_m)
	const quokka::valarray<double, fluxdim> F = s.u * U_m + s.P * D_m;
	return F;
}
} // namespace quokka::Riemann

#endif // EXACTRIEMANN_HPP_