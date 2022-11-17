#ifndef EXACTRIEMANN_HPP_ // NOLINT
#define EXACTRIEMANN_HPP_

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
	for (int n = 0; n < max_iter; ++n) {
		const double P_prev = P_star;
		P_star += -f(P_prev) / fprime(P_prev);
		const double rel_diff = std::abs(P_star - P_prev) / (0.5 * (P_star + P_prev));
		if (rel_diff < reltol) {
			riemann_success = true;
			break;
		}
	}
	AMREX_ALWAYS_ASSERT(riemann_success);

	return P_star;
}
} // namespace detail

// Exact Riemann solver following Chapter 4 of Toro (1998).
//
template <FluxDir DIR, int N_scalars, int fluxdim>
AMREX_FORCE_INLINE AMREX_GPU_DEVICE auto Exact(quokka::HydroState<N_scalars> const &sL, quokka::HydroState<N_scalars> const &sR, const double gamma,
					       const double /*du*/, const double /*dw*/) -> quokka::valarray<double, fluxdim>
{
	// compute pressure in the star region

	const double P_star = detail::ComputePstar(sL, sR, gamma);

	// compute wave speeds

	const double q_L = (P_star <= sL.P) ? 1.0 : std::sqrt(1.0 + ((gamma + 1.0) / (2.0 * gamma)) * ((P_star / sL.P) - 1.0));
	const double q_R = (P_star <= sR.P) ? 1.0 : std::sqrt(1.0 + ((gamma + 1.0) / (2.0 * gamma)) * ((P_star / sR.P) - 1.0));

	double S_L = sL.u - q_L * sL.cs;
	double S_R = sR.u + q_R * sR.cs;

	/// sample RP solution at x = 0

	/// compute fluxes

	const std::initializer_list<double> state_mid = {sL.rho, sL.rho * sL.vx, sL.rho * sL.vy, sL.rho * sL.vz, sL.E, sL.Eint};

	// N.B.: quokka::valarray is written to allow assigning <= fluxdim
	// components, so this works even if there are more components than
	// enumerated in the initializer list. The remaining components are
	// assigned a default value of zero.
	quokka::valarray<double, fluxdim> U_m = state_mid;

	// The remaining components are passive scalars, so just copy them from
	// x1LeftState and x1RightState into the (left, right) state vectors U_L and
	// U_R
	for (int n = 0; n < N_scalars; ++n) {
		const int nstart = static_cast<int>(state_mid.size());
		U_L[nstart + n] = sL.scalar[n];
		U_R[nstart + n] = sR.scalar[n];
	}

	// compute F(U_m)
	quokka::valarray<double, fluxdim> F{};

	return F;
}
} // namespace quokka::Riemann

#endif // EXACTRIEMANN_HPP_