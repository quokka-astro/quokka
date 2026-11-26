// SPDX-FileCopyrightText: © The Quokka Authors
// SPDX-License-Identifier: MIT

#ifndef ISM_COOLING_HPP_
#define ISM_COOLING_HPP_

/// \file ISMCooling.hpp
/// \brief Volumetric ISM cooling and heating rates, with analytic partial derivatives, for the photochemistry network.
///
/// All rates are in erg cm^-3 s^-1, with cooling >= 0 and heating >= 0. The independent variables are the temperature T, the
/// electron density ne, and the densities nHI and nHp, with n_H = nHI + nHp. Every function returns a Rate: the value and its partial
/// derivatives with respect to (T, ne, nHI, nHp). The partial with respect to ne is taken at fixed ne; the caller chains it through
/// ElectronDensityMax. Zp is the metallicity in solar units, and G0 is the FUV field in Habing units.

#include "AMReX_Extension.H"
#include "AMReX_GpuQualifiers.H"
#include "AMReX_REAL.H"

#include <cmath>

namespace quokka::ISMCooling
{

using amrex::Real;

/// A rate and its partial derivatives.
struct Rate {
	Real val;  ///< erg cm^-3 s^-1
	Real dT;   ///< d/dT
	Real dne;  ///< d/d(ne), at the ne passed in
	Real dnHI; ///< d/d(nHI)
	Real dnHp; ///< d/d(nHp)
};

/// The electron density used by the cooling and heating rates, and its derivatives.
struct ElectronDensity {
	Real ne;
	Real dT;
	Real dnHI;
	Real dnHp;
	int branch; ///< which term set the maximum: 0 network, 1 CIE, 2 C+, 3 CR
};

/// Floor [cm^-3] on n_e^max, which keeps the rates finite when there is no gas at all.
constexpr Real ne_floor = 1.0e-30;

/// A coefficient and its derivative with respect to T.
struct Coefficient {
	Real val;
	Real dT; ///< d/dT
};

/// Case-B recombination coefficient of H [cm^3 s^-1], 2.63e-13 (T/1e4)^-0.7, or the constant 2.63e-13 if temperatureDependent is false.
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto CaseBRecombinationCoefficient(Real T, bool temperatureDependent) -> Coefficient
{
	constexpr Real coeff = 2.63e-13;
	if (!temperatureDependent) {
		return {coeff, 0.};
	}
	return {coeff * std::pow(T / 1.0e4, -0.7), -coeff * 0.7 * std::pow(T / 1.0e4, -1.7) / 1.0e4};
}

namespace detail
{

constexpr Real k_B = 1.380649e-16;   // erg K^-1
constexpr Real eV = 1.602176634e-12; // erg
constexpr Real x_C_tot = 1.6e-4;     // per H, times Zp
constexpr Real x_O_tot = 3.2e-4;     // per H, times Zp
constexpr Real T_PE_max = 1.0e4;     // K, upper limit of the PE fit
constexpr Real Zd_over_001 = 0.68;   // Z_d / 0.01 = 0.5 * 0.0136 * Zp / 0.01, per Zp
constexpr Real E_CR = 1.6022e-11;    // erg per CR ionization
constexpr Real ln10 = 2.302585092994046;

/// Forward-mode dual number: a value and its partial derivatives with respect to (T, ne, nHI, nHp).
struct Dual {
	Real v, dT, dne, dnHI, dnHp;
	AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE Dual(Real v_ = 0., Real dT_ = 0., Real dne_ = 0., Real dnHI_ = 0.,
						      Real dnHp_ = 0.) // NOLINT(google-explicit-constructor, hicpp-explicit-conversions)
	    : v(v_), dT(dT_), dne(dne_), dnHI(dnHI_), dnHp(dnHp_)
	{
	}
};

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto operator+(const Dual &a, const Dual &b) -> Dual
{
	return {a.v + b.v, a.dT + b.dT, a.dne + b.dne, a.dnHI + b.dnHI, a.dnHp + b.dnHp};
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto operator-(const Dual &a, const Dual &b) -> Dual
{
	return {a.v - b.v, a.dT - b.dT, a.dne - b.dne, a.dnHI - b.dnHI, a.dnHp - b.dnHp};
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto operator-(const Dual &a) -> Dual { return {-a.v, -a.dT, -a.dne, -a.dnHI, -a.dnHp}; }

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto operator*(const Dual &a, const Dual &b) -> Dual
{
	return {a.v * b.v, a.dT * b.v + a.v * b.dT, a.dne * b.v + a.v * b.dne, a.dnHI * b.v + a.v * b.dnHI, a.dnHp * b.v + a.v * b.dnHp};
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto operator*(Real c, const Dual &a) -> Dual { return {c * a.v, c * a.dT, c * a.dne, c * a.dnHI, c * a.dnHp}; }

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto operator*(const Dual &a, Real c) -> Dual { return c * a; }

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto operator/(const Dual &a, const Dual &b) -> Dual
{
	const Real q = a.v / b.v;
	return {q, (a.dT - q * b.dT) / b.v, (a.dne - q * b.dne) / b.v, (a.dnHI - q * b.dnHI) / b.v, (a.dnHp - q * b.dnHp) / b.v};
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto operator/(const Dual &a, Real c) -> Dual { return (1. / c) * a; }

/// Apply a scalar function with value f and derivative df to a dual number.
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto Chain(const Dual &a, Real f, Real df) -> Dual { return {f, df * a.dT, df * a.dne, df * a.dnHI, df * a.dnHp}; }

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto Exp(const Dual &a) -> Dual
{
	const Real e = std::exp(a.v);
	return Chain(a, e, e);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto Log(const Dual &a) -> Dual { return Chain(a, std::log(a.v), 1. / a.v); }

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto Sqrt(const Dual &a) -> Dual
{
	const Real s = std::sqrt(a.v);
	return Chain(a, s, 0.5 / s);
}

/// a^p for a constant exponent p. At a = 0 the derivative factor is set to 0 (the argument is then constant in every variable
/// where this is used, e.g. psi = 0 for G0 = 0), which avoids 0 * infinity.
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto Pow(const Dual &a, Real p) -> Dual
{
	const Real f = std::pow(a.v, p);
	return Chain(a, f, (a.v == 0.) ? 0. : p * f / a.v);
}

/// a^b for a positive base and a dual exponent.
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto Pow(const Dual &a, const Dual &b) -> Dual { return Exp(b * Log(a)); }

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto ToRate(const Dual &a) -> Rate { return {a.v, a.dT, a.dne, a.dnHI, a.dnHp}; }

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto Variables(Real nHI, Real nHp, Real ne, Real T, Dual &dT, Dual &dne, Dual &dnHI, Dual &dnHp) -> void
{
	dT = Dual(T, 1., 0., 0., 0.);
	dne = Dual(ne, 0., 1., 0., 0.);
	dnHI = Dual(nHI, 0., 0., 1., 0.);
	dnHp = Dual(nHp, 0., 0., 0., 1.);
}

/// Cooling per H (times xs) of a two-level atom with collisional rates q01, q10 (Athena++ Cooling2Level_).
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto Cooling2Level(const Dual &q01, const Dual &q10, Real A10, Real E10, Real xs) -> Dual
{
	return (q01 / (q01 + q10 + A10)) * (A10 * E10 * xs);
}

/// Cooling per H (times xs) of a three-level atom (Athena++ Cooling3Level_).
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto Cooling3Level(const Dual &q01, const Dual &q10, const Dual &q02, const Dual &q20, const Dual &q12,
							    const Dual &q21, Real A10, Real A20, Real A21, Real E10, Real E20, Real E21, const Dual &xs) -> Dual
{
	const Dual R10 = q10 + A10;
	const Dual R20 = q20 + A20;
	const Dual R21 = q21 + A21;
	const Dual a0 = R10 * R20 + R10 * R21 + q12 * R20;
	const Dual a1 = q01 * R20 + q01 * R21 + R21 * q02;
	const Dual a2 = q02 * R10 + q02 * q12 + q12 * q01;
	const Dual de = a0 + a1 + a2;
	return ((a1 / de) * (A10 * E10) + (a2 / de) * (A20 * E20 + A21 * E21)) * xs;
}

} // namespace detail

/// Ly-alpha cooling by collisional excitation of HI (NCR Eq. 42).
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto LyaCooling(Real nHI, Real ne, Real T) -> Rate
{
	using namespace detail;
	Dual vT, vne, vnHI, vnHp;
	Variables(nHI, 0., ne, T, vT, vne, vnHI, vnHp);
	const Dual T4 = vT / 1.0e4;
	const Dual k12 = 5.31e-8 * Pow(T4, 0.15) / (1. + Pow(T4 / 5., 0.65)) * Exp(-11.84 / T4);
	return ToRate(10.2 * eV * vnHI * vne * k12);
}

/// Nebular metal-line cooling in ionized gas (NCR Eq. 47, NCR's printed coefficients).
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto NebularCooling(Real nHp, Real ne, Real T, Real Zp) -> Rate
{
	using namespace detail;
	constexpr Real a[7] = {0.692, -0.586, 0.816, -0.505, 0.118, 7.66e-3, -5.08e-3};
	Dual vT, vne, vnHI, vnHp;
	Variables(0., nHp, ne, T, vT, vne, vnHI, vnHp);
	const Dual T4 = vT / 1.0e4;
	const Dual lnT4 = Log(T4);
	Dual poly = a[6];
	for (int i = 5; i >= 0; --i) {
		poly = poly * lnT4 + a[i];
	}
	const Dual fneb = Exp(ln10 * poly);
	const Dual ne2 = vne / 100.;
	const Dual denom = Sqrt(T4) * (1. + 0.12 * Pow(ne2, 0.38 - 0.12 * lnT4));
	return ToRate(Zp * vne * vnHp * (3.68e-23 * Exp(-3.86 / T4) * fneb / denom));
}

// The C II and O I fine-structure cooling below is ported from Gong's Athena++ code, src/chemistry/utils/thermo.cpp
// (Thermo::q10CII_, CoolingCII, CoolingOI, Cooling2Level_, Cooling3Level_; Athena++ is BSD-3 licensed, Copyright (C) 2014 James M. Stone
// and other code contributors). The collision-rate fits are those of Gong, Ostriker & Wolfire (2017, ApJ 843, 38; "G17"), from Draine
// (2011) ISM book Eq. 17.16-17 and Table F.6, and Bell et al. (1998) for electrons. Collision partners are HI and e only (n_H2 = 0).

/// C II 158 micron cooling (RIGEL A.4 Eq. A.3). All carbon is C+, n_C+ = x_C,tot Zp n_H.
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto CIICooling(Real nHI, Real nHp, Real ne, Real T, Real Zp) -> Rate
{
	using namespace detail;
	constexpr Real A10 = 2.3e-6;
	constexpr Real E10 = 1.26e-14;
	constexpr Real g0 = 2.;
	constexpr Real g1 = 4.;
	Dual vT, vne, vnHI, vnHp;
	Variables(nHI, nHp, ne, T, vT, vne, vnHI, vnHp);
	const Dual T2 = vT / 100.;
	const Dual k10e = 4.53e-8 * Sqrt(1.0e4 / vT);
	const Dual k10HI = 7.58e-10 * Pow(T2, 0.1281 + 0.0087 * Log(T2));
	const Dual q10 = k10e * vne + k10HI * vnHI;
	const Dual q01 = (g1 / g0) * q10 * Exp(-E10 / (k_B * vT));
	return ToRate((vnHI + vnHp) * Cooling2Level(q01, q10, A10, E10, x_C_tot * Zp));
}

/// O I 63 and 146 micron cooling (RIGEL A.4). Neutral O follows HI, n_O = x_O,tot Zp n_HI.
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto OICooling(Real nHI, Real ne, Real T, Real Zp) -> Rate
{
	using namespace detail;
	constexpr Real g0 = 5.;
	constexpr Real g1 = 3.;
	constexpr Real g2 = 1.;
	constexpr Real A10 = 8.910e-05;
	constexpr Real A20 = 1.340e-10;
	constexpr Real A21 = 1.750e-05;
	constexpr Real E10 = 3.144e-14;
	constexpr Real E20 = 4.509e-14;
	constexpr Real E21 = 1.365e-14;
	Dual vT, vne, vnHI, vnHp;
	Variables(nHI, 0., ne, T, vT, vne, vnHI, vnHp);
	const Dual T2 = vT / 100.;
	const Dual lnT2 = Log(T2);
	const Dual k10HI = 3.57e-10 * Pow(T2, 0.419 - 0.003 * lnT2);
	const Dual k20HI = 3.19e-10 * Pow(T2, 0.369 - 0.006 * lnT2);
	const Dual k21HI = 4.34e-10 * Pow(T2, 0.755 - 0.160 * lnT2);
	const Dual k10e = 5.12e-10 * Pow(vT, -0.075);
	const Dual k20e = 4.86e-10 * Pow(vT, -0.026);
	const Dual k21e = 1.08e-14 * Pow(vT, 0.926);
	const Dual q10 = k10HI * vnHI + k10e * vne;
	const Dual q20 = k20HI * vnHI + k20e * vne;
	const Dual q21 = k21HI * vnHI + k21e * vne;
	const Dual q01 = (g1 / g0) * q10 * Exp(-E10 / (k_B * vT));
	const Dual q02 = (g2 / g0) * q20 * Exp(-E20 / (k_B * vT));
	const Dual q12 = (g2 / g1) * q21 * Exp(-E21 / (k_B * vT));
	return ToRate(Cooling3Level(q01, q10, q02, q20, q12, q21, A10, A20, A21, E10, E20, E21, (x_O_tot * Zp) * vnHI));
}

/// Cooling by electron recombination on small grains and PAHs (NCR Sect. 4.6.3, WD01 Eq. 45; coefficients from Athena++ DPE_).
/// Our G0 is in Habing units, so x = G0 sqrt(T) / ne + 50. Scales linearly with Zp.
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto GrainRecombinationCooling(Real nHI, Real nHp, Real ne, Real T, Real G0, Real Zp) -> Rate
{
	using namespace detail;
	constexpr Real D0 = 0.4535;
	constexpr Real D1 = 2.234;
	constexpr Real D2 = -6.266;
	constexpr Real D3 = 1.442;
	constexpr Real D4 = 0.05089;
	Dual vT, vne, vnHI, vnHp;
	Variables(nHI, nHp, ne, T, vT, vne, vnHI, vnHp);
	const Dual x = G0 * Sqrt(vT) / vne + 50.;
	const Dual lnx = Log(x);
	const Dual rate = 1.0e-28 * Zp * vne * (vnHI + vnHp) * Pow(vT, D0 + D1 / lnx) * Exp(D2 + (D3 - D4 * lnx) * lnx);
	return ToRate(rate);
}

/// Photoelectric heating (RIGEL A.7 Eqs. A.8-A.9). The fit is capped at T = 1e4 K: the capped temperature enters psi and the
/// (T/1e4)^0.7 term, and dT = 0 above the cap.
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto PEHeating(Real nHI, Real nHp, Real ne, Real T, Real G0, Real Zp) -> Rate
{
	using namespace detail;
	Dual vT, vne, vnHI, vnHp;
	Variables(nHI, nHp, ne, T, vT, vne, vnHI, vnHp);
	const Dual Tt = (T < T_PE_max) ? vT : Dual(T_PE_max);
	const Dual psi = G0 * Sqrt(Tt) / (0.5 * vne);
	const Dual eps = 0.0487 / (1. + 0.004 * Pow(psi, 0.73)) + 0.0365 * Pow(Tt / 1.0e4, 0.7) / (1. + 2.0e-4 * psi);
	return ToRate((1.3e-24 * Zd_over_001 * Zp * G0) * eps * (vnHI + vnHp));
}

/// Cosmic-ray heating (RIGEL A.8 Eq. A.10), with zeta_HeI = 1.1 zeta_HI and n_HeI = yHe n_HI.
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto CRHeating(Real nHI, Real zetaHI, Real yHe) -> Rate
{
	const Real perHI = detail::E_CR * zetaHI * (1. + 1.1 * yHe);
	return {perHI * nHI, 0., 0., perHI, 0.};
}

/// Electron density for the cooling and heating rates (SPEC 5.4): the maximum of the network electrons, the CIE value, the C+ floor,
/// and the CR floor sqrt(zeta n_H / alphaB). Derivatives are those of the branch that is active; ties go to the lower branch number.
/// The CIE term is xeCIE(n_H, T) n_H, with the given partials dxeCIE_dT and dxeCIE_dnH of the electron fraction.
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto ElectronDensityMax(Real nHI, Real nHp, Real T, Real xeCIE, Real dxeCIE_dT, Real dxeCIE_dnH, Real Zp, Real zetaHI,
								 Real alphaB, Real dalphaB_dT) -> ElectronDensity
{
	using detail::x_C_tot;
	const Real nH = nHI + nHp;
	const Real cie = xeCIE * nH;
	const Real cplus = x_C_tot * Zp * nH;
	const Real cr = std::sqrt(zetaHI * nH / alphaB);

	ElectronDensity e{nHp, 0., 0., 1., 0};
	if (cie > e.ne) {
		const Real dnH = xeCIE + nH * dxeCIE_dnH;
		e = {cie, dxeCIE_dT * nH, dnH, dnH, 1};
	}
	if (cplus > e.ne) {
		e = {cplus, 0., x_C_tot * Zp, x_C_tot * Zp, 2};
	}
	if (cr > e.ne) {
		const Real dnH = (nH > 0.) ? cr / (2. * nH) : 0.;
		e = {cr, -cr / (2. * alphaB) * dalphaB_dT, dnH, dnH, 3};
	}
	return e;
}

} // namespace quokka::ISMCooling

#endif // ISM_COOLING_HPP_
