#ifndef DEUTERIUM_STELLAR_MODEL_HPP_
#define DEUTERIUM_STELLAR_MODEL_HPP_

/// \file deuterium_stellar_model.hpp
/// \brief One-zone protostellar evolution models with deuterium burning (Nakano et al. 1995, 2000; Tan & McKee 2004).
/// Orion2007StellarModel follows the 2007 standalone ORION StarParticleData code ("Orion2007"), an earlier version of
/// the model published by Offner et al. (2009, Appendix B). Orion2StellarModel follows the later ORION2 code ("Orion2").

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>

#include "AMReX_Array.H"
#include "AMReX_Extension.H"
#include "AMReX_GpuQualifiers.H"
#include "AMReX_REAL.H"
#include "fundamental_constants.H"
#include "math/root_finding.hpp"
#include "particles/star_particle_indices.H"
#include "util/time_units.hpp"

namespace quokka
{

namespace StellarPhysics
{
// Constants
constexpr amrex::Real L_solar = 3.828e33; // erg/s (IAU 2015 nominal)
constexpr amrex::Real mu = 0.613;	  // mean molecular weight of fully ionized gas of solar composition
constexpr amrex::Real m_H = C::m_p;	  // hydrogen mass used with mu

// Model parameters
constexpr amrex::Real M_rad_min = 0.01 * C::M_solar;	     // minimum mass at which the model is used
constexpr amrex::Real f_acc = 0.5;			     // fraction of accretion power radiated rather than driving a wind
constexpr amrex::Real f_k = 0.5;			     // fraction of infall kinetic energy radiated in the inner disk
constexpr amrex::Real f_rad = 0.18;			     // shell burning starts when L_D <= f_rad L_ZAMS
constexpr amrex::Real shell_factor = 2.1;		     // radius increase when shell burning starts
constexpr amrex::Real T_Hayashi = 3000.0;		     // Hayashi-track surface temperature (K)
constexpr amrex::Real T_deut = 1.5e6;			     // central temperature for deuterium ignition (K)
constexpr amrex::Real psi_ion = 17.0 * C::ev2erg * C::n_A;   // energy per gram to dissociate and ionize the accreted gas
constexpr amrex::Real psi_deut = 100.0 * C::ev2erg * C::n_A; // energy per gram released by burning its deuterium
constexpr amrex::Real dlogm = 1.0e-3;			     // fractional mass step of the numerical d/dlog(m)

// Lane-Emden factors rho_mean/rho_c and P_c/(G m^2/r^4) for n = 1.5, 1.6, ..., 3.1 (Kippenhahn & Weigert)
constexpr int n_poly_table = 17;
static AMREX_GPU_CONSTANT const amrex::GpuArray<amrex::Real, n_poly_table> rho_factor_table = {0.166931,  0.14742,   0.129933,	0.114265,  0.100242,  0.0877,
											       0.0764968, 0.0665109, 0.0576198, 0.0497216, 0.0427224, 0.0365357,
											       0.0310837, 0.0262952, 0.0221057, 0.0184553, 0.01529};
static AMREX_GPU_CONSTANT const amrex::GpuArray<amrex::Real, n_poly_table> pressure_factor_table = {
    0.770087, 0.889001, 1.02979, 1.19731, 1.39753, 1.63818, 1.92909, 2.2825, 2.71504, 3.24792, 3.90921, 4.73657, 5.78067, 7.11088, 8.82286, 11.0515, 13.9885};

// Mean gas-pressure fraction beta of a polytrope for m = 5, 7.5, ..., 50 M_sun (rows) and n = 1.5, 2, 2.5, 3 (columns)
constexpr int n_beta_mass = 19;
constexpr int n_beta_poly = 4;
static AMREX_GPU_CONSTANT const amrex::GpuArray<amrex::Real, n_beta_mass * n_beta_poly> beta_table = {
    0.98785,  0.988928, 0.98947,  0.989634, 0.97438,  0.976428, 0.977462, 0.977774, 0.957927, 0.960895, 0.962397, 0.962846, 0.939787,
    0.943497, 0.945369, 0.945922, 0.92091,  0.925151, 0.927276, 0.927896, 0.901932, 0.906512, 0.908785, 0.909436, 0.883254, 0.888017,
    0.890353, 0.891013, 0.865111, 0.86994,  0.872277, 0.872927, 0.847635, 0.852445, 0.854739, 0.855367, 0.830886, 0.835619, 0.837842,
    0.838441, 0.814885, 0.8195,	  0.821635, 0.822201, 0.799625, 0.804095, 0.806133, 0.806664, 0.785082, 0.789394, 0.791328, 0.791825,
    0.771226, 0.775371, 0.777202, 0.777665, 0.758022, 0.761997, 0.763726, 0.764156, 0.745433, 0.749238, 0.750869, 0.751268, 0.733423,
    0.73706,  0.738596, 0.738966, 0.721954, 0.725429, 0.726874, 0.727216, 0.710993, 0.714311, 0.715671, 0.715987};

// Initial polytropic index from the accretion rate, clamped to [1.5, 3]
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto nInit(amrex::Real mdot) -> amrex::Real
{
	if (mdot <= 0.0) {
		return 1.5;
	}
	const amrex::Real aG_init = 1.475 + 0.07 * std::log10(8.0e-5 * mdot * yr_in_s / C::M_solar);
	return std::min(std::max(5.0 - 3.0 / aG_init, 1.5), 3.0);
}

// Initial radius from the accretion rate
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto radInit(amrex::Real mdot) -> amrex::Real
{
	return 2.5 * C::R_solar * std::pow(mdot * yr_in_s / C::M_solar * 1.0e5, 0.2);
}

// Gravitational energy coefficient of a polytrope, E_grav = -aG G m^2 / r
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto aG(amrex::Real n) -> amrex::Real { return 3.0 / (5.0 - n); }

// Linear interpolation of a Lane-Emden factor table in n
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto polytropeFactor(amrex::GpuArray<amrex::Real, n_poly_table> const &table, amrex::Real n) -> amrex::Real
{
	const int itab = std::min(std::max(static_cast<int>(std::floor((n - 1.5) / 0.1)), 0), n_poly_table - 2);
	const amrex::Real wgt = (n - (1.5 + 0.1 * itab)) / 0.1;
	return table[itab] * (1.0 - wgt) + table[itab + 1] * wgt;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto rhoCentral(amrex::Real m, amrex::Real r, amrex::Real n) -> amrex::Real
{
	return m / (4.0 / 3.0 * M_PI * r * r * r) / polytropeFactor(rho_factor_table, n);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto pressureCentral(amrex::Real m, amrex::Real r, amrex::Real n) -> amrex::Real
{
	return polytropeFactor(pressure_factor_table, n) * C::Gconst * m * m / (r * r * r * r);
}

// Root of P_c = rho_c k_B T / (mu m_H) + a T^4 / 3 on [0, T_max]
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto solveCentralTemperature(amrex::Real rho_c, amrex::Real P_c, amrex::Real T_max) -> amrex::Real
{
	auto f = [=](amrex::Real T) { return P_c - rho_c * C::k_B * T / (mu * m_H) - C::a_rad * T * T * T * T / 3.0; };
	int max_iter = 100;
	const auto [T_lo, T_hi] = quokka::math::toms748_solve(f, 0.0, T_max, quokka::math::eps_tolerance<amrex::Real>(1.0e-12), max_iter);
	return 0.5 * (T_lo + T_hi);
}

// Central temperature from P_c = rho_c k_B T_c / (mu m_H) + a T_c^4 / 3
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto temperatureCentral(amrex::Real m, amrex::Real r, amrex::Real n) -> amrex::Real
{
	const amrex::Real rho_c = rhoCentral(m, r, n);
	const amrex::Real P_c = pressureCentral(m, r, n);
	const amrex::Real T_gas = P_c * mu * m_H / (C::k_B * rho_c);
	const amrex::Real T_rad = std::pow(3.0 * P_c / C::a_rad, 0.25);
	return solveCentralTemperature(rho_c, P_c, std::max(T_gas, T_rad));
}

// Ratio of gas pressure to total pressure at the centre
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto betaCentral(amrex::Real m, amrex::Real r, amrex::Real n) -> amrex::Real
{
	return rhoCentral(m, r, n) * C::k_B * temperatureCentral(m, r, n) / (mu * m_H * pressureCentral(m, r, n));
}

// Mean ratio of gas pressure to total pressure: Eddington quartic for n = 3, otherwise table lookup
// (beta = 1 below 5 M_sun; masses above 50 M_sun use the last table row)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto beta(amrex::Real m, amrex::Real r, amrex::Real n) -> amrex::Real
{
	if (n == 3.0) {
		// P_c^3 = (3/a) (k_B / (mu m_H))^4 (1 - beta) / beta^4 rho_c^4
		const amrex::Real P_c = pressureCentral(m, r, n);
		const amrex::Real x = C::k_B * rhoCentral(m, r, n) / (mu * m_H);
		const amrex::Real coef = 3.0 / C::a_rad * (x * x * x * x) / (P_c * P_c * P_c);
		auto f = [=](amrex::Real b) { return 1.0 - coef * (1.0 - b) / (b * b * b * b); };
		int max_iter = 100;
		const auto [b_lo, b_hi] = quokka::math::toms748_solve(f, 1.0e-4, 1.0, quokka::math::eps_tolerance<amrex::Real>(1.0e-12), max_iter);
		return 0.5 * (b_lo + b_hi);
	}
	constexpr amrex::Real m_min = 5.0 * C::M_solar;
	constexpr amrex::Real m_step = 2.5 * C::M_solar;
	if (m < m_min) {
		return 1.0;
	}
	const int midx = std::min(static_cast<int>(std::floor((m - m_min) / m_step)), n_beta_mass - 2);
	const amrex::Real mwgt = std::min((m - (m_min + midx * m_step)) / m_step, 1.0);
	const int nidx = std::min(std::max(static_cast<int>(std::floor((n - 1.5) / 0.5)), 0), n_beta_poly - 2);
	const amrex::Real nwgt = (n - (1.5 + 0.5 * nidx)) / 0.5;
	auto b = [](int i, int j) { return beta_table[i * n_beta_poly + j]; };
	return b(midx, nidx) * (1.0 - mwgt) * (1.0 - nwgt) + b(midx + 1, nidx) * mwgt * (1.0 - nwgt) + b(midx, nidx + 1) * (1.0 - mwgt) * nwgt +
	       b(midx + 1, nidx + 1) * mwgt * nwgt;
}

// d log(beta) / d log(m) at fixed r and n
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto dlogBeta_dlogM(amrex::Real m, amrex::Real r, amrex::Real n, amrex::Real beta1) -> amrex::Real
{
	return (beta(m * (1.0 + dlogm), r, n) / beta1 - 1.0) / dlogm;
}

// d log(beta / beta_c) / d log(m) at fixed r and n (zero for n = 3, where beta = beta_c)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto dlogBetaOverBetac_dlogM(amrex::Real m, amrex::Real r, amrex::Real n, amrex::Real beta1) -> amrex::Real
{
	if (n == 3.0) {
		return 0.0;
	}
	const amrex::Real m2 = m * (1.0 + dlogm);
	const amrex::Real ratio1 = beta1 / betaCentral(m, r, n);
	const amrex::Real ratio2 = beta(m2, r, n) / betaCentral(m2, r, n);
	return (ratio2 / ratio1 - 1.0) / dlogm;
}

// Zero-age main-sequence luminosity, Tout et al. (1996), solar metallicity
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto luminosityZAMS(amrex::Real m) -> amrex::Real
{
	const amrex::Real x = m / C::M_solar;
	const amrex::Real num = 0.39704170 * std::pow(x, 5.5) + 8.52762600 * std::pow(x, 11.0);
	const amrex::Real den = 0.00025546 + std::pow(x, 3.0) + 5.43288900 * std::pow(x, 5.0) + 5.56357900 * std::pow(x, 7.0) + 0.78866060 * std::pow(x, 8.0) +
				0.00586685 * std::pow(x, 9.5);
	return num / den * L_solar;
}

// Zero-age main-sequence radius, Tout et al. (1996), solar metallicity
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto radiusZAMS(amrex::Real m) -> amrex::Real
{
	const amrex::Real x = m / C::M_solar;
	const amrex::Real num = 1.71535900 * std::pow(x, 2.5) + 6.59778800 * std::pow(x, 6.5) + 10.08855000 * std::pow(x, 11.0) +
				1.01249500 * std::pow(x, 19.0) + 0.07490166 * std::pow(x, 19.5);
	const amrex::Real den =
	    0.01077422 + 3.08223400 * std::pow(x, 2.0) + 17.84778000 * std::pow(x, 8.5) + std::pow(x, 18.5) + 0.00022582 * std::pow(x, 19.5);
	return num / den * C::R_solar;
}

// Luminosity radiated at the accretion shock
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto luminosityAcc(amrex::Real m, amrex::Real r, amrex::Real mdot) -> amrex::Real
{
	return f_acc * f_k * C::Gconst * m * mdot / r;
}

// Luminosity radiated by the inner disk
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto luminosityDisk(amrex::Real m, amrex::Real r, amrex::Real mdot) -> amrex::Real
{
	return (1.0 - f_k) * C::Gconst * m * mdot / r;
}

// Luminosity of the protostar without the disk: L_ZAMS + L_acc, but no less than the Hayashi-track luminosity
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto luminosityStar(amrex::Real m, amrex::Real r, amrex::Real mdot) -> amrex::Real
{
	const amrex::Real l_star = luminosityZAMS(m) + luminosityAcc(m, r, mdot);
	const amrex::Real l_hayashi = 4.0 * M_PI * r * r * C::sigma_SB * std::pow(T_Hayashi, 4);
	return (l_star > l_hayashi) ? l_star : l_hayashi;
}
} // namespace StellarPhysics

/// Orion2007 deuterium-burning protostellar model. The star is a polytrope of index n whose radius follows from energy
/// conservation; it evolves through the phases of BurnState. Per-particle state: mdeut and n (extra reals) and
/// burn_state (extra int). A zero-initialized particle starts in PreCollapse with zero luminosity.
struct Orion2007StellarModel {
	static constexpr int nExtraReal = 2; // mdeut, n
	static constexpr int nExtraInt = 1;  // burn_state
	static constexpr std::array<const char *, nExtraReal> extraRealNames = {"mdeut", "n"};
	static constexpr std::array<const char *, nExtraInt> extraIntNames = {"burn_state"};

	enum class BurnState : int {
		PreCollapse = 0,	 // m < 0.01 M_sun or no accretion yet; model not started
		NoBurning,		 // contraction before deuterium ignition
		CoreDeuteriumFixedTc,	 // deuterium burning holds T_c at T_deut until the core supply is exhausted
		CoreDeuteriumVariableTc, // deuterium burned as fast as it is accreted
		ShellDeuterium,		 // radiative core; deuterium burned in a shell
		MainSequence		 // radius set to the ZAMS radius
	};

	static constexpr amrex::Real max_dr_fraction = 0.01; // largest change of r per sub-step from any one term of dr/dt
	static constexpr int max_substeps = 10000;

	// Deuterium-burning luminosity
	AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE static auto luminosityDeut(BurnState state, amrex::Real m, amrex::Real r, amrex::Real n, amrex::Real mdot)
	    -> amrex::Real
	{
		if (state == BurnState::CoreDeuteriumFixedTc) {
			const amrex::Real beta1 = StellarPhysics::beta(m, r, n);
			return StellarPhysics::luminosityStar(m, r, mdot) + mdot * StellarPhysics::psi_ion +
			       C::Gconst * m * mdot / r *
				   (1.0 - StellarPhysics::f_k -
				    StellarPhysics::aG(n) * beta1 / 2.0 * (1.0 + StellarPhysics::dlogBetaOverBetac_dlogM(m, r, n, beta1)));
		}
		if (state == BurnState::CoreDeuteriumVariableTc || state == BurnState::ShellDeuterium) {
			return mdot * StellarPhysics::psi_deut;
		}
		return 0.0;
	}

	struct RadiusRate {
		amrex::Real drdt;  // dr/dt
		amrex::Real scale; // largest single term of dr/dt; limits the sub-step
	};

	// dr/dt from the discretized Nakano et al. (2000) Eq. 5.8
	AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE static auto radiusRate(BurnState state, amrex::Real m, amrex::Real r, amrex::Real n, amrex::Real mdot)
	    -> RadiusRate
	{
		const amrex::Real beta1 = StellarPhysics::beta(m, r, n);
		const amrex::Real aGb = StellarPhysics::aG(n) * beta1;
		const amrex::Real accretion_term =
		    2.0 * mdot / m * r * (1.0 - (1.0 - StellarPhysics::f_k) / aGb + 0.5 * StellarPhysics::dlogBeta_dlogM(m, r, n, beta1));
		const amrex::Real lum_coef = 2.0 / aGb * r * r / (C::Gconst * m * m);
		const amrex::Real l_star = StellarPhysics::luminosityStar(m, r, mdot);
		const amrex::Real l_ion = mdot * StellarPhysics::psi_ion;
		const amrex::Real l_deut = luminosityDeut(state, m, r, n, mdot);
		return {.drdt = accretion_term - lum_coef * (l_star + l_ion - l_deut),
			.scale = std::max(std::abs(accretion_term), lum_coef * (l_star + l_ion + std::abs(l_deut)))};
	}

	// Stellar-evolution step over dt, using the mass and accretion rate set by the accretion module.
	//   rdata:    [in/out] particle real components; the extras start at StarParticleFixedComps + n_groups
	//   idata:    [in/out] particle integer components
	//   n_groups: [in]     number of radiation groups (= number of lum slots)
	//   dt:       [in]     timestep
	AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE static void evolve(amrex::Real *rdata, int *idata, int n_groups, amrex::Real dt)
	{
		// NOLINTBEGIN(cppcoreguidelines-pro-bounds-pointer-arithmetic)
		const amrex::Real m = rdata[StarParticleMassIdx];
		const amrex::Real mdot = rdata[StarParticleMdotIdx];
		amrex::Real &r = rdata[StarParticleRadiusIdx];
		amrex::Real &mdeut = rdata[StarParticleFixedComps + n_groups];
		amrex::Real &n = rdata[StarParticleFixedComps + n_groups + 1];
		auto state = static_cast<BurnState>(idata[0]);
		amrex::Real *lum = &rdata[StarParticleLumIdx];

		for (int g = 0; g < n_groups; ++g) {
			lum[g] = 0.0;
		}
		amrex::Real t_left = dt;
		if (state == BurnState::PreCollapse) {
			if (m < StellarPhysics::M_rad_min || mdot <= 0.0) {
				return;
			}
			n = StellarPhysics::nInit(mdot);
			r = StellarPhysics::radInit(mdot);
			state = BurnState::NoBurning;
			// the model starts when the mass passes M_rad_min, part way through this step
			t_left = std::min(dt, (m - StellarPhysics::M_rad_min) / mdot);
		}

		// Sub-cycle the explicit radius update. The mass grows linearly in time to m at the end of dt, and, as in a
		// single step, each sub-step uses the mass at its end. The sub-step is set so that no single term of dr/dt,
		// evaluated at the mass at its start, changes r by more than max_dr_fraction (the contraction and accretion
		// terms nearly cancel while the star contracts, so the net rate alone does not bound the step). The part of dt
		// left after max_substeps sub-steps is not integrated.
		const amrex::Real m_min = StellarPhysics::M_rad_min; // local copy: std::max takes references
		for (int k = 0; k < max_substeps && t_left > 0.0; ++k) {
			amrex::Real h = t_left;
			RadiusRate rate{.drdt = 0.0, .scale = 0.0};
			if (state != BurnState::MainSequence) {
				const amrex::Real m_start = std::max(m - mdot * t_left, m_min);
				h = std::min(t_left, max_dr_fraction * r / radiusRate(state, m_start, r, n, mdot).scale);
			}
			const amrex::Real ms = std::max(m - mdot * (t_left - h), m_min);
			if (state != BurnState::MainSequence) {
				rate = radiusRate(state, ms, r, n, mdot);
			}
			t_left -= h;
			r += rate.drdt * h;

			switch (state) {
				case BurnState::NoBurning:
					mdeut = ms;
					n = StellarPhysics::nInit(mdot);
					if (StellarPhysics::temperatureCentral(ms, r, n) > StellarPhysics::T_deut) {
						state = BurnState::CoreDeuteriumFixedTc;
						n = 1.5; // the star becomes fully convective
					}
					break;
				case BurnState::CoreDeuteriumFixedTc:
					mdeut += mdot * h - luminosityDeut(state, ms, r, n, mdot) * h / StellarPhysics::psi_deut;
					if (mdeut <= mdot * h) {
						state = BurnState::CoreDeuteriumVariableTc; // core deuterium exhausted
						mdeut = 0.0;
					}
					break;
				case BurnState::CoreDeuteriumVariableTc:
					mdeut = 0.0;
					if (luminosityDeut(state, ms, r, n, mdot) <= StellarPhysics::f_rad * StellarPhysics::luminosityZAMS(ms)) {
						state = BurnState::ShellDeuterium; // a radiative barrier forms
						n = 3.0;
						r *= StellarPhysics::shell_factor;
					}
					break;
				case BurnState::ShellDeuterium:
					mdeut = 0.0;
					if (r <= StellarPhysics::radiusZAMS(ms)) {
						state = BurnState::MainSequence;
						r = StellarPhysics::radiusZAMS(ms);
					}
					break;
				default: // BurnState::MainSequence
					mdeut = 0.0;
					r = StellarPhysics::radiusZAMS(ms);
					break;
			}
		}

		idata[0] = static_cast<int>(state);
		if (n_groups > 0) {
			lum[0] = StellarPhysics::luminosityStar(m, r, mdot) + StellarPhysics::luminosityDisk(m, r, mdot);
		}
		// NOLINTEND(cppcoreguidelines-pro-bounds-pointer-arithmetic)
	}
};

namespace Orion2Physics
{
// Parameters of ORION2 that differ from Orion2007 (the others are those of StellarPhysics)
constexpr amrex::Real M_rad_min = 0.04 * C::M_solar;	       // minimum mass at which the model is used
constexpr amrex::Real f_rad = 0.33;			       // shell burning starts when L_D <= f_rad L_ZAMS
constexpr amrex::Real psi_ion = 16.8 * C::ev2erg * C::n_A;     // energy per gram to dissociate and ionize the accreted gas
constexpr amrex::Real dm_fraction = 0.01;		       // mass step DM = 0.01 m of the numerical d/dlog(m)
constexpr amrex::Real r_negative_reset = 0.2 * C::R_solar;     // radius set when the update gives r < 0
constexpr amrex::Real timescale_fraction = 0.01;	       // sub-step bound: |r/(dr/dt)| / 100 and |m/mdot| / 100

// Mean gas-pressure fraction beta of a polytrope for log10(m/M_sun) = 0, 0.1, ..., 2.5 (rows) and n = 1.5, 1.6, ..., 3 (columns)
constexpr int n_beta_mass = 26;
constexpr int n_beta_poly = 16;
static AMREX_GPU_CONSTANT const amrex::GpuArray<amrex::Real, n_beta_mass * n_beta_poly> beta_table = {
    0.999492, 0.999504, 0.999515, 0.999525, 0.999534, 0.999541, 0.999548, 0.999553, 0.999558, 0.999562, 0.999566, 0.999569, 0.999571, 0.999572, 0.999573, 0.999573,
    0.999196, 0.999215, 0.999233, 0.999248, 0.999262, 0.999274, 0.999284, 0.999293, 0.999301, 0.999307, 0.999313, 0.999317, 0.99932, 0.999323, 0.999324, 0.999324,
    0.998728, 0.998759, 0.998787, 0.998811, 0.998832, 0.998851, 0.998867, 0.998881, 0.998894, 0.998904, 0.998912, 0.998919, 0.998924, 0.998928, 0.99893, 0.998931,
    0.997991, 0.99804, 0.998083, 0.998121, 0.998155, 0.998184, 0.99821, 0.998232, 0.998251, 0.998267, 0.998281, 0.998291, 0.9983, 0.998305, 0.998309, 0.99831,
    0.996834, 0.99691, 0.996978, 0.997037, 0.997089, 0.997135, 0.997176, 0.99721, 0.99724, 0.997265, 0.997286, 0.997303, 0.997316, 0.997325, 0.99733, 0.997332,
    0.995025, 0.995143, 0.995248, 0.99534, 0.995421, 0.995493, 0.995555, 0.995609, 0.995655, 0.995694, 0.995726, 0.995752, 0.995772, 0.995786, 0.995795, 0.995797,
    0.992221, 0.992402, 0.992562, 0.992703, 0.992827, 0.992936, 0.993032, 0.993114, 0.993185, 0.993245, 0.993294, 0.993334, 0.993365, 0.993386, 0.993399, 0.993403,
    0.987924, 0.988195, 0.988435, 0.988647, 0.988833, 0.988998, 0.989141, 0.989265, 0.989372, 0.989462, 0.989536, 0.989596, 0.989642, 0.989674, 0.989693, 0.9897,
    0.981449, 0.981844, 0.982193, 0.982503, 0.982776, 0.983017, 0.983227, 0.983409, 0.983565, 0.983697, 0.983806, 0.983894, 0.983961, 0.984008, 0.984036, 0.984045,
    0.971919, 0.972472, 0.972964, 0.973399, 0.973785, 0.974124, 0.97442, 0.974677, 0.974897, 0.975084, 0.975237, 0.975361, 0.975455, 0.975521, 0.97556, 0.975573,
    0.95832, 0.959059, 0.959716, 0.9603, 0.960817, 0.961271, 0.961669, 0.962014, 0.962309, 0.962559, 0.962764, 0.962929, 0.963054, 0.963142, 0.963194, 0.963211,
    0.939649, 0.940579, 0.941407, 0.942143, 0.942794, 0.943367, 0.943868, 0.944301, 0.944673, 0.944985, 0.945242, 0.945448, 0.945603, 0.945712, 0.945776, 0.945797,
    0.91513, 0.916223, 0.917197, 0.918062, 0.918826, 0.919497, 0.920083, 0.920588, 0.92102, 0.921382, 0.92168, 0.921915, 0.922094, 0.922217, 0.92229, 0.922313,
    0.884431, 0.885628, 0.886692, 0.887634, 0.888465, 0.889192, 0.889824, 0.890367, 0.890829, 0.891214, 0.891528, 0.891776, 0.891962, 0.892091, 0.892165, 0.892189,
    0.847787, 0.849009, 0.85009, 0.851042, 0.851877, 0.852605, 0.853233, 0.85377, 0.854223, 0.854598, 0.854902, 0.85514, 0.855317, 0.855439, 0.855508, 0.855531,
    0.805975, 0.80714, 0.808165, 0.809062, 0.809842, 0.810516, 0.811094, 0.811583, 0.811993, 0.812329, 0.812598, 0.812808, 0.812962, 0.813066, 0.813126, 0.813145,
    0.76016, 0.761206, 0.762117, 0.762907, 0.763588, 0.76417, 0.764664, 0.765078, 0.765421, 0.765699, 0.76592, 0.766089, 0.766212, 0.766295, 0.766341, 0.766356,
    0.711692, 0.71258, 0.713344, 0.713999, 0.714557, 0.715029, 0.715424, 0.71575, 0.716017, 0.71623, 0.716397, 0.716523, 0.716614, 0.716673, 0.716706, 0.716716,
    0.661917, 0.662632, 0.66324, 0.663753, 0.664184, 0.664542, 0.664838, 0.665078, 0.665271, 0.665422, 0.665538, 0.665623, 0.665683, 0.665722, 0.665742, 0.665748,
    0.612041, 0.612589, 0.613047, 0.613427, 0.61374, 0.613995, 0.614201, 0.614364, 0.614491, 0.614589, 0.614661, 0.614712, 0.614746, 0.614766, 0.614776, 0.614779,
    0.563063, 0.563461, 0.563787, 0.564051, 0.564262, 0.564429, 0.56456, 0.564659, 0.564733, 0.564786, 0.564822, 0.564846, 0.564859, 0.564866, 0.564868, 0.564868,
    0.51575, 0.516022, 0.516238, 0.516406, 0.516535, 0.516632, 0.516702, 0.516751, 0.516782, 0.516801, 0.51681, 0.516813, 0.51681, 0.516806, 0.516802, 0.516801,
    0.470653, 0.470824, 0.470952, 0.471044, 0.471109, 0.471151, 0.471175, 0.471186, 0.471186, 0.47118, 0.471169, 0.471156, 0.471142, 0.471131, 0.471122, 0.471119,
    0.428134, 0.428225, 0.428286, 0.428321, 0.428337, 0.428339, 0.428329, 0.428312, 0.42829, 0.428265, 0.42824, 0.428216, 0.428195, 0.428178, 0.428167, 0.428163,
    0.388399, 0.388431, 0.38844, 0.388434, 0.388415, 0.388387, 0.388354, 0.388317, 0.38828, 0.388243, 0.388208, 0.388177, 0.388151, 0.388131, 0.388118, 0.388114,
    0.351535, 0.351522, 0.351496, 0.35146, 0.351416, 0.351369, 0.351319, 0.35127, 0.351222, 0.351178, 0.351137, 0.351102, 0.351073, 0.351051, 0.351037, 0.351033};

// Initial polytropic index from the accretion rate, clamped to [1.5, 3]
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto nInit(amrex::Real mdot) -> amrex::Real
{
	if (mdot <= 0.0) {
		return 1.5;
	}
	const amrex::Real aG_init = 1.475 + 0.07 * std::log10(mdot * yr_in_s / C::M_solar);
	return std::min(std::max(5.0 - 3.0 / aG_init, 1.5), 3.0);
}

// Initial radius from the accretion rate, no less than 2 R_sun
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto radInit(amrex::Real mdot) -> amrex::Real
{
	return C::R_solar * std::max(2.5 * std::pow(mdot * yr_in_s / C::M_solar * 1.0e5, 0.2), 2.0);
}

// Central temperature; same equation as StellarPhysics::temperatureCentral, bracketed by [0, 2 max(T_gas, T_rad)] as in ORION2
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto temperatureCentral(amrex::Real m, amrex::Real r, amrex::Real n) -> amrex::Real
{
	const amrex::Real rho_c = StellarPhysics::rhoCentral(m, r, n);
	const amrex::Real P_c = StellarPhysics::pressureCentral(m, r, n);
	const amrex::Real T_gas = P_c * StellarPhysics::mu * StellarPhysics::m_H / (C::k_B * rho_c);
	const amrex::Real T_rad = std::pow(3.0 * P_c / C::a_rad, 0.25);
	return StellarPhysics::solveCentralTemperature(rho_c, P_c, 2.0 * std::max(T_gas, T_rad));
}

// Ratio of gas pressure to total pressure at the centre
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto betaCentral(amrex::Real m, amrex::Real r, amrex::Real n) -> amrex::Real
{
	const amrex::Real rho_c = StellarPhysics::rhoCentral(m, r, n);
	const amrex::Real P_c = StellarPhysics::pressureCentral(m, r, n);
	return rho_c * C::k_B * temperatureCentral(m, r, n) / (StellarPhysics::mu * StellarPhysics::m_H) / P_c;
}

// Mean ratio of gas pressure to total pressure, by bilinear interpolation in beta_table: 1 for m <= 1 M_sun, and
// beta(last row) (m / 10^2.6 M_sun)^(-1/2) above the table. Two bugs of ORION2's beta() are fixed here:
//  (a) ORION2 computes the row as floor(logm - logm0 / dlogm) = floor(logm), so between 1 and 10 M_sun it extrapolates
//      from the first row with weights up to 10; the row is floor((logm - logm0) / dlogm), clamped to n_beta_mass - 2
//      (with the mass weight clamped to 1) so that row + 1 stays in the table for 2.5 <= logm < 2.6;
//  (b) for n = 3 exactly, ORION2 uses column nn - 1 with weight 1 and so reads column nn, outside the table; here
//      column nn - 2 with weight 1 is used, as ORION2 does above the table.
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto beta(amrex::Real m, amrex::Real n) -> amrex::Real
{
	constexpr amrex::Real logm0 = 0.0;
	constexpr amrex::Real dlogm_tab = 0.1;
	constexpr amrex::Real n0 = 1.5;
	constexpr amrex::Real dn = 0.1;
	const amrex::Real logm = std::log10(m / C::M_solar);
	if (logm <= logm0) {
		return 1.0;
	}
	// column and weight in n; n = 1.5 and n = 3 are handled exactly to avoid roundoff
	int nidx = std::min(std::max(static_cast<int>(std::floor((n - n0) / dn)), 0), n_beta_poly - 2);
	amrex::Real nwgt = (n - (n0 + nidx * dn)) / dn;
	if (n == 1.5) {
		nidx = 0;
		nwgt = 0.0;
	} else if (n == 3.0) {
		nidx = n_beta_poly - 2;
		nwgt = 1.0;
	}
	auto b = [](int i, int j) { return beta_table[i * n_beta_poly + j]; };
	if (logm < logm0 + n_beta_mass * dlogm_tab) {
		const int midx = std::min(static_cast<int>(std::floor((logm - logm0) / dlogm_tab)), n_beta_mass - 2);
		const amrex::Real mwgt = std::min((logm - (logm0 + midx * dlogm_tab)) / dlogm_tab, 1.0);
		return b(midx, nidx) * (1.0 - mwgt) * (1.0 - nwgt) + b(midx + 1, nidx) * mwgt * (1.0 - nwgt) + b(midx, nidx + 1) * (1.0 - mwgt) * nwgt +
		       b(midx + 1, nidx + 1) * mwgt * nwgt;
	}
	// above the table: interpolate the last row in n, then beta ~ m^(-1/2)
	const amrex::Real beta_table_end = (1.0 - nwgt) * b(n_beta_mass - 1, nidx) + nwgt * b(n_beta_mass - 1, nidx + 1);
	return beta_table_end / std::sqrt(std::pow(10.0, logm - logm0 - n_beta_mass * dlogm_tab));
}

// d log(beta) / d log(m) at fixed n, one-sided difference with DM = 0.01 m
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto dlogBeta_dlogM(amrex::Real m, amrex::Real n, amrex::Real beta1) -> amrex::Real
{
	const amrex::Real dm = dm_fraction * m;
	const amrex::Real beta2 = beta(m + dm, n);
	return m / beta1 * (beta2 - beta1) / dm;
}

// d log(beta / beta_c) / d log(m) at fixed r and n (zero for n = 3, where beta = beta_c)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto dlogBetaOverBetac_dlogM(amrex::Real m, amrex::Real r, amrex::Real n, amrex::Real beta1) -> amrex::Real
{
	if (n == 3.0) {
		return 0.0;
	}
	const amrex::Real dm = dm_fraction * m;
	const amrex::Real beta2 = beta(m + dm, n);
	const amrex::Real betac1 = betaCentral(m, r, n);
	const amrex::Real betac2 = betaCentral(m + dm, r, n);
	return m / (beta1 / betac1) * ((beta2 / betac2) - (beta1 / betac1)) / dm;
}

// Luminosity of the protostar without accretion or disk: L_ZAMS, but no less than the Hayashi-track luminosity
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto luminosityStar(amrex::Real m, amrex::Real r) -> amrex::Real
{
	const amrex::Real l_star = StellarPhysics::luminosityZAMS(m);
	const amrex::Real T_eff = std::pow(l_star / (4.0 * M_PI * r * r * C::sigma_SB), 0.25);
	if (T_eff > StellarPhysics::T_Hayashi) {
		return l_star;
	}
	return 4.0 * M_PI * r * r * C::sigma_SB * std::pow(StellarPhysics::T_Hayashi, 4);
}

// Luminosity radiated at the accretion shock, net of the energy used to dissociate and ionize the gas (may be negative)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto luminosityAcc(amrex::Real m, amrex::Real r, amrex::Real mdot) -> amrex::Real
{
	return StellarPhysics::f_acc * (StellarPhysics::f_k * C::Gconst * m / r - psi_ion) * mdot;
}

// Luminosity radiated by the inner disk, net of the energy used to dissociate and ionize the gas (may be negative)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto luminosityDisk(amrex::Real m, amrex::Real r, amrex::Real mdot) -> amrex::Real
{
	return ((1.0 - StellarPhysics::f_k) * (C::Gconst * m / r - psi_ion)) * mdot;
}
} // namespace Orion2Physics

/// Orion2 deuterium-burning protostellar model: a port of StarParticleData of the ORION2 code (the model of Offner et
/// al. 2009, Appendix B), with two bugs of its beta() fixed (see Orion2Physics::beta). Per-particle state: mdeut, n
/// and netlum (extra reals; netlum is the negative luminosity carried to the next step) and burn_state (extra int,
/// numbered as in Orion2007StellarModel). A zero-initialized particle starts in PreCollapse with zero luminosity.
struct Orion2StellarModel {
	static constexpr int nExtraReal = 3; // mdeut, n, netlum
	static constexpr int nExtraInt = 1;  // burn_state
	static constexpr std::array<const char *, nExtraReal> extraRealNames = {"mdeut", "n", "netlum"};
	static constexpr std::array<const char *, nExtraInt> extraIntNames = {"burn_state"};

	// ORION2 names: Uninitialized, None, VariableCoreDeuterium, SteadyCoreDeuterium, ShellDeuterium, ZAMS
	using BurnState = Orion2007StellarModel::BurnState;

	// Deuterium-burning luminosity; the mean beta (beta1) is used only in CoreDeuteriumFixedTc
	AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE static auto luminosityDeut(BurnState state, amrex::Real m, amrex::Real r, amrex::Real n, amrex::Real mdot,
									     amrex::Real beta1) -> amrex::Real
	{
		if (state == BurnState::CoreDeuteriumFixedTc) {
			return Orion2Physics::luminosityStar(m, r) + mdot * Orion2Physics::psi_ion +
			       C::Gconst * m * mdot / r *
				   (1.0 - StellarPhysics::f_k -
				    StellarPhysics::aG(n) * beta1 / 2.0 * (1.0 + Orion2Physics::dlogBetaOverBetac_dlogM(m, r, n, beta1)));
		}
		if (state == BurnState::CoreDeuteriumVariableTc || state == BurnState::ShellDeuterium) {
			return mdot * StellarPhysics::psi_deut;
		}
		return 0.0;
	}

	// dr/dt from the discretized Nakano et al. (2000) Eq. 5.8, in the form of ORION2's updateState
	AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE static auto radiusRate(BurnState state, amrex::Real m, amrex::Real r, amrex::Real n, amrex::Real mdot)
	    -> amrex::Real
	{
		const amrex::Real beta1 = Orion2Physics::beta(m, n);
		const amrex::Real aG = StellarPhysics::aG(n);
		const amrex::Real f_k = StellarPhysics::f_k;
		return 2.0 * mdot / m * r * (f_k / (aG * beta1) + 1.0 - 1.0 / (aG * beta1)) +
		       beta1 / m * Orion2Physics::dlogBeta_dlogM(m, n, beta1) * mdot * r / beta1 -
		       2.0 / (beta1 * aG) * r * r / (C::Gconst * m * m) *
			   (Orion2Physics::luminosityStar(m, r) + mdot * Orion2Physics::psi_ion - luminosityDeut(state, m, r, n, mdot, beta1));
	}

	// Stellar-evolution step over dt, using the mass and accretion rate set by the accretion module.
	//   rdata:    [in/out] particle real components; the extras start at StarParticleFixedComps + n_groups
	//   idata:    [in/out] particle integer components
	//   n_groups: [in]     number of radiation groups (= number of lum slots)
	//   dt:       [in]     timestep
	AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE static void evolve(amrex::Real *rdata, int *idata, int n_groups, amrex::Real dt)
	{
		// NOLINTBEGIN(cppcoreguidelines-pro-bounds-pointer-arithmetic)
		const amrex::Real m = rdata[StarParticleMassIdx];
		const amrex::Real mdot = rdata[StarParticleMdotIdx];
		amrex::Real &r = rdata[StarParticleRadiusIdx];
		amrex::Real &mdeut = rdata[StarParticleFixedComps + n_groups];
		amrex::Real &n = rdata[StarParticleFixedComps + n_groups + 1];
		amrex::Real &netlum = rdata[StarParticleFixedComps + n_groups + 2];
		auto state = static_cast<BurnState>(idata[0]);
		amrex::Real *lum = &rdata[StarParticleLumIdx];

		for (int g = 0; g < n_groups; ++g) {
			lum[g] = 0.0;
		}
		if (state == BurnState::PreCollapse) {
			mdeut = m; // ORION2 counts all accreted mass as deuterium-bearing
			if (m < Orion2Physics::M_rad_min || mdot <= 0.0) {
				netlum = 0.0;
				return;
			}
			n = Orion2Physics::nInit(mdot);
			r = Orion2Physics::radInit(mdot);
			state = BurnState::NoBurning;
		} else {
			mdeut += mdot * dt;
		}

		// Radius update at fixed m and mdot. If |r/(dr/dt)|/100 or |m/mdot|/100 (tested in this order) is shorter than
		// dt, take ceil(dt / that time) equal forward-Euler sub-steps, otherwise one step.
		if (state != BurnState::MainSequence) {
			const amrex::Real drdt = radiusRate(state, m, r, n, mdot);
			const amrex::Real rdot_time = std::abs(r / drdt) * Orion2Physics::timescale_fraction;
			const amrex::Real mdot_time = std::abs(m / mdot) * Orion2Physics::timescale_fraction;
			int64_t n_sub = 1;
			if (rdot_time < dt) {
				n_sub = static_cast<int64_t>(std::ceil(dt / rdot_time));
			} else if (mdot_time < dt) {
				n_sub = static_cast<int64_t>(std::ceil(dt / mdot_time));
			}
			const amrex::Real h = dt / static_cast<amrex::Real>(n_sub);
			for (int64_t k = 0; k < n_sub; ++k) {
				r += h * radiusRate(state, m, r, n, mdot);
			}
			if (r < 0.0) {
				r = Orion2Physics::r_negative_reset;
			}
		}

		// Phase update, once per step with the full dt
		switch (state) {
			case BurnState::NoBurning:
				n = Orion2Physics::nInit(mdot);
				if (Orion2Physics::temperatureCentral(m, r, n) > StellarPhysics::T_deut) {
					state = BurnState::CoreDeuteriumFixedTc;
					n = 1.5; // the star becomes fully convective
				}
				break;
			case BurnState::CoreDeuteriumFixedTc:
				mdeut -= luminosityDeut(state, m, r, n, mdot, Orion2Physics::beta(m, n)) * dt / StellarPhysics::psi_deut;
				if (mdeut <= mdot * dt) {
					state = BurnState::CoreDeuteriumVariableTc; // core deuterium exhausted
					mdeut = 0.0;
				}
				break;
			case BurnState::CoreDeuteriumVariableTc:
				mdeut = 0.0;
				if (luminosityDeut(state, m, r, n, mdot, 1.0) <= Orion2Physics::f_rad * StellarPhysics::luminosityZAMS(m)) {
					state = BurnState::ShellDeuterium; // a radiative barrier forms
					n = 3.0;
					r *= StellarPhysics::shell_factor;
				}
				break;
			case BurnState::ShellDeuterium:
				mdeut = 0.0;
				if (r <= StellarPhysics::radiusZAMS(m)) {
					state = BurnState::MainSequence;
					r = StellarPhysics::radiusZAMS(m);
				}
				break;
			default: // BurnState::MainSequence: the radius is kept
				mdeut = 0.0;
				break;
		}
		idata[0] = static_cast<int>(state);

		// Luminosity, including a negative luminosity carried over from the previous step; a negative total is
		// carried to the next step and zero is stored
		amrex::Real L = Orion2Physics::luminosityStar(m, r) + Orion2Physics::luminosityAcc(m, r, mdot) + Orion2Physics::luminosityDisk(m, r, mdot);
		if (netlum < 0.0) {
			L += netlum;
		}
		if (L < 0.0) {
			netlum = L;
			L = 0.0;
		} else {
			netlum = 0.0;
		}
		if (n_groups > 0) {
			lum[0] = L;
		}
		// NOLINTEND(cppcoreguidelines-pro-bounds-pointer-arithmetic)
	}
};

} // namespace quokka

#endif // DEUTERIUM_STELLAR_MODEL_HPP_
