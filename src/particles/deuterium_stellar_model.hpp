#ifndef DEUTERIUM_STELLAR_MODEL_HPP_
#define DEUTERIUM_STELLAR_MODEL_HPP_

/// \file deuterium_stellar_model.hpp
/// \brief One-zone protostellar evolution model with deuterium burning (Offner et al. 2009, Appendix B;
/// Nakano et al. 1995, 2000; Tan & McKee 2004). The update follows the ORION2 StarParticleData module.

#include <algorithm>
#include <array>
#include <cmath>

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

// Central temperature from P_c = rho_c k_B T_c / (mu m_H) + a T_c^4 / 3
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto temperatureCentral(amrex::Real m, amrex::Real r, amrex::Real n) -> amrex::Real
{
	const amrex::Real rho_c = rhoCentral(m, r, n);
	const amrex::Real P_c = pressureCentral(m, r, n);
	const amrex::Real T_gas = P_c * mu * m_H / (C::k_B * rho_c);
	const amrex::Real T_rad = std::pow(3.0 * P_c / C::a_rad, 0.25);
	auto f = [=](amrex::Real T) { return P_c - rho_c * C::k_B * T / (mu * m_H) - C::a_rad * T * T * T * T / 3.0; };
	int max_iter = 100;
	const auto [T_lo, T_hi] = quokka::math::toms748_solve(f, 0.0, std::max(T_gas, T_rad), quokka::math::eps_tolerance<amrex::Real>(1.0e-12), max_iter);
	return 0.5 * (T_lo + T_hi);
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

/// Deuterium-burning protostellar model. The star is a polytrope of index n whose radius follows from energy
/// conservation; it evolves through the phases of BurnState. Per-particle state: mdeut and n (extra reals) and
/// burn_state (extra int). A zero-initialized particle starts in PreCollapse with zero luminosity.
struct DeuteriumBurningStellarModel {
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

} // namespace quokka

#endif // DEUTERIUM_STELLAR_MODEL_HPP_
