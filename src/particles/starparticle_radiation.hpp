#ifndef STAR_PARTICLE_DATA_HPP_
#define STAR_PARTICLE_DATA_HPP_

#include "AMReX_Extension.H"
#include "AMReX_GpuQualifiers.H"
#include "AMReX_REAL.H"
#include "fundamental_constants.H"
#include "math/interpolate.hpp"
#include "math/root_finding.hpp"
#include "particles/particle_radiation.hpp"
#include <cmath>

namespace quokka
{

// Constants for stellar astrophysics

namespace StellarConstants
{
static constexpr amrex::Real M_rad_min = 0.01 * C::M_solar; // minimum mass for model

// Model parameters
static constexpr amrex::Real F_acc = 0.5;	  // fraction of accreted energy radiated
static constexpr amrex::Real F_k = 0.5;		  // fraction of energy from inner disk
static constexpr amrex::Real F_rad = 0.33;	  // radiative barrier parameter
static constexpr amrex::Real shell_factor = 2.1;  // radius increase for shell burning
static constexpr amrex::Real T_Hayashi = 3000.0;  // Hayashi temperature
static constexpr amrex::Real T_deuterium = 1.5e6; // deuterium ignition temperature
} // namespace StellarConstants

// Central density/pressure tables (GPU-compatible static data)

namespace StellarTables
{
// NOLINTBEGIN(hicpp-avoid-c-arrays)
// rho_mean/rho_c table for n=1.5 to 3.0 (step 0.1)
static constexpr int n_rho_table = 17;
AMREX_GPU_CONSTANT constexpr amrex::Real n_table[n_rho_table] = { // NOLINT
    1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0, 3.1};
AMREX_GPU_CONSTANT constexpr amrex::Real rho_factor_table[n_rho_table] = { // NOLINT
    0.166931,  0.14742,	  0.129933,  0.114265,	0.100242,  0.0877,    0.0764968, 0.0665109, 0.0576198,
    0.0497216, 0.0427224, 0.0365357, 0.0310837, 0.0262952, 0.0221057, 0.0184553, 0.01529};

// P_c/(G M^2/R^4) table for n=1.5 to 3.0 (step 0.1)
AMREX_GPU_CONSTANT constexpr amrex::Real pressure_factor_table[n_rho_table] = { // NOLINT
    0.770087, 0.889001, 1.02979, 1.19731, 1.39753, 1.63818, 1.92909, 2.2825, 2.71504, 3.24792, 3.90921, 4.73657, 5.78067, 7.11088, 8.82286, 11.0515, 13.9885};

// Beta table for interpolation (M: 5-50 Msun, n: 1.5-3.0)
static constexpr int n_beta_mass = 19;
static constexpr int n_beta_poly = 4;
AMREX_GPU_CONSTANT constexpr amrex::Real beta_table[n_beta_mass][n_beta_poly] = { // NOLINT
    {0.98785, 0.988928, 0.98947, 0.989634},   {0.97438, 0.976428, 0.977462, 0.977774},	{0.957927, 0.960895, 0.962397, 0.962846},
    {0.939787, 0.943497, 0.945369, 0.945922}, {0.92091, 0.925151, 0.927276, 0.927896},	{0.901932, 0.906512, 0.908785, 0.909436},
    {0.883254, 0.888017, 0.890353, 0.891013}, {0.865111, 0.86994, 0.872277, 0.872927},	{0.847635, 0.852445, 0.854739, 0.855367},
    {0.830886, 0.835619, 0.837842, 0.838441}, {0.814885, 0.8195, 0.821635, 0.822201},	{0.799625, 0.804095, 0.806133, 0.806664},
    {0.785082, 0.789394, 0.791328, 0.791825}, {0.771226, 0.775371, 0.777202, 0.777665}, {0.758022, 0.761997, 0.763726, 0.764156},
    {0.745433, 0.749238, 0.750869, 0.751268}, {0.733423, 0.73706, 0.738596, 0.738966},	{0.721954, 0.725429, 0.726874, 0.727216},
    {0.710993, 0.714311, 0.715671, 0.715987}};

// Tout et al. (1996) main sequence fitting parameters
namespace Tout96
{
static constexpr amrex::Real alpha = 0.39704170;
static constexpr amrex::Real beta = 8.52762600;
static constexpr amrex::Real gamma = 0.00025546;
static constexpr amrex::Real delta = 5.43288900;
static constexpr amrex::Real epsilon = 5.56357900;
static constexpr amrex::Real zeta = 0.78866060;
static constexpr amrex::Real eta = 0.00586685;
static constexpr amrex::Real theta = 1.71535900;
static constexpr amrex::Real iota = 6.59778800;
static constexpr amrex::Real kappa = 10.08855000;
static constexpr amrex::Real lambda = 1.01249500;
static constexpr amrex::Real gmu = 0.07490166;
static constexpr amrex::Real nu = 0.01077422;
static constexpr amrex::Real xi = 3.08223400;
static constexpr amrex::Real upsilon = 17.84778000;
static constexpr amrex::Real gpi = 0.00022582;
} // namespace Tout96
// NOLINTEND(hicpp-avoid-c-arrays)
} // namespace StellarTables

// Utility functions for stellar physics

namespace StellarPhysics
{
// Solar properties
using C::M_solar;
static constexpr amrex::Real L_solar = 3.828e+33; // erg/s, CODATA 2022
using C::R_solar;

// Physical constants
using C::a_rad; // radiation constant
using C::Gconst;
using C::k_B;
using C::m_u;
using C::sigma_SB;
static constexpr amrex::Real mu = 0.613; // mean molecular weight

// Uses burningState enum from particle_types.hpp

// Initialize polytropic index from accretion rate
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto n_init(amrex::Real mdot) -> amrex::Real
{
	if (mdot == 0.0) {
		return 1.5;
	}

	amrex::Real log_term = std::log10(mdot * seconds_per_year / M_solar);
	amrex::Real aG_init = 1.475 + 0.07 * log_term;
	amrex::Real n_val = 5.0 - 3.0 / aG_init;

	// Clamp to valid range
	n_val = std::max(1.5, std::min(3.0, n_val));
	return n_val;
}

// Initialize stellar radius from accretion rate
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto rad_init(amrex::Real mdot) -> amrex::Real
{
	amrex::Real mdot_norm = mdot * seconds_per_year / M_solar * 1.0e5;
	amrex::Real rad_factor = 2.5 * std::pow(mdot_norm, 0.2);
	rad_factor = std::max(rad_factor, 2.0); // Minimum factor of 2

	return R_solar * rad_factor;
}

// Polytropic gravitational energy parameter
AMREX_GPU_DEVICE AMREX_FORCE_INLINE auto aG(amrex::Real n) -> amrex::Real { return 3.0 / (5.0 - n); }

// Interpolate from central density table
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto rho_factor_interp(amrex::Real n) -> amrex::Real
{
	return interpolate_value<BoundaryPolicy::Clamp>(n, &StellarTables::n_table[0], &StellarTables::rho_factor_table[0], StellarTables::n_rho_table);
}

// Interpolate from central pressure table
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto pressure_factor_interp(amrex::Real n) -> amrex::Real
{
	return interpolate_value<BoundaryPolicy::Clamp>(n, &StellarTables::n_table[0], &StellarTables::pressure_factor_table[0], StellarTables::n_rho_table);
}

// Central density of polytropic star
AMREX_GPU_DEVICE AMREX_FORCE_INLINE auto rho_central(amrex::Real mass, amrex::Real radius, amrex::Real n) -> amrex::Real
{
	amrex::Real volume = (4.0 / 3.0) * M_PI * radius * radius * radius;
	amrex::Real rho_mean = mass / volume;
	amrex::Real rho_factor = rho_factor_interp(n);

	return rho_mean / rho_factor;
}

// Central pressure of polytropic star
AMREX_GPU_DEVICE AMREX_FORCE_INLINE auto pressure_central(amrex::Real mass, amrex::Real radius, amrex::Real n) -> amrex::Real
{
	amrex::Real p_factor = pressure_factor_interp(n);
	return p_factor * Gconst * mass * mass / (radius * radius * radius * radius);
}

// Central temperature from equation of state
AMREX_GPU_DEVICE AMREX_FORCE_INLINE auto temperature_central(amrex::Real mass, amrex::Real radius, amrex::Real n) -> amrex::Real
{
	amrex::Real rho_c = rho_central(mass, radius, n);
	amrex::Real P_c = pressure_central(mass, radius, n);

	// Gas temperature if radiation pressure negligible
	amrex::Real T_gas = P_c * mu * m_u / (k_B * rho_c);

	// Radiation temperature if gas pressure negligible
	amrex::Real T_rad = std::pow(3.0 * P_c / a_rad, 0.25);

	// Solve full EOS: P = ρkT/μm_u + aT⁴/3 using TOMS 748 root finder
	auto pressure_func = [=](amrex::Real T) { return P_c - rho_c * k_B * T / (mu * m_u) - (a_rad * std::pow(T, 4)) / 3.0; };

	// Lower bound at 1 K (f > 0); upper bound at 2x the pure-gas or pure-radiation estimate (f < 0)
	const amrex::Real T_low = 1.0;
	const amrex::Real T_high = (T_rad > T_gas) ? 2.0 * T_rad : 2.0 * T_gas;
	int max_iter = 60;
	const auto [T_lo, T_hi] = quokka::math::toms748_solve(pressure_func, T_low, T_high, quokka::math::eps_tolerance<amrex::Real>(1.0e-7), max_iter);
	return 0.5 * (T_lo + T_hi);
}

// Beta parameter (gas pressure / total pressure) for n=3 (Eddington quartic)
AMREX_GPU_DEVICE AMREX_FORCE_INLINE auto beta_n3(amrex::Real mass, amrex::Real radius, amrex::Real n) -> amrex::Real
{
	amrex::Real rho_c = rho_central(mass, radius, n);
	amrex::Real P_c = pressure_central(mass, radius, n);

	amrex::Real coefficient = 3.0 / a_rad * std::pow(k_B * rho_c / (mu * m_u), 4);

	auto beta_func = [=](amrex::Real beta) { return std::pow(P_c, 3) - coefficient * (1.0 - beta) / std::pow(beta, 4); };

	// f(1e-4) < 0 (radiation-dominated), f(1.0) = P_c^3 > 0 (gas-dominated)
	int max_iter = 60;
	const auto [beta_lo, beta_hi] = quokka::math::toms748_solve(beta_func, 1.0e-4, 1.0, quokka::math::eps_tolerance<amrex::Real>(1.0e-7), max_iter);
	return 0.5 * (beta_lo + beta_hi);
}

// Beta parameter from interpolation table
AMREX_GPU_DEVICE AMREX_FORCE_INLINE auto beta_table_interp(amrex::Real mass, amrex::Real n) -> amrex::Real
{
	// Table parameters
	constexpr amrex::Real M_min = 5.0 * M_solar;
	constexpr amrex::Real M_max = 50.0 * M_solar;
	constexpr amrex::Real M_step = 2.5 * M_solar;
	constexpr amrex::Real n_min = 1.5;
	constexpr amrex::Real n_max = 3.0;
	constexpr amrex::Real n_step = 0.5;

	// Clamp to table bounds
	if (mass < M_min) {
		return 1.0; // Fully gas pressure supported for low mass
	}
	if (mass >= M_max || n >= n_max) {
		return -1.0; // Out of table bounds
	}

	// Mass index
	amrex::Real m_idx_real = (mass - M_min) / M_step;
	int m_idx = static_cast<int>(std::floor(m_idx_real));
	amrex::Real m_weight = m_idx_real - m_idx;

	// Polytropic index
	amrex::Real n_idx_real = (n - n_min) / n_step;
	int n_idx = static_cast<int>(std::floor(n_idx_real));
	amrex::Real n_weight = n_idx_real - n_idx;

	// Clamp indices
	m_idx = std::max(0, std::min(StellarTables::n_beta_mass - 2, m_idx));
	n_idx = std::max(0, std::min(StellarTables::n_beta_poly - 2, n_idx));

	// Bilinear interpolation
	amrex::Real beta00 = StellarTables::beta_table[m_idx][n_idx];
	amrex::Real beta10 = StellarTables::beta_table[m_idx + 1][n_idx];
	amrex::Real beta01 = StellarTables::beta_table[m_idx][n_idx + 1];
	amrex::Real beta11 = StellarTables::beta_table[m_idx + 1][n_idx + 1];

	return beta00 * (1.0 - m_weight) * (1.0 - n_weight) + beta10 * m_weight * (1.0 - n_weight) + beta01 * (1.0 - m_weight) * n_weight +
	       beta11 * m_weight * n_weight;
}

// Unified beta function
AMREX_GPU_DEVICE AMREX_FORCE_INLINE auto beta_total(amrex::Real mass, amrex::Real radius, amrex::Real n) -> amrex::Real
{
	if (n == 3.0) {
		return beta_n3(mass, radius, n);
	}
	return beta_table_interp(mass, n);
}

// ZAMS luminosity from Tout et al. (1996)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto luminosity_ZAMS(amrex::Real mass) -> amrex::Real
{
	amrex::Real m_sol = mass / M_solar;

	amrex::Real numerator = StellarTables::Tout96::alpha * std::pow(m_sol, 5.5) + StellarTables::Tout96::beta * std::pow(m_sol, 11.0);
	amrex::Real denominator = StellarTables::Tout96::gamma + std::pow(m_sol, 3.0) + StellarTables::Tout96::delta * std::pow(m_sol, 5.0) +
				  StellarTables::Tout96::epsilon * std::pow(m_sol, 7.0) + StellarTables::Tout96::zeta * std::pow(m_sol, 8.0) +
				  StellarTables::Tout96::eta * std::pow(m_sol, 9.5);

	return (numerator / denominator) * L_solar;
}

// ZAMS radius from Tout et al. (1996)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto radius_ZAMS(amrex::Real mass) -> amrex::Real
{
	amrex::Real m_sol = mass / M_solar;

	amrex::Real numerator = StellarTables::Tout96::theta * std::pow(m_sol, 2.5) + StellarTables::Tout96::iota * std::pow(m_sol, 6.5) +
				StellarTables::Tout96::kappa * std::pow(m_sol, 11.0) + StellarTables::Tout96::lambda * std::pow(m_sol, 19.0) +
				StellarTables::Tout96::gmu * std::pow(m_sol, 19.5);
	amrex::Real denominator = StellarTables::Tout96::nu + StellarTables::Tout96::xi * std::pow(m_sol, 2.0) +
				  StellarTables::Tout96::upsilon * std::pow(m_sol, 8.5) + std::pow(m_sol, 18.5) +
				  StellarTables::Tout96::gpi * std::pow(m_sol, 19.5);

	return (numerator / denominator) * R_solar;
}

// Stellar luminosity (ZAMS + accretion)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto luminosity_star(amrex::Real mass, amrex::Real radius, amrex::Real mdot) -> amrex::Real
{
	amrex::Real L_zams = luminosity_ZAMS(mass);
	amrex::Real L_acc = StellarConstants::F_acc * StellarConstants::F_k * Gconst * mass * mdot / radius;
	amrex::Real L_total = L_zams + L_acc;

	// Hayashi limit check
	amrex::Real T_eff = std::pow(L_total / (4.0 * M_PI * radius * radius * sigma_SB), 0.25);
	if (T_eff > StellarConstants::T_Hayashi) {
		return L_total;
	}
	return 4.0 * M_PI * radius * radius * sigma_SB * std::pow(StellarConstants::T_Hayashi, 4);
}

// Disk luminosity
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto luminosity_disk(amrex::Real mass, amrex::Real radius, amrex::Real mdot) -> amrex::Real
{
	return (1.0 - StellarConstants::F_k) * Gconst * mass * mdot / radius;
}

// Total luminosity
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto luminosity_total(amrex::Real mass, amrex::Real radius, amrex::Real mdot, burningState burn_state) -> amrex::Real
{
	if (burn_state == burningState::Uninitialized) {
		return 0.0;
	}

	amrex::Real L_star = luminosity_star(mass, radius, mdot);
	amrex::Real L_disk = luminosity_disk(mass, radius, mdot);

	return L_star + L_disk;
}
} // namespace StellarPhysics

// Main update function for stellar particles

#if AMREX_SPACEDIM == 3

class StellarUpdate
{
      public:
	template <typename problem_t, typename ParticleType, int Nout>
	AMREX_GPU_DEVICE AMREX_FORCE_INLINE static void updateStellarProperties(ParticleType &p, amrex::Real /*current_time*/, amrex::Real dt,
										LuminosityGpuConstTables<Nout> const & /*gpu_tables*/) noexcept
	{
		// Call the internal update function
		updateStellarPropertiesImpl(p, dt);
	}

      private:
	template <typename ParticleType> AMREX_GPU_DEVICE AMREX_FORCE_INLINE static void updateStellarPropertiesImpl(ParticleType &p, amrex::Real dt) noexcept
	{
		// Get current values using correct StarParticle indices
		const amrex::Real mass = p.rdata(StarParticleMassIdx);
		amrex::Real mdeut = p.rdata(StarParticleMdeutIdx);
		const amrex::Real mdot = p.rdata(StarParticleMdotIdx); // already computed by UpdateParticleMassAndMomentumInBox
		amrex::Real n = p.rdata(StarParticleNIdx);
		amrex::Real radius = p.rdata(StarParticleRadiusIdx);
		auto burn_state = static_cast<burningState>(p.idata(StarParticleBurnStateIdx));

		// Initialize if needed
		if (burn_state == burningState::Uninitialized) {
			if (mass < StellarConstants::M_rad_min || mdot == 0.0) {
				return;
			}

			n = StellarPhysics::n_init(mdot);
			radius = StellarPhysics::rad_init(mdot);
			burn_state = burningState::None;

			p.rdata(StarParticleNIdx) = n;
			p.rdata(StarParticleRadiusIdx) = radius;
			p.idata(StarParticleBurnStateIdx) = static_cast<int>(burn_state);
		}

		// Update deuterium mass using already-computed mdot (after early return check)
		mdeut += mdot * dt;

		// Update burning state
		if (burn_state == burningState::None) {
			n = StellarPhysics::n_init(mdot);
			if (StellarPhysics::temperature_central(mass, radius, n) > StellarConstants::T_deuterium) {
				burn_state = burningState::VariableCoreDeuterium;
				n = 1.5;
			}
		} else if (burn_state == burningState::VariableCoreDeuterium) {
			mdeut -= 0.1 * mdot * dt;
			if (mdeut <= mdot * dt) {
				burn_state = burningState::SteadyCoreDeuterium;
				mdeut = 0.0;
			}
		} else if (burn_state == burningState::SteadyCoreDeuterium) {
			mdeut = 0.0;
			if (StellarPhysics::luminosity_star(mass, radius, mdot) < StellarConstants::F_rad * StellarPhysics::luminosity_ZAMS(mass)) {
				burn_state = burningState::ShellDeuterium;
				n = 3.0;
				radius *= StellarConstants::shell_factor;
			}
		} else if (burn_state == burningState::ShellDeuterium) {
			mdeut = 0.0;
			if (radius <= StellarPhysics::radius_ZAMS(mass)) {
				burn_state = burningState::ZAMS;
				radius = StellarPhysics::radius_ZAMS(mass);
			}
		} else if (burn_state == burningState::ZAMS) {
			mdeut = 0.0;
		}

		// Compute and store luminosity
		const amrex::Real lum = StellarPhysics::luminosity_total(mass, radius, mdot, burn_state);
		p.rdata(StarParticleLumIdx) = lum;

		// Update particle data
		p.rdata(StarParticleMdeutIdx) = mdeut;
		p.rdata(StarParticleNIdx) = n;
		p.rdata(StarParticleRadiusIdx) = radius;
		p.idata(StarParticleBurnStateIdx) = static_cast<int>(burn_state);
	}
};

#endif // AMREX_SPACEDIM == 3

} // namespace quokka

#endif // STAR_PARTICLE_DATA_HPP_
