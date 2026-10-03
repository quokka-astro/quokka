#ifndef CHABRIER_IMF_HPP_ // NOLINT
#define CHABRIER_IMF_HPP_
//==============================================================================
// Quokka -- two-moment radiation hydrodynamics on AMR grids
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file chabrier_imf.hpp
/// \brief Integrals of the Chabrier (2005) system IMF used by stochastic star formation.
///
/// The stochastic star-formation model splits a newly formed stellar population into individually
/// resolved massive stars above a threshold mass and one (or more) low-mass composite particles
/// below it. Two numbers follow from that threshold:
///
///   - the fraction of the population's *mass* that sits above it, and
///   - the *mean mass* of a star above it, which sets how many individual stars to draw.
///
/// Both used to be hard-coded constants, which is only valid for one particular threshold. They are
/// computed here instead so that `particles.min_mass_individual_stars` can be set freely.
///
/// The IMF is the one in extern/ChabrierIMFCalculations.nb: a lognormal in log10(M) below
/// m_break, joined continuously to a power law above it. It is expressed as dN/dlog10(M), so the
/// high-mass exponent is gamma + 1, with gamma = -2.35 the Salpeter slope of dN/dM.
///
/// All integrals below are analytic (erf for the lognormal part, a plain power-law antiderivative
/// above m_break), so no quadrature and no accuracy tuning is involved. They are host-only: the
/// results are evaluated once when the input file is parsed and then passed to device code by value.

#include <cmath>
#include <numbers>

#include "AMReX_BLassert.H"
#include "AMReX_REAL.H"

namespace quokka::ChabrierIMF
{

// Parameters of the Chabrier (2005) system IMF, in solar masses. These must match
// extern/ChabrierIMFCalculations.nb; changing them will trip validateAgainstReferenceValues().
inline constexpr double m_min = 0.08;	     // lower end of the IMF [Msun]
inline constexpr double m_max = 120.0;	     // upper end of the IMF [Msun]
inline constexpr double m_break = 1.0;	     // lognormal-to-power-law transition [Msun]
inline constexpr double m_peak = 0.2;	     // peak of the lognormal part [Msun]
inline constexpr double sigma_log10m = 0.55; // dispersion of the lognormal part, in log10(M)
inline constexpr double gamma_high = -2.35;  // high-mass slope of dN/dM (Salpeter)

//! \brief Continuity constant: the value of dN/dlog10(M) at m_break, coming in from the lognormal side.
[[nodiscard]] inline auto breakAmplitude() -> double
{
	const double log_ratio = std::log10(m_break) - std::log10(m_peak);
	return std::exp(-(log_ratio * log_ratio) / (2.0 * sigma_log10m * sigma_log10m));
}

//! \brief Mass-weighted integral of the lognormal branch, \f$\int M \, (dN/d\log_{10}M) \, d\log_{10}M\f$.
//!
//! Completing the square in \f$u = \log_{10} M\f$ turns \f$10^u \exp[-(u-u_p)^2/2\sigma^2]\f$ into a
//! Gaussian centred at \f$u_p + \sigma^2 \ln 10\f$, so the integral is an erf difference.
[[nodiscard]] inline auto lognormalMassIntegral(const double m_lo, const double m_hi) -> double
{
	const double ln10 = std::numbers::ln10;
	const double u_peak = std::log10(m_peak);
	const double shift = sigma_log10m * sigma_log10m * ln10;
	const double norm = sigma_log10m * std::sqrt(std::numbers::pi / 2.0) * m_peak * std::exp(0.5 * shift * shift / (sigma_log10m * sigma_log10m));
	auto const erf_at = [&](const double mass) { return std::erf((std::log10(mass) - u_peak - shift) / (sigma_log10m * std::numbers::sqrt2)); };
	return norm * (erf_at(m_hi) - erf_at(m_lo));
}

//! \brief Number-weighted integral of the lognormal branch, \f$\int (dN/d\log_{10}M) \, d\log_{10}M\f$.
[[nodiscard]] inline auto lognormalNumberIntegral(const double m_lo, const double m_hi) -> double
{
	const double u_peak = std::log10(m_peak);
	const double norm = sigma_log10m * std::sqrt(std::numbers::pi / 2.0);
	auto const erf_at = [&](const double mass) { return std::erf((std::log10(mass) - u_peak) / (sigma_log10m * std::numbers::sqrt2)); };
	return norm * (erf_at(m_hi) - erf_at(m_lo));
}

//! \brief Mass-weighted integral of the power-law branch. Valid only for m_break <= m_lo <= m_hi.
[[nodiscard]] inline auto powerLawMassIntegral(const double m_lo, const double m_hi) -> double
{
	const double prefactor = breakAmplitude() / (std::numbers::ln10 * std::pow(m_break, gamma_high + 1.0));
	const double exponent = gamma_high + 2.0;
	return prefactor * (std::pow(m_hi, exponent) - std::pow(m_lo, exponent)) / exponent;
}

//! \brief Number-weighted integral of the power-law branch. Valid only for m_break <= m_lo <= m_hi.
[[nodiscard]] inline auto powerLawNumberIntegral(const double m_lo, const double m_hi) -> double
{
	const double prefactor = breakAmplitude() / (std::numbers::ln10 * std::pow(m_break, gamma_high + 1.0));
	const double exponent = gamma_high + 1.0;
	return prefactor * (std::pow(m_hi, exponent) - std::pow(m_lo, exponent)) / exponent;
}

//! \brief Total mass per unit normalisation of the whole IMF, from m_min to m_max.
[[nodiscard]] inline auto totalMass() -> double { return lognormalMassIntegral(m_min, m_break) + powerLawMassIntegral(m_break, m_max); }

//! \brief Fraction of the IMF's total mass in stars above \p m_threshold (in Msun).
//!
//! This is `fstar_high`: the share of a newly formed stellar population that goes into individually
//! sampled stars, the rest going into the low-mass composite particle.
[[nodiscard]] inline auto massFractionAbove(const double m_threshold) -> double
{
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_threshold >= m_break && m_threshold < m_max,
					 "particles.min_mass_individual_stars must lie in [1, 120) Msun: the individual-star masses are drawn from a "
					 "pure power law, which only matches the Chabrier IMF above the lognormal break at 1 Msun.");
	return powerLawMassIntegral(m_threshold, m_max) / totalMass();
}

//! \brief Mean mass (in Msun) of a star drawn from the IMF above \p m_threshold (in Msun).
//!
//! This is `m_star_high_avg`, which converts the mass budget above the threshold into an expected
//! number of individual stars.
[[nodiscard]] inline auto meanMassAbove(const double m_threshold) -> double
{
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_threshold >= m_break && m_threshold < m_max,
					 "particles.min_mass_individual_stars must lie in [1, 120) Msun: the individual-star masses are drawn from a "
					 "pure power law, which only matches the Chabrier IMF above the lognormal break at 1 Msun.");
	return powerLawMassIntegral(m_threshold, m_max) / powerLawNumberIntegral(m_threshold, m_max);
}

//! \brief Check the IMF integrals against independently known reference values.
//!
//! Called once when the input file is parsed, so any change to the IMF parameters or to the integrals
//! is caught immediately and everywhere, rather than silently rescaling star formation.
//!
//! Two reference points, both external to this file:
//!
//!  - massFractionAbove(9) must equal the value printed by extern/ChabrierIMFCalculations.nb,
//!    0.20549466073679384, which is where the old hard-coded fstar_high = 0.2055 came from.
//!
//!  - meanMassAbove(8) must equal 19.3986 Msun, of which the old hard-coded m_star_high_avg = 19.39 is
//!    the truncation to two decimals. Note the threshold: that constant corresponds to a cut at
//!    8 Msun, not the 9 Msun used for fstar_high (the mean mass above 9 Msun is 21.34). The two legacy
//!    constants were therefore computed at different thresholds, and pairing them understated the mean
//!    mass by ~10%, forming ~10% too many massive stars per star-formation event. Deriving both from a
//!    single threshold here fixes that; see the PR description.
inline void validateAgainstReferenceValues()
{
	constexpr double notebook_mass_fraction_above_9 = 0.20549466073679384;
	constexpr double reference_mean_mass_above_8 = 19.3986;

	const double mass_fraction_above_9 = massFractionAbove(9.0);
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(std::abs(mass_fraction_above_9 - notebook_mass_fraction_above_9) < 1.0e-12,
					 "Chabrier IMF mass fraction above 9 Msun does not match extern/ChabrierIMFCalculations.nb.");

	const double mean_mass_above_8 = meanMassAbove(8.0);
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(std::abs(mean_mass_above_8 - reference_mean_mass_above_8) < 1.0e-3,
					 "Chabrier IMF mean mass above 8 Msun does not match the legacy m_star_high_avg constant.");
}

} // namespace quokka::ChabrierIMF

#endif // CHABRIER_IMF_HPP_
