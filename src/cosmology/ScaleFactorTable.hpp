#ifndef SCALE_FACTOR_TABLE_HPP_
#define SCALE_FACTOR_TABLE_HPP_
/// \file ScaleFactorTable.hpp
/// \brief Host-only lookup table for the scale factor a(t) and the conformal time tau(t), built once from the cosmological parameters.

#include <array>
#include <cmath>

#include "AMReX_Algorithm.H"
#include "AMReX_BLassert.H"
#include "AMReX_INT.H"
#include "AMReX_REAL.H"
#include "AMReX_Vector.H"

#include "cosmology/Cosmology.hpp"

namespace quokka::cosmology
{

/// Integrals over [x0, x1] in x = ln a: physical time dt = dx / H and conformal time dtau = dt / a = dx / (a H).
struct LnAIntegrals {
	amrex::Real dt;
	amrex::Real dtau;
};

/// @brief Physical and conformal time between ln a = x0 and ln a = x1, 3-point Gauss-Legendre with shared nodes.
inline auto IntegralsBetweenLnA(amrex::Real const x0, amrex::Real const x1, CosmologyParams const &cosmo) -> LnAIntegrals
{
	const std::array<amrex::Real, 3> nodes = {-std::sqrt(0.6), 0.0, std::sqrt(0.6)};
	const std::array<amrex::Real, 3> weights = {5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0};
	const amrex::Real half_width = 0.5 * (x1 - x0);
	const amrex::Real midpoint = 0.5 * (x1 + x0);

	amrex::Real sum_t = 0.0;
	amrex::Real sum_tau = 0.0;
	for (int k = 0; k < 3; ++k) {
		const amrex::Real a = std::exp(midpoint + half_width * nodes[k]); // map GL node from [-1, 1] to [x0, x1]
		const amrex::Real inv_H = 1.0 / (cosmo.H0 * HubbleFactor(a, cosmo));
		sum_t += weights[k] * inv_H;
		sum_tau += weights[k] * inv_H / a;
	}
	return LnAIntegrals{.dt = half_width * sum_t, .dtau = half_width * sum_tau};
}

/// @brief Table of a(t) and tau(t) with t = tau = 0 at a = a_init. Built once; a pure function of the parameters.
class ScaleFactorTable
{
      public:
	/// Build the table on a grid uniform in ln a from a_init to (at least) a_max with spacing dlna.
	/// If H0 == 0 no table is built: a(t) = a_init and tau(t) = t / a_init for every t.
	void Build(CosmologyParams const &cosmo, amrex::Real const a_init, amrex::Real const a_max, amrex::Real const dlna)
	{
		AMREX_ALWAYS_ASSERT_WITH_MESSAGE(a_init > 0.0, "ScaleFactorTable: a_init must be positive.");
		a_init_ = a_init;
		cosmo_ = cosmo;
		t_.clear(); // clear old data
		a_.clear();
		adot_.clear();
		tau_.clear();
		tau_dot_.clear();

		if (cosmo.H0 == 0.0) {
			is_static_ = true;
			return;
		}
		is_static_ = false;

		AMREX_ALWAYS_ASSERT_WITH_MESSAGE(a_max > a_init, "ScaleFactorTable: a_max must be larger than a_init.");
		AMREX_ALWAYS_ASSERT_WITH_MESSAGE(dlna > 0.0, "ScaleFactorTable: dlna must be positive.");

		const amrex::Real x_init = std::log(a_init);
		const int n_intervals = static_cast<int>(std::ceil((std::log(a_max) - x_init) / dlna));
		t_.reserve(n_intervals + 1);
		a_.reserve(n_intervals + 1);
		adot_.reserve(n_intervals + 1);
		tau_.reserve(n_intervals + 1);
		tau_dot_.reserve(n_intervals + 1);

		amrex::Real t = 0.0;
		amrex::Real tau = 0.0;
		amrex::Real x_prev = x_init; // left end of the next interval: shared with the previous right end, so the sum telescopes
		for (int i = 0; i <= n_intervals; ++i) {
			const amrex::Real x = x_init + i * dlna;
			const amrex::Real a = (i == 0) ? a_init : std::exp(x); // exact a_init at t = 0
			if (i > 0) {
				const LnAIntegrals d = IntegralsBetweenLnA(x_prev, x, cosmo);
				t += d.dt;
				tau += d.dtau;
			}
			x_prev = x;
			const amrex::Real adot = a * cosmo.H0 * HubbleFactor(a, cosmo); // da/dt = a H(a)
			AMREX_ALWAYS_ASSERT_WITH_MESSAGE(adot > 0.0, "ScaleFactorTable: H(a) must stay positive (no recollapse).");
			t_.push_back(t);
			a_.push_back(a);
			adot_.push_back(adot);
			tau_.push_back(tau);
			tau_dot_.push_back(1.0 / a); // dtau/dt = 1/a
		}
	}

	/// Scale factor at physical time t (cubic Hermite interpolation with exact derivatives). Host only.
	[[nodiscard]] auto ScaleFactorAt(amrex::Real const t) const -> amrex::Real
	{
		if (is_static_) {
			return a_init_;
		}
		return Interpolate(a_, adot_, t);
	}

	/// Conformal time tau(t) = int_0^t dt' / a(t') (cubic Hermite interpolation with exact derivatives). Host only.
	[[nodiscard]] auto ConformalTimeAt(amrex::Real const t) const -> amrex::Real
	{
		if (is_static_) {
			return t / a_init_;
		}
		return Interpolate(tau_, tau_dot_, t);
	}

	/// Transport time step dtau = tau(t + dt) - tau(t): a difference of table values, nothing accumulates. Host only.
	[[nodiscard]] auto Dtau(amrex::Real const t, amrex::Real const dt) const -> amrex::Real { return ConformalTimeAt(t + dt) - ConformalTimeAt(t); }

	/// Hubble rate H(a(t)) [1/s]. Host only.
	[[nodiscard]] auto HubbleAt(amrex::Real const t) const -> amrex::Real
	{
		if (is_static_) {
			return 0.0;
		}
		return cosmo_.H0 * HubbleFactor(ScaleFactorAt(t), cosmo_);
	}

      private:
	/// Cubic Hermite interpolation of y(t) on the nodes t_, with exact derivatives dydt at the nodes.
	[[nodiscard]] auto Interpolate(amrex::Vector<amrex::Real> const &y, amrex::Vector<amrex::Real> const &dydt, amrex::Real const t) const -> amrex::Real
	{
		AMREX_ALWAYS_ASSERT_WITH_MESSAGE(t >= 0.0 && t <= t_.back(), "ScaleFactorTable: time outside the table (increase cosmology.a_max).");

		// interval [t_[i], t_[i+1]] with t_[i] <= t < t_[i+1]; t == t_.back() falls in the last interval
		const amrex::Long i = amrex::bisect(t_.data(), amrex::Long{0}, t_.size() - 1, t);

		const amrex::Real width = t_[i + 1] - t_[i];
		const amrex::Real s = (t - t_[i]) / width; // map to s in [0,1], where Hermite relation is valid
		const amrex::Real s2 = s * s;
		const amrex::Real s3 = s2 * s;
		const amrex::Real h00 = 2.0 * s3 - 3.0 * s2 + 1.0;
		const amrex::Real h10 = s3 - 2.0 * s2 + s;
		const amrex::Real h01 = -2.0 * s3 + 3.0 * s2;
		const amrex::Real h11 = s3 - s2;

		return h00 * y[i] + h10 * width * dydt[i] + h01 * y[i + 1] + h11 * width * dydt[i + 1];
	}

	bool is_static_ = false;
	amrex::Real a_init_ = 1.0;
	CosmologyParams cosmo_{};	     ///< parameters the table was built from
	amrex::Vector<amrex::Real> t_;	     ///< physical time since a_init [s]
	amrex::Vector<amrex::Real> a_;	     ///< scale factor at t_
	amrex::Vector<amrex::Real> adot_;    ///< da/dt at t_ [1/s]
	amrex::Vector<amrex::Real> tau_;     ///< conformal time int dt/a since a_init [s]
	amrex::Vector<amrex::Real> tau_dot_; ///< dtau/dt = 1/a at t_ [1/s]
};

} // namespace quokka::cosmology

#endif // SCALE_FACTOR_TABLE_HPP_