#ifndef SCALE_FACTOR_TABLE_HPP_
#define SCALE_FACTOR_TABLE_HPP_
/// \file ScaleFactorTable.hpp
/// \brief Host-only lookup table for the scale factor a(t), built once from the cosmological parameters.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <iterator>
#include <vector>

#include "AMReX.H"
#include "AMReX_REAL.H"

#include "cosmology/Cosmology.hpp"

namespace quokka::cosmology
{

/// @brief Physical time elapsed between ln a = x0 and ln a = x1: int dx / H(e^x), 3-point Gauss-Legendre.
inline auto TimeBetweenLnA(amrex::Real const x0, amrex::Real const x1, CosmologyParams const &cosmo) -> amrex::Real
{
	const std::array<amrex::Real, 3> nodes = {-std::sqrt(0.6), 0.0, std::sqrt(0.6)};
	const std::array<amrex::Real, 3> weights = {5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0};
	const amrex::Real half_width = 0.5 * (x1 - x0);
	const amrex::Real midpoint = 0.5 * (x1 + x0);

	amrex::Real sum = 0.0;
	for (int k = 0; k < 3; ++k) {
		const amrex::Real a = std::exp(midpoint + half_width * nodes[k]); // map GL node from [-1, 1] to [x0, x1]
		sum += weights[k] / (cosmo.H0 * HubbleFactor(a, cosmo));
	}
	return half_width * sum;
}

/// @brief Table of a(t) with t = 0 at a = a_init. Built once; a pure function of the parameters.
class ScaleFactorTable
{
      public:
	/// Build the table on a grid uniform in ln a from a_init to (at least) a_max with spacing dlna.
	/// If H0 == 0 no table is built and a(t) = a_init for every t.
	void Build(CosmologyParams const &cosmo, amrex::Real const a_init, amrex::Real const a_max, amrex::Real const dlna)
	{
		AMREX_ALWAYS_ASSERT_WITH_MESSAGE(a_init > 0.0, "ScaleFactorTable: a_init must be positive.");
		a_init_ = a_init;
		t_.clear(); // clear old data
		a_.clear();
		adot_.clear();

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

		amrex::Real t = 0.0;
		for (int i = 0; i <= n_intervals; ++i) {
			const amrex::Real x = x_init + i * dlna;
			const amrex::Real a = (i == 0) ? a_init : std::exp(x); // exact a_init at t = 0
			if (i > 0) {
				t += TimeBetweenLnA(x - dlna, x, cosmo);
			}
			const amrex::Real adot = a * cosmo.H0 * HubbleFactor(a, cosmo); // da/dt = a H(a)
			AMREX_ALWAYS_ASSERT_WITH_MESSAGE(adot > 0.0, "ScaleFactorTable: H(a) must stay positive (no recollapse).");
			t_.push_back(t);
			a_.push_back(a);
			adot_.push_back(adot);
		}
	}

	/// Scale factor at physical time t (cubic Hermite interpolation with exact derivatives). Host only.
	[[nodiscard]] auto ScaleFactorAt(amrex::Real const t) const -> amrex::Real
	{
		if (is_static_) {
			return a_init_;
		}
		AMREX_ALWAYS_ASSERT_WITH_MESSAGE(t >= 0.0 && t <= t_.back(), "ScaleFactorTable: time outside the table (increase cosmology.a_max).");

		// first node strictly after t; the interval is [t_[i], t_[i+1]]
		const auto it = std::upper_bound(t_.begin(), t_.end(), t);
		const std::size_t i = (it == t_.end()) ? t_.size() - 2 : static_cast<std::size_t>(std::distance(t_.begin(), it)) - 1;

		const amrex::Real width = t_[i + 1] - t_[i];
		const amrex::Real s = (t - t_[i]) / width; // map to s in [0,1], where Hermite relation is valid
		const amrex::Real s2 = s * s;
		const amrex::Real s3 = s2 * s;
		const amrex::Real h00 = 2.0 * s3 - 3.0 * s2 + 1.0;
		const amrex::Real h10 = s3 - 2.0 * s2 + s;
		const amrex::Real h01 = -2.0 * s3 + 3.0 * s2;
		const amrex::Real h11 = s3 - s2;

		return h00 * a_[i] + h10 * width * adot_[i] + h01 * a_[i + 1] + h11 * width * adot_[i + 1];
	}

      private:
	bool is_static_ = false;
	amrex::Real a_init_ = 1.0;
	std::vector<amrex::Real> t_;	///< physical time since a_init [s]
	std::vector<amrex::Real> a_;	///< scale factor at t_
	std::vector<amrex::Real> adot_; ///< da/dt at t_ [1/s]
};

} // namespace quokka::cosmology

#endif // SCALE_FACTOR_TABLE_HPP_
