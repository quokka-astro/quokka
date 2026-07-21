#ifndef QUOKKA_CONSTEXPR_MATH_HPP_
#define QUOKKA_CONSTEXPR_MATH_HPP_

#include <limits>

namespace quokka::math
{

// Newton iteration for constants that must remain usable in compile-time
// problem definitions. Runtime kernels should use std::sqrt directly.
[[nodiscard]] constexpr auto sqrt(double value) noexcept -> double
{
	if (value < 0.0) {
		return std::numeric_limits<double>::quiet_NaN();
	}
	if (value == 0.0) {
		return 0.0;
	}
	double estimate = value >= 1.0 ? value : 1.0;
	for (int iteration = 0; iteration < 64; ++iteration) {
		estimate = 0.5 * (estimate + value / estimate);
	}
	return estimate;
}

} // namespace quokka::math

#endif // QUOKKA_CONSTEXPR_MATH_HPP_
