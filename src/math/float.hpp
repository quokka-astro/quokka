#ifndef FLOAT_HPP_
#define FLOAT_HPP_

#include <cmath>
#include <stdexcept>
#include <exception>

#include "AMReX_BLassert.H"
#include "AMReX_GpuQualifiers.H"

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto round_to_sigfigs_stable(amrex::Real x, int N) -> amrex::Real
{
	if (std::isnan(x) || std::isinf(x) || x == 0.0) {
		return x;
	}
	
	AMREX_ASSERT(N > 0);

	const amrex::Real ax = std::fabs(x);

	// Approximate base-10 exponent from binary exponent
	// ax = m * 2^b, where m in [0.5, 1)
	const int b;
	const amrex::Real m = std::frexp(ax, &b); // ax = m * 2^(b)
	// log10(ax) = log10(m) + b * log10(2)
	static const amrex::Real LOG10_2 = 0.30102999566398119521;
	const amrex::Real e_est = std::floor(std::log10(m) + b * LOG10_2);

	// It’s possible e_est is off by 1 due to rounding; correct it
	const amrex::Real pow10_e = std::pow(10.0, e_est);
	if (ax / pow10_e >= 10.0) {
			e_est += 1.0;
			pow10_e *= 10.0;
	} else if (ax / pow10_e < 1.0) {
			e_est -= 1.0;
			pow10_e /= 10.0;
	}

	const amrex::Real s = std::pow(10.0, N - 1 - e_est);
	const amrex::Real rounded = std::round(ax * s) / s;
	return std::copysign(rounded, x);
}

#endif // FLOAT_HPP_