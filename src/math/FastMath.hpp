#ifndef FASTMATH_HPP_ // NOLINT
#define FASTMATH_HPP_
//======================================================================
// © 2021-2023. Triad National Security, LLC. All rights reserved.  This
// program was produced under U.S. Government contract 89233218CNA000001
// for Los Alamos National Laboratory (LANL), which is operated by Triad
// National Security, LLC for the U.S.  Department of Energy/National
// Nuclear Security Administration. All rights in the program are
// reserved by Triad National Security, LLC, and the U.S. Department of
// Energy/National Nuclear Security Administration. The Government is
// granted for itself and others acting on its behalf a nonexclusive,
// paid-up, irrevocable worldwide license in this material to reproduce,
// prepare derivative works, distribute copies to the public, perform
// publicly and display publicly, and to permit others to do so.
//======================================================================

// Code taken from singularity-eos
// https://github.com/lanl/singularity-eos

#include <cassert>
#include <cmath>

#include "AMReX_Extension.H"
#include "AMReX_GpuQualifiers.H"

// this speeds up the *total walltime* for problems with cooling by ~30% on a
// single V100
#define USE_FASTMATH

#ifdef USE_FASTMATH
namespace FastMath
{

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto fastlg(const double x) -> double
{
	int n = 0;
	assert(x > 0 && "log divergent for x <= 0");
	const double y = frexp(x, &n);
	return 2 * (y - 1) + n;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto fastpow2(const double x) -> double
{
	const int flr = std::floor(x);
	const double remainder = x - flr;
	const double mantissa = 0.5 * (remainder + 1);
	const int exponent = flr + 1;
	return ldexp(mantissa, exponent);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto lg(const double x) -> double
{
	assert(x > 0 && "log divergent for x <= 0");
	return fastlg(x);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto pow2(const double x) -> double { return fastpow2(x); }

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto log10(const double x) -> double
{
	constexpr double LOG2OLOG10 = 0.301029995663981195;
	return LOG2OLOG10 * lg(x);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto pow10(const double x) -> double
{
	constexpr double LOG10OLOG2 = 3.321928094887362626;
	return pow2(LOG10OLOG2 * x);
}

// Inverse of pow2 using Newton iteration
// Finds y such that pow2(y) = x to machine precision
// Uses lg(x) as initial guess
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto inverse_pow2(const double x) -> double
{
	constexpr double LN2 = 0.693147180559945309417232121458;
	constexpr int MAX_ITER = 20;
	constexpr double TOLERANCE = 1.0e-15;

	// Initial guess using fast log
	double y = lg(x);

	// Newton iteration: solve pow2(y) - x = 0
	// f(y) = pow2(y) - x
	// f'(y) = pow2(y) * ln(2)
	// y_new = y - f(y)/f'(y) = y - (pow2(y) - x) / (pow2(y) * ln(2))
	for (int iter = 0; iter < MAX_ITER; ++iter) {
		const double pow2_y = pow2(y);
		const double f = pow2_y - x;

		// Check convergence
		if (std::abs(f) < TOLERANCE * std::abs(x)) {
			break;
		}

		// Newton update
		const double df = pow2_y * LN2;
		y = y - f / df;
	}

	return y;
}

} // namespace FastMath
#else
namespace FastMath
{

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto log10(const double x) -> double { return std::log10(x); }

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto pow10(const double x) -> double { return std::pow(10., x); }
} // namespace FastMath
#endif // USE_FASTMATH

#endif // FASTMATH_HPP_