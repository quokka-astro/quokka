#ifndef FASTMATH_HPP_ // NOLINT
#define FASTMATH_HPP_
//======================================================================
// © 2022. Triad National Security, LLC. All rights reserved.  This
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

#include "AMReX_Extension.H"
#include <cassert>
#include <cmath>

namespace FastMath {

AMREX_FORCE_INLINE
auto as_int(double f) {
  return *reinterpret_cast<int64_t*>(&f); // NOLINT
}
AMREX_FORCE_INLINE
auto as_double(int64_t i) {
  return *reinterpret_cast<double*>(&i); // NOLINT
}

// Reference implementations, however the integer cast implementation
// below is probably faster.
/*
AMREX_FORCE_INLINE
double lg(const double x) {
  int n;
  assert(x > 0 && "log divergent for x <= 0");
  const double y = frexp(x, &n);
  return 2 * (y - 1) + n;
}
AMREX_FORCE_INLINE
double pow2(const double x) {
  const int flr = std::floor(x);
  const double remainder = x - flr;
  const double mantissa = 0.5 * (remainder + 1);
  const double exponent = flr + 1;
  return ldexp(mantissa, exponent);
}
*/

AMREX_FORCE_INLINE
auto lg(const double x) -> double {
  // Magic numbers constexpr because C++ doesn't constexpr reinterpret casts
  // these are floating point numbers as reinterpreted as integers.
  // as_int(1.0)
  constexpr int64_t one_as_int = 4607182418800017408;
  // 1./static_cast<double>(as_int(2.0) - as_int(1.0))
  constexpr double scale_down = 2.22044604925031e-16;
  return static_cast<double>(as_int(x) - one_as_int) * scale_down;
}

AMREX_FORCE_INLINE
auto pow2(const double x) -> double {
  // Magic numbers constexpr because C++ doesn't constexpr reinterpret casts
  // these are floating point numbers as reinterpreted as integers.
  // as_int(1.0)
  constexpr int64_t one_as_int = 4607182418800017408;
  // as_int(2.0) - as_int(1.0)
  constexpr double scale_up = 4503599627370496;
  return as_double(static_cast<int64_t>(x*scale_up) + one_as_int);
}

AMREX_FORCE_INLINE
auto ln(const double x) -> double {
  constexpr double ILOG2E = 0.6931471805599453;
  return ILOG2E * lg(x);
}

AMREX_FORCE_INLINE
auto exp(const double x) -> double {
  constexpr double LOG2E = 1.4426950408889634;
  return pow2(LOG2E * x);
}

AMREX_FORCE_INLINE
auto log10(const double x) -> double {
  constexpr double LOG2OLOG10 = 0.301029995663981195;
  return LOG2OLOG10 * lg(x);
}

AMREX_FORCE_INLINE
auto pow10(const double x) -> double {
  constexpr double LOG10OLOG2 = 3.321928094887362626;
  return pow2(LOG10OLOG2 * x);
}

AMREX_FORCE_INLINE
auto tanh(const double x) -> double {
  const double expx = exp(2 * x);
  return (expx - 1) / (expx + 1);
}

} // namespace FastMath

#endif // FASTMATH_HPP_