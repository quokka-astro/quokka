#ifndef BRACKETING_ROOT_FINDING_HPP_
#define BRACKETING_ROOT_FINDING_HPP_
//==============================================================================
// Bracketing root finders ported from SciML/NonlinearSolve.jl
// (lib/BracketingNonlinearSolve/src/{brent,modAB}.jl, MIT license).
//==============================================================================
/// \file bracketing_root_finding.hpp
/// \brief Brent and modified Anderson-Bjork (ModAB) bracketing root finders.
///
/// Both solvers share the calling convention of quokka::math::toms748_solve:
/// they take a functor f, a bracket [ax, bx] with f(ax) * f(bx) <= 0, a
/// termination functor tol(a, b) (e.g. quokka::math::eps_tolerance) and an
/// iteration budget max_iter. On return, max_iter holds the number of
/// iterations used and the returned pair (a, b), with a <= b, still brackets
/// the root. If f(x) == 0 is hit exactly, a == b == x.
///
/// The bracket is returned unconverged (tol(a, b) == false) if the iteration
/// budget is exhausted, if f returns NaN (ModAB), or if the bracket cannot be
/// split further in floating point.

#include <cmath>
#include <limits>
#include <utility>

#include "AMReX_Algorithm.H"
#include "AMReX_BLassert.H"
#include "AMReX_Extension.H"
#include "AMReX_GpuQualifiers.H"

#include "math/math_impl.hpp"

namespace quokka::math
{

namespace detail
{
/// Midpoint of [x1, x2] that cannot overflow.
template <class T> AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto safe_midpoint(T x1, T x2) -> T { return x1 / 2 + x2 / 2; }

/// True if the bracket [a, b] cannot be split further in floating point.
template <class T> AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto at_floating_point_limit(T a, T b) -> bool
{
	const T m = safe_midpoint(a, b);
	return (m == a) || (m == b);
}

/// Secant (regula falsi) point of (x1, y1), (x2, y2), clamped to [x1, x2] and safe against overflow of |y1| + |y2|.
template <class T> AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto safe_secant(T x1, T y1, T x2, T y2) -> T
{
	T a = std::abs(y1);
	T b = std::abs(y2);
	T den = a + b;
	if (std::isinf(den)) {
		if (std::isinf(a) || std::isinf(b)) {
			return safe_midpoint(x1, x2);
		}
		a /= 2;
		b /= 2;
		den = a + b;
	}
	const T x = (b / den) * x1 + (a / den) * x2;
	return amrex::min(amrex::max(x, x1), x2);
}

/// Anderson-Bjork scaling factor for the retained endpoint.
template <class T> AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto ab_factor(T y3, T y) -> T
{
	const T m = 1 - y3 / y;
	return m > 0 ? m : T(0.5);
}
} // namespace detail

/// Brent's method (inverse quadratic interpolation / secant / bisection).
/// Port of Brent() from NonlinearSolve.jl, plus the minimum-step safeguard of Numerical Recipes' zbrent
/// (without it, the far end of the bracket converges only by bisection when f is exactly rounded, e.g. with FMA).
/// fax and fbx are f(ax) and f(bx).
template <class F, class T, class Tol>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto brent_solve(F f, T ax, T bx, T fax, T fbx, Tol tol, int &max_iter) -> std::pair<T, T>
{
	const T eps = std::numeric_limits<T>::epsilon();
	const int budget = max_iter;
	max_iter = 0;

	T left = ax;
	T right = bx;
	T fl = fax;
	T fr = fbx;

	if (fl == 0) {
		return std::make_pair(left, left);
	}
	if (fr == 0) {
		return std::make_pair(right, right);
	}
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(sgn(fl) != sgn(fr), "brent_solve: parameters a and b do not bracket the root!");

	auto ordered = [](T a, T b) { return (a < b) ? std::make_pair(a, b) : std::make_pair(b, a); };

	// keep 'right' as the best estimate
	if (std::abs(fl) < std::abs(fr)) {
		std::swap(left, right);
		std::swap(fl, fr);
	}

	// c is the previous iterate, d the one before it
	T c = left;
	T fc = fl;
	T d = c;
	bool cond = true; // whether the previous step was a bisection

	while (max_iter < budget) {
		const auto [lo, hi] = ordered(left, right);
		if (tol(lo, hi)) {
			break;
		}

		T s{};
		if ((fl != fc) && (fr != fc)) {
			// inverse quadratic interpolation
			s = left * fr * fc / ((fl - fr) * (fl - fc)) + right * fl * fc / ((fr - fl) * (fr - fc)) + c * fl * fr / ((fc - fl) * (fc - fr));
		} else {
			// secant
			s = right - fr * (right - left) / (fr - fl);
		}

		const T q = (3 * left + right) / 4;
		// !isfinite(s): the interpolation overflowed (e.g. |f| ~ 1e200); NaN would pass every comparison below
		if (!std::isfinite(s) || (s < amrex::min(q, right)) || (s > amrex::max(q, right)) || (cond && std::abs(s - right) >= std::abs(right - c) / 2) ||
		    (!cond && std::abs(s - right) >= std::abs(c - d) / 2) || (cond && std::abs(right - c) <= eps) || (!cond && std::abs(c - d) <= eps)) {
			// bisection
			s = detail::safe_midpoint(left, right);
			if ((s == left) || (s == right)) {
				break; // floating-point limit
			}
			cond = true;
		} else {
			cond = false;
			// Numerical Recipes (zbrent) safeguard: step at least tol1 towards the contrapoint, so the bracket
			// still collapses when every interpolated iterate lands on the same side of the root.
			const T tol1 = amrex::max(2 * eps * std::abs(right), std::numeric_limits<T>::min());
			if (std::abs(s - right) < tol1) {
				s = right + ((left > right) ? tol1 : -tol1);
				if (!((s > lo) && (s < hi))) {
					s = detail::safe_midpoint(left, right);
				}
			}
		}

		const T fs = f(s);
		++max_iter;
		if (fs == 0) {
			return std::make_pair(s, s);
		}

		if (sgn(fl) * sgn(fs) < 0) {
			d = c;
			c = right;
			fc = fr;
			right = s;
			fr = fs;
		} else {
			left = s;
			fl = fs;
		}

		if (std::abs(fl) < std::abs(fr)) {
			d = c;
			c = right;
			fc = fr;
			right = left;
			fr = fl;
			left = c;
			fl = fc;
		}
	}

	return ordered(left, right);
}

/// Brent's method; evaluates f(ax) and f(bx) itself.
template <class F, class T, class Tol> AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto brent_solve(F f, T ax, T bx, Tol tol, int &max_iter) -> std::pair<T, T>
{
	return brent_solve(f, ax, bx, f(ax), f(bx), tol, max_iter);
}

/// ModAB (Modified Anderson-Bjork)
///
/// Use the ModAB method to find a root of a bracketed function, with a convergence rate between 1.7 and 1.8.
///
/// This method was introduced in the paper "Modified Anderson-Bjork's method for solving non-linear equations
/// in structural mechanics" (https://doi.org/10.1088/1757-899X/1276/1/012010) by N Ganchovski and A Traykov.
///
/// This implementation includes the latest improvements made in 2026 by the following paper:
/// Ganchovski, N.; Smith, O.; Rackauckas, C.; Tomov, L.; Traykov, A. Improvements to the Modified
/// Anderson-Bjorck (modAB) Root-Finding Algorithm. Algorithms 2026, 19, 332. (https://doi.org/10.3390/a19050332)
///
/// Port of ModAB() from NonlinearSolve.jl. fax and fbx are f(ax) and f(bx).
template <class F, class T, class Tol>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto modab_solve(F f, T ax, T bx, T fax, T fbx, Tol tol, int &max_iter) -> std::pair<T, T>
{
	const int budget = max_iter;
	max_iter = 0;

	T x1 = ax;
	T x2 = bx;
	T y1 = fax;
	T y2 = fbx;
	if (x2 < x1) {
		std::swap(x1, x2);
		std::swap(y1, y2);
	}

	if (y1 == 0) {
		return std::make_pair(x1, x1);
	}
	if (y2 == 0) {
		return std::make_pair(x2, x2);
	}
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(sgn(y1) != sgn(y2), "modab_solve: parameters a and b do not bracket the root!");

	bool bisecting = true;
	int side = 0;	       // side that moved in the previous Anderson-Bjork step
	T threshold = x2 - x1; // fall back to bisection if Anderson-Bjork does not shrink the bracket below this
	constexpr T C = 2;     // safety factor for the threshold (two iterations)
	T f1 = y1;	       // unmodified residuals at the bracket ends
	T f2 = y2;	       // (y1, y2 hold the Anderson-Bjork-scaled ones)
	T yMin = 0;	       // smallest unmodified residual of the bracket at the previous AB step

	while ((max_iter < budget) && !tol(x1, x2)) {
		T x3{};
		T y3{};
		if (bisecting) {
			x3 = detail::safe_midpoint(x1, x2);
			y3 = f(x3);
			++max_iter;
			if (std::isfinite(f2 - f1)) {
				const T ym = (f1 + f2) / 2;		  // chord ordinate at the midpoint
				const T r = 1 - std::abs(ym / (f2 - f1)); // symmetry factor
				const T k = r * r;			  // deviation factor
				if (std::abs(ym - y3) < k * std::abs(y3) + k * std::abs(ym)) {
					// close enough to linear: switch to Anderson-Bjork, starting from the true residuals
					bisecting = false;
					threshold = C * (x2 - x1);
					y1 = f1;
					y2 = f2;
				}
			}
		} else {
			x3 = detail::safe_secant(x1, y1, x2, y2);
			if (x3 == x1) {
				y3 = f1;
			} else if (x3 == x2) {
				y3 = f2;
			} else {
				y3 = f(x3);
				++max_iter;
			}
			threshold /= 2;
			yMin = amrex::min(std::abs(f1), std::abs(f2));
		}

		if (y3 == 0) {
			return std::make_pair(x3, x3);
		}
		if (std::isnan(y3)) {
			break;
		}

		if (bisecting) {
			if (sgn(f1) == sgn(y3)) {
				x1 = x3;
				f1 = y3;
			} else {
				x2 = x3;
				f2 = y3;
			}
		} else {
			if (sgn(f1) == sgn(y3)) {
				if (side == 1) {
					y2 *= detail::ab_factor(y3, y1);
				}
				x1 = x3;
				y1 = y3;
				f1 = y3;
				side = 1;
			} else {
				if (side == -1) {
					y1 *= detail::ab_factor(y3, y2);
				}
				x2 = x3;
				y2 = y3;
				f2 = y3;
				side = -1;
			}
			if ((x2 - x1 > threshold) && (std::abs(y3) > yMin / 2)) {
				bisecting = true;
				side = 0;
			}
		}

		if (detail::at_floating_point_limit(x1, x2)) {
			break;
		}
	}

	return std::make_pair(x1, x2);
}

/// ModAB; evaluates f(ax) and f(bx) itself.
template <class F, class T, class Tol> AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto modab_solve(F f, T ax, T bx, Tol tol, int &max_iter) -> std::pair<T, T>
{
	return modab_solve(f, ax, bx, f(ax), f(bx), tol, max_iter);
}

} // namespace quokka::math

#endif // BRACKETING_ROOT_FINDING_HPP_
