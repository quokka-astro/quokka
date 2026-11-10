#ifndef ROSENBROCK_DENSE_LU_HPP_
#define ROSENBROCK_DENSE_LU_HPP_
/// \file dense_lu.hpp
/// \brief Dense LU factorisation with partial pivoting and the matching solve, for small systems in device code.
///
/// A 0-based port of LINPACK's dgefa/dgesl as used by Microphysics (util/linpack.H), with pivoting always on.
///
/// In device code every loop is unrolled at compile time (amrex::constexpr_for), and the pivot row is selected by
/// comparing it with each candidate row instead of indexing with it. All matrix and vector indices are then compile-time
/// constants, so the compiler can keep the matrix and the right-hand side in registers instead of local memory. On the
/// host, where that form is not faster, the loops index with the pivot row as in dgefa/dgesl. Both forms do the same
/// arithmetic in the same order, so they give identical results.

#include <cmath>

#include "AMReX_Array.H"
#include "AMReX_ConstexprFor.H"
#include "AMReX_GpuQualifiers.H"
#include "AMReX_REAL.H"

namespace rosenbrock
{

template <int N> using Vector = amrex::GpuArray<amrex::Real, N>;
template <int N> using Matrix = amrex::Array2D<amrex::Real, 0, N - 1, 0, N - 1>;
template <int N> using Pivots = amrex::GpuArray<int, N>;

namespace detail
{

/// lu_factor with loops and indexing by the pivot row (used on the host)
template <int N> AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto lu_factor_loops(Matrix<N> &a, Pivots<N> &pivot) -> int
{
	int info = 0;
	for (int k = 0; k < N - 1; ++k) {
		// find the pivot row l
		int l = k;
		amrex::Real dmax = std::abs(a(k, k));
		for (int i = k + 1; i < N; ++i) {
			const amrex::Real ai = std::abs(a(i, k));
			if (ai > dmax) {
				l = i;
				dmax = ai;
			}
		}
		pivot[k] = l;

		// a zero pivot means this column is already triangular
		if (a(l, k) == 0.0) {
			info = k + 1;
			continue;
		}
		if (l != k) {
			const amrex::Real t = a(l, k);
			a(l, k) = a(k, k);
			a(k, k) = t;
		}

		// multipliers
		const amrex::Real t = -1.0 / a(k, k);
		for (int j = k + 1; j < N; ++j) {
			a(j, k) *= t;
		}

		// row elimination with column indexing
		for (int j = k + 1; j < N; ++j) {
			const amrex::Real alj = a(l, j);
			if (l != k) {
				a(l, j) = a(k, j);
				a(k, j) = alj;
			}
			for (int i = k + 1; i < N; ++i) {
				a(i, j) += alj * a(i, k);
			}
		}
	}
	pivot[N - 1] = N - 1;
	if (a(N - 1, N - 1) == 0.0) {
		info = N;
	}
	return info;
}

/// lu_solve with loops and indexing by the pivot row (used on the host)
template <int N> AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void lu_solve_loops(Matrix<N> const &a, Pivots<N> const &pivot, Vector<N> &b)
{
	// L y = b
	for (int k = 0; k < N - 1; ++k) {
		const int l = pivot[k];
		const amrex::Real t = b[l];
		if (l != k) {
			b[l] = b[k];
			b[k] = t;
		}
		for (int j = k + 1; j < N; ++j) {
			b[j] += t * a(j, k);
		}
	}
	// U x = y
	for (int k = N - 1; k >= 0; --k) {
		b[k] = b[k] / a(k, k);
		const amrex::Real t = -b[k];
		for (int j = 0; j < k; ++j) {
			b[j] += t * a(j, k);
		}
	}
}

/// lu_factor unrolled at compile time, the pivot row selected by comparison (used in device code)
template <int N> AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto lu_factor_unrolled(Matrix<N> &a, Pivots<N> &pivot) -> int
{
	int info = 0;
	amrex::constexpr_for<0, N - 1>([&](auto kc) {
		constexpr int k = decltype(kc)::value;

		// find the pivot row l and the pivot a(l, k)
		int l = k;
		amrex::Real dmax = std::abs(a(k, k));
		amrex::Real akk = a(k, k);
		amrex::constexpr_for<k + 1, N>([&](auto ic) {
			constexpr int i = decltype(ic)::value;
			const amrex::Real ai = std::abs(a(i, k));
			if (ai > dmax) {
				l = i;
				dmax = ai;
				akk = a(i, k);
			}
		});
		pivot[k] = l;

		// a zero pivot means this column is already triangular
		if (akk == 0.0) {
			info = k + 1;
			return;
		}
		// swap a(l, k) and a(k, k)
		amrex::constexpr_for<k + 1, N>([&](auto ic) {
			constexpr int i = decltype(ic)::value;
			if (i == l) {
				a(i, k) = a(k, k);
			}
		});
		a(k, k) = akk;

		// multipliers
		const amrex::Real t = -1.0 / a(k, k);
		amrex::constexpr_for<k + 1, N>([&](auto jc) {
			constexpr int j = decltype(jc)::value;
			a(j, k) *= t;
		});

		// row elimination with column indexing
		amrex::constexpr_for<k + 1, N>([&](auto jc) {
			constexpr int j = decltype(jc)::value;
			// alj = a(l, j); then swap a(l, j) and a(k, j)
			amrex::Real alj = a(k, j);
			amrex::constexpr_for<k + 1, N>([&](auto ic) {
				constexpr int i = decltype(ic)::value;
				if (i == l) {
					alj = a(i, j);
					a(i, j) = a(k, j);
				}
			});
			a(k, j) = alj;
			amrex::constexpr_for<k + 1, N>([&](auto ic) {
				constexpr int i = decltype(ic)::value;
				a(i, j) += alj * a(i, k);
			});
		});
	});
	pivot[N - 1] = N - 1;
	if (a(N - 1, N - 1) == 0.0) {
		info = N;
	}
	return info;
}

/// lu_solve unrolled at compile time, the pivot row selected by comparison (used in device code)
template <int N> AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void lu_solve_unrolled(Matrix<N> const &a, Pivots<N> const &pivot, Vector<N> &b)
{
	// L y = b
	amrex::constexpr_for<0, N - 1>([&](auto kc) {
		constexpr int k = decltype(kc)::value;
		const int l = pivot[k];
		// t = b[l]; then swap b[l] and b[k]
		amrex::Real t = b[k];
		amrex::constexpr_for<k + 1, N>([&](auto ic) {
			constexpr int i = decltype(ic)::value;
			if (i == l) {
				t = b[i];
				b[i] = b[k];
			}
		});
		b[k] = t;
		amrex::constexpr_for<k + 1, N>([&](auto jc) {
			constexpr int j = decltype(jc)::value;
			b[j] += t * a(j, k);
		});
	});
	// U x = y
	amrex::constexpr_for<0, N>([&](auto kr) {
		constexpr int k = N - 1 - decltype(kr)::value;
		b[k] = b[k] / a(k, k);
		const amrex::Real t = -b[k];
		amrex::constexpr_for<0, k>([&](auto jc) {
			constexpr int j = decltype(jc)::value;
			b[j] += t * a(j, k);
		});
	});
}

} // namespace detail

/// Factors a = L U in place (gaussian elimination with partial pivoting).
/// \return 0 on success, or k + 1 if the pivot U(k, k) is zero (the matrix is singular).
// Which device form to use. The unrolled one was chosen on a V100, where keeping the matrix in registers is the win.
// On gfx90a the kernel is occupancy-bound instead, and the unrolled form's O(N^3) selects (it compares against the
// pivot row rather than indexing with it) cost more than the dynamic indexing they avoid -- measured 3% faster with
// the loops for the 6-equation photochemistry system. So: unrolled on CUDA, loops on HIP, either forced by defining
// ROSENBROCK_LU_UNROLLED or ROSENBROCK_LU_LOOPS.
#if defined(ROSENBROCK_LU_LOOPS)
#define ROSENBROCK_LU_USE_UNROLLED 0
#elif defined(ROSENBROCK_LU_UNROLLED)
#define ROSENBROCK_LU_USE_UNROLLED 1
#elif defined(__HIP_DEVICE_COMPILE__)
#define ROSENBROCK_LU_USE_UNROLLED 0
#else
#define ROSENBROCK_LU_USE_UNROLLED 1
#endif

template <int N> AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto lu_factor(Matrix<N> &a, Pivots<N> &pivot) -> int
{
#if AMREX_DEVICE_COMPILE && ROSENBROCK_LU_USE_UNROLLED
	return detail::lu_factor_unrolled<N>(a, pivot);
#else
	return detail::lu_factor_loops<N>(a, pivot);
#endif
}

/// Solves a x = b, with a as returned by lu_factor; b is overwritten by x.
template <int N> AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void lu_solve(Matrix<N> const &a, Pivots<N> const &pivot, Vector<N> &b)
{
#if AMREX_DEVICE_COMPILE && ROSENBROCK_LU_USE_UNROLLED
	detail::lu_solve_unrolled<N>(a, pivot, b);
#else
	detail::lu_solve_loops<N>(a, pivot, b);
#endif
}

} // namespace rosenbrock

#endif // ROSENBROCK_DENSE_LU_HPP_
