#ifndef ROSENBROCK_DENSE_LU_HPP_
#define ROSENBROCK_DENSE_LU_HPP_
/// \file dense_lu.hpp
/// \brief Dense LU factorisation with partial pivoting and the matching solve, for small systems in device code.
///
/// A 0-based port of LINPACK's dgefa/dgesl as used by Microphysics (util/linpack.H), with pivoting always on.

#include <cmath>

#include "AMReX_Array.H"
#include "AMReX_GpuQualifiers.H"
#include "AMReX_REAL.H"

namespace rosenbrock
{

template <int N> using Vector = amrex::GpuArray<amrex::Real, N>;
template <int N> using Matrix = amrex::Array2D<amrex::Real, 0, N - 1, 0, N - 1>;
template <int N> using Pivots = amrex::GpuArray<int, N>;

/// Factors a = L U in place (gaussian elimination with partial pivoting).
/// \return 0 on success, or k + 1 if the pivot U(k, k) is zero (the matrix is singular).
template <int N> AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto lu_factor(Matrix<N> &a, Pivots<N> &pivot) -> int
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

/// Solves a x = b, with a as returned by lu_factor; b is overwritten by x.
template <int N> AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void lu_solve(Matrix<N> const &a, Pivots<N> const &pivot, Vector<N> &b)
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

} // namespace rosenbrock

#endif // ROSENBROCK_DENSE_LU_HPP_
