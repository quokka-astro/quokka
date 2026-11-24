// SPDX-FileCopyrightText: © The Quokka Authors
// SPDX-License-Identifier: MIT

#ifndef LINEAR_ALGEBRA_HPP_
#define LINEAR_ALGEBRA_HPP_
/// \file linear_algebra.hpp
/// \brief LU factorization and solve of a small dense matrix (LINPACK's dgefa and dgesl), with optional partial
/// pivoting. Ported from Microphysics' util/linpack.H, with 0-based indices.
///
/// Every loop is unrolled at compile time (amrex::constexpr_for).
///
/// TODO(james471): two other forms of this LU exist and need testing and implementing for different network sizes and machine architectures. Unimplemented
///  claude findings:
///  - LINPACK's loops, indexing with the pivot row (Microphysics' util/linpack.H; 0-based in #2425's
///    src/math/linear_algebra.hpp at 2bfff3fa9, with AMREX_UNROLL_LOOP(1) on the inner loops as Microphysics). On a
///    CPU with pivoting they were faster than this form from 12 equations on a Xeon Platinum 8274 (up to 4.5 times at
///    23) and at 23 on an Apple M-series; far slower on GPUs (3-30 times).
///  - By columns (left-looking; optimized branch james471/claude-optim/gow_r2a8nopiv, 72558f0a9 + fa572fa3c, which
///    store the pivots' reciprocals: divide by the pivot instead to keep the results identical). Faster than this form
///    once the matrix spills out of registers: from 13 equations on an H200, from 15 on an MI250X (3-9 % faster for
///    PrimordialChem and PopIII, 15 equations), and 0-16 % faster on a CPU at 4-10 equations.
/// This form is the fastest of the three on GPUs at the sizes of the DType networks (5-9 equations), and faster than
/// the loops on CPUs there.

#include <cmath>
#include <cstdint>

#include "AMReX_Array.H"
#include "AMReX_ConstexprFor.H"
#include "AMReX_Extension.H"
#include "AMReX_GpuQualifiers.H"
#include "AMReX_REAL.H"

namespace quokka::math
{

using amrex::literals::operator""_rt;

// The matrix a is any type whose a(i, j), for i, j = 0, ..., num_eqs - 1, returns a reference to an amrex::Real, such
// as amrex::Array2D<amrex::Real, 0, num_eqs - 1, 0, num_eqs - 1>.

/// Factors a = L U by Gaussian elimination, overwriting a with the factors. Unlike LINPACK, the diagonal of the factored
/// matrix holds the reciprocals of the pivots, so that dgesl multiplies instead of dividing (each pivot's reciprocal is
/// computed here anyway, for the multipliers). With allow_pivot it pivots on the largest entry of each column and records
/// the row interchanges in pivot; without it, pivot is not written. info is
///  - 0 when the factors can be used: every entry of a is finite and every pivot is nonzero;
///  - -1 when an entry of a is not finite (NaN or infinite); a is then not factored;
///  - otherwise one more than the index of a zero pivot, in which case dgesl would divide by zero.
template <int num_eqs, bool allow_pivot, typename Matrix>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void dgefa(Matrix &a, amrex::GpuArray<std::int16_t, num_eqs> &pivot, int &info)
{
	info = 0;

	// Every entry must be finite.
	// TODO(james471): this is a second pass over the whole matrix, cheap while the matrix is in registers (the
	// networks so far, up to about 15 equations on GPUs). For large networks, whose matrix is in memory, reconsider
	// it, e.g. by folding the test into the caller's pass that forms the matrix.
	bool finite = true;
	amrex::constexpr_for<0, num_eqs>([&](auto jc) {
		constexpr int j = decltype(jc)::value;
		amrex::constexpr_for<0, num_eqs>([&](auto ic) {
			constexpr int i = decltype(ic)::value;
			finite &= std::isfinite(a(i, j));
		});
	});
	if (!finite) {
		info = -1;
		return;
	}

	amrex::constexpr_for<0, num_eqs - 1>([&](auto kc) {
		constexpr int k = decltype(kc)::value;

		// find l = pivot index, and the pivot akk = a(l, k)
		[[maybe_unused]] int l = k; // used and changed only with allow_pivot, NOLINT(misc-const-correctness)
		amrex::Real akk = a(k, k);
		if constexpr (allow_pivot) {
			amrex::Real dmax = std::abs(a(k, k));
			amrex::constexpr_for<k + 1, num_eqs>([&](auto ic) {
				constexpr int i = decltype(ic)::value;
				const amrex::Real ai = std::abs(a(i, k));
				if (ai > dmax) {
					l = i;
					dmax = ai;
					akk = a(i, k);
				}
			});
			pivot[k] = static_cast<std::int16_t>(l);
		}

		// zero pivot implies this column already triangularized
		if (akk == 0.0e0_rt) {
			info = k + 1;
			return;
		}

		if constexpr (allow_pivot) {
			// interchange a(l, k) and a(k, k)
			amrex::constexpr_for<k + 1, num_eqs>([&](auto ic) {
				constexpr int i = decltype(ic)::value;
				a(i, k) = (i == l) ? a(k, k) : a(i, k);
			});
			a(k, k) = akk;
		}

		// compute multipliers, and store the reciprocal of the pivot for dgesl
		const amrex::Real t = -1.0e0_rt / a(k, k);
		amrex::constexpr_for<k + 1, num_eqs>([&](auto jc) {
			constexpr int j = decltype(jc)::value;
			a(j, k) *= t;
		});
		a(k, k) = -t;

		// row elimination with column indexing
		amrex::constexpr_for<k + 1, num_eqs>([&](auto jc) {
			constexpr int j = decltype(jc)::value;
			amrex::Real alj = a(k, j);
			if constexpr (allow_pivot) {
				// alj = a(l, j); then interchange a(l, j) and a(k, j)
				amrex::constexpr_for<k + 1, num_eqs>([&](auto ic) {
					constexpr int i = decltype(ic)::value;
					const bool swap = (i == l);
					alj = swap ? a(i, j) : alj;
					a(i, j) = swap ? a(k, j) : a(i, j);
				});
				a(k, j) = alj;
			}
			amrex::constexpr_for<k + 1, num_eqs>([&](auto ic) {
				constexpr int i = decltype(ic)::value;
				a(i, j) += alj * a(i, k);
			});
		});
	});

	if constexpr (allow_pivot) {
		pivot[num_eqs - 1] = static_cast<std::int16_t>(num_eqs - 1);
	}

	if (a(num_eqs - 1, num_eqs - 1) == 0.0e0_rt) {
		info = num_eqs;
	} else {
		a(num_eqs - 1, num_eqs - 1) = 1.0e0_rt / a(num_eqs - 1, num_eqs - 1);
	}
}

/// Solves a x = b and overwrites b with x, where a and pivot hold the factors from dgefa<num_eqs, allow_pivot>.
/// Without allow_pivot, pivot is not read.
template <int num_eqs, bool allow_pivot, typename Matrix>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void dgesl(const Matrix &a, const amrex::GpuArray<std::int16_t, num_eqs> &pivot,
						    amrex::GpuArray<amrex::Real, num_eqs> &b)
{
	// first solve l * y = b
	amrex::constexpr_for<0, num_eqs - 1>([&](auto kc) {
		constexpr int k = decltype(kc)::value;
		amrex::Real t = b[k];
		if constexpr (allow_pivot) {
			const int l = pivot[k];
			// t = b[l]; then interchange b[l] and b[k]
			amrex::constexpr_for<k + 1, num_eqs>([&](auto ic) {
				constexpr int i = decltype(ic)::value;
				const bool swap = (i == l);
				t = swap ? b[i] : t;
				b[i] = swap ? b[k] : b[i];
			});
			b[k] = t;
		}
		amrex::constexpr_for<k + 1, num_eqs>([&](auto jc) {
			constexpr int j = decltype(jc)::value;
			b[j] += t * a(j, k);
		});
	});

	// now solve u * x = y (dgefa stored the reciprocal of each pivot on the diagonal)
	amrex::constexpr_for<0, num_eqs>([&](auto kb) {
		constexpr int k = num_eqs - 1 - decltype(kb)::value;
		b[k] *= a(k, k);
		const amrex::Real t = -b[k];
		amrex::constexpr_for<0, k>([&](auto jc) {
			constexpr int j = decltype(jc)::value;
			b[j] += t * a(j, k);
		});
	});
}

} // namespace quokka::math

#endif // LINEAR_ALGEBRA_HPP_
