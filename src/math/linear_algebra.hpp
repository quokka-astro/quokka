// SPDX-FileCopyrightText: © The Quokka Authors
// SPDX-License-Identifier: MIT

#ifndef LINEAR_ALGEBRA_HPP_
#define LINEAR_ALGEBRA_HPP_
/// \file linear_algebra.hpp
/// \brief LU factorization and solve of a dense matrix (LINPACK's dgefa and dgesl). Ported from Microphysics'
/// util/linpack.H, with 0-based indices.
///
/// In device code every loop is unrolled at compile time (amrex::constexpr_for), and the pivot row is selected by
/// comparing it with each candidate row instead of indexing with it. All matrix and vector indices are then compile-time
/// constants, so the compiler can keep the matrix and the right-hand side in registers instead of local memory (a
/// matrix too large for the registers has its pivot rows interchanged by index, see dgefa_unrolled). On the host, where
/// that form is not faster, the loops index with the pivot row as in dgefa/dgesl. All forms do the same arithmetic in
/// the same order, so they give identical results.

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
// as amrex::Array2D<amrex::Real, 0, num_eqs - 1, 0, num_eqs - 1>. The unroll length of the marked loops is the CMake
// cache variable QUOKKA_NET_LOOP_UNROLL_LEN (default 1, no unrolling), Microphysics' NET_LOOP_UNROLL_LEN; it applies
// to the host loops only.

namespace detail
{

/// dgesl with loops and indexing by the pivot row (used on the host)
template <int num_eqs, bool allow_pivot, typename Matrix>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void dgesl_loops(const Matrix &a, const amrex::GpuArray<std::int16_t, num_eqs> &pivot,
							  amrex::GpuArray<amrex::Real, num_eqs> &b)
{
	constexpr int nm1 = num_eqs - 1;

	// solve a * x = b
	// first solve l * y = b
	if constexpr (nm1 >= 1) {
		for (int k = 0; k < nm1; ++k) {

			amrex::Real t{};
			if constexpr (allow_pivot) {
				const int l = pivot[k];
				t = b[l];
				if (l != k) {
					b[l] = b[k];
					b[k] = t;
				}
			} else {
				t = b[k];
			}

			AMREX_UNROLL_LOOP(QUOKKA_NET_LOOP_UNROLL_LEN)
			for (int j = k + 1; j < num_eqs; ++j) {
				b[j] += t * a(j, k);
			}
		}
	}

	// now solve u * x = y (dgefa stored the reciprocal of each pivot on the diagonal)
	for (int k = num_eqs - 1; k >= 0; --k) {
		b[k] = b[k] * a(k, k);
		const amrex::Real t = -b[k];
		AMREX_UNROLL_LOOP(QUOKKA_NET_LOOP_UNROLL_LEN)
		for (int j = 0; j < k; ++j) {
			b[j] += t * a(j, k);
		}
	}
}

/// dgefa with loops and indexing by the pivot row (used on the host)
template <int num_eqs, bool allow_pivot, typename Matrix>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void dgefa_loops(Matrix &a, amrex::GpuArray<std::int16_t, num_eqs> &pivot, int &info)
{
	// dgefa factors a matrix by gaussian elimination.
	// a is returned in the form a = l * u where
	// l is a product of permutation and unit lower
	// triangular matrices and u is upper triangular.

	// gaussian elimination with partial pivoting

	info = 0;
	constexpr int nm1 = num_eqs - 1;

	if constexpr (nm1 >= 1) {

		amrex::Real t{};

		for (int k = 0; k < nm1; ++k) {

			// find l = pivot index
			int l = k; // changed only with allow_pivot, NOLINT(misc-const-correctness)

			if constexpr (allow_pivot) {
				amrex::Real dmax = std::abs(a(k, k));
				AMREX_UNROLL_LOOP(QUOKKA_NET_LOOP_UNROLL_LEN)
				for (int i = k + 1; i < num_eqs; ++i) {
					const amrex::Real ai = std::abs(a(i, k));
					if (ai > dmax) {
						l = i;
						dmax = ai;
					}
				}

				pivot[k] = static_cast<std::int16_t>(l);
			}

			// zero pivot implies this column already triangularized
			if (a(l, k) != 0.0e0_rt) {

				if constexpr (allow_pivot) {
					// interchange if necessary
					if (l != k) {
						t = a(l, k);
						a(l, k) = a(k, k);
						a(k, k) = t;
					}
				}

				// compute multipliers, and store the reciprocal of the pivot for dgesl
				t = -1.0e0_rt / a(k, k);
				a(k, k) = -t;
				AMREX_UNROLL_LOOP(QUOKKA_NET_LOOP_UNROLL_LEN)
				for (int j = k + 1; j < num_eqs; ++j) {
					a(j, k) *= t;
				}

				// row elimination with column indexing
				for (int j = k + 1; j < num_eqs; ++j) {
					t = a(l, j);

					if constexpr (allow_pivot) {
						if (l != k) {
							a(l, j) = a(k, j);
							a(k, j) = t;
						}
					}

					AMREX_UNROLL_LOOP(QUOKKA_NET_LOOP_UNROLL_LEN)
					for (int i = k + 1; i < num_eqs; ++i) {
						a(i, j) += t * a(i, k);
					}
				}

			} else {
				info = k + 1;
			}
		}
	}

	if constexpr (allow_pivot) {
		pivot[nm1] = static_cast<std::int16_t>(nm1);
	}

	if (a(nm1, nm1) == 0.0e0_rt) {
		info = num_eqs;
	} else {
		a(nm1, nm1) = 1.0e0_rt / a(nm1, nm1);
	}
}

/// dgesl unrolled at compile time, the pivot row selected by comparison (used in device code)
template <int num_eqs, bool allow_pivot, typename Matrix>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void dgesl_unrolled(const Matrix &a, const amrex::GpuArray<std::int16_t, num_eqs> &pivot,
							     amrex::GpuArray<amrex::Real, num_eqs> &b)
{
	constexpr int nm1 = num_eqs - 1;

	// first solve l * y = b
	amrex::constexpr_for<0, nm1>([&](auto kc) {
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
	amrex::constexpr_for<0, num_eqs>([&](auto kr) {
		constexpr int k = num_eqs - 1 - decltype(kr)::value;
		b[k] = b[k] * a(k, k);
		const amrex::Real t = -b[k];
		amrex::constexpr_for<0, k>([&](auto jc) {
			constexpr int j = decltype(jc)::value;
			b[j] += t * a(j, k);
		});
	});
}

/// dgefa unrolled at compile time, the pivot row selected by comparison (used in device code)
template <int num_eqs, bool allow_pivot, typename Matrix>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void dgefa_unrolled(Matrix &a, amrex::GpuArray<std::int16_t, num_eqs> &pivot, int &info)
{
	info = 0;
	constexpr int nm1 = num_eqs - 1;

	// A matrix of more than 128 entries cannot be in registers (256 VGPRs hold 128 doubles) and is in local memory
	// whatever its indices are. Its pivot row is then interchanged by indexing with it, 2 (n - k) loads and stores per
	// step, instead of a select and a store on each of the (n - k)^2 entries of the trailing matrix: with 23 equations
	// those selects were 29 % of the photochemistry kernel's samples on an MI250X.
	constexpr bool swap_by_index = num_eqs * num_eqs > 128;

	amrex::constexpr_for<0, nm1>([&](auto kc) {
		constexpr int k = decltype(kc)::value;

		// find l = pivot index, and the pivot akk = a(l, k)
		int l = k; // changed only with allow_pivot, NOLINT(misc-const-correctness)
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
			if constexpr (swap_by_index) {
				a(l, k) = a(k, k);
			} else {
				amrex::constexpr_for<k + 1, num_eqs>([&](auto ic) {
					constexpr int i = decltype(ic)::value;
					a(i, k) = (i == l) ? a(k, k) : a(i, k);
				});
			}
			a(k, k) = akk;
		}

		// compute multipliers, and store the reciprocal of the pivot for dgesl
		const amrex::Real t = -1.0e0_rt / a(k, k);
		a(k, k) = -t;
		amrex::constexpr_for<k + 1, num_eqs>([&](auto jc) {
			constexpr int j = decltype(jc)::value;
			a(j, k) *= t;
		});

		// row elimination with column indexing
		amrex::constexpr_for<k + 1, num_eqs>([&](auto jc) {
			constexpr int j = decltype(jc)::value;
			amrex::Real alj = a(k, j);
			if constexpr (allow_pivot && swap_by_index) {
				// alj = a(l, j); then interchange a(l, j) and a(k, j)
				const amrex::Real akj = alj;
				alj = a(l, j);
				a(l, j) = akj;
				a(k, j) = alj;
			} else if constexpr (allow_pivot) {
				// the same with selects, the matrix staying in registers
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
		pivot[nm1] = static_cast<std::int16_t>(nm1);
	}

	if (a(nm1, nm1) == 0.0e0_rt) {
		info = num_eqs;
	} else {
		a(nm1, nm1) = 1.0e0_rt / a(nm1, nm1);
	}
}

} // namespace detail

/// Solves a x = b and overwrites b with x, where a and pivot hold the factors from dgefa<num_eqs, allow_pivot>.
/// Without allow_pivot, pivot is not read.
template <int num_eqs, bool allow_pivot, typename Matrix>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void dgesl(const Matrix &a, const amrex::GpuArray<std::int16_t, num_eqs> &pivot,
						    amrex::GpuArray<amrex::Real, num_eqs> &b)
{
#if AMREX_DEVICE_COMPILE
	detail::dgesl_unrolled<num_eqs, allow_pivot>(a, pivot, b);
#else
	detail::dgesl_loops<num_eqs, allow_pivot>(a, pivot, b);
#endif
}

/// Factors a = L U by Gaussian elimination, overwriting a with the factors. With allow_pivot it pivots on the largest
/// entry of each column and records the row swaps in pivot; without it, pivot is not written. info is 0 when every
/// pivot is nonzero, and otherwise one more than the index of the last zero pivot, in which case dgesl must not be
/// called. Unlike LINPACK, the diagonal of the factored matrix holds the reciprocals of the pivots, so that dgesl
/// multiplies instead of dividing (each pivot's reciprocal is computed here anyway, for the multipliers).
template <int num_eqs, bool allow_pivot, typename Matrix>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void dgefa(Matrix &a, amrex::GpuArray<std::int16_t, num_eqs> &pivot, int &info)
{
#if AMREX_DEVICE_COMPILE
	detail::dgefa_unrolled<num_eqs, allow_pivot>(a, pivot, info);
#else
	detail::dgefa_loops<num_eqs, allow_pivot>(a, pivot, info);
#endif
}

} // namespace quokka::math

#endif // LINEAR_ALGEBRA_HPP_
