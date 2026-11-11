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
/// constants, so the compiler can keep the matrix and the right-hand side in registers instead of local memory. A
/// matrix too large for the registers is factored column by column instead, each column passing through the registers
/// once (dgefa_unrolled_columns). On the host, where neither form is faster, the loops index with the pivot row as in
/// dgefa/dgesl. All forms do the same arithmetic in the same order, so they give identical results.

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
			amrex::constexpr_for<k + 1, num_eqs>([&](auto ic) {
				constexpr int i = decltype(ic)::value;
				a(i, k) = (i == l) ? a(k, k) : a(i, k);
			});
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
		pivot[nm1] = static_cast<std::int16_t>(nm1);
	}

	if (a(nm1, nm1) == 0.0e0_rt) {
		info = num_eqs;
	} else {
		a(nm1, nm1) = 1.0e0_rt / a(nm1, nm1);
	}
}

/// A matrix of more than 128 entries cannot be in registers (256 VGPRs hold 128 doubles) and is in local memory
/// whatever its indices are; dgefa_unrolled_columns and dgesl_unrolled_columns are used for it in device code.
template <int num_eqs> constexpr bool in_local_memory = num_eqs * num_eqs > 128;

/// dgefa unrolled at compile time for a matrix in local memory (see in_local_memory). The elimination goes column by
/// column: each column is loaded into registers once, updated with the multipliers of the columns before it and stored
/// once, so the matrix is read about n^3/6 times and written n^2 times, against n^3/3 reads and writes of the trailing
/// matrix in dgefa_unrolled (and as many selects for its row interchanges, 29 % of the photochemistry kernel's samples
/// with 23 equations on an MI250X). The pivot rows are interchanged by indexing with the pivot row, 2 n loads and
/// stores per step, throughout the row, multipliers included (LAPACK's convention, not LINPACK's), so dgesl must be
/// dgesl_unrolled_columns. The arithmetic is the same as dgefa_unrolled's in the same order, so the factors have the
/// same values; only after a zero pivot (info != 0, factors unusable) do they differ, as that step is not skipped here.
template <int num_eqs, bool allow_pivot, typename Matrix>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void dgefa_unrolled_columns(Matrix &a, amrex::GpuArray<std::int16_t, num_eqs> &pivot, int &info)
{
	info = 0;

	amrex::constexpr_for<0, num_eqs>([&](auto jc) {
		constexpr int j = decltype(jc)::value;

		// column j, with the row interchanges of the steps before it already applied to it
		amrex::GpuArray<amrex::Real, num_eqs> col{};
		amrex::constexpr_for<0, num_eqs>([&](auto ic) {
			constexpr int i = decltype(ic)::value;
			col[i] = a(i, j);
		});

		// the eliminations of the steps before it: step k adds a(k, j) times the multipliers of column k to the rows below k
		amrex::constexpr_for<0, j>([&](auto kc) {
			constexpr int k = decltype(kc)::value;
			const amrex::Real akj = col[k];
			amrex::constexpr_for<k + 1, num_eqs>([&](auto ic) {
				constexpr int i = decltype(ic)::value;
				col[i] += akj * a(i, k);
			});
		});

		// find l = pivot index, and the pivot akk = a(l, j)
		int l = j; // changed only with allow_pivot, NOLINT(misc-const-correctness)
		amrex::Real akk = col[j];
		if constexpr (allow_pivot) {
			amrex::Real dmax = std::abs(col[j]);
			amrex::constexpr_for<j + 1, num_eqs>([&](auto ic) {
				constexpr int i = decltype(ic)::value;
				const amrex::Real ai = std::abs(col[i]);
				if (ai > dmax) {
					l = i;
					dmax = ai;
					akk = col[i];
				}
			});
			pivot[j] = static_cast<std::int16_t>(l);

			// interchange rows l and j: in the column by selects, in the rest of the matrix by index. The branch skips
			// the 2 n loads and stores when the pivot is on the diagonal already (the usual case of an integrator
			// matrix, dominated by its diagonal at a small step); a lane that needs them only waits for the others.
			if (l != j) {
				amrex::constexpr_for<j + 1, num_eqs>([&](auto ic) {
					constexpr int i = decltype(ic)::value;
					col[i] = (i == l) ? col[j] : col[i];
				});
				col[j] = akk;
				amrex::constexpr_for<0, num_eqs>([&](auto cc) {
					constexpr int c = decltype(cc)::value;
					if constexpr (c != j) {
						const amrex::Real t = a(j, c);
						a(j, c) = a(l, c);
						a(l, c) = t;
					}
				});
			}
		}

		// zero pivot implies this column already triangularized
		if (akk == 0.0e0_rt) {
			info = j + 1;
		} else {
			// compute multipliers, and store the reciprocal of the pivot for dgesl
			const amrex::Real t = -1.0e0_rt / akk;
			col[j] = -t;
			amrex::constexpr_for<j + 1, num_eqs>([&](auto ic) {
				constexpr int i = decltype(ic)::value;
				col[i] *= t;
			});
		}

		amrex::constexpr_for<0, num_eqs>([&](auto ic) {
			constexpr int i = decltype(ic)::value;
			a(i, j) = col[i];
		});
	});
}

/// dgesl for the factors of dgefa_unrolled_columns: their rows are interchanged throughout, so b is permuted the same
/// way first, then substituted. The same operations as dgesl_unrolled in the same order, so the same solution.
template <int num_eqs, bool allow_pivot, typename Matrix>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void dgesl_unrolled_columns(const Matrix &a, const amrex::GpuArray<std::int16_t, num_eqs> &pivot,
								     amrex::GpuArray<amrex::Real, num_eqs> &b)
{
	constexpr int nm1 = num_eqs - 1;

	if constexpr (allow_pivot) {
		amrex::constexpr_for<0, nm1>([&](auto kc) {
			constexpr int k = decltype(kc)::value;
			const int l = pivot[k];
			const amrex::Real bk = b[k];
			amrex::Real bl = bk;
			// bl = b[l]; then interchange b[l] and b[k]
			amrex::constexpr_for<k + 1, num_eqs>([&](auto ic) {
				constexpr int i = decltype(ic)::value;
				const bool swap = (i == l);
				bl = swap ? b[i] : bl;
				b[i] = swap ? bk : b[i];
			});
			b[k] = bl;
		});
	}

	// solve l * y = b
	amrex::constexpr_for<0, nm1>([&](auto kc) {
		constexpr int k = decltype(kc)::value;
		const amrex::Real t = b[k];
		amrex::constexpr_for<k + 1, num_eqs>([&](auto jc) {
			constexpr int j = decltype(jc)::value;
			b[j] += t * a(j, k);
		});
	});

	// now solve u * x = y (the diagonal holds the reciprocal of each pivot)
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

} // namespace detail

/// Solves a x = b and overwrites b with x, where a and pivot hold the factors from dgefa<num_eqs, allow_pivot>.
/// Without allow_pivot, pivot is not read.
template <int num_eqs, bool allow_pivot, typename Matrix>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void dgesl(const Matrix &a, const amrex::GpuArray<std::int16_t, num_eqs> &pivot,
						    amrex::GpuArray<amrex::Real, num_eqs> &b)
{
#if AMREX_DEVICE_COMPILE
	if constexpr (detail::in_local_memory<num_eqs>) {
		detail::dgesl_unrolled_columns<num_eqs, allow_pivot>(a, pivot, b);
	} else {
		detail::dgesl_unrolled<num_eqs, allow_pivot>(a, pivot, b);
	}
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
	if constexpr (detail::in_local_memory<num_eqs>) {
		detail::dgefa_unrolled_columns<num_eqs, allow_pivot>(a, pivot, info);
	} else {
		detail::dgefa_unrolled<num_eqs, allow_pivot>(a, pivot, info);
	}
#else
	detail::dgefa_loops<num_eqs, allow_pivot>(a, pivot, info);
#endif
}

} // namespace quokka::math

#endif // LINEAR_ALGEBRA_HPP_
