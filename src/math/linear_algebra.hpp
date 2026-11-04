#ifndef LINEAR_ALGEBRA_HPP_
#define LINEAR_ALGEBRA_HPP_
/// \file linear_algebra.hpp
/// \brief LU factorization and solve of a dense matrix with 1-based indices (LINPACK's dgefa and dgesl). Ported from
/// Microphysics' util/linpack.H.
/// Transitional: it keeps Microphysics' 1-based indexing
/// while the port is in progress, and will be replaced by a 0-based version at the end of the Microphysics port.

#include <cmath>
#include <cstdint>

#include "AMReX_Array.H"
#include "AMReX_Extension.H"
#include "AMReX_GpuQualifiers.H"
#include "AMReX_REAL.H"

namespace quokka::math
{

using amrex::literals::operator""_rt;

// The matrix a is any type whose a(i, j), for i, j = 1, ..., num_eqs, returns a reference to an amrex::Real, such as
// quokka::microphysics::MathArray2D<num_eqs>. The unroll length of the marked loops is the CMake cache variable
// QUOKKA_NET_LOOP_UNROLL_LEN (default 1, no unrolling), Microphysics' NET_LOOP_UNROLL_LEN.

/// Solves a x = b and overwrites b with x, where a and pivot hold the factors from dgefa<num_eqs, allow_pivot>.
/// Without allow_pivot, pivot is not read.
template <int num_eqs, bool allow_pivot, typename Matrix>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void dgesl(const Matrix &a, const amrex::Array1D<std::int16_t, 1, num_eqs> &pivot,
						    amrex::Array1D<amrex::Real, 1, num_eqs> &b)
{
	constexpr int nm1 = num_eqs - 1;

	// solve a * x = b
	// first solve l * y = b
	if constexpr (nm1 >= 1) {
		for (int k = 1; k <= nm1; ++k) {

			amrex::Real t{};
			if constexpr (allow_pivot) {
				const int l = pivot(k);
				t = b(l);
				if (l != k) {
					b(l) = b(k);
					b(k) = t;
				}
			} else {
				t = b(k);
			}

			AMREX_UNROLL_LOOP(QUOKKA_NET_LOOP_UNROLL_LEN)
			for (int j = k + 1; j <= num_eqs; ++j) {
				b(j) += t * a(j, k);
			}
		}
	}

	// now solve u * x = y
	for (int kb = 1; kb <= num_eqs; ++kb) {

		const int k = num_eqs + 1 - kb;
		b(k) = b(k) / a(k, k);
		const amrex::Real t = -b(k);
		AMREX_UNROLL_LOOP(QUOKKA_NET_LOOP_UNROLL_LEN)
		for (int j = 1; j <= k - 1; ++j) {
			b(j) += t * a(j, k);
		}
	}
}

/// Factors a = L U by Gaussian elimination, overwriting a with the factors. With allow_pivot it pivots on the largest
/// entry of each column and records the row swaps in pivot; without it, pivot is not written. info is 0 when every
/// pivot is nonzero, and otherwise the index of the last zero pivot, in which case dgesl would divide by zero.
template <int num_eqs, bool allow_pivot, typename Matrix>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void dgefa(Matrix &a, amrex::Array1D<std::int16_t, 1, num_eqs> &pivot, int &info)
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

		for (int k = 1; k <= nm1; ++k) {

			// find l = pivot index
			int l = k; // changed only with allow_pivot, NOLINT(misc-const-correctness)

			if constexpr (allow_pivot) {
				amrex::Real dmax = std::abs(a(k, k));
				AMREX_UNROLL_LOOP(QUOKKA_NET_LOOP_UNROLL_LEN)
				for (int i = k + 1; i <= num_eqs; ++i) {
					const amrex::Real ai = std::abs(a(i, k));
					if (ai > dmax) {
						l = i;
						dmax = ai;
					}
				}

				pivot(k) = static_cast<std::int16_t>(l);
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

				// compute multipliers
				t = -1.0e0_rt / a(k, k);
				AMREX_UNROLL_LOOP(QUOKKA_NET_LOOP_UNROLL_LEN)
				for (int j = k + 1; j <= num_eqs; ++j) {
					a(j, k) *= t;
				}

				// row elimination with column indexing
				for (int j = k + 1; j <= num_eqs; ++j) {
					t = a(l, j);

					if constexpr (allow_pivot) {
						if (l != k) {
							a(l, j) = a(k, j);
							a(k, j) = t;
						}
					}

					AMREX_UNROLL_LOOP(QUOKKA_NET_LOOP_UNROLL_LEN)
					for (int i = k + 1; i <= num_eqs; ++i) {
						a(i, j) += t * a(i, k);
					}
				}

			} else {
				info = k;
			}
		}
	}

	if constexpr (allow_pivot) {
		pivot(num_eqs) = static_cast<std::int16_t>(num_eqs);
	}

	if (a(num_eqs, num_eqs) == 0.0e0_rt) {
		info = num_eqs;
	}
}

} // namespace quokka::math

#endif // LINEAR_ALGEBRA_HPP_
