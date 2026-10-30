#ifndef MICROPHYSICS_ARRAY_UTILS_HPP_
#define MICROPHYSICS_ARRAY_UTILS_HPP_
/// \file array_utils.hpp
/// \brief The dense matrix of the microphysics integrators, with 1-based indices. Ported from Microphysics'
/// interfaces/ArrayUtilities.H. Transitional: it keeps Microphysics' 1-based indexing while the port is in progress,
/// and will be removed at the end of the Microphysics port.

#include "AMReX_Array.H"
#include "AMReX_BLassert.H"
#include "AMReX_Extension.H"
#include "AMReX_GpuQualifiers.H"
#include "AMReX_REAL.H"

namespace quokka::microphysics
{

using amrex::literals::operator""_rt;

/// N x N matrix with 1-based indices, stored column by column. Microphysics' ArrayUtil::MathArray2D took the element
/// type and all four index bounds as template parameters, and was always used as MathArray2D<amrex::Real, 1, N, 1, N>.
/// The unroll length of the loop in zero() is the CMake cache variable QUOKKA_NET_LOOP_UNROLL_LEN (default 1, no
/// unrolling), Microphysics' NET_LOOP_UNROLL_LEN.
template <int N> struct MathArray2D {
	/// position of element (i, j) in arr
	AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE static constexpr auto offset(const int i, const int j) noexcept -> int { return (i - 1) + (j - 1) * N; }

	AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void zero()
	{
		AMREX_UNROLL_LOOP(QUOKKA_NET_LOOP_UNROLL_LEN)
		for (int i = 0; i < N * N; ++i) {
			arr[i] = 0.0_rt;
		}
	}

	AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void mul(const amrex::Real x) noexcept
	{
		for (int i = 0; i < N * N; ++i) {
			arr[i] *= x;
		}
	}

	AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void set(const int i, const int j, const amrex::Real x) noexcept
	{
		AMREX_ASSERT(i >= 1 && i <= N && j >= 1 && j <= N);
		arr[offset(i, j)] = x;
	}

	AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void add(const int i, const int j, const amrex::Real x) noexcept
	{
		AMREX_ASSERT(i >= 1 && i <= N && j >= 1 && j <= N);
		arr[offset(i, j)] += x;
	}

	AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void mul(const int i, const int j, const amrex::Real x) noexcept
	{
		AMREX_ASSERT(i >= 1 && i <= N && j >= 1 && j <= N);
		arr[offset(i, j)] *= x;
	}

	[[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto get(const int i, const int j) const noexcept -> amrex::Real
	{
		AMREX_ASSERT(i >= 1 && i <= N && j >= 1 && j <= N);
		return arr[offset(i, j)];
	}

	AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void add_identity() noexcept
	{
		for (int i = 1; i <= N; ++i) {
			arr[offset(i, i)] += 1.0_rt;
		}
	}

	AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto operator()(const int i, const int j) const noexcept -> const amrex::Real &
	{
		AMREX_ASSERT(i >= 1 && i <= N && j >= 1 && j <= N);
		return arr[offset(i, j)];
	}

	AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto operator()(const int i, const int j) noexcept -> amrex::Real &
	{
		AMREX_ASSERT(i >= 1 && i <= N && j >= 1 && j <= N);
		return arr[offset(i, j)];
	}

	amrex::GpuArray<amrex::Real, N * N> arr;
};

} // namespace quokka::microphysics

#endif // MICROPHYSICS_ARRAY_UTILS_HPP_
