#ifndef OPTIONALARRAYCAPTURE_HPP_
#define OPTIONALARRAYCAPTURE_HPP_
//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file OptionalArrayCapture.hpp
/// \brief Provides a CUDA-friendly wrapper for conditionally available Array4 vectors.

#include "AMReX_GpuQualifiers.H"

#include <type_traits>
#include <utility>

namespace quokka::detail
{
	template <bool Enabled, typename ArrayVec> class OptionalArrayCapture;

	template <typename ArrayVec> class OptionalArrayCapture<true, ArrayVec>
	{
	      public:
		using value_type = std::remove_reference_t<decltype(std::declval<ArrayVec>()[0])>;

		AMREX_GPU_HOST_DEVICE OptionalArrayCapture() = default;
		explicit AMREX_GPU_HOST_DEVICE OptionalArrayCapture(ArrayVec arrays_in) : arrays_(arrays_in) {}

		AMREX_GPU_DEVICE AMREX_FORCE_INLINE auto operator[](int idx) const noexcept -> value_type { return arrays_[idx]; }

	      private:
		ArrayVec arrays_{};
	};

	template <typename ArrayVec> class OptionalArrayCapture<false, ArrayVec>
	{
	      public:
		using value_type = std::remove_reference_t<decltype(std::declval<ArrayVec>()[0])>;

		AMREX_GPU_HOST_DEVICE constexpr OptionalArrayCapture() noexcept = default;
		explicit AMREX_GPU_HOST_DEVICE OptionalArrayCapture(ArrayVec const &) noexcept {}

		AMREX_GPU_DEVICE AMREX_FORCE_INLINE auto operator[](int) const noexcept -> value_type { return value_type{}; }
	};

	template <bool Enabled, typename MultiFab> auto make_optional_array_capture(MultiFab const &mf)
	{
		using ArraysType = decltype(mf.const_arrays());
		if constexpr (Enabled) {
			return OptionalArrayCapture<true, ArraysType>(mf.const_arrays());
		} else {
			return OptionalArrayCapture<false, ArraysType>();
		}
	}

	template <bool Enabled, typename Provider> auto make_optional_array_capture_from_provider(Provider &&provider)
	{
		using ArraysType = decltype(provider());
		if constexpr (Enabled) {
			return OptionalArrayCapture<true, ArraysType>(provider());
		} else {
			return OptionalArrayCapture<false, ArraysType>();
		}
	}

} // namespace quokka::detail

#endif // OPTIONALARRAYCAPTURE_HPP_
