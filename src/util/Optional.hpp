#ifndef OPTIONAL_HPP_
#define OPTIONAL_HPP_
//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file Optional.hpp
/// \brief Implements a minimal GPU-compatible optional class for EOS.hpp usage.

#include "AMReX_GpuQualifiers.H"
#include <utility>

namespace quokka
{

template <typename T> class optional
{
      private:
	// NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,hicpp-avoid-c-arrays,modernize-avoid-c-arrays)
	alignas(T) mutable char storage_[sizeof(T)]{};
	bool has_value_{false};

      public:
	// Default constructor - creates empty optional
	AMREX_GPU_HOST_DEVICE constexpr optional() noexcept = default;

	// Value constructor - creates optional containing a value
	// NOLINTNEXTLINE(google-explicit-constructor,hicpp-explicit-conversions)
	AMREX_GPU_HOST_DEVICE constexpr optional(const T &value) : has_value_(true) { new (storage_) T(value); }

	// Copy constructor
	AMREX_GPU_HOST_DEVICE optional(const optional &other) : has_value_(other.has_value_)
	{
		if (has_value_) {
			new (storage_) T(*other);
		}
	}

	// Move constructor
	AMREX_GPU_HOST_DEVICE optional(optional &&other) noexcept : has_value_(other.has_value_)
	{
		if (has_value_) {
			new (storage_) T(std::move(*other));
			reinterpret_cast<T *>(other.storage_)->~T(); // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
			other.has_value_ = false;
		}
	}

	// Copy assignment operator
	AMREX_GPU_HOST_DEVICE auto operator=(const optional &other) -> optional &
	{
		if (this != &other) {
			if (has_value_) {
				reinterpret_cast<T *>(storage_)->~T(); // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
			}
			has_value_ = other.has_value_;
			if (has_value_) {
				new (storage_) T(*other);
			}
		}
		return *this;
	}

	// Move assignment operator
	AMREX_GPU_HOST_DEVICE auto operator=(optional &&other) noexcept -> optional &
	{
		if (this != &other) {
			if (has_value_) {
				reinterpret_cast<T *>(storage_)->~T(); // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
			}
			has_value_ = other.has_value_;
			if (has_value_) {
				new (storage_) T(std::move(*other));
				reinterpret_cast<T *>(other.storage_)->~T(); // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
				other.has_value_ = false;
			}
		}
		return *this;
	}

	// Destructor
	AMREX_GPU_HOST_DEVICE ~optional()
	{
		if (has_value_) {
			reinterpret_cast<T *>(storage_)->~T(); // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
		}
	}

	// Boolean conversion - check if optional has a value
	AMREX_GPU_HOST_DEVICE constexpr explicit operator bool() const noexcept { return has_value_; }

	// Dereference operator - access the contained value
	AMREX_GPU_HOST_DEVICE constexpr auto operator*() const & noexcept -> const T &
	{
		return *reinterpret_cast<const T *>(storage_); // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
	}
}; // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)

} // namespace quokka

#endif // OPTIONAL_HPP_