#ifndef OPTIONAL_HPP_
#define OPTIONAL_HPP_
//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file Optional.hpp
/// \brief Implements a GPU-compatible optional class.

#include <type_traits>
#include <utility>
#include "AMReX_GpuQualifiers.H"

namespace quokka
{

template <typename T>
class optional
{
  private:
	alignas(T) mutable char storage_[sizeof(T)];
	bool has_value_;

  public:
	using value_type = T;

	// Default constructor
	AMREX_GPU_HOST_DEVICE constexpr optional() noexcept : has_value_(false) {}

	// Nullopt constructor
	AMREX_GPU_HOST_DEVICE constexpr optional(std::nullopt_t) noexcept : has_value_(false) {}

	// Copy constructor
	AMREX_GPU_HOST_DEVICE constexpr optional(const optional& other) : has_value_(other.has_value_)
	{
		if (has_value_) {
			new (storage_) T(other.value());
		}
	}

	// Move constructor
	AMREX_GPU_HOST_DEVICE constexpr optional(optional&& other) noexcept(std::is_nothrow_move_constructible_v<T>)
		: has_value_(other.has_value_)
	{
		if (has_value_) {
			new (storage_) T(std::move(other.value()));
			other.reset();
		}
	}

	// Value constructor
	template <typename U, typename = std::enable_if_t<std::is_constructible_v<T, U&&>>>
	AMREX_GPU_HOST_DEVICE constexpr optional(U&& value)
		: has_value_(true)
	{
		new (storage_) T(std::forward<U>(value));
	}

	// Destructor
	AMREX_GPU_HOST_DEVICE ~optional()
	{
		if (has_value_) {
			reinterpret_cast<T*>(storage_)->~T();
		}
	}

	// Copy assignment
	AMREX_GPU_HOST_DEVICE constexpr optional& operator=(const optional& other)
	{
		if (this != &other) {
			if (has_value_ && other.has_value_) {
				value() = other.value();
			} else if (other.has_value_) {
				new (storage_) T(other.value());
				has_value_ = true;
			} else {
				reset();
			}
		}
		return *this;
	}

	// Move assignment
	AMREX_GPU_HOST_DEVICE constexpr optional& operator=(optional&& other) noexcept(std::is_nothrow_move_assignable_v<T> && std::is_nothrow_move_constructible_v<T>)
	{
		if (this != &other) {
			if (has_value_ && other.has_value_) {
				value() = std::move(other.value());
			} else if (other.has_value_) {
				new (storage_) T(std::move(other.value()));
				has_value_ = true;
			} else {
				reset();
			}
			other.reset();
		}
		return *this;
	}

	// Value assignment
	template <typename U, typename = std::enable_if_t<std::is_constructible_v<T, U&&>>>
	AMREX_GPU_HOST_DEVICE constexpr optional& operator=(U&& value)
	{
		if (has_value_) {
			this->value() = std::forward<U>(value);
		} else {
			new (storage_) T(std::forward<U>(value));
			has_value_ = true;
		}
		return *this;
	}

	// Nullopt assignment
	AMREX_GPU_HOST_DEVICE constexpr optional& operator=(std::nullopt_t) noexcept
	{
		reset();
		return *this;
	}

	// Dereference operators (the problematic ones from std::optional)
	AMREX_GPU_HOST_DEVICE constexpr const T& operator*() const& noexcept
	{
		return *reinterpret_cast<const T*>(storage_);
	}

	AMREX_GPU_HOST_DEVICE constexpr T& operator*() & noexcept
	{
		return *reinterpret_cast<T*>(storage_);
	}

	AMREX_GPU_HOST_DEVICE constexpr const T&& operator*() const&& noexcept
	{
		return std::move(*reinterpret_cast<const T*>(storage_));
	}

	AMREX_GPU_HOST_DEVICE constexpr T&& operator*() && noexcept
	{
		return std::move(*reinterpret_cast<T*>(storage_));
	}

	// Arrow operators
	AMREX_GPU_HOST_DEVICE constexpr const T* operator->() const noexcept
	{
		return reinterpret_cast<const T*>(storage_);
	}

	AMREX_GPU_HOST_DEVICE constexpr T* operator->() noexcept
	{
		return reinterpret_cast<T*>(storage_);
	}

	// has_value() and operator bool
	AMREX_GPU_HOST_DEVICE constexpr bool has_value() const noexcept
	{
		return has_value_;
	}

	AMREX_GPU_HOST_DEVICE constexpr explicit operator bool() const noexcept
	{
		return has_value_;
	}

	// value() functions
	AMREX_GPU_HOST_DEVICE constexpr const T& value() const&
	{
		return *reinterpret_cast<const T*>(storage_);
	}

	AMREX_GPU_HOST_DEVICE constexpr T& value() &
	{
		return *reinterpret_cast<T*>(storage_);
	}

	AMREX_GPU_HOST_DEVICE constexpr const T&& value() const&&
	{
		return std::move(*reinterpret_cast<const T*>(storage_));
	}

	AMREX_GPU_HOST_DEVICE constexpr T&& value() &&
	{
		return std::move(*reinterpret_cast<T*>(storage_));
	}

	// value_or() functions
	template <typename U>
	AMREX_GPU_HOST_DEVICE constexpr T value_or(U&& default_value) const&
	{
		return has_value_ ? value() : static_cast<T>(std::forward<U>(default_value));
	}

	template <typename U>
	AMREX_GPU_HOST_DEVICE constexpr T value_or(U&& default_value) &&
	{
		return has_value_ ? std::move(value()) : static_cast<T>(std::forward<U>(default_value));
	}

	// reset() function
	AMREX_GPU_HOST_DEVICE constexpr void reset() noexcept
	{
		if (has_value_) {
			reinterpret_cast<T*>(storage_)->~T();
			has_value_ = false;
		}
	}

	// emplace() function
	template <typename... Args>
	AMREX_GPU_HOST_DEVICE constexpr T& emplace(Args&&... args)
	{
		reset();
		new (storage_) T(std::forward<Args>(args)...);
		has_value_ = true;
		return value();
	}
};

// Comparison operators
template <typename T, typename U>
AMREX_GPU_HOST_DEVICE constexpr bool operator==(const optional<T>& lhs, const optional<U>& rhs)
{
	return lhs.has_value() == rhs.has_value() && (!lhs.has_value() || *lhs == *rhs);
}

template <typename T, typename U>
AMREX_GPU_HOST_DEVICE constexpr bool operator!=(const optional<T>& lhs, const optional<U>& rhs)
{
	return !(lhs == rhs);
}

template <typename T, typename U>
AMREX_GPU_HOST_DEVICE constexpr bool operator<(const optional<T>& lhs, const optional<U>& rhs)
{
	return rhs.has_value() && (!lhs.has_value() || *lhs < *rhs);
}

template <typename T, typename U>
AMREX_GPU_HOST_DEVICE constexpr bool operator<=(const optional<T>& lhs, const optional<U>& rhs)
{
	return !(rhs < lhs);
}

template <typename T, typename U>
AMREX_GPU_HOST_DEVICE constexpr bool operator>(const optional<T>& lhs, const optional<U>& rhs)
{
	return rhs < lhs;
}

template <typename T, typename U>
AMREX_GPU_HOST_DEVICE constexpr bool operator>=(const optional<T>& lhs, const optional<U>& rhs)
{
	return !(lhs < rhs);
}

// Comparison with nullopt
template <typename T>
AMREX_GPU_HOST_DEVICE constexpr bool operator==(const optional<T>& opt, std::nullopt_t) noexcept
{
	return !opt.has_value();
}

template <typename T>
AMREX_GPU_HOST_DEVICE constexpr bool operator==(std::nullopt_t, const optional<T>& opt) noexcept
{
	return !opt.has_value();
}

template <typename T>
AMREX_GPU_HOST_DEVICE constexpr bool operator!=(const optional<T>& opt, std::nullopt_t) noexcept
{
	return opt.has_value();
}

template <typename T>
AMREX_GPU_HOST_DEVICE constexpr bool operator!=(std::nullopt_t, const optional<T>& opt) noexcept
{
	return opt.has_value();
}

// Comparison with values
template <typename T, typename U>
AMREX_GPU_HOST_DEVICE constexpr bool operator==(const optional<T>& opt, const U& value)
{
	return opt.has_value() && *opt == value;
}

template <typename T, typename U>
AMREX_GPU_HOST_DEVICE constexpr bool operator==(const T& value, const optional<U>& opt)
{
	return opt.has_value() && value == *opt;
}

template <typename T, typename U>
AMREX_GPU_HOST_DEVICE constexpr bool operator!=(const optional<T>& opt, const U& value)
{
	return !opt.has_value() || *opt != value;
}

template <typename T, typename U>
AMREX_GPU_HOST_DEVICE constexpr bool operator!=(const T& value, const optional<U>& opt)
{
	return !opt.has_value() || value != *opt;
}

} // namespace quokka

#endif // OPTIONAL_HPP_