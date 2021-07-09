#ifndef ARRAYVIEW_HPP_
#define ARRAYVIEW_HPP_
//==============================================================================
// AMRAdvection
// Copyright 2021 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file ArrayView.hpp
/// \brief A container for an array of Reals with template magic to permute indices

// library headers
#include <AMReX_Utility.H>

// These functions are defined such that, e.g., Array4View<X2>::operator(LOOP_ORDER_X2(i,j,k)) ==
// arr_(i,j,k). Therefore, they do NOT have the same index ordering as that inside the corresponding
// Array4View<>::operator()!

enum class FluxDir { X1 = 0, X2 = 1, X3 = 2 };

namespace quokka
{
template <FluxDir N> AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto reorderMultiIndex(int, int, int);

template <>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto reorderMultiIndex<FluxDir::X1>(int i, int j, int k)
{
	return std::make_tuple(i, j, k);
}

template <>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto reorderMultiIndex<FluxDir::X2>(int i, int j, int k)
{
	return std::make_tuple(j, k, i);
}

template <>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto reorderMultiIndex<FluxDir::X3>(int i, int j, int k)
{
	return std::make_tuple(k, i, j);
}

template <class T, FluxDir N, class Enable = void> struct Array4View {
	amrex::Array4<T> arr_;
	constexpr static FluxDir indexOrder = N;

	explicit Array4View(amrex::Array4<T> arr) : arr_(arr) {}
};

// X1-flux

// if T is non-const
template <class T> struct Array4View<T, FluxDir::X1, std::enable_if_t<!std::is_const_v<T>>> {
	amrex::Array4<T> arr_;
	constexpr static FluxDir indexOrder = FluxDir::X1;

	explicit Array4View(amrex::Array4<T> arr) : arr_(arr) {}

	AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto operator()(int i, int j, int k,
								 int n) const noexcept -> T &
	{
		return arr_(i, j, k, n);
	}

	AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto operator()(int i, int j, int k) const noexcept
	    -> T &
	{
		return arr_(i, j, k);
	}
};

// if T is const
template <class T> struct Array4View<T, FluxDir::X1, std::enable_if_t<std::is_const_v<T>>> {
	amrex::Array4<T> arr_;
	constexpr static FluxDir indexOrder = FluxDir::X1;

	explicit Array4View(amrex::Array4<T> arr) : arr_(arr) {}

	AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto operator()(int i, int j, int k,
								 int n) const noexcept -> T
	{
		return arr_(i, j, k, n);
	}

	AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto operator()(int i, int j, int k) const noexcept
	    -> T
	{
		return arr_(i, j, k);
	}
};

// X2-flux

// if T is non-const
template <class T> struct Array4View<T, FluxDir::X2, std::enable_if_t<!std::is_const_v<T>>> {
	amrex::Array4<T> arr_;
	constexpr static FluxDir indexOrder = FluxDir::X2;

	explicit Array4View(amrex::Array4<T> arr) : arr_(arr) {}

	AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto operator()(int i, int j, int k,
								 int n) const noexcept -> T &
	{
		return arr_(k, i, j, n);
	}

	AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto operator()(int i, int j, int k) const noexcept
	    -> T &
	{
		return arr_(k, i, j);
	}
};

// if T is const
template <class T> struct Array4View<T, FluxDir::X2, std::enable_if_t<std::is_const_v<T>>> {
	amrex::Array4<T> arr_;
	constexpr static FluxDir indexOrder = FluxDir::X2;

	explicit Array4View(amrex::Array4<T> arr) : arr_(arr) {}

	AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto operator()(int i, int j, int k,
								 int n) const noexcept -> T
	{
		return arr_(k, i, j, n);
	}

	AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto operator()(int i, int j, int k) const noexcept
	    -> T
	{
		return arr_(k, i, j);
	}
};

// X3-flux

// if T is non-const
template <class T> struct Array4View<T, FluxDir::X3, std::enable_if_t<!std::is_const_v<T>>> {
	amrex::Array4<T> arr_;
	constexpr static FluxDir indexOrder = FluxDir::X3;

	explicit Array4View(amrex::Array4<T> arr) : arr_(arr) {}

	AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto operator()(int i, int j, int k,
								 int n) const noexcept -> T &
	{
		return arr_(j, k, i, n);
	}

	AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto operator()(int i, int j, int k) const noexcept
	    -> T &
	{
		return arr_(j, k, i);
	}
};

// if T is const
template <class T> struct Array4View<T, FluxDir::X3, std::enable_if_t<std::is_const_v<T>>> {
	amrex::Array4<T> arr_;
	constexpr static FluxDir indexOrder = FluxDir::X3;

	explicit Array4View(amrex::Array4<T> arr) : arr_(arr) {}

	AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto operator()(int i, int j, int k,
								 int n) const noexcept -> T
	{
		return arr_(j, k, i, n);
	}

	AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto operator()(int i, int j, int k) const noexcept
	    -> T
	{
		return arr_(j, k, i);
	}
};
} // namespace quokka

#endif // ARRAYVIEW_HPP_