#ifndef VALARRAY_HPP_
#define VALARRAY_HPP_
//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file valarray.hpp
/// \brief A container for a vector with addition, multiplication with expression templates
/// (This is necessary because std::valarray is not defined in CUDA C++!)

// library headers
#include <AMReX_GpuQualifiers.H>

namespace quokka
{
template <typename T, int d> class valarray
{
      public:
	AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE valarray() {}
	AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE valarray(std::initializer_list<T> list)
	{
		AMREX_ASSERT(list.size() == d);
		T const *input =
		    std::data(list); // requires nvcc to be in C++17 mode! (if it fails, the
				     // compiler flags are wrong, probably due to a CMake issue.)
		for (size_t i = 0; i < list.size(); ++i) {
			values[i] = input[i];
		}
	}

	AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE T &operator[](size_t i) { return values[i]; }

	AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE T operator[](size_t i) const { return values[i]; }

	AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE size_t size() const { return d; }

      private:
	T values[d];
};
} // namespace quokka

template <typename T, int d>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE quokka::valarray<T, d>
operator+(quokka::valarray<T, d> const &a, quokka::valarray<T, d> const &b)
{
	quokka::valarray<T, d> sum;
	for (size_t i = 0; i < a.size(); ++i) {
		sum[i] = a[i] + b[i];
	}
	return sum;
}

template <typename T, int d>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE quokka::valarray<T, d>
operator-(quokka::valarray<T, d> const &a, quokka::valarray<T, d> const &b)
{
	quokka::valarray<T, d> diff;
	for (size_t i = 0; i < a.size(); ++i) {
		diff[i] = a[i] - b[i];
	}
	return diff;
}

template <typename T, int d>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE quokka::valarray<T, d>
operator*(quokka::valarray<T, d> const &a, quokka::valarray<T, d> const &b)
{
	quokka::valarray<T, d> prod;
	for (size_t i = 0; i < a.size(); ++i) {
		prod[i] = a[i] * b[i];
	}
	return prod;
}

template <typename T, int d>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE quokka::valarray<T, d>
operator*(T const &scalar, quokka::valarray<T, d> const &v)
{
	quokka::valarray<T, d> scalarprod;
	for (size_t i = 0; i < v.size(); ++i) {
		scalarprod[i] = scalar * v[i];
	}
	return scalarprod;
}

template <typename T, int d>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE quokka::valarray<T, d>
operator/(quokka::valarray<T, d> const &v, T const &scalar)
{
	quokka::valarray<T, d> scalardiv;
	for (size_t i = 0; i < v.size(); ++i) {
		scalardiv[i] = v[i] / scalar;
	}
	return scalardiv;
}

#endif // VALARRAY_HPP_