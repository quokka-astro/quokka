#ifndef MATH_IMPL_HPP_ // NOLINT
#define MATH_IMPL_HPP_
//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file math_impl.hpp
/// \brief Implements functions for various math operations on GPU not supported by CUDA C++
/// standard library.

#include "AMReX_GpuQualifiers.H"
#include "AMReX_Extension.H"

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto clamp(double v, double lo, double hi) -> double
{
	return (v < lo) ? lo : (hi < v) ? hi : v;
}

#endif // MATH_IMPL_HPP_
