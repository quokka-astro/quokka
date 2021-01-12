//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file hyperbolic_system.cpp
/// \brief Implements classes and functions for use with hyperbolic systems of
/// conservation laws.

#include "hyperbolic_system.hpp"

// Convenience function to allocate stand-alone amrex::Array4 objects
void array_t::AllocateArray(int ncomp, int dim1, int dim2, int dim3)
{
	auto size = dim1 * dim2 * dim3 * ncomp;
	auto p = new double[size];
	amrex::Dim3 lower = {0, 0, 0};
	amrex::Dim3 upper = {dim1, dim2, dim3};
	arr_ = amrex::Array4<double>(p, lower, upper, ncomp);
}

// Return a shallow slice corresponding to an individual component.
// [Array4 objects can be accessed with arr(i,j,k) if there is only one component]
auto array_t::SliceArray(int ncomp) -> array_t
{
	return array_t(arr_, ncomp);
}
