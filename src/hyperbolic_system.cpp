//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file hyperbolic_system.cpp
/// \brief Implements utility functions for hyperbolic conservation laws.

#include "hyperbolic_system.hpp"

template <typename T> int sgn(T val) { return (T(0) < val) - (val < T(0)); }

auto HyperbolicSystem::minmod(double a, double b) -> double
{
	auto result = 0.0;

	if ((sgn(a) == sgn(b)) && (a != b) && (a != 0.0) && (b != 0.0)) {
		if (std::abs(a) < std::abs(b)) {
			result = a;
		} else {
			result = b;
		}
	}

	return result;
}
