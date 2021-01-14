//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file hyperbolic_system.cpp
/// \brief Implements classes and functions for use with hyperbolic systems of
/// conservation laws.

#include "hyperbolic_system.hpp"

// explicitly instantiate class templates here
extern template struct templatedArray<X1>;
extern template struct templatedArray<X2>;
extern template struct templatedArray<X3>;
