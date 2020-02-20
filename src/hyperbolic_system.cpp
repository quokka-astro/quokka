//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file hyperbolic_system.cpp
/// \brief Implements classes and functions for use with hyperbolic systems of
/// conservation laws.

#include "hyperbolic_system.hpp"

auto HyperbolicSystem::time() -> double { return time_; }

auto HyperbolicSystem::nx() -> int { return nx_; }

auto HyperbolicSystem::nghost() -> int { return nghost_; }

void HyperbolicSystem::set_cflNumber(double cflNumber)
{
	assert((cflNumber > 0.0) && (cflNumber <= 1.0)); // NOLINT
	cflNumber_ = cflNumber;
}
