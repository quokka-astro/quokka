//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file testStromgrenSphereTempIndependentRecombination.cpp
/// \brief The static Stromgren sphere problem of testStromgrenSphere.cpp, built as its own executable for
/// inputs/StromgrenSphereTempIndependentRecombination.toml.
///

#include "microphysics/microphysics_traits.hpp"

struct StromgrenSphereTempIndependentRecombination {};
using StromgrenSphereProblem = StromgrenSphereTempIndependentRecombination; // StromgrenSphere.hpp is written against this alias

template <> struct Microphysics_Traits<StromgrenSphereTempIndependentRecombination> : DefaultMicrophysicsTraits {
	// ROS2S is too slow for this problem
	using Integrator = quokka::microphysics::Rosenbrock<quokka::microphysics::Rodas4P>;
	static constexpr bool photochemistry = true;	  // built with PHOTOCHEMISTRY
	static constexpr bool integrate_rad_flux = false; // built with SKIP_PHOTOCHEMFLUX
};

#include "StromgrenSphere.hpp"
