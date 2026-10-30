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
#include "networks/photoionization_H_caseB_handwritten/network.hpp"

struct StromgrenSphereTempIndependentRecombination {};
using StromgrenSphereProblem = StromgrenSphereTempIndependentRecombination; // StromgrenSphere.hpp is written against this alias

template <> struct Microphysics_Traits<StromgrenSphereTempIndependentRecombination> : DefaultMicrophysicsTraits {
	using Network = quokka::networks::PhotoionizationHCaseB;
	// ROS2S is too slow for this problem
	using Integrator = quokka::microphysics::Rosenbrock<quokka::microphysics::Rodas4P>;
	static constexpr bool photochemistry = true;
	static constexpr bool integrate_rad_flux = false;
};

#include "StromgrenSphere.hpp"
