//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file testStromgrenSphere.cpp
/// \brief Defines a test problem for the static Stromgren sphere with no temperature dependence.
///

#include "microphysics/microphysics_traits.hpp"
#include "networks/photoionization_H_caseB_handwritten/network.hpp"

struct StromgrenSphere {};
using StromgrenSphereProblem = StromgrenSphere; // StromgrenSphere.hpp is written against this alias

template <> struct Microphysics_Traits<StromgrenSphere> : DefaultMicrophysicsTraits {
	using Network = quokka::networks::PhotoionizationHCaseB;
	using Integrator = quokka::microphysics::Rosenbrock<quokka::microphysics::ROS2S>;
	static constexpr bool photochemistry = true;
	static constexpr bool integrate_rad_flux = false;
};

#include "StromgrenSphere.hpp"
