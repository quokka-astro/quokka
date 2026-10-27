//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file testStromgrenSphere.cpp
/// \brief Defines a test problem for the static Stromgren sphere with no temperature dependence.
///

#include "microphysics/microphysics_traits.hpp"

struct StromgrenSphere {};
using StromgrenSphereProblem = StromgrenSphere; // StromgrenSphere.hpp is written against this alias

template <> struct Microphysics_Traits<StromgrenSphere> : DefaultMicrophysicsTraits {
	using Integrator = quokka::microphysics::Rosenbrock<quokka::microphysics::ROS2S>;
	static constexpr bool photochemistry = true;	  // built with PHOTOCHEMISTRY
	static constexpr bool integrate_rad_flux = false; // built with SKIP_PHOTOCHEMFLUX
};

#include "StromgrenSphere.hpp"
