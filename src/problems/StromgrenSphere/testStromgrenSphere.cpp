//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file testStromgrenSphere.cpp
/// \brief Defines a test problem for the static Stromgren sphere with no temperature dependence.
///

struct StromgrenSphere {};
using StromgrenSphereProblem = StromgrenSphere; // StromgrenSphere.hpp is written against this alias

#include "StromgrenSphere.hpp"
