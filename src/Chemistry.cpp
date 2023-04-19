//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file Chemistry.cpp
/// \brief Implements methods for primordial chemistry using Microphysics
///

#include "Chemistry.hpp"
#include "burn_type.H"
#include "burner.H"

namespace quokka::chemistry
{

void chemburner(burn_t chemstate, const Real dt) { return burner(chemstate, dt); }

} // namespace quokka::chemistry
