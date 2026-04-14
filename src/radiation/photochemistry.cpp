//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file Photochemistry.cpp
/// \brief Implements methods for primordial chemistry using Microphysics
///

#include "radiation/photochemistry.hpp"
#include "burn_type.H"
#include "burner.H"

namespace quokka::photochemistry
{

AMREX_GPU_DEVICE void photochem_burner(burn_t &photochemstate, const Real dt)
{
	burner(photochemstate, dt);
}
} // namespace quokka::photochemistry
