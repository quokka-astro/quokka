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

AMREX_GPU_DEVICE void photochem_burner(burn_t &photochemstate, const Real dt) { 
    // std::cout << "Entering photochem_burner with dt: " << dt << std::endl;
    // std::cout << "photochem_burner()--> state.xn[1]: " << photochemstate.xn[1] << std::endl;
    burner(photochemstate, dt); 
    // std::cout << "photochem_burner()--> state.xn[1] after burn: " << photochemstate.xn[1] << std::endl;
}
} // namespace quokka::photochemistry
