// SPDX-FileCopyrightText: © The Quokka Authors
// SPDX-License-Identifier: MIT

/// \file Photochemistry.cpp
/// \brief Implements methods for primordial chemistry using Microphysics
///

#include "radiation/photochemistry.hpp"
#include "burn_type.H"
#include "burner.H"

namespace quokka::photochemistry
{

AMREX_GPU_DEVICE void photochem_burner(burn_t &photochemstate, const Real dt) { burner(photochemstate, dt); }
} // namespace quokka::photochemistry
