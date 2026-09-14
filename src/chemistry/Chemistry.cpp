// SPDX-FileCopyrightText: © The Quokka Authors
// SPDX-License-Identifier: MIT

/// \file Chemistry.cpp
/// \brief Implements methods for primordial chemistry using Microphysics
///

#include "chemistry/Chemistry.hpp"
#include "burn_type.H"
#include "burner.H"

namespace quokka::chemistry
{

AMREX_GPU_DEVICE void chemburner(burn_t &chemstate, const Real dt) { burner(chemstate, dt); }

} // namespace quokka::chemistry
