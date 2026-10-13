// SPDX-FileCopyrightText: © The Quokka Authors
// SPDX-License-Identifier: MIT

#ifndef ARRAYVIEW_HPP_
#define ARRAYVIEW_HPP_
/// \file ArrayView.hpp
/// \brief A container for an array of Reals with template magic to permute indices

#include <AMReX.H>

#if AMREX_SPACEDIM == 1
#include "ArrayView_3d.hpp" // same as 3D
#endif

#if AMREX_SPACEDIM == 2
#include "ArrayView_2d.hpp"
#endif

#if AMREX_SPACEDIM == 3
#include "ArrayView_3d.hpp"
#endif

#endif // ARRAYVIEW_HPP_
