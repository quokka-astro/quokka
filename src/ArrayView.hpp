#ifndef ARRAYVIEW_HPP_
#define ARRAYVIEW_HPP_
//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
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
