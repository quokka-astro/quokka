#ifndef INTERPOLATE2D_HPP_ // NOLINT
#define INTERPOLATE2D_HPP_
//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file Interpolate2D.cpp
/// \brief Defines methods for interpolating 2D table data using bilinear
/// interpolation.
///

#include <vector>

#include "AMReX_TableData.H"
#include "AMReX_BLassert.H"

auto find_index_in_sorted_vec(double x, std::vector<double> &v) -> long;

auto interpolate2d(double x, double y, std::vector<double> &xv,
                   std::vector<double> &yv, amrex::Table2D<double> &table)
    -> double;

#endif // INTERPOLATE2D_HPP_