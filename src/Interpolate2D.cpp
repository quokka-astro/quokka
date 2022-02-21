//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file Interpolate2D.cpp
/// \brief Implements methods for interpolating 2D table data using bilinear
/// interpolation.
///

#include "Interpolate2D.hpp"

auto find_index_in_sorted_vec(double x, std::vector<double> &v) -> long {
  // find the largest index whose value is less than or equal to 'x' in the
  // sorted vector 'vec'

  auto before = std::lower_bound(v.begin(), v.end(), x);

  if ((before != v.begin()) && (before != v.end())) {
    --before;
  }

  // return index
  return std::distance(v.begin(), before);
}

auto interpolate2d(double x, double y, std::vector<double> &xv,
                   std::vector<double> &yv, amrex::Table2D<double> &table)
    -> double {
  // return the bilinearly-interpolated value in table

  // compute indices
  int ix = static_cast<int>(find_index_in_sorted_vec(x, xv));
  int iy = static_cast<int>(find_index_in_sorted_vec(y, yv));
  int iix = (ix == xv.size() - 1) ? ix : ix + 1;
  int iiy = (iy == yv.size() - 1) ? iy : iy + 1;

  AMREX_ASSERT(ix <= table.end[0]);
  AMREX_ASSERT(iix <= table.end[0]);
  AMREX_ASSERT(iy <= table.end[1]);
  AMREX_ASSERT(iiy <= table.end[1]);

  // get values
  double x1 = xv[ix];
  double x2 = xv[iix];
  double y1 = yv[iy];
  double y2 = yv[iiy];

  // compute weights
  double volinv = 1.0 / ((x2 - x1) * (y2 - y1));
  double w11 = (x2 - x) * (y2 - y) * volinv;
  double w12 = (x2 - x) * (y - y1) * volinv;
  double w21 = (x - x1) * (y2 - y) * volinv;
  double w22 = (x - x1) * (y - y1) * volinv;

  double value = w11 * table(ix, iy) + w12 * table(ix, iiy) +
                 w21 * table(iix, iy) + w22 * table(iix, iiy);
  return value;
}
