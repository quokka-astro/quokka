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

#include "AMReX_BLassert.H"
#include "AMReX_TableData.H"

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto
interpolate2d(double x, double y, amrex::Table1D<const double> const &xv,
              amrex::Table1D<const double> const &yv,
              amrex::Table2D<const double> const &table) -> double {
  // return the bilinearly-interpolated value in table
  // NOTE: table must be uniformly spaced in xv and yv

  double dx = (xv(xv.end - 1) - xv(xv.begin)) / (xv.end - xv.begin);
  double dy = (yv(yv.end - 1) - yv(yv.begin)) / (yv.end - yv.begin);

  x = std::clamp(x, xv(xv.begin), xv(xv.end - 1));
  y = std::clamp(y, yv(yv.begin), yv(yv.end - 1));

  // compute indices
  int ix = static_cast<int>((x - xv(xv.begin)) / dx);
  int iy = static_cast<int>((y - yv(yv.begin)) / dy);
  int iix = (ix == xv.end - 1) ? ix : ix + 1;
  int iiy = (iy == yv.end - 1) ? iy : iy + 1;

  // get values
  double x1 = xv(ix);
  double x2 = xv(iix);
  double y1 = yv(iy);
  double y2 = yv(iiy);

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

#endif // INTERPOLATE2D_HPP_