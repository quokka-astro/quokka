#ifndef RK4_HPP_ // NOLINT
#define RK4_HPP_
//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file rk4.hpp
/// \brief Implements functions for explicitly integrating ODEs with error
/// control.
///

#include <array>

#include "AMReX_Extension.H"
#include "AMReX_GpuQualifiers.H"
#include "AMReX_REAL.H"
#include "valarray.hpp"

using Real = amrex::Real;

// Cash-Karp constants
static const std::array<Real, 5> ah = {1.0 / 5.0, 0.3, 3.0 / 5.0, 1.0,
                                       7.0 / 8.0};

static const Real b21 = 1.0 / 5.0;
static const std::array<Real, 2> b3 = {3.0 / 40.0, 9.0 / 40.0};
static const std::array<Real, 3> b4 = {0.3, -0.9, 1.2};
static const std::array<Real, 4> b5 = {-11.0 / 54.0, 2.5, -70.0 / 27.0,
                                       35.0 / 27.0};
static const std::array<Real, 5> b6 = {1631.0 / 55296.0, 175.0 / 512.0,
                                       575.0 / 13824.0, 44275.0 / 110592.0,
                                       253.0 / 4096.0};

static const Real c1 = 37.0 / 378.0;
static const Real c3 = 250.0 / 621.0;
static const Real c4 = 125.0 / 594.0;
static const Real c6 = 512.0 / 1771.0;

// ec == (c - c*) gives the error estimate from the embedded method
static const std::array<Real, 7> ec = {0.0,
                                       37.0 / 378.0 - 2825.0 / 27648.0,
                                       0.0,
                                       250.0 / 621.0 - 18575.0 / 48384.0,
                                       125.0 / 594.0 - 13525.0 / 55296.0,
                                       -277.0 / 14336.0,
                                       512.0 / 1771.0 - 0.25};

template <typename F, int N>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
rk4_single_step(F &&rhs, Real t0, quokka::valarray<Real, N> &y, Real dt,
                quokka::valarray<Real, N> &yerr, void *user_data) {
  // Compute one step of the Runge-Kutta Cash-Karp method

  // Initial time t0, initial values y
  // Right-hand side given by function 'rhs' with signature:
  // 	rhs(Real t, std::array<Real, N> y_data, std::array<Real, N> y_rhs,
  // 		void *user_data);

  // stage 1
  quokka::valarray<Real, N> k1{};
  quokka::valarray<Real, N> y_arg = y;
  int ierr = rhs(t0, y_arg, k1, user_data);
  k1 *= dt;

  // stage 2
  quokka::valarray<Real, N> k2{};
  y_arg = y + b21 * k1;
  ierr = rhs(t0 + dt * ah[0], y_arg, k2, user_data);
  k2 *= dt;

  // stage 3
  quokka::valarray<Real, N> k3{};
  y_arg = y + b3[0] * k1 + b3[1] * k2;
  ierr = rhs(t0 + dt * ah[1], y_arg, k3, user_data);
  k3 *= dt;

  // stage 4
  quokka::valarray<Real, N> k4{};
  y_arg = y + b4[0] * k1 + b4[1] * k2 + b4[2] * k3;
  ierr = rhs(t0 + dt * ah[2], y_arg, k4, user_data);
  k4 *= dt;

  // stage 5
  quokka::valarray<Real, N> k5{};
  y_arg = y + b5[0] * k1 + b5[1] * k2 + b5[2] * k3 + b5[3] * k4;
  ierr = rhs(t0 + dt * ah[3], y_arg, k5, user_data);
  k5 *= dt;

  // stage 6
  quokka::valarray<Real, N> k6{};
  y_arg = y + b6[0] * k1 + b6[1] * k2 + b6[2] * k3 + b6[3] * k4 + b6[4] * k5;
  ierr = rhs(t0 + dt * ah[4], y_arg, k6, user_data);
  k6 *= dt;

  // compute 5th-order solution in-place
  y = y + c1 * k1 + c3 * k3 + c4 * k4 + c6 * k6;

  // error estimate
  yerr = ec[1] * k1 + ec[2] * k2 + ec[3] * k3 + ec[4] * k4 + ec[5] * k5 +
         ec[6] * k6;
}

#endif // RK4_HPP_