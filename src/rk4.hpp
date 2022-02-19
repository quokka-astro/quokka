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

#include <algorithm>
#include <array>
#include <cmath>

#include "AMReX.H"
#include "AMReX_BLassert.H"
#include "AMReX_Extension.H"
#include "AMReX_GpuQualifiers.H"
#include "AMReX_REAL.H"
#include "valarray.hpp"

using Real = amrex::Real;

// Cash-Karp constants
AMREX_GPU_DEVICE constexpr std::array<Real, 5> ah = {1.0 / 5.0, 0.3, 3.0 / 5.0,
                                                     1.0, 7.0 / 8.0};

AMREX_GPU_DEVICE constexpr Real b21 = 1.0 / 5.0;
AMREX_GPU_DEVICE constexpr std::array<Real, 2> b3 = {3.0 / 40.0, 9.0 / 40.0};
AMREX_GPU_DEVICE constexpr std::array<Real, 3> b4 = {0.3, -0.9, 1.2};
AMREX_GPU_DEVICE constexpr std::array<Real, 4> b5 = {-11.0 / 54.0, 2.5,
                                                     -70.0 / 27.0, 35.0 / 27.0};
AMREX_GPU_DEVICE constexpr std::array<Real, 5> b6 = {
    1631.0 / 55296.0, 175.0 / 512.0, 575.0 / 13824.0, 44275.0 / 110592.0,
    253.0 / 4096.0};

AMREX_GPU_DEVICE constexpr Real c1 = 37.0 / 378.0;
AMREX_GPU_DEVICE constexpr Real c3 = 250.0 / 621.0;
AMREX_GPU_DEVICE constexpr Real c4 = 125.0 / 594.0;
AMREX_GPU_DEVICE constexpr Real c6 = 512.0 / 1771.0;

// ec == (c - c*) gives the error estimate from the embedded method
AMREX_GPU_DEVICE constexpr std::array<Real, 7> ec = {
    0.0,
    37.0 / 378.0 - 2825.0 / 27648.0,
    0.0,
    250.0 / 621.0 - 18575.0 / 48384.0,
    125.0 / 594.0 - 13525.0 / 55296.0,
    -277.0 / 14336.0,
    512.0 / 1771.0 - 0.25};

template <typename F, int N>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
rk12_single_step(F &&rhs, Real t0, quokka::valarray<Real, N> const &y, Real dt,
                 quokka::valarray<Real, N> &ynew,
                 quokka::valarray<Real, N> &yerr, void *user_data) {
  // Compute one step of the RK Heun-Euler (1)2 method

  // stage 1 [Forward Euler step at t0]
  quokka::valarray<Real, N> k1{};
  quokka::valarray<Real, N> y_arg = y;
  int ierr = rhs(t0, y_arg, k1, user_data);
  k1 *= dt;

  // stage 2 [Forward Euler step at t1]
  quokka::valarray<Real, N> k2{};
  y_arg = y + k1;
  ierr = rhs(t0 + dt, y_arg, k2, user_data);
  k2 *= dt;

  // N.B.: equivalent to RK2-SSP
  ynew = y + 0.5 * k1 + 0.5 * k2;

  // difference between RK2-SSP and Forward Euler
  yerr = -0.5 * k1 + 0.5 * k2;
}

template <typename F, int N>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
rk23_single_step(F &&rhs, Real t0, quokka::valarray<Real, N> const &y, Real dt,
                 quokka::valarray<Real, N> &ynew,
                 quokka::valarray<Real, N> &yerr, void *user_data) {
  // Compute one step of the RK Bogaki-Shampine (2)3 method
  // https://sundials.readthedocs.io/en/latest/arkode/Butcher_link.html#bogacki-shampine-4-2-3

  // stage 1
  quokka::valarray<Real, N> k1{};
  quokka::valarray<Real, N> y_arg = y;
  int ierr = rhs(t0, y_arg, k1, user_data);
  k1 *= dt;

  // stage 2
  quokka::valarray<Real, N> k2{};
  y_arg = y + 0.5 * k1;
  ierr = rhs(t0 + 0.5 * dt, y_arg, k2, user_data);
  k2 *= dt;

  // stage 3
  quokka::valarray<Real, N> k3{};
  y_arg = y + (3. / 4.) * k2;
  ierr = rhs(t0 + (3. / 4.) * dt, y_arg, k3, user_data);
  k3 *= dt;

  // stage 4
  quokka::valarray<Real, N> k4{};
  y_arg = y + (2. / 9.) * k1 + (1. / 3.) * k2 + (4. / 9.) * k3;
  ierr = rhs(t0 + dt, y_arg, k4, user_data);
  k4 *= dt;

  ynew = y_arg; // use FSAL (first same as last) property

  yerr = (2. / 9. - 7. / 24.) * k1 + (1. / 3. - 1. / 4.) * k2 +
         (4. / 9. - 1. / 3.) * k3 - (1. / 8.) * k4;
}

template <typename F, int N>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
rk45_single_step(F &&rhs, Real t0, quokka::valarray<Real, N> const &y, Real dt,
                 quokka::valarray<Real, N> &ynew,
                 quokka::valarray<Real, N> &yerr, void *user_data) {
  // Compute one step of the RK Cash-Karp (4)5 method
  // https://sundials.readthedocs.io/en/latest/arkode/Butcher_link.html#cash-karp-6-4-5

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

  // compute 5th-order solution
  ynew = y + c1 * k1 + c3 * k3 + c4 * k4 + c6 * k6;

  // error estimate
  yerr = ec[1] * k1 + ec[2] * k2 + ec[3] * k3 + ec[4] * k4 + ec[5] * k5 +
         ec[6] * k6;
}

template <int N>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto
error_norm(quokka::valarray<Real, N> const &y0,
           quokka::valarray<Real, N> const &yerr, Real reltol,
           quokka::valarray<Real, N> const &abstol) -> Real {
  // compute a weighted rms error norm
  // https://sundials.readthedocs.io/en/latest/arkode/Mathematics_link.html#error-norms

  Real err_sq = 0;
  for (int i = 0; i < N; ++i) {
    Real w_i = 1. / (reltol * y0[i] + abstol[i]);
    err_sq += (yerr[i] * yerr[i]) * (w_i * w_i);
  }
  const Real err = std::sqrt(err_sq / N);
  return err;
}

template <typename F, int N>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
rk_adaptive_integrate(F &&rhs, Real t0, quokka::valarray<Real, N> &y0, Real t1,
                      void *user_data, Real reltol,
                      quokka::valarray<Real, N> const &abstol) {
  // Integrate dy/dt = rhs(y, t) from t0 to t1,
  // with local truncation error bounded by relative tolerance 'reltol'
  // and absolute tolerances 'abstol'.

  // initial timestep
  quokka::valarray<Real, N> ydot0{};
  rhs(t0, y0, ydot0, user_data);
  const Real dt_guess = 0.1 * std::abs(min(y0 / ydot0));

  // adaptive timestep controller
  const int maxRetries = 7;
  const int p = 2; // integration order of (high-order) method
  const Real eta_max = 20.;
  const Real eta_max_errfail_prevstep = 1.0;
  const Real eta_max_errfail_again = 0.3;
  const Real eta_min_errfail_multiple = 0.1;

  // integration loop
  const int maxSteps = 1000;
  Real time = t0;
  Real dt = dt_guess;
  quokka::valarray<Real, N> &y = y0;
  quokka::valarray<Real, N> yerr{};
  quokka::valarray<Real, N> ynew{};

  for (int i = 0; i < maxSteps; ++i) {
    if ((time + dt) > t1) {
      // reduce dt to end at t1
      dt = t1 - time;
    }

    // std::cout << "Step i = " << i << " dt = " << dt << std::endl;

    bool step_success = false;
    for (int k = 0; k < maxRetries; ++k) {
      // compute single step of chosen RK method
      rk12_single_step(rhs, time, y, dt, ynew, yerr, user_data);

      // compute error norm from embedded error estimate
      const Real epsilon = error_norm(y, yerr, reltol, abstol);

      // compute new timestep with 'I' controller
      // https://sundials.readthedocs.io/en/latest/arkode/Mathematics_link.html#i-controller
      Real eta = std::pow(epsilon, -1.0 / static_cast<Real>(p));

      if (epsilon < 1.0) { // error passed
        y = ynew;
        time += dt; // increment time
        if (k == 0) {
          eta = std::min(eta, eta_max); // limit timestep increase
        } else {
          eta = std::min(eta, eta_max_errfail_prevstep);
        }
        dt *= eta; // use new timestep
        step_success = true;
        // std::cout << "\tsuccess k = " << k << " epsilon = " << epsilon << "
        // eta = " << eta << std::endl;
        break;
      }

      // error is too large, use smaller timestep and redo
      if (k == 1) {
        eta = std::min(eta, eta_max_errfail_again);
      } else if (k > 1) {
        eta = std::clamp(eta, eta_min_errfail_multiple, eta_max_errfail_again);
      }
      dt *= eta; // use new timestep
      // std::cout << "\tfailed step k = " << k << " epsilon = " << epsilon << "
      // eta = " << eta << std::endl;
    }
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        step_success, "ODE integrator failed to reach accuracy tolerance after "
                      "maximum step-size iterations reached!");

    if (std::abs((time - t1) / t1) < 1.0e-3) {
      // we are at t1 within a reasonable tolerance, so stop
      break;
    }
  }

  AMREX_ALWAYS_ASSERT_WITH_MESSAGE(std::abs((time - t1) / t1) < 1.0e-3,
                                   "ODE integration exceeded maxSteps!");
}

#endif // RK4_HPP_