#ifndef ODEINTEGRATE_HPP_ // NOLINT
#define ODEINTEGRATE_HPP_
//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file ODEIntegrate.hpp
/// \brief Implements functions for explicitly integrating ODEs with error
/// control.
///

#include <algorithm>
#include <cmath>

#include "AMReX.H"
#include "AMReX_BLassert.H"
#include "AMReX_Extension.H"
#include "AMReX_GpuQualifiers.H"
#include "AMReX_REAL.H"

#include "valarray.hpp"

using Real = amrex::Real;

template <typename F, int N>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
forward_euler_integrate(F &&rhs, Real t0, quokka::valarray<Real, N> &y0,
                        Real t1, void *user_data) {
  // Compute multiple forward Euler steps, limiting the fractional change in y.

  quokka::valarray<Real, N> ydot{};
  quokka::valarray<Real, N> &y = y0;
  Real time = t0;
  const int maxSteps = 1000;
  const Real rel_change = 0.1; // limit to 10% change in y per substep
  bool success = false;

  for (int i = 0; i < maxSteps; ++i) {
    // evaluate rhs
    int ierr = rhs(t0, y, ydot, user_data);

    // compute dt such that dt = 0.1 (y/ydot), or dt = t1 - t.
    const Real dt = std::min(rel_change * min(abs(y / ydot)), t1 - time);

    // advance single step
    y = y + dt * ydot;
    time += dt;

    //std::cout << "[" << i << "] t = " << time << " dt = " << dt << std::endl;

    if (time >= t1) {
      success = true;
      break;
    }
  }
  AMREX_ALWAYS_ASSERT(success);
}

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
  const Real dt_guess = 0.1 * min(abs(y0 / ydot0));

  AMREX_ALWAYS_ASSERT(dt_guess > 0.);

  // adaptive timestep controller
  const int maxRetries = 7;
  const int p = 2; // integration order of (high-order) method
  const Real eta_max = 20.;
  const Real eta_max_errfail_prevstep = 1.0;
  const Real eta_max_errfail_again = 0.3;
  const Real eta_min_errfail_multiple = 0.1;

  // integration loop
  const int maxSteps = 2000;
  Real time = t0;
  Real dt = dt_guess;
  const Real hmin = 1.0e-3;
  quokka::valarray<Real, N> &y = y0;
  quokka::valarray<Real, N> yerr{};
  quokka::valarray<Real, N> ynew{};

  bool success = false;
  for (int i = 0; i < maxSteps; ++i) {
    if ((time + dt) > t1) {
      // reduce dt to end at t1
      dt = t1 - time;
    }

    // std::cout << "Step i = " << i << " t = " << time << " dt = " << dt <<
    // std::endl;

    bool step_success = false;
    for (int k = 0; k < maxRetries; ++k) {
      // check that timestep is not infinitesimal
      AMREX_ALWAYS_ASSERT(dt > (hmin * (t1 - t0)));

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
        // std::cout << "\tsuccess k = " << k << " epsilon = " << epsilon
        //            << " eta = " << eta << std::endl;
        break;
      }

      // error is too large, use smaller timestep and redo
      if (k == 1) {
        eta = std::min(eta, eta_max_errfail_again);
      } else if (k > 1) {
        eta = std::clamp(eta, eta_min_errfail_multiple, eta_max_errfail_again);
      }
      dt *= eta; // use new timestep
      // std::cout << "\tfailed step k = " << k << " epsilon = " << epsilon
      //           << " eta = " << eta << std::endl;
    }
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        step_success, "ODE integrator failed to reach accuracy tolerance after "
                      "maximum step-size iterations reached!");

    if (time >= t1) {
      // we are at t1
      success = true;
      break;
    }
  }
  AMREX_ALWAYS_ASSERT_WITH_MESSAGE(success,
                                   "ODE integration exceeded maxSteps!");
}


#endif // ODEINTEGRATE_HPP_