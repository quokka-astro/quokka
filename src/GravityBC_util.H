#ifndef GRAVITY_UTIL_H
#define GRAVITY_UTIL_H
//==============================================================================
// Poisson gravity solver, adapted from Castro's gravity module:
//   Commit history:
//   https://github.com/AMReX-Astro/Castro/commits/main/Source/gravity/Gravity_util.H
// Used under the terms of the open-source license (BSD 3-clause) given here:
//   https://github.com/AMReX-Astro/Castro/blob/main/license.txt
//==============================================================================
/// \file Gravity_util.H
/// \brief Provides helper functions used by Gravity.H and Gravity.cpp
///

#include <cmath>

#include "AMReX_Array4.H"
#include "AMReX_Extension.H"
#include "AMReX_GpuQualifiers.H"
#include "AMReX_GpuReduce.H"
#include "AMReX_REAL.H"

#include "Gravity.H"

AMREX_GPU_HOST_DEVICE AMREX_INLINE auto factorial(int n) -> amrex::Real {
  amrex::Real fact = 1.0;

  for (int i = 2; i <= n; ++i) {
    fact *= static_cast<amrex::Real>(i);
  }

  return fact;
}

AMREX_GPU_HOST_DEVICE AMREX_INLINE void
calcAssocLegPolyLM(int l, int m, amrex::Real &assocLegPolyLM,
                   amrex::Real &assocLegPolyLM1, amrex::Real &assocLegPolyLM2,
                   amrex::Real x) {
  // Calculate the associated Legendre polynomials. There are a number of
  // recurrence relations, but many are unstable. We'll use one that is known
  // to be stable for the reasonably low values of l we care about in a
  // simulation: (l-m)P_l^m(x) = x(2l-1)P_{l-1}^m(x) - (l+m-1)P_{l-2}^m(x). This
  // uses the following two expressions as initial conditions: P_m^m(x) = (-1)^m
  // (2m-1)! (1-x^2)^(m/2) P_{m+1}^m(x) = x (2m+1) P_m^m (x)

  if (l == m) {

    // P_m^m

    assocLegPolyLM = std::pow(-1, m) * std::pow((1.0 - x) * (1.0 + x), m * 0.5);

    for (int n = (2 * m - 1); n >= 3; n = n - 2) {

      assocLegPolyLM *= n;
    }

  } else if (l == m + 1) {

    // P_{m+1}^m

    assocLegPolyLM1 = assocLegPolyLM;
    assocLegPolyLM = x * (2 * m + 1) * assocLegPolyLM1;

  } else {

    assocLegPolyLM2 = assocLegPolyLM1;
    assocLegPolyLM1 = assocLegPolyLM;
    assocLegPolyLM =
        (x * (2 * l - 1) * assocLegPolyLM1 - (l + m - 1) * assocLegPolyLM2) /
        (l - m);
  }
}

AMREX_GPU_HOST_DEVICE AMREX_INLINE void
calcLegPolyL(int l, amrex::Real &legPolyL, amrex::Real &legPolyL1,
             amrex::Real &legPolyL2, amrex::Real x) {
  // Calculate the Legendre polynomials. We use a stable recurrence relation:
  // (l+1) P_{l+1}(x) = (2l+1) x P_l(x) - l P_{l-1}(x).
  // This uses initial conditions:
  // P_0(x) = 1
  // P_1(x) = x

  if (l == 0) {

    legPolyL = 1.0;

  } else if (l == 1) {

    legPolyL1 = legPolyL;
    legPolyL = x;

  } else {

    legPolyL2 = legPolyL1;
    legPolyL1 = legPolyL;
    legPolyL = ((2 * l - 1) * x * legPolyL1 - (l - 1) * legPolyL2) / l;
  }
}

AMREX_GPU_DEVICE AMREX_INLINE void multipole_add(
    amrex::Real cosTheta, amrex::Real phiAngle, amrex::Real r, amrex::Real rho,
    amrex::Real vol, amrex::Array4<amrex::Real> const &qL0,
    amrex::Array4<amrex::Real> const &qLC,
    amrex::Array4<amrex::Real> const &qLS,
    amrex::Array4<amrex::Real> const &qU0,
    amrex::Array4<amrex::Real> const &qUC,
    amrex::Array4<amrex::Real> const &qUS, int npts, int nlo, int index,
    amrex::Gpu::Handler const &handler, bool parity = false) {

  amrex::Real legPolyL = NAN;
  amrex::Real legPolyL1 = NAN;
  amrex::Real legPolyL2 = NAN;
  amrex::Real assocLegPolyLM = NAN;
  amrex::Real assocLegPolyLM1 = NAN;
  amrex::Real assocLegPolyLM2 = NAN;

  amrex::Real rho_r_L = NAN;
  amrex::Real rho_r_U = NAN;

  for (int n = nlo; n <= npts - 1; ++n) {

    for (int l = 0; l <= gravity::lnum; ++l) {

      calcLegPolyL(l, legPolyL, legPolyL1, legPolyL2, cosTheta);

      amrex::Real dQL0 = 0.0;
      amrex::Real dQU0 = 0.0;

      if (index <= n) {

        rho_r_L = rho * (std::pow(r, l));

        dQL0 = legPolyL * rho_r_L * vol * multipole::volumeFactor;
        if (parity) {
          dQL0 = dQL0 * multipole::parity_q0(l);
        }

      } else {

        rho_r_U = rho * (std::pow(r, -l - 1));

        dQU0 = legPolyL * rho_r_U * vol * multipole::volumeFactor;
        if (parity) {
          dQU0 = dQU0 * multipole::parity_q0(l);
        }
      }

      amrex::Gpu::deviceReduceSum(&qL0(l, 0, n), dQL0, handler);
      amrex::Gpu::deviceReduceSum(&qU0(l, 0, n), dQU0, handler);
    }

    // For the associated Legendre polynomial loop, we loop over m and then l.
    // It means that we have to recompute rho_r_L or rho_r_U again, but the
    // recursion relation we use for the polynomials depends on l being the
    // innermost loop index.

    for (int m = 1; m <= gravity::lnum; ++m) {
      for (int l = 1; l <= gravity::lnum; ++l) {

        if (m > l) {
          continue;
        }

        calcAssocLegPolyLM(l, m, assocLegPolyLM, assocLegPolyLM1,
                           assocLegPolyLM2, cosTheta);

        amrex::Real dQLC = 0.0;
        amrex::Real dQLS = 0.0;

        amrex::Real dQUC = 0.0;
        amrex::Real dQUS = 0.0;

        if (index <= n) {

          rho_r_L = rho * (std::pow(r, l));

          dQLC = assocLegPolyLM * std::cos(m * phiAngle) * rho_r_L * vol *
                 multipole::factArray(l, m);
          if (parity) {
            dQLC = dQLC * multipole::parity_qC_qS(l, m);
          }

          dQLS = assocLegPolyLM * std::sin(m * phiAngle) * rho_r_L * vol *
                 multipole::factArray(l, m);
          if (parity) {
            dQLS = dQLS * multipole::parity_qC_qS(l, m);
          }

        } else {

          rho_r_U = rho * (std::pow(r, -l - 1));

          dQUC = assocLegPolyLM * std::cos(m * phiAngle) * rho_r_U * vol *
                 multipole::factArray(l, m);
          if (parity) {
            dQUC = dQUC * multipole::parity_qC_qS(l, m);
          }

          dQUS = assocLegPolyLM * std::sin(m * phiAngle) * rho_r_U * vol *
                 multipole::factArray(l, m);
          if (parity) {
            dQUS = dQUS * multipole::parity_qC_qS(l, m);
          }
        }

        amrex::Gpu::deviceReduceSum(&qLC(l, m, n), dQLC, handler);
        amrex::Gpu::deviceReduceSum(&qLS(l, m, n), dQLS, handler);
        amrex::Gpu::deviceReduceSum(&qUC(l, m, n), dQUC, handler);
        amrex::Gpu::deviceReduceSum(&qUS(l, m, n), dQUS, handler);
      }
    }
  }
}

AMREX_GPU_DEVICE AMREX_INLINE void multipole_symmetric_add(
    amrex::Real x, amrex::Real y, amrex::Real z,
    const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> &problo,
    const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> &coordCenter,
    amrex::Real rho, amrex::Real vol, amrex::Array4<amrex::Real> const &qU0,
    amrex::Array4<amrex::Real> const &qUC,
    amrex::Array4<amrex::Real> const &qUS,
    amrex::Array4<amrex::Real> const &qL0,
    amrex::Array4<amrex::Real> const &qLC,
    amrex::Array4<amrex::Real> const &qLS, int npts, int nlo, int index,
    amrex::Gpu::Handler const &handler) {

  amrex::Real xLo = (2.0 * (problo[0] - coordCenter[0])) / multipole::rmax - x;

#if AMREX_SPACEDIM >= 2
  amrex::Real yLo = (2.0 * (problo[1] - coordCenter[1])) / multipole::rmax - y;
#else
  Real yLo = 0.0;
#endif

#if AMREX_SPACEDIM == 3
  amrex::Real zLo = (2.0 * (problo[2] - coordCenter[2])) / multipole::rmax - z;
#else
  Real zLo = 0.0;
#endif

  if (multipole::doSymmetricAddLo(0)) {

    amrex::Real r = std::sqrt(xLo * xLo + y * y + z * z);
    amrex::Real phiAngle = std::atan2(y, xLo);
    amrex::Real cosTheta = z / r;

    multipole_add(cosTheta, phiAngle, r, rho, vol, qL0, qLC, qLS, qU0, qUC, qUS,
                  npts, nlo, index, handler);

    if (multipole::doSymmetricAddLo(1)) {

      amrex::Real r = std::sqrt(xLo * xLo + yLo * yLo + z * z);
      amrex::Real phiAngle = std::atan2(yLo, xLo);
      amrex::Real cosTheta = z / r;

      multipole_add(cosTheta, phiAngle, r, rho, vol, qL0, qLC, qLS, qU0, qUC,
                    qUS, npts, nlo, index, handler);
    }

    if (multipole::doSymmetricAddLo(2)) {

      amrex::Real r = std::sqrt(xLo * xLo + y * y + zLo * zLo);
      amrex::Real phiAngle = std::atan2(y, xLo);
      amrex::Real cosTheta = zLo / r;

      multipole_add(cosTheta, phiAngle, r, rho, vol, qL0, qLC, qLS, qU0, qUC,
                    qUS, npts, nlo, index, handler);
    }

    if (multipole::doSymmetricAddLo(1) && multipole::doSymmetricAddLo(2)) {

      amrex::Real r = std::sqrt(xLo * xLo + yLo * yLo + zLo * zLo);
      amrex::Real phiAngle = std::atan2(yLo, xLo);
      amrex::Real cosTheta = zLo / r;

      multipole_add(cosTheta, phiAngle, r, rho, vol, qL0, qLC, qLS, qU0, qUC,
                    qUS, npts, nlo, index, handler);
    }
  }

  if (multipole::doSymmetricAddLo(1)) {

    amrex::Real r = std::sqrt(x * x + yLo * yLo + z * z);
    amrex::Real phiAngle = std::atan2(yLo, x);
    amrex::Real cosTheta = z / r;

    multipole_add(cosTheta, phiAngle, r, rho, vol, qL0, qLC, qLS, qU0, qUC, qUS,
                  npts, nlo, index, handler);

    if (multipole::doSymmetricAddLo(2)) {

      amrex::Real r = std::sqrt(x * x + yLo * yLo + zLo * zLo);
      amrex::Real phiAngle = std::atan2(yLo, x);
      amrex::Real cosTheta = zLo / r;

      multipole_add(cosTheta, phiAngle, r, rho, vol, qL0, qLC, qLS, qU0, qUC,
                    qUS, npts, nlo, index, handler);
    }
  }

  if (multipole::doSymmetricAddLo(2)) {

    amrex::Real r = std::sqrt(x * x + y * y + zLo * zLo);
    amrex::Real phiAngle = std::atan2(y, x);
    amrex::Real cosTheta = zLo / r;

    multipole_add(cosTheta, phiAngle, r, rho, vol, qL0, qLC, qLS, qU0, qUC, qUS,
                  npts, nlo, index, handler);
  }
}

#endif
