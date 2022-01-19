//==============================================================================
// Poisson gravity solver, adapted from Castro's gravity module:
//   Commit history:
//   https://github.com/AMReX-Astro/Castro/commits/main/Source/gravity/Gravity.cpp
// Used under the terms of the open-source license (BSD 3-clause) given here:
//   https://github.com/AMReX-Astro/Castro/blob/main/license.txt
//==============================================================================
/// \file GravityBC.hpp
/// \brief Implements a class for solving the Poisson equation.
///

#include "Gravity.hpp"
#include "GravityBC_util.hpp"

using Real = amrex::Real;

template <typename T> void Gravity<T>::init_multipole_grav() {
  if (gravity::lnum < 0) {
    amrex::Abort("lnum negative");
  }

  if (gravity::lnum > multipole::lnum_max) {
    amrex::Abort("lnum greater than lnum_max");
  }

  const auto problo = sim->Geom(0).ProbLoArray();
  const auto probhi = sim->Geom(0).ProbHiArray();

  // Compute pre-factors now to save computation time, for qC and qS

  for (int l = 0; l <= multipole::lnum_max; ++l) {
    multipole::parity_q0(l) = 1.0;
    for (int m = 0; m <= multipole::lnum_max; ++m) {
      multipole::factArray(l, m) = 0.0;
      multipole::parity_qC_qS(l, m) = 1.0;
    }
  }

  for (int l = 0; l <= multipole::lnum_max; ++l) {

    // The odd l Legendre polynomials are odd in their argument, so
    // a symmetric reflection about the z axis leads to a total cancellation.

    multipole::parity_q0(l) = 1.0;

    if (l % 2 != 0) {
      if (multipole::doReflectionLo(2) || multipole::doReflectionHi(2)) {
        multipole::parity_q0(l) = 0.0;
      }
    }

    for (int m = 1; m <= l; ++m) {

      // The parity properties of the associated Legendre polynomials are:
      // P_l^m (-x) = (-1)^(l+m) P_l^m (x)
      // Therefore, a complete cancellation occurs if l+m is odd and
      // we are reflecting about the z axis.

      // Additionally, the cosine and sine terms flip sign when reflected
      // about the x or y axis, so if we have a reflection about x or y
      // then the terms have a complete cancellation.

      multipole::parity_qC_qS(l, m) = 1.0;

      if ((l + m) % 2 != 0 &&
          (multipole::doReflectionLo(2) || multipole::doReflectionHi(2))) {
        multipole::parity_qC_qS(l, m) = 0.0;
      }

      if (multipole::doReflectionLo(0) || multipole::doReflectionLo(1) ||
          multipole::doReflectionHi(0) || multipole::doReflectionHi(1)) {
        multipole::parity_qC_qS(l, m) = 0.0;
      }

      multipole::factArray(l, m) =
          2.0 * factorial(l - m) / factorial(l + m) * multipole::volumeFactor;
    }
  }

  // Normalize all distances to the maximum possible physical distance from the
  // center, which is the diagonal from the center to the edge of the box. Then
  // r^l will always be less than or equal to one.

  Real maxWidth = probhi[0] - problo[0];
  maxWidth = amrex::max(maxWidth, probhi[1] - problo[1]);
  maxWidth = amrex::max(maxWidth, probhi[2] - problo[2]);

  multipole::rmax =
      0.5 * maxWidth * std::sqrt(static_cast<Real>(AMREX_SPACEDIM));
}

template <typename T>
void Gravity<T>::fill_multipole_BCs(int crse_level, int fine_level,
                                    const amrex::Vector<amrex::MultiFab *> &Rhs,
                                    amrex::MultiFab &phi) {
  BL_PROFILE("Gravity<T>::fill_multipole_BCs()");

  // Multipole BCs only make sense to construct if we are starting from the
  // coarse level.
  BL_ASSERT(crse_level == 0);
  BL_ASSERT(gravity::lnum >= 0);

  const Real strt = amrex::ParallelDescriptor::second();
  const int npts = numpts[crse_level];

  // Storage arrays for the multipole moments.
  // We will initialize them to zero, and then
  // sum up the results over grids.
  // Note that since Boxes are defined with
  // AMREX_SPACEDIM dimensions, we cannot presently
  // use this array to fill the interior of the
  // domain in 2D, since we can only have one
  // radial index for calculating the multipole moments.

  amrex::Box boxq0(amrex::IntVect(D_DECL(0, 0, 0)),
                   amrex::IntVect(D_DECL(gravity::lnum, 0, npts - 1)));
  amrex::Box boxqC(
      amrex::IntVect(D_DECL(0, 0, 0)),
      amrex::IntVect(D_DECL(gravity::lnum, gravity::lnum, npts - 1)));
  amrex::Box boxqS(
      amrex::IntVect(D_DECL(0, 0, 0)),
      amrex::IntVect(D_DECL(gravity::lnum, gravity::lnum, npts - 1)));

  amrex::FArrayBox qL0(boxq0);
  amrex::FArrayBox qLC(boxqC);
  amrex::FArrayBox qLS(boxqS);

  amrex::FArrayBox qU0(boxq0);
  amrex::FArrayBox qUC(boxqC);
  amrex::FArrayBox qUS(boxqS);

  qL0.setVal<amrex::RunOn::Device>(0.0);
  qLC.setVal<amrex::RunOn::Device>(0.0);
  qLS.setVal<amrex::RunOn::Device>(0.0);
  qU0.setVal<amrex::RunOn::Device>(0.0);
  qUC.setVal<amrex::RunOn::Device>(0.0);
  qUS.setVal<amrex::RunOn::Device>(0.0);

  // Use all available data in constructing the boundary conditions,
  // unless the user has indicated that a maximum level at which
  // to stop using the more accurate data.

  for (int lev = crse_level; lev <= fine_level; ++lev) {

    // Create a local copy of the RHS so that we can mask it.

    amrex::MultiFab source(Rhs[lev - crse_level]->boxArray(),
                           Rhs[lev - crse_level]->DistributionMap(), 1, 0);

    amrex::MultiFab::Copy(source, *Rhs[lev - crse_level], 0, 0, 1, 0);

    if (lev < fine_level) {
      auto mask =
          amrex::makeFineMask(sim->boxArray(lev), sim->DistributionMap(lev),
                              sim->boxArray(lev + 1), sim->refRatio(lev),
                              1.0,  // coarse
                              0.0); // fine
      amrex::MultiFab::Multiply(source, mask, 0, 0, 1, 0);
    }

    // Loop through the grids and compute the individual contributions
    // to the various moments. The multipole moment constructor
    // is coded to only add to the moment arrays, so it is safe
    // to directly hand the arrays to them.

    const auto dx = sim->Geom(lev).CellSizeArray();
    const auto problo = sim->Geom(lev).ProbLoArray();
    const auto probhi = sim->Geom(lev).ProbHiArray();
    const auto coord_center = coordCenter;

    for (amrex::MFIter mfi(source); mfi.isValid(); ++mfi) {
      const amrex::Box &bx = mfi.tilebox();

      auto qL0_arr = qL0.array();
      auto qLC_arr = qLC.array();
      auto qLS_arr = qLS.array();
      auto qU0_arr = qU0.array();
      auto qUC_arr = qUC.array();
      auto qUS_arr = qUS.array();

      auto rho = source[mfi].array();

      // assume Cartesian coordinates
      auto vol = AMREX_D_TERM(dx[0], *dx[1], *dx[2]);

      amrex::ParallelFor(
          amrex::Gpu::KernelInfo().setReduction(true), bx,
          [=] AMREX_GPU_DEVICE(int i, int j, int k,
                               amrex::Gpu::Handler const &handler) {
            // If we're using this to construct boundary values, then only
            // fill the outermost bin.

            int nlo = npts - 1;

            // Note that we don't currently support dx != dy != dz, so this is
            // acceptable.

            Real drInv = multipole::rmax / dx[0];

            Real rmax_cubed_inv =
                1.0 / (multipole::rmax * multipole::rmax * multipole::rmax);

            Real x = (problo[0] + (static_cast<Real>(i) + 0.5) * dx[0] -
                      coord_center[0]) /
                     multipole::rmax;

            Real y = (problo[1] + (static_cast<Real>(j) + 0.5) * dx[1] -
                      coord_center[1]) /
                     multipole::rmax;

            Real z = (problo[2] + (static_cast<Real>(k) + 0.5) * dx[2] -
                      coord_center[2]) /
                     multipole::rmax;

            Real r = std::sqrt(x * x + y * y + z * z);

            Real cosTheta = NAN;
            Real phiAngle = NAN;
            int index = 0;

            index = static_cast<int>(r * drInv);
            cosTheta = z / r;
            phiAngle = std::atan2(y, x);

            // Now, compute the multipole moments.

            multipole_add(cosTheta, phiAngle, r, rho(i, j, k),
                          vol * rmax_cubed_inv, qL0_arr, qLC_arr, qLS_arr,
                          qU0_arr, qUC_arr, qUS_arr, npts, nlo, index, handler,
                          true);
          });
    }

  } // end loop over levels

  // Now, do a global reduce over all processes.

  if (!amrex::ParallelDescriptor::UseGpuAwareMpi()) {
    qL0.prefetchToHost();
    qLC.prefetchToHost();
    qLS.prefetchToHost();
  }

  amrex::ParallelDescriptor::ReduceRealSum(qL0.dataPtr(),
                                           static_cast<int>(boxq0.numPts()));
  amrex::ParallelDescriptor::ReduceRealSum(qLC.dataPtr(),
                                           static_cast<int>(boxqC.numPts()));
  amrex::ParallelDescriptor::ReduceRealSum(qLS.dataPtr(),
                                           static_cast<int>(boxqS.numPts()));

  if (!amrex::ParallelDescriptor::UseGpuAwareMpi()) {
    qL0.prefetchToDevice();
    qLC.prefetchToDevice();
    qLS.prefetchToDevice();
  }

  // Finally, construct the boundary conditions using the
  // complete multipole moments, for all points on the
  // boundary that are held on this process.

  const amrex::Box &domain = sim->Geom(crse_level).Domain();
  const auto dx = sim->Geom(crse_level).CellSizeArray();
  const auto problo = sim->Geom(crse_level).ProbLoArray();
  const auto coord_center = coordCenter;

  for (amrex::MFIter mfi(phi); mfi.isValid(); ++mfi) {
    const amrex::Box &bx = mfi.growntilebox();

    auto qL0_arr = qL0.array();
    auto qLC_arr = qLC.array();
    auto qLS_arr = qLS.array();
    auto phi_arr = phi[mfi].array();

    amrex::ParallelFor(bx, [=] AMREX_GPU_HOST_DEVICE(int i, int j, int k) {
      const int *domlo = domain.loVect();
      const int *domhi = domain.hiVect();

      // If we're using this to construct boundary values, then only use
      // the outermost bin.

      int nlo = npts - 1;

      Real rmax_cubed = multipole::rmax * multipole::rmax * multipole::rmax;

      Real x = NAN;
      if (i > domhi[0]) {
        x = problo[0] + (static_cast<Real>(i)) * dx[0] - coord_center[0];
      } else if (i < domlo[0]) {
        x = problo[0] + (static_cast<Real>(i + 1)) * dx[0] - coord_center[0];
      } else {
        x = problo[0] + (static_cast<Real>(i) + 0.5) * dx[0] - coord_center[0];
      }

      x = x / multipole::rmax;

      Real y = NAN;
      if (j > domhi[1]) {
        y = problo[1] + (static_cast<Real>(j)) * dx[1] - coord_center[1];
      } else if (j < domlo[1]) {
        y = problo[1] + (static_cast<Real>(j + 1)) * dx[1] - coord_center[1];
      } else {
        y = problo[1] + (static_cast<Real>(j) + 0.5) * dx[1] - coord_center[1];
      }

      y = y / multipole::rmax;

      Real z = NAN;
      if (k > domhi[2]) {
        z = problo[2] + (static_cast<Real>(k)) * dx[2] - coord_center[2];
      } else if (k < domlo[2]) {
        z = problo[2] + (static_cast<Real>(k + 1)) * dx[2] - coord_center[2];
      } else {
        z = problo[2] + (static_cast<Real>(k) + 0.5) * dx[2] - coord_center[2];
      }

      z = z / multipole::rmax;

      // Only adjust ghost zones here

      if (i < domlo[0] || i > domhi[0] || j < domlo[1] || j > domhi[1] ||
          k < domlo[2] || k > domhi[2]) {

        // There are some cases where r == 0. This might occur, for example,
        // when we have symmetric BCs and our corner is at one edge.
        // In this case, we'll set phi to zero for safety, to avoid NaN issues.
        // These cells should not be accessed anyway during the gravity solve.

        Real r = std::sqrt(x * x + y * y + z * z);

        if (r < 1.0e-12) {
          phi_arr(i, j, k) = 0.0;
          return;
        }

        Real cosTheta = NAN;
        Real phiAngle = NAN;

        // assume 3D cartesian coordinates
        cosTheta = z / r;
        phiAngle = std::atan2(y, x);

        phi_arr(i, j, k) = 0.0;

        // Compute the potentials on the ghost cells.

        Real legPolyL = NAN;
        Real legPolyL1 = NAN;
        Real legPolyL2 = NAN;
        Real assocLegPolyLM = NAN;
        Real assocLegPolyLM1 = NAN;
        Real assocLegPolyLM2 = NAN;

        for (int n = nlo; n <= npts - 1; ++n) {

          for (int l = 0; l <= gravity::lnum; ++l) {

            calcLegPolyL(l, legPolyL, legPolyL1, legPolyL2, cosTheta);

            Real r_U = std::pow(r, -l - 1);

            // Make sure we undo the volume scaling here.

            phi_arr(i, j, k) += qL0_arr(l, 0, n) * legPolyL * r_U * rmax_cubed;
          }

          for (int m = 1; m <= gravity::lnum; ++m) {
            for (int l = 1; l <= gravity::lnum; ++l) {

              if (m > l) {
                continue;
              }

              calcAssocLegPolyLM(l, m, assocLegPolyLM, assocLegPolyLM1,
                                 assocLegPolyLM2, cosTheta);

              Real r_U = std::pow(r, -l - 1);

              // Make sure we undo the volume scaling here.

              phi_arr(i, j, k) += (qLC_arr(l, m, n) * std::cos(m * phiAngle) +
                                   qLS_arr(l, m, n) * std::sin(m * phiAngle)) *
                                  assocLegPolyLM * r_U * rmax_cubed;
            }
          }
        }

        phi_arr(i, j, k) = -C::Gconst * phi_arr(i, j, k) / multipole::rmax;
      }
    });
  }

  if (gravity::verbose != 0) {
    const int IOProc = amrex::ParallelDescriptor::IOProcessorNumber();
    Real end = amrex::ParallelDescriptor::second() - strt;
    amrex::ParallelDescriptor::ReduceRealMax(end, IOProc);
    if (amrex::ParallelDescriptor::IOProcessor()) {
      std::cout << "Gravity<T>::fill_multipole_BCs() time = " << end
                << std::endl
                << std::endl;
    }
  }
}

template <typename T> void Gravity<T>::make_mg_bc() {
  const amrex::Geometry &geom = this->sim->Geom(0);

  for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
    if (geom.isPeriodic(idim)) {
      mlmg_lobc[idim] = amrex::MLLinOp::BCType::Periodic;
      mlmg_hibc[idim] = amrex::MLLinOp::BCType::Periodic;
    } else {
      if (phys_bc->lo(idim) == Symmetry) {
        amrex::Abort(
            "Reflecting boundaries not supported for Poisson gravity!");
      } else {
        mlmg_lobc[idim] = amrex::MLLinOp::BCType::Dirichlet;
      }
      if (phys_bc->hi(idim) == Symmetry) {
        amrex::Abort(
            "Reflecting boundaries not supported for Poisson gravity!");
      } else {
        mlmg_hibc[idim] = amrex::MLLinOp::BCType::Dirichlet;
      }
    }
  }
}
