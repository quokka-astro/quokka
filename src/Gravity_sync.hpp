//==============================================================================
// Poisson gravity solver, adapted from Castro's gravity module.
//   Note that Quokka's modified version only handles Poisson gravity in 3D
//   Cartesian geometry, which makes the code significantly simpler. There are
//   also significant style changes to modernize to C++17. Also, we do not use
//   amrex::AmrLevel here since Quokka uses amrex::AmrCore instead.
//
// Commit history:
//   https://github.com/AMReX-Astro/Castro/commits/main/Source/gravity/Gravity.cpp
//
// Used under the terms of the open-source license (BSD 3-clause) given here:
//   https://github.com/AMReX-Astro/Castro/blob/main/license.txt
//==============================================================================
/// \file Gravity_sync.hpp
/// \brief Implements methods to do the elliptic synchronization solve for AMR
/// subcycling-in-time Poisson solves.
///

#include "Gravity.hpp"

template <typename T>
void Gravity<T>::gravity_sync(int crse_level, int fine_level,
                              const Vector<MultiFab *> &drho,
                              const Vector<MultiFab *> &dphi) {
  BL_PROFILE("Gravity<T>::gravity_sync()");

  // There is no need to do a synchronization if
  // we didn't solve on the fine levels.

  if (fine_level > gravity::max_solve_level) {
    return;
  }
  fine_level = amrex::min(fine_level, gravity::max_solve_level);

  AMREX_ASSERT(sim->finestLevel() > crse_level);
  if (gravity::verbose > 1) {
    amrex::Print() << " ... gravity_sync at crse_level " << crse_level << '\n';
    amrex::Print() << " ...     up to finest_level     " << fine_level << '\n';
  }

  const Geometry &crse_geom = sim->Geom(crse_level);
  const Box &crse_domain = crse_geom.Domain();

  int nlevs = fine_level - crse_level + 1;

  // Construct delta(phi) and delta(grad_phi). delta(phi)
  // needs a ghost zone for holding the boundary condition
  // in the same way that phi does.

  Vector<std::unique_ptr<MultiFab>> delta_phi(nlevs);

  for (int lev = crse_level; lev <= fine_level; ++lev) {
    delta_phi[lev - crse_level] = std::make_unique<MultiFab>(
        sim->boxArray(lev), sim->DistributionMap(lev), 1, 1);
    delta_phi[lev - crse_level]->setVal(0.0);
  }

  Vector<Vector<std::unique_ptr<MultiFab>>> ec_gdPhi(nlevs);

  for (int lev = crse_level; lev <= fine_level; ++lev) {
    ec_gdPhi[lev - crse_level].resize(AMREX_SPACEDIM);

    const DistributionMapping &dm = sim->DistributionMap(lev);
    for (int n = 0; n < AMREX_SPACEDIM; ++n) {
      ec_gdPhi[lev - crse_level][n] = std::make_unique<MultiFab>(
          amrex::convert(sim->boxArray(lev), IntVect::TheDimensionVector(n)),
          dm, 1, 0);
      ec_gdPhi[lev - crse_level][n]->setVal(0.0);
    }
  }

  // Construct a container for the right-hand-side (4 * pi * G * drho + dphi).
  // dphi appears in the construction of the boundary conditions because it
  // indirectly represents a change in mass on the domain (the mass motion that
  // occurs on the fine grid, whose gravitational effects are now indirectly
  // being propagated to the coarse grid).

  // We will temporarily leave the RHS divided by (4 * pi * G) because that
  // is the form expected by the boundary condition routine.

  Vector<std::unique_ptr<MultiFab>> rhs(nlevs);

  for (int lev = crse_level; lev <= fine_level; ++lev) {
    rhs[lev - crse_level] = std::make_unique<MultiFab>(
        sim->boxArray(lev), sim->DistributionMap(lev), 1, 0);
    MultiFab::Copy(*rhs[lev - crse_level], *dphi[lev - crse_level], 0, 0, 1, 0);
    rhs[lev - crse_level]->mult(1.0 / Ggravity);
    MultiFab::Add(*rhs[lev - crse_level], *drho[lev - crse_level], 0, 0, 1, 0);
  }

  // Construct the boundary conditions for the Poisson solve.

  if (crse_level == 0 && !crse_geom.isAllPeriodic()) {

    if (gravity::verbose > 1) {
      amrex::Print() << " ... Making bc's for delta_phi at crse_level 0"
                     << std::endl;
    }

    fill_multipole_BCs(crse_level, fine_level, amrex::GetVecOfPtrs(rhs),
                       *delta_phi[crse_level]);
  }

  // Restore the factor of (4 * pi * G) for the Poisson solve.
  for (int lev = crse_level; lev <= fine_level; ++lev) {
    rhs[lev - crse_level]->mult(Ggravity);
  }

  // In the all-periodic case we enforce that the RHS sums to zero.
  // We only do this if we're periodic and the coarse level covers the whole
  // domain. In principle this could be true for level > 0, so we'll test on
  // whether the number of points on the level is equal to the number of points
  // possible on the level. Note that since we did the average-down, we can
  // stick with the data on the coarse level since the averaging down is
  // conservative.

  if (crse_geom.isAllPeriodic() &&
      (sim->boxArray(crse_level).numPts() == crse_domain.numPts())) {

    // We assume that if we're fully periodic then we're going to be in
    // Cartesian coordinates, so to get the average value of the RHS we can
    // divide the sum of the RHS by the number of points.

    Real local_correction = rhs[0]->sum() / sim->boxArray(crse_level).numPts();

    if (gravity::verbose > 1) {
      amrex::Print() << "WARNING: Adjusting RHS in gravity_sync solve by "
                     << local_correction << '\n';
    }

    for (int lev = fine_level; lev >= crse_level; --lev) {
      rhs[lev - crse_level]->plus(-local_correction, 0, 1, 0);
    }
  }

  // Do multi-level solve for delta_phi.

  solve_for_delta_phi(crse_level, fine_level, amrex::GetVecOfPtrs(rhs),
                      amrex::GetVecOfPtrs(delta_phi),
                      amrex::GetVecOfVecOfPtrs(ec_gdPhi));

  // In the all-periodic case we enforce that delta_phi averages to zero.

  if (crse_geom.isAllPeriodic() &&
      (sim->boxArray(crse_level).numPts() == crse_domain.numPts())) {

    Real local_correction =
        delta_phi[0]->sum() / sim->boxArray(crse_level).numPts();

    for (int lev = crse_level; lev <= fine_level; ++lev) {
      delta_phi[lev - crse_level]->plus(-local_correction, 0, 1, 1);
    }
  }

  // Add delta_phi to phi_new, and grad(delta_phi) to grad(delta_phi_curr) on
  // each level. Update the cell-centered gravity too.

  for (int lev = crse_level; lev <= fine_level; lev++) {

    phi_new_[lev].plus(*delta_phi[lev - crse_level], 0, 1, 0);

    for (int n = 0; n < AMREX_SPACEDIM; n++) {
      grad_phi_curr[lev][n]->plus(*ec_gdPhi[lev - crse_level][n], 0, 1, 0);
    }

    get_new_grav_vector(lev, g_new_[lev], sim->tNew_[lev]);
  }

  int is_new = 1;

  for (int lev = fine_level - 1; lev >= crse_level; --lev) {

    // Average phi_new from fine to coarse level

    const IntVect &ratio = sim->refRatio(lev);

    amrex::average_down(phi_new_[lev + 1], phi_new_[lev], 0, 1, ratio);

    // Average the edge-based grad_phi from finer to coarser level

    average_fine_ec_onto_crse_ec(lev, is_new);

    // Average down the gravitational acceleration too.

    amrex::average_down(g_new_[lev + 1], g_new_[lev], 0, 1, ratio);
  }
}

template <typename T>
void Gravity<T>::solve_for_delta_phi(
    int crse_level, int fine_level, const Vector<MultiFab *> &rhs,
    const Vector<MultiFab *> &delta_phi,
    const Vector<Vector<MultiFab *>> &grad_delta_phi) {
  BL_PROFILE("Gravity<T>::solve_for_delta_phi");

  AMREX_ASSERT(grad_delta_phi.size() == fine_level - crse_level + 1);
  AMREX_ASSERT(delta_phi.size() == fine_level - crse_level + 1);

  if (gravity::verbose > 1) {
    amrex::Print() << "... solving for delta_phi at crse_level = " << crse_level
                   << std::endl;
    amrex::Print() << "...                    up to fine_level = " << fine_level
                   << std::endl;
  }

  Vector<const MultiFab *> crhs{rhs.begin(), rhs.end()};
  Vector<std::array<MultiFab *, AMREX_SPACEDIM>> gp;
  for (const auto &x : grad_delta_phi) {
    gp.push_back({AMREX_D_DECL(x[0], x[1], x[2])});
  }

  Real rel_eps = 0.0;
  Real abs_eps =
      *(std::max_element(level_solver_resnorm.begin() + crse_level,
                         level_solver_resnorm.begin() + fine_level + 1));

  actual_solve_with_mlmg(crse_level, fine_level, delta_phi, crhs, gp, {},
                         nullptr, rel_eps, abs_eps);
}
