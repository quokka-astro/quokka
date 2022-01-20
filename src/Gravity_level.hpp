/// \file Gravity_level.hpp
/// \brief Implements a class for solving the Poisson equation for 3D, Cartesian
/// geometry problems.
///

#include "Gravity.hpp"

template <typename T>
void Gravity<T>::construct_old_gravity(Real time, int level) {
  // Note: this function does NOT do a composite solve! That must be done
  //  by calling Gravity<T>::multilevel_solve_for_new_phi()

  BL_PROFILE("Quokka::construct_old_gravity()");

  MultiFab &grav_old = g_old_[level];
  MultiFab &phi_old = phi_old_[level];

  std::unique_ptr<MultiFab> &comp_minus_level_phi = corr_phi_[level];
  Vector<std::unique_ptr<MultiFab>> &comp_minus_level_grad_phi =
      corr_grad_phi_[level];

  // Do level solve at beginning of time step in order to compute the
  // difference between the multilevel and the single level solutions.
  // [Note that we don't need to do this solve for single-level runs, since
  //  there is no difference between the composite and level solves in this
  //  case.]

  if (get_gravity_type() == GravityMode::Poisson && sim->finestLevel() > 0) {

    // Create a copy of the current (composite) data on this level.

    MultiFab comp_phi;
    Vector<std::unique_ptr<MultiFab>> comp_gphi(AMREX_SPACEDIM);

    // When level == sim->finestLevel(), the composite correction is zero, so
    // only compute it for lower levels

    if (NoComposite() != 1 && DoCompositeCorrection() != 0 &&
        level < sim->finestLevel() && level <= get_max_solve_level()) {

      comp_phi.define(phi_old.boxArray(), phi_old.DistributionMap(),
                      phi_old.nComp(), phi_old.nGrow());
      MultiFab::Copy(comp_phi, phi_old, 0, 0, phi_old.nComp(), phi_old.nGrow());

      for (int n = 0; n < AMREX_SPACEDIM; ++n) {
        comp_gphi[n] = std::make_unique<MultiFab>(
            amrex::convert(sim->boxArray(level),
                           IntVect::TheDimensionVector(n)),
            sim->DistributionMap(level), 1, 0);
        MultiFab::Copy(*comp_gphi[n], *grad_phi_prev[level][n], 0, 0, 1, 0);
      }
    }

    if (gravity::verbose != 0) {
      amrex::Print()
          << "... doing old-time level Poisson gravity solve at level " << level
          << std::endl
          << std::endl;
    }

    int is_new = 0;

    // This is a placeholder solve to get the difference between the composite
    // and level solutions.

    solve_for_phi(level, phi_old, amrex::GetVecOfPtrs(grad_phi_prev[level]),
                  is_new);

    // When level == sim->finestLevel(), the composite correction is zero, so
    // only compute it for lower levels

    if (NoComposite() != 1 && DoCompositeCorrection() != 0 &&
        level < sim->finestLevel() && level <= get_max_solve_level()) {

      // Subtract the level solve from the composite solution.

      create_comp_minus_level_grad_phi(
          level, comp_phi, amrex::GetVecOfPtrs(comp_gphi), comp_minus_level_phi,
          comp_minus_level_grad_phi);

      // Copy the composite data back. This way the forcing (prior to the hydro
      // solve) uses the most accurate data we have.

      MultiFab::Copy(phi_old, comp_phi, 0, 0, phi_old.nComp(), phi_old.nGrow());

      for (int n = 0; n < AMREX_SPACEDIM; ++n) {
        MultiFab::Copy(*grad_phi_prev[level][n], *comp_gphi[n], 0, 0, 1, 0);
      }
    }

    if (test_results_of_solves() == 1) {

      if (gravity::verbose != 0) {
        amrex::Print()
            << "... testing grad_phi_prev (with composite solution)\n";
      }

      test_level_grad_phi_prev(level);
    }
  }

  // Set the old-time gravity vector
  get_old_grav_vector(level, grav_old, time);
}

template <typename T>
void Gravity<T>::construct_new_gravity(Real time, int level) {
  // Note: this function does NOT do a composite solve NOR a sync solve!
  //    That must be done by: multilevel_solve_for_new_phi()
  //    and gravity_sync()!

  BL_PROFILE("Quokka::construct_new_gravity()");

  MultiFab &grav_new = g_new_[level];
  MultiFab &phi_new = phi_new_[level];

  std::unique_ptr<MultiFab> &comp_minus_level_phi = corr_phi_[level];
  Vector<std::unique_ptr<MultiFab>> &comp_minus_level_grad_phi =
      corr_grad_phi_[level];

  // If we're doing Poisson gravity, do the new-time level solve here.

  if (get_gravity_type() == GravityMode::Poisson) {

    // Use the "old" phi from the current time step as a guess for this solve.

    MultiFab &phi_old = phi_old_[level];

    MultiFab::Copy(phi_new, phi_old, 0, 0, 1, phi_new.nGrow());

    // Subtract off the (composite - level) contribution

    if (NoComposite() != 1 && DoCompositeCorrection() != 0 &&
        level < sim->finestLevel() && level <= get_max_solve_level()) {
      // When level == sim->finestLevel(), the composite correction is zero, so
      // only compute it for lower levels, if they exist.
      phi_new.minus(*comp_minus_level_phi, 0, 1, 0);
    }

    if (gravity::verbose != 0) {
      amrex::Print()
          << "... doing new-time level Poisson gravity solve at level " << level
          << std::endl;
    }

    // Do the level solve

    int is_new = 1;

    solve_for_phi(level, phi_new, amrex::GetVecOfPtrs(get_grad_phi_curr(level)),
                  is_new);

    if (test_results_of_solves() == 1) {
      if (gravity::verbose != 0) {
        amrex::Print() << "... testing grad_phi_curr\n";
      }
      test_level_grad_phi_curr(level);
    }

    // [When level == sim->finestLevel(), the composite correction is zero, so
    // only compute it for lower levels, if they exist.]
    if (NoComposite() != 1 && DoCompositeCorrection() != 0 &&
        level < sim->finestLevel() && level <= get_max_solve_level()) {

      // Add back the (composite - level) contribution.
      phi_new.plus(*comp_minus_level_phi, 0, 1, 0);

      for (int n = 0; n < AMREX_SPACEDIM; ++n) {
        get_grad_phi_curr(level)[n]->plus(*comp_minus_level_grad_phi[n], 0, 1,
                                          0);
      }

      if (test_results_of_solves() == 1) {

        if (gravity::verbose != 0) {
          amrex::Print()
              << "... testing grad_phi_curr after composite correction\n";
        }

        test_level_grad_phi_curr(level);
      }
    }
  }

  // Set new-time gravity vector
  get_new_grav_vector(level, grav_new, time);

  if (get_gravity_type() == GravityMode::Poisson &&
      level < sim->finestLevel() && level <= get_max_solve_level()) {

    if (NoComposite() != 1 && DoCompositeCorrection() == 1) {

      if (NoSync() == 0) {
        // Now that we have calculated the force, if we are going to do a sync
        // solve then subtract off the (composite - level) contribution, as it
        // interferes with the sync solve.

        phi_new.minus(*comp_minus_level_phi, 0, 1, 0);

        for (int n = 0; n < AMREX_SPACEDIM; ++n) {
          get_grad_phi_curr(level)[n]->minus(*comp_minus_level_grad_phi[n], 0,
                                             1, 0);
        }
      }

      // Clear the pointers for the correction MultiFabs (deallocated via
      // unique_ptr)
      comp_minus_level_phi.reset();
      comp_minus_level_grad_phi.clear();
    }
  }
}

template <typename T>
void Gravity<T>::create_comp_minus_level_grad_phi(
    int level, MultiFab &comp_phi, const Vector<MultiFab *> &comp_gphi,
    std::unique_ptr<MultiFab> &comp_minus_level_phi,
    Vector<std::unique_ptr<MultiFab>> &comp_minus_level_grad_phi) {
  BL_PROFILE("Gravity<T>::create_comp_minus_level_grad_phi()");

  if (gravity::verbose > 1) {
    amrex::Print()
        << "... creating MultiFabs for differences between level and "
           "composite solves at level "
        << level << "\n";
  }

  // create comp_minus_level_phi

  comp_minus_level_phi = std::make_unique<MultiFab>(
      sim->boxArray(level), sim->DistributionMap(level), 1, 0);

  MultiFab::Copy(*comp_minus_level_phi, comp_phi, 0, 0, 1, 0);
  comp_minus_level_phi->minus(phi_old_[level], 0, 1, 0);

  // create comp_minus_level_grad_phi

  comp_minus_level_grad_phi.resize(AMREX_SPACEDIM);

  for (int n = 0; n < AMREX_SPACEDIM; ++n) {
    comp_minus_level_grad_phi[n] = std::make_unique<MultiFab>(
        amrex::convert(sim->boxArray(level), IntVect::TheDimensionVector(n)),
        sim->DistributionMap(level), 1, 0);

    MultiFab::Copy(*comp_minus_level_grad_phi[n], *comp_gphi[n], 0, 0, 1, 0);
    comp_minus_level_grad_phi[n]->minus(*grad_phi_prev[level][n], 0, 1, 0);
  }
}

template <typename T>
void Gravity<T>::get_old_grav_vector(int level, MultiFab &grav_vector,
                                     Real time) {
  BL_PROFILE("Gravity<T>::get_old_grav_vector()");

  // Fill data from the level below if we're not doing a solve on this level

  if (level > gravity::max_solve_level) {
    sim->FillCoarsePatch(level, time, grav_vector, g_old_, g_new_, 0, 3);
    return;
  }

  if (gravity::gravity_type == GravityMode::Poisson) {

    int ng = grav_vector.nGrow();
    const Geometry &geom = sim->Geom(level);
    amrex::average_face_to_cellcenter(
        grav_vector, amrex::GetVecOfConstPtrs(grad_phi_prev[level]), geom);
    grav_vector.mult(-1.0, ng); // g = - grad(phi)

  } else {
    amrex::Abort("Unknown gravity_type in get_old_grav_vector");
  }
}

template <typename T>
void Gravity<T>::get_new_grav_vector(int level, MultiFab &grav_vector,
                                     Real time) {
  BL_PROFILE("Gravity<T>::get_new_grav_vector()");

  // Fill data from the level below if we're not doing a solve on this level

  if (level > gravity::max_solve_level) {
    sim->FillCoarsePatch(level, time, grav_vector, g_old_, g_new_, 0, 3);
    return;
  }

  if (gravity::gravity_type == GravityMode::Poisson) {

    int ng = grav_vector.nGrow();
    const Geometry &geom = sim->Geom(level);
    amrex::average_face_to_cellcenter(
        grav_vector, amrex::GetVecOfConstPtrs(grad_phi_curr[level]), geom);
    grav_vector.mult(-1.0, ng); // g = - grad(phi)

  } else {
    amrex::Abort("Unknown gravity_type in get_new_grav_vector");
  }
}
