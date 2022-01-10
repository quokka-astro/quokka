//==============================================================================
// Poisson gravity solver, adapted from Castro's gravity module.
//   Note that Quokka's modified version only handles Poisson gravity in 3D
//   Cartesian geometry, which makes the code significantly simpler. There are
//   also significant style changes to modernize to C++17. Also, we do not use
//   amrex::AmrLevel here since Quokka uses amrex::AmrCore instead.
// Commit history:
//   https://github.com/AMReX-Astro/Castro/commits/main/Source/gravity/Gravity.cpp
// Used under the terms of the open-source license (BSD 3-clause) given here:
//   https://github.com/AMReX-Astro/Castro/blob/main/license.txt
//==============================================================================
/// \file gravity.cpp
/// \brief Implements a class for solving the Poisson equation for 3D, Cartesian
/// geometry problems.
///

#include <cmath>
#include <limits>
#include <memory>

#include <AMReX_FillPatchUtil.H>
#include <AMReX_MLMG.H>
#include <AMReX_MLPoisson.H>
#include <AMReX_ParmParse.H>

#include "hydro_system.hpp"
#include <Gravity.H>

using namespace amrex;

Real Gravity::mass_offset = 0.0;

// **************************************************************************************
// //

// Ggravity is defined as 4 * pi * G, where G is the gravitational constant.

// In CGS, this constant is currently
//      Gconst   =  6.67428e-8           cm^3/g/s^2 , which results in
//      Ggravity =  83.8503442814844e-8  cm^3/g/s^2

// **************************************************************************************
// //

static Real Ggravity = 0.;

///
/// Multipole gravity data
///
AMREX_GPU_MANAGED Real multipole::volumeFactor;
AMREX_GPU_MANAGED Real multipole::parityFactor;

AMREX_GPU_MANAGED Real multipole::rmax;

AMREX_GPU_MANAGED Array1D<bool, 0, 2> multipole::doSymmetricAddLo;
AMREX_GPU_MANAGED Array1D<bool, 0, 2> multipole::doSymmetricAddHi;
AMREX_GPU_MANAGED bool multipole::doSymmetricAdd;

AMREX_GPU_MANAGED Array1D<bool, 0, 2> multipole::doReflectionLo;
AMREX_GPU_MANAGED Array1D<bool, 0, 2> multipole::doReflectionHi;

AMREX_GPU_MANAGED Array2D<Real, 0, multipole::lnum_max, 0, multipole::lnum_max>
    multipole::factArray;
AMREX_GPU_MANAGED Array1D<Real, 0, multipole::lnum_max> multipole::parity_q0;
AMREX_GPU_MANAGED Array2D<Real, 0, multipole::lnum_max, 0, multipole::lnum_max>
    multipole::parity_qC_qS;

Gravity::Gravity(Amr *Parent, int /*_finest_level*/, BCRec *_phys_bc,
                 int _Density)
    : parent(Parent), LevelData(MAX_LEV), grad_phi_curr(MAX_LEV),
      grad_phi_prev(MAX_LEV), grids(Parent->boxArray()),
      dmap(Parent->DistributionMap()), abs_tol(MAX_LEV), rel_tol(MAX_LEV),
      level_solver_resnorm(MAX_LEV), volume(MAX_LEV), area(MAX_LEV),
      phys_bc(_phys_bc) {
  Density = _Density;
  read_params();
  finest_level_allocated = -1;

  if (gravity::gravity_type == "PoissonGrav") {
    make_mg_bc();
  }
  if (gravity::gravity_type == "PoissonGrav") {
    init_multipole_grav();
  }
  max_rhs = 0.0;

  // uninitialized:
  //  mlmg_lobc, mlmg_hibc -- set by Gravity::make_mg_bc()
  //  numpts_at_level -- set by Gravity::set_numpts_in_gravity()
}

void Gravity::read_params() {
  static bool done = false;

  if (!done) {
    const Geometry &dgeom = DefaultGeometry();

    ParmParse pp("gravity");

    if ((gravity::gravity_type != "ConstantGrav") &&
        (gravity::gravity_type != "PoissonGrav")) {
      std::cout << "Sorry -- dont know this gravity type" << std::endl;
      amrex::Abort("Options are ConstantGrav or PoissonGrav");
    }

#if (AMREX_SPACEDIM < 3)
    if (gravity::gravity_type == "PoissonGrav") {
      amrex::Abort(
          " gravity::gravity_type = PoissonGrav doesn't work in 1D or 2D!");
    }
#endif

    if (pp.contains("get_g_from_phi") && (gravity::get_g_from_phi == 0) &&
        gravity::gravity_type == "PoissonGrav") {
      if (ParallelDescriptor::IOProcessor()) {
        std::cout << "Warning: gravity::gravity_type = PoissonGrav assumes "
                     "get_g_from_phi is true"
                  << std::endl;
      }
    }

    int nlevs = parent->maxLevel() + 1;

    // Allow run-time input of solver tolerance. If the user
    // provides no value, set a reasonable default value on the
    // coarse level, and then increase it by ref_ratio**2 as the
    // levels get finer to account for the change in the absolute
    // scale of the Laplacian. If the user provides one value, use
    // that on the coarse level, and increase it the same way for
    // the fine levels. If the user provides more than one value,
    // we expect them to provide one for every level, and we do
    // not apply the ref_ratio effect.

    int n_abs_tol = pp.countval("abs_tol");

    if (n_abs_tol <= 1) {
      Real tol = NAN;
      if (n_abs_tol == 1) {
        pp.get("abs_tol", tol);
      } else {
        tol = 1.e-11;
      }

      abs_tol[0] = tol;

      // Account for the fact that on finer levels, the scale of the
      // Laplacian changes due to the zone size changing. We assume
      // dx == dy == dz, so it is fair to say that on each level the
      // tolerance should increase by the factor ref_ratio**2, since
      // in absolute terms the Laplacian increases by that ratio too.
      // The actual tolerance we'll send in is the effective tolerance
      // on the finest level that we solve for.

      for (int lev = 1; lev < nlevs; ++lev) {
        abs_tol[lev] =
            abs_tol[lev - 1] * std::pow(parent->refRatio(lev - 1)[0], 2);
      }

    } else if (n_abs_tol >= nlevs) {

      pp.getarr("abs_tol", abs_tol, 0, nlevs);

    } else {

      amrex::Abort(
          "If you are providing multiple values for abs_tol, you must provide "
          "at least one value for every level up to amr.max_level.");
    }

    // For the relative tolerance, we can again accept a single
    // scalar (same for all levels) or one for all levels. The
    // default value is zero, so that we only use the absolute
    // tolerance.  The multigrid always chooses the looser of the
    // two criteria in determining whether the solve has
    // converged.

    // Note that the parameter rel_tol used to be known as ml_tol,
    // so if we detect that the user has set ml_tol but not
    // rel_tol, we'll accept that for specifying the relative
    // tolerance. ml_tol is now considered deprecated and will be
    // removed in a future release.

    std::string rel_tol_name = "rel_tol";
    int n_rel_tol = pp.countval(rel_tol_name.c_str());

    if (n_rel_tol <= 1) {
      Real tol = NAN;
      if (n_rel_tol == 1) {
        pp.get(rel_tol_name.c_str(), tol);
      } else {
        tol = 0.0;
      }
      for (int lev = 0; lev < MAX_LEV; ++lev) {
        rel_tol[lev] = tol;
      }

    } else if (n_rel_tol >= nlevs) {
      pp.getarr(rel_tol_name.c_str(), rel_tol, 0, nlevs);
    } else {
      amrex::Abort(
          "If you are providing multiple values for rel_tol, you must provide "
          "at least one value for every level up to amr.max_level.");
    }

    Ggravity = 4.0 * M_PI * C::Gconst;
    if (gravity::verbose > 1 && ParallelDescriptor::IOProcessor()) {
      std::cout << "Getting Gconst from constants: " << C::Gconst << std::endl;
      std::cout << "Using " << Ggravity << " for 4 pi G in Gravity.cpp "
                << std::endl;
    }

    done = true;
  }
}

void Gravity::set_numpts_in_gravity(int numpts) { numpts_at_level = numpts; }

void Gravity::install_level(int level, AmrLevel *level_data, MultiFab &_volume,
                            MultiFab *_area) {
  if (gravity::verbose > 1 && ParallelDescriptor::IOProcessor()) {
    std::cout << "Installing Gravity level " << level << '\n';
  }

  LevelData[level] = level_data;

  volume[level] = &_volume;

  area[level] = _area;

  level_solver_resnorm[level] = 0.0;

  const Geometry &geom = level_data->Geom();

  if (gravity::gravity_type == "PoissonGrav") {

    const DistributionMapping &dm = level_data->DistributionMap();

    grad_phi_prev[level].resize(AMREX_SPACEDIM);
    for (int n = 0; n < AMREX_SPACEDIM; ++n) {
      grad_phi_prev[level][n] =
          std::make_unique<MultiFab>(level_data->getEdgeBoxArray(n), dm, 1, 1);
    }

    grad_phi_curr[level].resize(AMREX_SPACEDIM);
    for (int n = 0; n < AMREX_SPACEDIM; ++n) {
      grad_phi_curr[level][n] =
          std::make_unique<MultiFab>(level_data->getEdgeBoxArray(n), dm, 1, 1);
    }
  }

  finest_level_allocated = level;
}

auto Gravity::get_gravity_type() -> std::string {
  return gravity::gravity_type;
}

auto Gravity::get_max_solve_level() -> int { return gravity::max_solve_level; }

auto Gravity::NoSync() -> int { return gravity::no_sync; }

auto Gravity::NoComposite() -> int { return gravity::no_composite; }

auto Gravity::DoCompositeCorrection() -> int {
  return gravity::do_composite_phi_correction;
}

// if so, check the residuals manually
auto Gravity::test_results_of_solves() -> int { return test_solves; }

auto Gravity::get_grad_phi_prev(int level)
    -> Vector<std::unique_ptr<MultiFab>> & {
  return grad_phi_prev[level];
}

auto Gravity::get_grad_phi_prev_comp(int level, int comp) -> MultiFab * {
  return grad_phi_prev[level][comp].get();
}

auto Gravity::get_grad_phi_curr(int level)
    -> Vector<std::unique_ptr<MultiFab>> & {
  return grad_phi_curr[level];
}

void Gravity::plus_grad_phi_curr(int level,
                                 Vector<std::unique_ptr<MultiFab>> &addend) {
  for (int n = 0; n < AMREX_SPACEDIM; n++) {
    grad_phi_curr[level][n]->plus(*addend[n], 0, 1, 0);
  }
}

void Gravity::swapTimeLevels(int level) {
  BL_PROFILE("Gravity::swapTimeLevels()");

  if (gravity::gravity_type == "PoissonGrav") {
    for (int n = 0; n < AMREX_SPACEDIM; n++) {
      std::swap(grad_phi_prev[level][n], grad_phi_curr[level][n]);
      grad_phi_curr[level][n]->setVal(1.e50);
    }
  }
}

void Gravity::solve_for_phi(int level, MultiFab &phi,
                            const Vector<MultiFab *> &grad_phi, int is_new) {
  BL_PROFILE("Gravity::solve_for_phi()");

  if (gravity::verbose > 1 && ParallelDescriptor::IOProcessor()) {
    std::cout << " ... solve for phi at level " << level << std::endl;
  }

  const Real strt = ParallelDescriptor::second();

  if (is_new == 0) {
    sanity_check(level);
  }

  Real time = NAN;
  if (is_new == 1) {
    time = LevelData[level]->get_state_data(PhiGrav_Type).curTime();
  } else {
    time = LevelData[level]->get_state_data(PhiGrav_Type).prevTime();
  }

  // If we are below the max_solve_level, do the Poisson solve.
  // Otherwise, interpolate using a fillpatch from max_solve_level.

  if (level <= gravity::max_solve_level) {

    Vector<MultiFab *> phi_p(1, &phi);

    const auto &rhs = get_rhs(level, 1, is_new);

    Vector<Vector<MultiFab *>> grad_phi_p(1);
    grad_phi_p[0].resize(AMREX_SPACEDIM);
    for (int i = 0; i < AMREX_SPACEDIM; i++) {
      grad_phi_p[0][i] = grad_phi[i];
    }

    Vector<MultiFab *> res_null;

    level_solver_resnorm[level] =
        solve_phi_with_mlmg(level, level, phi_p, amrex::GetVecOfPtrs(rhs),
                            grad_phi_p, res_null, time);

  } else {

    LevelData[level]->FillCoarsePatch(phi, 0, time, PhiGrav_Type, 0, 1, 1);
  }

  if (gravity::verbose != 0) {
    const int IOProc = ParallelDescriptor::IOProcessorNumber();
    Real end = ParallelDescriptor::second() - strt;
    ParallelDescriptor::ReduceRealMax(end, IOProc);
    if (ParallelDescriptor::IOProcessor()) {
      std::cout << "Gravity::solve_for_phi() time = " << end << std::endl
                << std::endl;
    }
  }
}

void Gravity::gravity_sync(int crse_level, int fine_level,
                           const Vector<MultiFab *> &drho,
                           const Vector<MultiFab *> &dphi) {
  BL_PROFILE("Gravity::gravity_sync()");

  // There is no need to do a synchronization if
  // we didn't solve on the fine levels.

  if (fine_level > gravity::max_solve_level) {
    return;
  }
  fine_level = amrex::min(fine_level, gravity::max_solve_level);

  BL_ASSERT(parent->finestLevel() > crse_level);
  if (gravity::verbose > 1 && ParallelDescriptor::IOProcessor()) {
    std::cout << " ... gravity_sync at crse_level " << crse_level << '\n';
    std::cout << " ...     up to finest_level     " << fine_level << '\n';
  }

  const Geometry &crse_geom = parent->Geom(crse_level);
  const Box &crse_domain = crse_geom.Domain();

  int nlevs = fine_level - crse_level + 1;

  // Construct delta(phi) and delta(grad_phi). delta(phi)
  // needs a ghost zone for holding the boundary condition
  // in the same way that phi does.

  Vector<std::unique_ptr<MultiFab>> delta_phi(nlevs);

  for (int lev = crse_level; lev <= fine_level; ++lev) {
    delta_phi[lev - crse_level] =
        std::make_unique<MultiFab>(grids[lev], dmap[lev], 1, 1);
    delta_phi[lev - crse_level]->setVal(0.0);
  }

  Vector<Vector<std::unique_ptr<MultiFab>>> ec_gdPhi(nlevs);

  for (int lev = crse_level; lev <= fine_level; ++lev) {
    ec_gdPhi[lev - crse_level].resize(AMREX_SPACEDIM);

    const DistributionMapping &dm = LevelData[lev]->DistributionMap();
    for (int n = 0; n < AMREX_SPACEDIM; ++n) {
      ec_gdPhi[lev - crse_level][n] = std::make_unique<MultiFab>(
          LevelData[lev]->getEdgeBoxArray(n), dm, 1, 0);
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
        LevelData[lev]->boxArray(), LevelData[lev]->DistributionMap(), 1, 0);
    MultiFab::Copy(*rhs[lev - crse_level], *dphi[lev - crse_level], 0, 0, 1, 0);
    rhs[lev - crse_level]->mult(1.0 / Ggravity);
    MultiFab::Add(*rhs[lev - crse_level], *drho[lev - crse_level], 0, 0, 1, 0);
  }

  // Construct the boundary conditions for the Poisson solve.

  if (crse_level == 0 && !crse_geom.isAllPeriodic()) {

    if (gravity::verbose > 1 && ParallelDescriptor::IOProcessor()) {
      std::cout << " ... Making bc's for delta_phi at crse_level 0"
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
      (grids[crse_level].numPts() == crse_domain.numPts())) {

    // We assume that if we're fully periodic then we're going to be in
    // Cartesian coordinates, so to get the average value of the RHS we can
    // divide the sum of the RHS by the number of points. This correction should
    // probably be volume weighted if we somehow got here without being
    // Cartesian.

    Real local_correction = rhs[0]->sum() / grids[crse_level].numPts();

    if (gravity::verbose > 1 && ParallelDescriptor::IOProcessor()) {
      std::cout << "WARNING: Adjusting RHS in gravity_sync solve by "
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
      (grids[crse_level].numPts() == crse_domain.numPts())) {

    Real local_correction = delta_phi[0]->sum() / grids[crse_level].numPts();

    for (int lev = crse_level; lev <= fine_level; ++lev) {
      delta_phi[lev - crse_level]->plus(-local_correction, 0, 1, 1);
    }
  }

  // Add delta_phi to phi_new, and grad(delta_phi) to grad(delta_phi_curr) on
  // each level. Update the cell-centered gravity too.

  for (int lev = crse_level; lev <= fine_level; lev++) {

    LevelData[lev]
        ->get_new_data(PhiGrav_Type)
        .plus(*delta_phi[lev - crse_level], 0, 1, 0);

    for (int n = 0; n < AMREX_SPACEDIM; n++) {
      grad_phi_curr[lev][n]->plus(*ec_gdPhi[lev - crse_level][n], 0, 1, 0);
    }

    get_new_grav_vector(lev, LevelData[lev]->get_new_data(Gravity_Type),
                        LevelData[lev]->get_state_data(State_Type).curTime());
  }

  int is_new = 1;

  for (int lev = fine_level - 1; lev >= crse_level; --lev) {

    // Average phi_new from fine to coarse level

    const IntVect &ratio = parent->refRatio(lev);

    amrex::average_down(LevelData[lev + 1]->get_new_data(PhiGrav_Type),
                        LevelData[lev]->get_new_data(PhiGrav_Type), 0, 1,
                        ratio);

    // Average the edge-based grad_phi from finer to coarser level

    average_fine_ec_onto_crse_ec(lev, is_new);

    // Average down the gravitational acceleration too.

    amrex::average_down(LevelData[lev + 1]->get_new_data(Gravity_Type),
                        LevelData[lev]->get_new_data(Gravity_Type), 0, 1,
                        ratio);
  }
}

void Gravity::GetCrsePhi(int level, MultiFab &phi_crse, Real time) {
  BL_PROFILE("Gravity::GetCrsePhi()");

  BL_ASSERT(level != 0);

  const Real t_old =
      LevelData[level - 1]->get_state_data(PhiGrav_Type).prevTime();
  const Real t_new =
      LevelData[level - 1]->get_state_data(PhiGrav_Type).curTime();
  Real alpha = (time - t_old) / (t_new - t_old);
  Real omalpha = 1.0 - alpha;

  MultiFab const &phi_old = LevelData[level - 1]->get_old_data(PhiGrav_Type);
  MultiFab const &phi_new = LevelData[level - 1]->get_new_data(PhiGrav_Type);

  phi_crse.clear();
  phi_crse.define(grids[level - 1], dmap[level - 1], 1,
                  1); // BUT NOTE we don't trust phi's ghost cells.

  MultiFab::LinComb(phi_crse, alpha, phi_new, 0, omalpha, phi_old, 0, 0, 1, 1);

  const Geometry &geom = parent->Geom(level - 1);
  phi_crse.FillBoundary(geom.periodicity());
}

void Gravity::multilevel_solve_for_new_phi(int level, int finest_level_in) {
  BL_PROFILE("Gravity::multilevel_solve_for_new_phi()");

  if (gravity::verbose > 1 && ParallelDescriptor::IOProcessor()) {
    std::cout << "... multilevel solve for new phi at base level " << level
              << " to finest level " << finest_level_in << std::endl;
  }

  for (int lev = level; lev <= finest_level_in; lev++) {
    BL_ASSERT(grad_phi_curr[lev].size() == AMREX_SPACEDIM);
    for (int n = 0; n < AMREX_SPACEDIM; ++n) {
      grad_phi_curr[lev][n] =
          std::make_unique<MultiFab>(LevelData[lev]->getEdgeBoxArray(n),
                                     LevelData[lev]->DistributionMap(), 1, 1);
    }
  }

  int is_new = 1;
  actual_multilevel_solve(level, finest_level_in,
                          amrex::GetVecOfVecOfPtrs(grad_phi_curr), is_new);
}

void Gravity::actual_multilevel_solve(
    int crse_level, int finest_level_in,
    const Vector<Vector<MultiFab *>> &grad_phi, int is_new) {
  BL_PROFILE("Gravity::actual_multilevel_solve()");

  for (int ilev = crse_level; ilev <= finest_level_in; ++ilev) {
    sanity_check(ilev);
  }

  int nlevels = finest_level_in - crse_level + 1;

  Vector<MultiFab *> phi_p(nlevels);
  for (int ilev = 0; ilev < nlevels; ilev++) {
    int amr_lev = ilev + crse_level;
    if (is_new == 1) {
      phi_p[ilev] = &LevelData[amr_lev]->get_new_data(PhiGrav_Type);
    } else {
      phi_p[ilev] = &LevelData[amr_lev]->get_old_data(PhiGrav_Type);
    }
  }

  const auto &rhs = get_rhs(crse_level, nlevels, is_new);

  Vector<Vector<MultiFab *>> grad_phi_p(nlevels);
  for (int ilev = 0; ilev < nlevels; ilev++) {
    int amr_lev = ilev + crse_level;
    grad_phi_p[ilev] = grad_phi[amr_lev];
  }

  Real time = NAN;
  if (is_new == 1) {
    time = LevelData[crse_level]->get_state_data(PhiGrav_Type).curTime();
  } else {
    time = LevelData[crse_level]->get_state_data(PhiGrav_Type).prevTime();
  }

  int fine_level = amrex::min(finest_level_in, gravity::max_solve_level);

  if (fine_level >= crse_level) {

    Vector<MultiFab *> res_null;
    solve_phi_with_mlmg(crse_level, fine_level, phi_p, amrex::GetVecOfPtrs(rhs),
                        grad_phi_p, res_null, time);

    // Average phi from fine to coarse level
    for (int amr_lev = fine_level; amr_lev > crse_level; amr_lev--) {
      const IntVect &ratio = parent->refRatio(amr_lev - 1);
      if (is_new == 1) {
        amrex::average_down(LevelData[amr_lev]->get_new_data(PhiGrav_Type),
                            LevelData[amr_lev - 1]->get_new_data(PhiGrav_Type),
                            0, 1, ratio);
      } else if (is_new == 0) {
        amrex::average_down(LevelData[amr_lev]->get_old_data(PhiGrav_Type),
                            LevelData[amr_lev - 1]->get_old_data(PhiGrav_Type),
                            0, 1, ratio);
      }
    }

    // Average grad_phi from fine to coarse level
    for (int amr_lev = fine_level; amr_lev > crse_level; amr_lev--) {
      average_fine_ec_onto_crse_ec(amr_lev - 1, is_new);
    }
  }

  // For all levels on which we're not doing the solve, interpolate from
  // the coarsest level with correct data. Note that since FillCoarsePatch
  // fills from the coarse level just below it, we need to fill from the
  // lowest level upwards using successive interpolations.

  for (int amr_lev = gravity::max_solve_level + 1; amr_lev <= finest_level_in;
       amr_lev++) {

    // Interpolate the potential.

    if (is_new == 1) {

      MultiFab &phi = LevelData[amr_lev]->get_new_data(PhiGrav_Type);

      LevelData[amr_lev]->FillCoarsePatch(phi, 0, time, PhiGrav_Type, 0, 1, 1);

    } else {

      MultiFab &phi = LevelData[amr_lev]->get_old_data(PhiGrav_Type);

      LevelData[amr_lev]->FillCoarsePatch(phi, 0, time, PhiGrav_Type, 0, 1, 1);
    }

    // Interpolate the grad_phi.

    // Instantiate a bare physical BC function for grad_phi. It doesn't do
    // anything since the fine levels for Poisson gravity do not touch the
    // physical boundary.

    GradPhiPhysBCFunct gp_phys_bc;

    // We need to use a interpolater that works with data on faces.

    Interpolater *gp_interp = &face_linear_interp;

    // For the BCs, we will use the Gravity_Type BCs for convenience, but these
    // will not do anything because we do not fill on physical boundaries.

    const Vector<BCRec> &gp_bcs =
        LevelData[amr_lev]->get_desc_lst()[Gravity_Type].getBCs();

    for (int n = 0; n < AMREX_SPACEDIM; ++n) {
      amrex::InterpFromCoarseLevel(
          *grad_phi[amr_lev][n], time, *grad_phi[amr_lev - 1][n], 0, 0, 1,
          parent->Geom(amr_lev - 1), parent->Geom(amr_lev), gp_phys_bc, 0,
          gp_phys_bc, 0, parent->refRatio(amr_lev - 1), gp_interp, gp_bcs, 0);
    }
  }
}

void Gravity::get_old_grav_vector(int level, MultiFab &grav_vector, Real time) {
  BL_PROFILE("Gravity::get_old_grav_vector()");

  int ng = grav_vector.nGrow();

  // Fill data from the level below if we're not doing a solve on this level.

  if (level > gravity::max_solve_level) {

    LevelData[level]->FillCoarsePatch(grav_vector, 0, time, Gravity_Type, 0, 3,
                                      ng);

    return;
  }

  // Note that grav_vector coming into this routine always has three components.
  // So we'll define a temporary MultiFab with AMREX_SPACEDIM dimensions.
  // Then at the end we'll copy in all AMREX_SPACEDIM dimensions from this into
  // the outgoing grav_vector, leaving any higher dimensions unchanged.

  MultiFab grav(grids[level], dmap[level], AMREX_SPACEDIM, ng);
  grav.setVal(0.0, ng);

  if (gravity::gravity_type == "ConstantGrav") {

    // Set to constant value in the AMREX_SPACEDIM direction and zero in all
    // others.

    grav.setVal(gravity::const_grav, AMREX_SPACEDIM - 1, 1, ng);

  } else if (gravity::gravity_type == "PoissonGrav") {

    const Geometry &geom = parent->Geom(level);
    amrex::average_face_to_cellcenter(
        grav, amrex::GetVecOfConstPtrs(grad_phi_prev[level]), geom);
    grav.mult(-1.0, ng); // g = - grad(phi)

  } else {
    amrex::Abort("Unknown gravity_type in get_old_grav_vector");
  }

  // Do the copy to the output vector.

  for (int dir = 0; dir < 3; dir++) {
    if (dir < AMREX_SPACEDIM) {
      MultiFab::Copy(grav_vector, grav, dir, dir, 1, ng);
    } else {
      grav_vector.setVal(0., dir, 1, ng);
    }
  }

  if (gravity::gravity_type != "ConstantGrav") {
    // Fill ghost cells
    AmrLevel *amrlev = &parent->getLevel(level);
    AmrLevel::FillPatch(*amrlev, grav_vector, ng, time, Gravity_Type, 0,
                        AMREX_SPACEDIM);
  }
}

void Gravity::get_new_grav_vector(int level, MultiFab &grav_vector, Real time) {
  BL_PROFILE("Gravity::get_new_grav_vector()");

  int ng = grav_vector.nGrow();

  // Fill data from the level below if we're not doing a solve on this level.

  if (level > gravity::max_solve_level) {

    LevelData[level]->FillCoarsePatch(grav_vector, 0, time, Gravity_Type, 0, 3,
                                      ng);

    return;
  }

  // Note that grav_vector coming into this routine always has three components.
  // So we'll define a temporary MultiFab with AMREX_SPACEDIM dimensions.
  // Then at the end we'll copy in all AMREX_SPACEDIM dimensions from this into
  // the outgoing grav_vector, leaving any higher dimensions unchanged.

  MultiFab grav(grids[level], dmap[level], AMREX_SPACEDIM, ng);
  grav.setVal(0.0, ng);

  if (gravity::gravity_type == "ConstantGrav") {

    // Set to constant value in the AMREX_SPACEDIM direction
    grav.setVal(gravity::const_grav, AMREX_SPACEDIM - 1, 1, ng);

  } else if (gravity::gravity_type == "PoissonGrav") {

    const Geometry &geom = parent->Geom(level);
    amrex::average_face_to_cellcenter(
        grav, amrex::GetVecOfConstPtrs(grad_phi_curr[level]), geom);
    grav.mult(-1.0, ng); // g = - grad(phi)

  } else {
    amrex::Abort("Unknown gravity_type in get_new_grav_vector");
  }

  // Do the copy to the output vector.

  for (int dir = 0; dir < 3; dir++) {
    if (dir < AMREX_SPACEDIM) {
      MultiFab::Copy(grav_vector, grav, dir, dir, 1, ng);
    } else {
      grav_vector.setVal(0., dir, 1, ng);
    }
  }

  if (gravity::gravity_type != "ConstantGrav" && ng > 0) {
    // Fill ghost cells
    AmrLevel *amrlev = &parent->getLevel(level);
    AmrLevel::FillPatch(*amrlev, grav_vector, ng, time, Gravity_Type, 0,
                        AMREX_SPACEDIM);
  }
}

void Gravity::create_comp_minus_level_grad_phi(
    int level, MultiFab &comp_phi, const Vector<MultiFab *> &comp_gphi,
    MultiFab &comp_minus_level_phi,
    Vector<std::unique_ptr<MultiFab>> &comp_minus_level_grad_phi) {
  BL_PROFILE("Gravity::create_comp_minus_level_grad_phi()");

  if (gravity::verbose > 1 && ParallelDescriptor::IOProcessor()) {
    std::cout << "\n";
    std::cout
        << "... compute difference between level and composite solves at level "
        << level << "\n";
    std::cout << "\n";
  }

  comp_minus_level_phi.define(LevelData[level]->boxArray(),
                              LevelData[level]->DistributionMap(), 1, 0);

  MultiFab::Copy(comp_minus_level_phi, comp_phi, 0, 0, 1, 0);
  comp_minus_level_phi.minus(parent->getLevel(level).get_old_data(PhiGrav_Type),
                             0, 1, 0);

  comp_minus_level_grad_phi.resize(AMREX_SPACEDIM);
  for (int n = 0; n < AMREX_SPACEDIM; ++n) {
    comp_minus_level_grad_phi[n] =
        std::make_unique<MultiFab>(LevelData[level]->getEdgeBoxArray(n),
                                   LevelData[level]->DistributionMap(), 1, 0);
    MultiFab::Copy(*comp_minus_level_grad_phi[n], *comp_gphi[n], 0, 0, 1, 0);
    comp_minus_level_grad_phi[n]->minus(*grad_phi_prev[level][n], 0, 1, 0);
  }
}

void Gravity::average_fine_ec_onto_crse_ec(int level, int is_new) {
  BL_PROFILE("Gravity::average_fine_ec_onto_crse_ec()");

  // NOTE: this is called with level == the coarser of the two levels involved
  if (level == parent->finestLevel()) {
    return;
  }

  //
  // Coarsen() the fine stuff on processors owning the fine data.
  //
  BoxArray crse_gphi_fine_BA(grids[level + 1].size());

  IntVect fine_ratio = parent->refRatio(level);

  for (int i = 0; i < crse_gphi_fine_BA.size(); ++i) {
    crse_gphi_fine_BA.set(i, amrex::coarsen(grids[level + 1][i], fine_ratio));
  }

  Vector<std::unique_ptr<MultiFab>> crse_gphi_fine(AMREX_SPACEDIM);
  for (int n = 0; n < AMREX_SPACEDIM; ++n) {
    BoxArray eba = crse_gphi_fine_BA;
    eba.surroundingNodes(n);
    crse_gphi_fine[n] = std::make_unique<MultiFab>(eba, dmap[level + 1], 1, 0);
  }

  auto &grad_phi = (is_new) != 0 ? grad_phi_curr : grad_phi_prev;

  amrex::average_down_faces(amrex::GetVecOfConstPtrs(grad_phi[level + 1]),
                            amrex::GetVecOfPtrs(crse_gphi_fine), fine_ratio);

  const Geometry &cgeom = parent->Geom(level);

  for (int n = 0; n < AMREX_SPACEDIM; ++n) {
    grad_phi[level][n]->ParallelCopy(*crse_gphi_fine[n], cgeom.periodicity());
  }
}

// this is used for periodic domains only
void Gravity::set_mass_offset(Real time, bool multi_level) const {
  BL_PROFILE("Gravity::set_mass_offset()");

  const Geometry &geom = parent->Geom(0);

  if (!geom.isAllPeriodic()) {
    mass_offset = 0.0;
  } else {
    Real old_mass_offset = mass_offset;
    mass_offset = 0.0;

    if (multi_level) {
      for (int lev = 0; lev <= parent->finestLevel(); lev++) {
        // Castro *cs = dynamic_cast<Castro *>(&parent->getLevel(lev));
        mass_offset += cs->volWgtSum("density", time);
      }
    } else {
      // Castro *cs = dynamic_cast<Castro *>(&parent->getLevel(0));
      mass_offset = cs->volWgtSum("density", time, false,
                                  false); // do not mask off fine grids
    }

    mass_offset = mass_offset / geom.ProbSize();
    if (gravity::verbose > 1 && ParallelDescriptor::IOProcessor()) {
      std::cout << "Defining average density to be " << mass_offset
                << std::endl;
    }

    Real diff = std::abs(mass_offset - old_mass_offset);
    Real eps = 1.e-10 * std::abs(old_mass_offset);
    if (diff > eps && old_mass_offset > 0) {
      if (ParallelDescriptor::IOProcessor()) {
        std::cout << " ... new vs old mass_offset " << mass_offset << " "
                  << old_mass_offset << " ... diff is " << diff << std::endl;
        std::cout << " ... Gravity::set_mass_offset -- total mass has changed!"
                  << std::endl;
        ;
      }
    }
  }
}

auto Gravity::get_rhs(int crse_level, int nlevs, int is_new)
    -> Vector<std::unique_ptr<MultiFab>> {
  Vector<std::unique_ptr<MultiFab>> rhs(nlevs);

  for (int ilev = 0; ilev < nlevs; ++ilev) {
    int amr_lev = ilev + crse_level;
    rhs[ilev] = std::make_unique<MultiFab>(grids[amr_lev], dmap[amr_lev], 1, 0);
    MultiFab &state = (is_new == 1)
                          ? LevelData[amr_lev]->get_new_data(State_Type)
                          : LevelData[amr_lev]->get_old_data(State_Type);
    MultiFab::Copy(*rhs[ilev], state, density_index, 0, 1, 0);
  }
  return rhs;
}

void Gravity::sanity_check(int level) {
  // This is a sanity check on whether we are trying to fill multipole boundary
  // conditions for grids at this level > 0 -- this case is not currently
  // supported. Here we shrink the domain at this level by 1 in any direction
  // which is not symmetry or periodic, then ask if the grids at this level are
  // contained in the shrunken domain.  If not, then grids at this level touch
  // the domain boundary and we must abort.

  const Geometry &geom = parent->Geom(level);

  if (level > 0 && !geom.isAllPeriodic()) {
    Box shrunk_domain(geom.Domain());
    for (int dir = 0; dir < AMREX_SPACEDIM; dir++) {
      if (!geom.isPeriodic(dir)) {
        if (phys_bc->lo(dir) != Symmetry) {
          shrunk_domain.growLo(dir, -1);
        }
        if (phys_bc->hi(dir) != Symmetry) {
          shrunk_domain.growHi(dir, -1);
        }
      }
    }
    if (!shrunk_domain.contains(grids[level].minimalBox())) {
      amrex::Error("Oops -- don't know how to set boundary conditions for "
                   "grids at this level that touch the domain boundary!");
    }
  }
}

void Gravity::update_max_rhs() {
  BL_PROFILE("Gravity::update_max_rhs()");

  // Calculate the maximum value of the RHS over all levels.
  // This should only be called at a synchronization point where
  // all Castro levels have valid new time data at the same simulation time.
  // The RHS we will use is the density multiplied by 4*pi*G and also
  // multiplied by the metric terms, just as it would be in a real solve.

  int crse_level = 0;
  int nlevs = parent->finestLevel() + 1;
  int is_new = 1;

  const auto &rhs = get_rhs(crse_level, nlevs, is_new);

  const Geometry &geom0 = parent->Geom(0);

  if (geom0.isAllPeriodic()) {
    for (int lev = 0; lev < nlevs; ++lev) {
      rhs[lev]->plus(-mass_offset, 0, 1, 0);
    }
  }

  for (int lev = 0; lev < nlevs; ++lev) {
    rhs[lev]->mult(Ggravity);
  }

  max_rhs = 0.0;

  for (int lev = 0; lev < nlevs; ++lev) {
    max_rhs = std::max(max_rhs, rhs[lev]->max(0));
  }
}

auto Gravity::solve_phi_with_mlmg(int crse_level, int fine_level,
                                  const Vector<MultiFab *> &phi,
                                  const Vector<MultiFab *> &rhs,
                                  const Vector<Vector<MultiFab *>> &grad_phi,
                                  const Vector<MultiFab *> &res, Real time)
    -> Real {
  BL_PROFILE("Gravity::solve_phi_with_mlmg()");

  int nlevs = fine_level - crse_level + 1;

  if (crse_level == 0 && !(parent->Geom(0).isAllPeriodic())) {
    if (gravity::verbose > 1) {
      amrex::Print() << " ... Making bc's for phi at level 0\n";
    }
    fill_multipole_BCs(crse_level, fine_level, rhs, *phi[0]);
  }

  for (int ilev = 0; ilev < nlevs; ++ilev) {
    rhs[ilev]->mult(Ggravity);
  }

  MultiFab CPhi;
  const MultiFab *crse_bcdata = nullptr;
  if (crse_level > 0) {
    GetCrsePhi(crse_level, CPhi, time);
    crse_bcdata = &CPhi;
  }

  Real rel_eps = rel_tol[fine_level];

  // The absolute tolerance is determined by the error tolerance
  // chosen by the user (tol) multiplied by the maximum value of
  // the RHS (4 * pi * G * rho). If we're doing periodic BCs, we
  // subtract off the mass_offset corresponding to the average
  // density on the domain. This will automatically be zero for
  // non-periodic BCs. And this also accounts for the metric
  // terms that are applied in non-Cartesian coordinates.

  Real abs_eps = abs_tol[fine_level] * max_rhs;

  Vector<const MultiFab *> crhs{rhs.begin(), rhs.end()};
  Vector<std::array<MultiFab *, AMREX_SPACEDIM>> gp;
  for (const auto &x : grad_phi) {
    gp.push_back({AMREX_D_DECL(x[0], x[1], x[2])});
  }

  return actual_solve_with_mlmg(crse_level, fine_level, phi, crhs, gp, res,
                                crse_bcdata, rel_eps, abs_eps);
}

void Gravity::solve_for_delta_phi(
    int crse_level, int fine_level, const Vector<MultiFab *> &rhs,
    const Vector<MultiFab *> &delta_phi,
    const Vector<Vector<MultiFab *>> &grad_delta_phi) {
  BL_PROFILE("Gravity::solve_for_delta_phi");

  BL_ASSERT(grad_delta_phi.size() == fine_level - crse_level + 1);
  BL_ASSERT(delta_phi.size() == fine_level - crse_level + 1);

  if (gravity::verbose > 1 && ParallelDescriptor::IOProcessor()) {
    std::cout << "... solving for delta_phi at crse_level = " << crse_level
              << std::endl;
    std::cout << "...                    up to fine_level = " << fine_level
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

auto Gravity::actual_solve_with_mlmg(
    int crse_level, int fine_level, const amrex::Vector<amrex::MultiFab *> &phi,
    const amrex::Vector<const amrex::MultiFab *> &rhs,
    const amrex::Vector<std::array<amrex::MultiFab *, AMREX_SPACEDIM>>
        &grad_phi,
    const amrex::Vector<amrex::MultiFab *> &res,
    const amrex::MultiFab *const crse_bcdata, amrex::Real rel_eps,
    amrex::Real abs_eps) const -> Real {
  BL_PROFILE("Gravity::actual_solve_with_mlmg()");

  Real final_resnorm = -1.0;

  int nlevs = fine_level - crse_level + 1;

  Vector<Geometry> gmv;
  Vector<BoxArray> bav;
  Vector<DistributionMapping> dmv;
  for (int ilev = 0; ilev < nlevs; ++ilev) {
    gmv.push_back(parent->Geom(ilev + crse_level));
    bav.push_back(rhs[ilev]->boxArray());
    dmv.push_back(rhs[ilev]->DistributionMap());
  }

  LPInfo info;
  info.setAgglomeration(gravity::mlmg_agglomeration != 0);
  info.setConsolidation(gravity::mlmg_consolidation != 0);

  MLPoisson mlpoisson(gmv, bav, dmv, info);

  // BC
  mlpoisson.setDomainBC(mlmg_lobc, mlmg_hibc);
  if (mlpoisson.needsCoarseDataForBC()) {
    mlpoisson.setCoarseFineBC(crse_bcdata, parent->refRatio(crse_level - 1)[0]);
  }

  for (int ilev = 0; ilev < nlevs; ++ilev) {
    mlpoisson.setLevelBC(ilev, phi[ilev]);
  }

  MLMG mlmg(mlpoisson);
  mlmg.setVerbose(gravity::verbose -
                  1); // With normal verbosity we don't want MLMG information
  if (crse_level == 0) {
    mlmg.setMaxFmgIter(gravity::mlmg_max_fmg_iter);
  } else {
    mlmg.setMaxFmgIter(0); // Vcycle
  }

  AMREX_ALWAYS_ASSERT(!grad_phi.empty() or !res.empty());
  AMREX_ALWAYS_ASSERT(grad_phi.empty() or res.empty());

  if (!grad_phi.empty()) {
    if (!gmv[0].isAllPeriodic()) {
      mlmg.setAlwaysUseBNorm(1);
    }

    mlmg.setNSolve(gravity::mlmg_nsolve);
    final_resnorm = mlmg.solve(phi, rhs, rel_eps, abs_eps);

    mlmg.getGradSolution(grad_phi);
  } else if (!res.empty()) {
    mlmg.compResidual(res, phi, rhs);
  }

  return final_resnorm;
}
