//==============================================================================
// Poisson gravity solver, adapted from Castro's gravity module:
//   Commit history:
//   https://github.com/AMReX-Astro/Castro/commits/main/Source/gravity/Gravity.cpp
// Used under the terms of the open-source license (BSD 3-clause) given here:
//   https://github.com/AMReX-Astro/Castro/blob/main/license.txt
//==============================================================================
/// \file gravity.cpp
/// \brief Implements a class for solving the Poisson equation.
///

#include <cmath>
#include <limits>

#include <AMReX_FillPatchUtil.H>
#include <AMReX_MLMG.H>
#include <AMReX_MLPoisson.H>
#include <AMReX_ParmParse.H>

#include <fundamental_constants.H>
#include <Gravity.H>
#include <Gravity_util.H>

#define MAX_LEV 15

using namespace amrex;

#ifdef AMREX_DEBUG
int Gravity::test_solves = 1;
#else
int Gravity::test_solves = 0;
#endif
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

Gravity::Gravity(Amr *Parent, int _finest_level, BCRec *_phys_bc, int _Density)
    : parent(Parent), LevelData(MAX_LEV), grad_phi_curr(MAX_LEV),
      grad_phi_prev(MAX_LEV), grids(Parent->boxArray()),
      dmap(Parent->DistributionMap()), abs_tol(MAX_LEV), rel_tol(MAX_LEV),
      level_solver_resnorm(MAX_LEV), volume(MAX_LEV), area(MAX_LEV),
      phys_bc(_phys_bc) {
  Density = _Density;
  read_params();
  finest_level_allocated = -1;

  if (gravity::gravity_type == "PoissonGrav")
    make_mg_bc();
  if (gravity::gravity_type == "PoissonGrav")
    init_multipole_grav();
  max_rhs = 0.0;
}

Gravity::~Gravity() {}

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

    if (pp.contains("get_g_from_phi") && !gravity::get_g_from_phi &&
        gravity::gravity_type == "PoissonGrav")
      if (ParallelDescriptor::IOProcessor())
        std::cout << "Warning: gravity::gravity_type = PoissonGrav assumes "
                     "get_g_from_phi is true"
                  << std::endl;

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

      Real tol;

      if (n_abs_tol == 1) {

        pp.get("abs_tol", tol);

      } else {

        if (dgeom.IsCartesian())
          tol = 1.e-11;
        else
          tol = 1.e-10;
      }

      abs_tol[0] = tol;

      // Account for the fact that on finer levels, the scale of the
      // Laplacian changes due to the zone size changing. We assume
      // dx == dy == dz, so it is fair to say that on each level the
      // tolerance should increase by the factor ref_ratio**2, since
      // in absolute terms the Laplacian increases by that ratio too.
      // The actual tolerance we'll send in is the effective tolerance
      // on the finest level that we solve for.

      for (int lev = 1; lev < nlevs; ++lev)
        abs_tol[lev] =
            abs_tol[lev - 1] * std::pow(parent->refRatio(lev - 1)[0], 2);

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
      Real tol;
      if (n_rel_tol == 1) {
        pp.get(rel_tol_name.c_str(), tol);
      } else {
        tol = 0.0;
      }
      for (int lev = 0; lev < MAX_LEV; ++lev)
        rel_tol[lev] = tol;

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

void Gravity::output_job_info_params(std::ostream &jobInfoFile) {}

void Gravity::set_numpts_in_gravity(int numpts) { numpts_at_level = numpts; }

void Gravity::install_level(int level, AmrLevel *level_data, MultiFab &_volume,
                            MultiFab *_area) {
  if (gravity::verbose > 1 && ParallelDescriptor::IOProcessor())
    std::cout << "Installing Gravity level " << level << '\n';

  LevelData[level] = level_data;

  volume[level] = &_volume;

  area[level] = _area;

  level_solver_resnorm[level] = 0.0;

  const Geometry &geom = level_data->Geom();

  if (gravity::gravity_type == "PoissonGrav") {

    const DistributionMapping &dm = level_data->DistributionMap();

    grad_phi_prev[level].resize(AMREX_SPACEDIM);
    for (int n = 0; n < AMREX_SPACEDIM; ++n)
      grad_phi_prev[level][n].reset(
          new MultiFab(level_data->getEdgeBoxArray(n), dm, 1, 1));

    grad_phi_curr[level].resize(AMREX_SPACEDIM);
    for (int n = 0; n < AMREX_SPACEDIM; ++n)
      grad_phi_curr[level][n].reset(
          new MultiFab(level_data->getEdgeBoxArray(n), dm, 1, 1));
  }

  finest_level_allocated = level;
}

std::string Gravity::get_gravity_type() { return gravity::gravity_type; }

int Gravity::get_max_solve_level() { return gravity::max_solve_level; }

int Gravity::NoSync() { return gravity::no_sync; }

int Gravity::NoComposite() { return gravity::no_composite; }

int Gravity::DoCompositeCorrection() {
  return gravity::do_composite_phi_correction;
}

int Gravity::test_results_of_solves() { return test_solves; }

Vector<std::unique_ptr<MultiFab>> &Gravity::get_grad_phi_prev(int level) {
  return grad_phi_prev[level];
}

MultiFab *Gravity::get_grad_phi_prev_comp(int level, int comp) {
  return grad_phi_prev[level][comp].get();
}

Vector<std::unique_ptr<MultiFab>> &Gravity::get_grad_phi_curr(int level) {
  return grad_phi_curr[level];
}

void Gravity::plus_grad_phi_curr(int level,
                                 Vector<std::unique_ptr<MultiFab>> &addend) {
  for (int n = 0; n < AMREX_SPACEDIM; n++)
    grad_phi_curr[level][n]->plus(*addend[n], 0, 1, 0);
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

  if (gravity::verbose > 1 && ParallelDescriptor::IOProcessor())
    std::cout << " ... solve for phi at level " << level << std::endl;

  const Real strt = ParallelDescriptor::second();

  if (is_new == 0)
    sanity_check(level);

  Real time;
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

  if (gravity::verbose) {
    const int IOProc = ParallelDescriptor::IOProcessorNumber();
    Real end = ParallelDescriptor::second() - strt;

#ifdef BL_LAZY
    Lazy::QueueReduction([=]() mutable {
#endif
      ParallelDescriptor::ReduceRealMax(end, IOProc);
      if (ParallelDescriptor::IOProcessor())
        std::cout << "Gravity::solve_for_phi() time = " << end << std::endl
                  << std::endl;
#ifdef BL_LAZY
    });
#endif
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
  } else {
    fine_level = amrex::min(fine_level, gravity::max_solve_level);
  }

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
    delta_phi[lev - crse_level].reset(
        new MultiFab(grids[lev], dmap[lev], 1, 1));
    delta_phi[lev - crse_level]->setVal(0.0);
  }

  Vector<Vector<std::unique_ptr<MultiFab>>> ec_gdPhi(nlevs);

  for (int lev = crse_level; lev <= fine_level; ++lev) {
    ec_gdPhi[lev - crse_level].resize(AMREX_SPACEDIM);

    const DistributionMapping &dm = LevelData[lev]->DistributionMap();
    for (int n = 0; n < AMREX_SPACEDIM; ++n) {
      ec_gdPhi[lev - crse_level][n].reset(
          new MultiFab(LevelData[lev]->getEdgeBoxArray(n), dm, 1, 0));
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
    rhs[lev - crse_level].reset(new MultiFab(
        LevelData[lev]->boxArray(), LevelData[lev]->DistributionMap(), 1, 0));
    MultiFab::Copy(*rhs[lev - crse_level], *dphi[lev - crse_level], 0, 0, 1, 0);
    rhs[lev - crse_level]->mult(1.0 / Ggravity);
    MultiFab::Add(*rhs[lev - crse_level], *drho[lev - crse_level], 0, 0, 1, 0);
  }

  // Construct the boundary conditions for the Poisson solve.

  if (crse_level == 0 && !crse_geom.isAllPeriodic()) {

    if (gravity::verbose > 1 && ParallelDescriptor::IOProcessor())
      std::cout << " ... Making bc's for delta_phi at crse_level 0"
                << std::endl;

    fill_multipole_BCs(crse_level, fine_level, amrex::GetVecOfPtrs(rhs),
                       *delta_phi[crse_level]);
  }

  // Restore the factor of (4 * pi * G) for the Poisson solve.
  for (int lev = crse_level; lev <= fine_level; ++lev)
    rhs[lev - crse_level]->mult(Ggravity);

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

    if (gravity::verbose > 1 && ParallelDescriptor::IOProcessor())
      std::cout << "WARNING: Adjusting RHS in gravity_sync solve by "
                << local_correction << '\n';

    for (int lev = fine_level; lev >= crse_level; --lev)
      rhs[lev - crse_level]->plus(-local_correction, 0, 1, 0);
  }

  // Do multi-level solve for delta_phi.

  solve_for_delta_phi(crse_level, fine_level, amrex::GetVecOfPtrs(rhs),
                      amrex::GetVecOfPtrs(delta_phi),
                      amrex::GetVecOfVecOfPtrs(ec_gdPhi));

  // In the all-periodic case we enforce that delta_phi averages to zero.

  if (crse_geom.isAllPeriodic() &&
      (grids[crse_level].numPts() == crse_domain.numPts())) {

    Real local_correction = delta_phi[0]->sum() / grids[crse_level].numPts();

    for (int lev = crse_level; lev <= fine_level; ++lev)
      delta_phi[lev - crse_level]->plus(-local_correction, 0, 1, 1);
  }

  // Add delta_phi to phi_new, and grad(delta_phi) to grad(delta_phi_curr) on
  // each level. Update the cell-centered gravity too.

  for (int lev = crse_level; lev <= fine_level; lev++) {

    LevelData[lev]
        ->get_new_data(PhiGrav_Type)
        .plus(*delta_phi[lev - crse_level], 0, 1, 0);

    for (int n = 0; n < AMREX_SPACEDIM; n++)
      grad_phi_curr[lev][n]->plus(*ec_gdPhi[lev - crse_level][n], 0, 1, 0);

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

  if (gravity::verbose > 1 && ParallelDescriptor::IOProcessor())
    std::cout << "... multilevel solve for new phi at base level " << level
              << " to finest level " << finest_level_in << std::endl;

  for (int lev = level; lev <= finest_level_in; lev++) {
    BL_ASSERT(grad_phi_curr[lev].size() == AMREX_SPACEDIM);
    for (int n = 0; n < AMREX_SPACEDIM; ++n) {
      grad_phi_curr[lev][n].reset(
          new MultiFab(LevelData[lev]->getEdgeBoxArray(n),
                       LevelData[lev]->DistributionMap(), 1, 1));
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

  for (int ilev = crse_level; ilev <= finest_level_in; ++ilev)
    sanity_check(ilev);

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

  Real time;
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
    for (int amr_lev = fine_level; amr_lev > crse_level; amr_lev--)
      average_fine_ec_onto_crse_ec(amr_lev - 1, is_new);
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

void Gravity::test_residual(const Box &bx, Array4<Real> const &rhs,
                            Array4<Real> const &ecx, Array4<Real> const &ecy,
                            Array4<Real> const &ecz,
                            GpuArray<Real, AMREX_SPACEDIM> dx,
                            GpuArray<Real, AMREX_SPACEDIM> problo,
                            int coord_type) {
  // Test whether using the edge-based gradients
  // to compute Div(Grad(Phi)) satisfies Lap(phi) = RHS
  // Fill the RHS array with the residual

  AMREX_ALWAYS_ASSERT(coord_type == 0);

  amrex::ParallelFor(bx, [=] AMREX_GPU_HOST_DEVICE(int i, int j, int k) {
    Real lapphi = (ecx(i + 1, j, k) - ecx(i, j, k)) / dx[0];
    lapphi += (ecy(i, j + 1, k) - ecy(i, j, k)) / dx[1];
    lapphi += (ecz(i, j, k + 1) - ecz(i, j, k)) / dx[2];
    rhs(i, j, k) -= lapphi;
  });
}

void Gravity::test_level_grad_phi_prev(int level) {
  BL_PROFILE("Gravity::test_level_grad_phi_prev()");

  // Fill the RHS for the solve
  MultiFab &S_old = LevelData[level]->get_old_data(State_Type);
  MultiFab Rhs(grids[level], dmap[level], 1, 0);
  MultiFab::Copy(Rhs, S_old, URHO, 0, 1, 0);

  const Geometry &geom = parent->Geom(level);

  // This is a correction for fully periodic domains only
  if (geom.isAllPeriodic()) {
    if (gravity::verbose > 1 && ParallelDescriptor::IOProcessor() &&
        mass_offset != 0.0)
      std::cout << " ... subtracting average density from RHS at level ... "
                << level << " " << mass_offset << std::endl;
    Rhs.plus(-mass_offset, 0, 1, 0);
  }

  Rhs.mult(Ggravity);

  if (gravity::verbose > 1) {
    Real rhsnorm = Rhs.norm0();
    amrex::Print() << "... test_level_grad_phi_prev at level " << level
                   << std::endl;
    amrex::Print() << "       norm of RHS             " << rhsnorm << std::endl;
  }

  auto dx = parent->Geom(level).CellSizeArray();
  auto problo = parent->Geom(level).ProbLoArray();
  const int coord_type = geom.Coord();

  for (MFIter mfi(Rhs, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
    const Box &bx = mfi.tilebox();

    test_residual(bx, Rhs.array(mfi), (*grad_phi_prev[level][0]).array(mfi),
                  (*grad_phi_prev[level][1]).array(mfi),
                  (*grad_phi_prev[level][2]).array(mfi), dx, problo,
                  coord_type);
  }

  if (gravity::verbose > 1) {
    Real resnorm = Rhs.norm0();
    amrex::Print() << "       norm of residual        " << resnorm << std::endl;
  }
}

void Gravity::test_level_grad_phi_curr(int level) {
  BL_PROFILE("Gravity::test_level_grad_phi_curr()");

  // Fill the RHS for the solve
  MultiFab &S_new = LevelData[level]->get_new_data(State_Type);
  MultiFab Rhs(grids[level], dmap[level], 1, 0);
  MultiFab::Copy(Rhs, S_new, URHO, 0, 1, 0);

  const Geometry &geom = parent->Geom(level);

  // This is a correction for fully periodic domains only
  if (geom.isAllPeriodic()) {
    if (gravity::verbose > 1 && ParallelDescriptor::IOProcessor() &&
        mass_offset != 0.0)
      std::cout << " ... subtracting average density from RHS in solve ... "
                << mass_offset << std::endl;
    Rhs.plus(-mass_offset, 0, 1, 0);
  }

  Rhs.mult(Ggravity);

  if (gravity::verbose > 1) {
    Real rhsnorm = Rhs.norm0();
    if (ParallelDescriptor::IOProcessor()) {
      std::cout << "... test_level_grad_phi_curr at level " << level
                << std::endl;
      std::cout << "       norm of RHS             " << rhsnorm << std::endl;
    }
  }

  auto dx = geom.CellSizeArray();
  auto problo = geom.ProbLoArray();
  const int coord_type = geom.Coord();

  for (MFIter mfi(Rhs, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
    const Box &bx = mfi.tilebox();

    test_residual(bx, Rhs.array(mfi), (*grad_phi_curr[level][0]).array(mfi),
#if AMREX_SPACEDIM >= 2
                  (*grad_phi_curr[level][1]).array(mfi),
#endif
#if AMREX_SPACEDIM == 3
                  (*grad_phi_curr[level][2]).array(mfi),
#endif
                  dx, problo, coord_type);
  }

  if (gravity::verbose > 1) {
    Real resnorm = Rhs.norm0();
    amrex::Print() << "       norm of residual        " << resnorm << std::endl;
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
    comp_minus_level_grad_phi[n].reset(
        new MultiFab(LevelData[level]->getEdgeBoxArray(n),
                     LevelData[level]->DistributionMap(), 1, 0));
    MultiFab::Copy(*comp_minus_level_grad_phi[n], *comp_gphi[n], 0, 0, 1, 0);
    comp_minus_level_grad_phi[n]->minus(*grad_phi_prev[level][n], 0, 1, 0);
  }
}

void Gravity::average_fine_ec_onto_crse_ec(int level, int is_new) {
  BL_PROFILE("Gravity::average_fine_ec_onto_crse_ec()");

  // NOTE: this is called with level == the coarser of the two levels involved
  if (level == parent->finestLevel())
    return;

  //
  // Coarsen() the fine stuff on processors owning the fine data.
  //
  BoxArray crse_gphi_fine_BA(grids[level + 1].size());

  IntVect fine_ratio = parent->refRatio(level);

  for (int i = 0; i < crse_gphi_fine_BA.size(); ++i)
    crse_gphi_fine_BA.set(i, amrex::coarsen(grids[level + 1][i], fine_ratio));

  Vector<std::unique_ptr<MultiFab>> crse_gphi_fine(AMREX_SPACEDIM);
  for (int n = 0; n < AMREX_SPACEDIM; ++n) {
    BoxArray eba = crse_gphi_fine_BA;
    eba.surroundingNodes(n);
    crse_gphi_fine[n].reset(new MultiFab(eba, dmap[level + 1], 1, 0));
  }

  auto &grad_phi = (is_new) ? grad_phi_curr : grad_phi_prev;

  amrex::average_down_faces(amrex::GetVecOfConstPtrs(grad_phi[level + 1]),
                            amrex::GetVecOfPtrs(crse_gphi_fine), fine_ratio);

  const Geometry &cgeom = parent->Geom(level);

  for (int n = 0; n < AMREX_SPACEDIM; ++n) {
    grad_phi[level][n]->ParallelCopy(*crse_gphi_fine[n], cgeom.periodicity());
  }
}

void Gravity::test_composite_phi(int crse_level) {
  BL_PROFILE("Gravity::test_composite_phi()");

  if (gravity::verbose > 1 && ParallelDescriptor::IOProcessor()) {
    std::cout << "   " << '\n';
    std::cout << "... test_composite_phi at base level " << crse_level << '\n';
  }

  int finest_level_local = parent->finestLevel();
  int nlevels = finest_level_local - crse_level + 1;

  Vector<std::unique_ptr<MultiFab>> phi(nlevels);
  Vector<std::unique_ptr<MultiFab>> rhs(nlevels);
  Vector<std::unique_ptr<MultiFab>> res(nlevels);
  for (int ilev = 0; ilev < nlevels; ++ilev) {
    int amr_lev = crse_level + ilev;

    phi[ilev].reset(new MultiFab(grids[amr_lev], dmap[amr_lev], 1, 1));
    MultiFab::Copy(*phi[ilev], LevelData[amr_lev]->get_new_data(PhiGrav_Type),
                   0, 0, 1, 1);

    rhs[ilev].reset(new MultiFab(grids[amr_lev], dmap[amr_lev], 1, 1));
    MultiFab::Copy(*rhs[ilev], LevelData[amr_lev]->get_new_data(State_Type),
                   URHO, 0, 1, 0);

    res[ilev].reset(new MultiFab(grids[amr_lev], dmap[amr_lev], 1, 0));
    res[ilev]->setVal(0.);
  }

  Real time = LevelData[crse_level]->get_state_data(PhiGrav_Type).curTime();

  Vector<Vector<MultiFab *>> grad_phi_null;
  solve_phi_with_mlmg(crse_level, finest_level_local, amrex::GetVecOfPtrs(phi),
                      amrex::GetVecOfPtrs(rhs), grad_phi_null,
                      amrex::GetVecOfPtrs(res), time);

  // Average residual from fine to coarse level before printing the norm
  for (int amr_lev = finest_level_local - 1; amr_lev >= 0; --amr_lev) {
    const IntVect &ratio = parent->refRatio(amr_lev);
    int ilev = amr_lev - crse_level;
    amrex::average_down(*res[ilev + 1], *res[ilev], 0, 1, ratio);
  }

  for (int amr_lev = crse_level; amr_lev <= finest_level_local; ++amr_lev) {
    Real resnorm = res[amr_lev]->norm0();
    if (ParallelDescriptor::IOProcessor()) {
      std::cout << "      ... norm of composite residual at level " << amr_lev
                << "  " << resnorm << '\n';
    }
  }
  if (ParallelDescriptor::IOProcessor())
    std::cout << std::endl;
}

void Gravity::init_multipole_grav() {
  if (gravity::lnum < 0) {
    amrex::Abort("lnum negative");
  }

  if (gravity::lnum > multipole::lnum_max) {
    amrex::Abort("lnum greater than lnum_max");
  }

  int lo_bc[3];
  int hi_bc[3];

  for (int dir = 0; dir < AMREX_SPACEDIM; dir++) {
    lo_bc[dir] = phys_bc->lo(dir);
    hi_bc[dir] = phys_bc->hi(dir);
  }
  for (int dir = AMREX_SPACEDIM; dir < 3; dir++) {
    lo_bc[dir] = -1;
    hi_bc[dir] = -1;
  }

  const auto problo = parent->Geom(0).ProbLoArray();
  const auto probhi = parent->Geom(0).ProbHiArray();

  // If any of the boundaries are symmetric, we need to account for the mass
  // that is assumed to lie on the opposite side of the symmetric axis. If the
  // center in any direction coincides with the boundary, then we can simply
  // double the mass as a result of that reflection. Otherwise, we need to do a
  // more general solve. We include a logical that is set to true if any
  // boundary is symmetric, so that we can avoid unnecessary function calls.

  multipole::volumeFactor = 1.0;
  multipole::parityFactor = 1.0;

  multipole::doSymmetricAdd = false;

  for (int n = 0; n < 3; ++n) {
    multipole::doSymmetricAddLo(n) = false;
    multipole::doSymmetricAddHi(n) = false;

    multipole::doReflectionLo(n) = false;
    multipole::doReflectionHi(n) = false;
  }

  const Real edgeTolerance = 1.0e-2;

  for (int b = 0; b < AMREX_SPACEDIM; ++b) {

    if ((lo_bc[b] == Symmetry) && (parent->Geom(0).Coord() == 0)) {
      if (std::abs(problem::center[b] - problo[b]) < edgeTolerance) {
        multipole::volumeFactor *= 2.0;
        multipole::doReflectionLo(b) = true;
      } else {
        multipole::doSymmetricAddLo(b) = true;
        multipole::doSymmetricAdd = true;
      }
    }

    if ((hi_bc[b] == Symmetry) && (parent->Geom(0).Coord() == 0)) {
      if (std::abs(problem::center[b] - probhi[b]) < edgeTolerance) {
        multipole::volumeFactor *= 2.0;
        multipole::doReflectionHi(b) = true;
      } else {
        multipole::doSymmetricAddHi(b) = true;
        multipole::doSymmetricAdd = true;
      }
    }
  }

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
      if (AMREX_SPACEDIM == 3 &&
          (multipole::doReflectionLo(2) || multipole::doReflectionHi(2))) {
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

  // Now let's take care of a safety issue. The multipole calculation involves
  // taking powers of r^l, which can overflow the floating point exponent limit
  // if lnum is very large. Therefore, we will normalize all distances to the
  // maximum possible physical distance from the center, which is the diagonal
  // from the center to the edge of the box. Then r^l will always be less than
  // or equal to one. For large enough lnum, this may still result in roundoff
  // errors that don't make your answer any more precise, but at least it avoids
  // possible NaN issues from having numbers that are too large for double
  // precision. We will put the rmax factor back in at the end of
  // ca_put_multipole_phi.

  Real maxWidth = probhi[0] - problo[0];
  maxWidth = amrex::max(maxWidth, probhi[1] - problo[1]);
  maxWidth = amrex::max(maxWidth, probhi[2] - problo[2]);

  multipole::rmax =
      0.5 * maxWidth * std::sqrt(static_cast<Real>(AMREX_SPACEDIM));
}

void Gravity::fill_multipole_BCs(int crse_level, int fine_level,
                                 const Vector<MultiFab *> &Rhs, MultiFab &phi) {
  BL_PROFILE("Gravity::fill_multipole_BCs()");

  // Multipole BCs only make sense to construct if we are starting from the
  // coarse level.

  BL_ASSERT(crse_level == 0);

  BL_ASSERT(gravity::lnum >= 0);

  const Real strt = ParallelDescriptor::second();
  const int npts = numpts_at_level;

  // Storage arrays for the multipole moments.
  // We will initialize them to zero, and then
  // sum up the results over grids.
  // Note that since Boxes are defined with
  // AMREX_SPACEDIM dimensions, we cannot presently
  // use this array to fill the interior of the
  // domain in 2D, since we can only have one
  // radial index for calculating the multipole moments.

  Box boxq0(IntVect(D_DECL(0, 0, 0)),
            IntVect(D_DECL(gravity::lnum, 0, npts - 1)));
  Box boxqC(IntVect(D_DECL(0, 0, 0)),
            IntVect(D_DECL(gravity::lnum, gravity::lnum, npts - 1)));
  Box boxqS(IntVect(D_DECL(0, 0, 0)),
            IntVect(D_DECL(gravity::lnum, gravity::lnum, npts - 1)));

  FArrayBox qL0(boxq0);
  FArrayBox qLC(boxqC);
  FArrayBox qLS(boxqS);

  FArrayBox qU0(boxq0);
  FArrayBox qUC(boxqC);
  FArrayBox qUS(boxqS);

  qL0.setVal<RunOn::Device>(0.0);
  qLC.setVal<RunOn::Device>(0.0);
  qLS.setVal<RunOn::Device>(0.0);
  qU0.setVal<RunOn::Device>(0.0);
  qUC.setVal<RunOn::Device>(0.0);
  qUS.setVal<RunOn::Device>(0.0);

  // This section needs to be generalized for computing
  // full multipole gravity, not just BCs. At present this
  // does nothing.
  int boundary_only = 1;

  // Use all available data in constructing the boundary conditions,
  // unless the user has indicated that a maximum level at which
  // to stop using the more accurate data.

  for (int lev = crse_level; lev <= fine_level; ++lev) {

    // Create a local copy of the RHS so that we can mask it.

    MultiFab source(Rhs[lev - crse_level]->boxArray(),
                    Rhs[lev - crse_level]->DistributionMap(), 1, 0);

    MultiFab::Copy(source, *Rhs[lev - crse_level], 0, 0, 1, 0);

    if (lev < fine_level) {
      const MultiFab &mask =
          dynamic_cast<Castro *>(&(parent->getLevel(lev + 1)))
              ->build_fine_mask();
      MultiFab::Multiply(source, mask, 0, 0, 1, 0);
    }

    // Loop through the grids and compute the individual contributions
    // to the various moments. The multipole moment constructor
    // is coded to only add to the moment arrays, so it is safe
    // to directly hand the arrays to them.

    const Box &domain = parent->Geom(lev).Domain();
    const auto dx = parent->Geom(lev).CellSizeArray();
    const auto problo = parent->Geom(lev).ProbLoArray();
    const auto probhi = parent->Geom(lev).ProbHiArray();
    int coord_type = parent->Geom(lev).Coord();

    {
      for (MFIter mfi(source, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const Box &bx = mfi.tilebox();

        auto qL0_arr = qL0.array();
        auto qLC_arr = qLC.array();
        auto qLS_arr = qLS.array();
        auto qU0_arr = qU0.array();
        auto qUC_arr = qUC.array();
        auto qUS_arr = qUS.array();

        auto rho = source[mfi].array();
        auto vol = (*volume[lev])[mfi].array();

        amrex::ParallelFor(
            amrex::Gpu::KernelInfo().setReduction(true), bx,
            [=] AMREX_GPU_DEVICE(int i, int j, int k,
                                 amrex::Gpu::Handler const &handler) {
              // If we're using this to construct boundary values, then only
              // fill the outermost bin.

              int nlo = 0;
              if (boundary_only == 1) {
                nlo = npts - 1;
              }

              // Note that we don't currently support dx != dy != dz, so this is
              // acceptable.

              Real drInv = multipole::rmax / dx[0];

              Real rmax_cubed_inv =
                  1.0 / (multipole::rmax * multipole::rmax * multipole::rmax);

              Real x = (problo[0] + (static_cast<Real>(i) + 0.5) * dx[0] -
                        problem::center[0]) /
                       multipole::rmax;

              Real y = (problo[1] + (static_cast<Real>(j) + 0.5) * dx[1] -
                        problem::center[1]) /
                       multipole::rmax;

              Real z = (problo[2] + (static_cast<Real>(k) + 0.5) * dx[2] -
                        problem::center[2]) /
                       multipole::rmax;

              Real r = std::sqrt(x * x + y * y + z * z);

              Real cosTheta, phiAngle;
              int index;

              if (AMREX_SPACEDIM == 3) {
                index = static_cast<int>(r * drInv);
                cosTheta = z / r;
                phiAngle = std::atan2(y, x);
              } else if (AMREX_SPACEDIM == 2 && coord_type == 1) {
                index = nlo; // We only do the boundary potential in 2D.
                cosTheta = y / r;
                phiAngle = z;
              } else if (AMREX_SPACEDIM == 1 && coord_type == 2) {
                index = nlo; // We only do the boundary potential in 1D.
                cosTheta = 1.0;
                phiAngle = 0.0;
              }

              // Now, compute the multipole moments.

              multipole_add(cosTheta, phiAngle, r, rho(i, j, k),
                            vol(i, j, k) * rmax_cubed_inv, qL0_arr, qLC_arr,
                            qLS_arr, qU0_arr, qUC_arr, qUS_arr, npts, nlo,
                            index, handler, true);

              // Now add in contributions if we have any symmetric boundaries in
              // 3D. The symmetric boundary in 2D axisymmetric is handled
              // separately.

              if (multipole::doSymmetricAdd) {

                multipole_symmetric_add(x, y, z, problo, probhi, rho(i, j, k),
                                        vol(i, j, k) * rmax_cubed_inv, qL0_arr,
                                        qLC_arr, qLS_arr, qU0_arr, qUC_arr,
                                        qUS_arr, npts, nlo, index, handler);
              }
            });
      }
    }

  } // end loop over levels

  // Now, do a global reduce over all processes.

  if (!ParallelDescriptor::UseGpuAwareMpi()) {
    qL0.prefetchToHost();
    qLC.prefetchToHost();
    qLS.prefetchToHost();
  }

  ParallelDescriptor::ReduceRealSum(qL0.dataPtr(), boxq0.numPts());
  ParallelDescriptor::ReduceRealSum(qLC.dataPtr(), boxqC.numPts());
  ParallelDescriptor::ReduceRealSum(qLS.dataPtr(), boxqS.numPts());

  if (!ParallelDescriptor::UseGpuAwareMpi()) {
    qL0.prefetchToDevice();
    qLC.prefetchToDevice();
    qLS.prefetchToDevice();
  }

  if (boundary_only != 1) {

    if (!ParallelDescriptor::UseGpuAwareMpi()) {
      qU0.prefetchToHost();
      qUC.prefetchToHost();
      qUS.prefetchToHost();
    }

    ParallelDescriptor::ReduceRealSum(qU0.dataPtr(), boxq0.numPts());
    ParallelDescriptor::ReduceRealSum(qUC.dataPtr(), boxqC.numPts());
    ParallelDescriptor::ReduceRealSum(qUS.dataPtr(), boxqS.numPts());

    if (!ParallelDescriptor::UseGpuAwareMpi()) {
      qU0.prefetchToDevice();
      qUC.prefetchToDevice();
      qUS.prefetchToDevice();
    }
  }

  // Finally, construct the boundary conditions using the
  // complete multipole moments, for all points on the
  // boundary that are held on this process.

  const Box &domain = parent->Geom(crse_level).Domain();
  const auto dx = parent->Geom(crse_level).CellSizeArray();
  const auto problo = parent->Geom(crse_level).ProbLoArray();
  int coord_type = parent->Geom(crse_level).Coord();

  for (MFIter mfi(phi, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
    const Box &bx = mfi.growntilebox();

    auto qL0_arr = qL0.array();
    auto qLC_arr = qLC.array();
    auto qLS_arr = qLS.array();
    auto qU0_arr = qU0.array();
    auto qUC_arr = qUC.array();
    auto qUS_arr = qUS.array();
    auto phi_arr = phi[mfi].array();

    amrex::ParallelFor(bx, [=] AMREX_GPU_HOST_DEVICE(int i, int j, int k) {
      const int *domlo = domain.loVect();
      const int *domhi = domain.hiVect();

      // If we're using this to construct boundary values, then only use
      // the outermost bin.

      int nlo = 0;
      if (boundary_only == 1) {
        nlo = npts - 1;
      }

      Real rmax_cubed = multipole::rmax * multipole::rmax * multipole::rmax;

      Real x;
      if (i > domhi[0]) {
        x = problo[0] + (static_cast<Real>(i)) * dx[0] - problem::center[0];
      } else if (i < domlo[0]) {
        x = problo[0] + (static_cast<Real>(i + 1)) * dx[0] - problem::center[0];
      } else {
        x = problo[0] + (static_cast<Real>(i) + 0.5) * dx[0] -
            problem::center[0];
      }

      x = x / multipole::rmax;

      Real y;
      if (j > domhi[1]) {
        y = problo[1] + (static_cast<Real>(j)) * dx[1] - problem::center[1];
      } else if (j < domlo[1]) {
        y = problo[1] + (static_cast<Real>(j + 1)) * dx[1] - problem::center[1];
      } else {
        y = problo[1] + (static_cast<Real>(j) + 0.5) * dx[1] -
            problem::center[1];
      }

      y = y / multipole::rmax;

      Real z;
      if (k > domhi[2]) {
        z = problo[2] + (static_cast<Real>(k)) * dx[2] - problem::center[2];
      } else if (k < domlo[2]) {
        z = problo[2] + (static_cast<Real>(k + 1)) * dx[2] - problem::center[2];
      } else {
        z = problo[2] + (static_cast<Real>(k) + 0.5) * dx[2] -
            problem::center[2];
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

        Real cosTheta, phiAngle;
        if (AMREX_SPACEDIM == 3) {
          cosTheta = z / r;
          phiAngle = std::atan2(y, x);
        } else if (AMREX_SPACEDIM == 2 && coord_type == 1) {
          cosTheta = y / r;
          phiAngle = 0.0;
        }

        phi_arr(i, j, k) = 0.0;

        // Compute the potentials on the ghost cells.

        Real legPolyL, legPolyL1, legPolyL2;
        Real assocLegPolyLM, assocLegPolyLM1, assocLegPolyLM2;

        for (int n = nlo; n <= npts - 1; ++n) {

          for (int l = 0; l <= gravity::lnum; ++l) {

            calcLegPolyL(l, legPolyL, legPolyL1, legPolyL2, cosTheta);

            Real r_U = std::pow(r, -l - 1);

            // Make sure we undo the volume scaling here.

            phi_arr(i, j, k) += qL0_arr(l, 0, n) * legPolyL * r_U * rmax_cubed;
          }

          for (int m = 1; m <= gravity::lnum; ++m) {
            for (int l = 1; l <= gravity::lnum; ++l) {

              if (m > l)
                continue;

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

  if (gravity::verbose) {
    const int IOProc = ParallelDescriptor::IOProcessorNumber();
    Real end = ParallelDescriptor::second() - strt;

#ifdef BL_LAZY
    Lazy::QueueReduction([=]() mutable {
#endif
      ParallelDescriptor::ReduceRealMax(end, IOProc);
      if (ParallelDescriptor::IOProcessor())
        std::cout << "Gravity::fill_multipole_BCs() time = " << end << std::endl
                  << std::endl;
#ifdef BL_LAZY
    });
#endif
  }
}

void Gravity::make_mg_bc() {
  const Geometry &geom = parent->Geom(0);

  for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
    if (geom.isPeriodic(idim)) {
      mlmg_lobc[idim] = MLLinOp::BCType::Periodic;
      mlmg_hibc[idim] = MLLinOp::BCType::Periodic;
    } else {
      if (phys_bc->lo(idim) == Symmetry) {
        mlmg_lobc[idim] = MLLinOp::BCType::Neumann;
      } else {
        mlmg_lobc[idim] = MLLinOp::BCType::Dirichlet;
      }
      if (phys_bc->hi(idim) == Symmetry) {
        mlmg_hibc[idim] = MLLinOp::BCType::Neumann;
      } else {
        mlmg_hibc[idim] = MLLinOp::BCType::Dirichlet;
      }
    }
  }
}

void Gravity::set_mass_offset(Real time, bool multi_level) {
  BL_PROFILE("Gravity::set_mass_offset()");

  const Geometry &geom = parent->Geom(0);

  if (!geom.isAllPeriodic()) {
    mass_offset = 0.0;
  } else {
    Real old_mass_offset = mass_offset;
    mass_offset = 0.0;

    if (multi_level) {
      for (int lev = 0; lev <= parent->finestLevel(); lev++) {
        Castro *cs = dynamic_cast<Castro *>(&parent->getLevel(lev));
        mass_offset += cs->volWgtSum("density", time);
      }
    } else {
      Castro *cs = dynamic_cast<Castro *>(&parent->getLevel(0));
      mass_offset = cs->volWgtSum("density", time, false,
                                  false); // do not mask off fine grids
    }

    mass_offset = mass_offset / geom.ProbSize();
    if (gravity::verbose > 1 && ParallelDescriptor::IOProcessor())
      std::cout << "Defining average density to be " << mass_offset
                << std::endl;

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

Vector<std::unique_ptr<MultiFab>> Gravity::get_rhs(int crse_level, int nlevs,
                                                   int is_new) {
  Vector<std::unique_ptr<MultiFab>> rhs(nlevs);

  for (int ilev = 0; ilev < nlevs; ++ilev) {
    int amr_lev = ilev + crse_level;
    rhs[ilev].reset(new MultiFab(grids[amr_lev], dmap[amr_lev], 1, 0));
    MultiFab &state = (is_new == 1)
                          ? LevelData[amr_lev]->get_new_data(State_Type)
                          : LevelData[amr_lev]->get_old_data(State_Type);
    MultiFab::Copy(*rhs[ilev], state, URHO, 0, 1, 0);
  }
  return rhs;
}

void Gravity::sanity_check(int level) {
  // This is a sanity check on whether we are trying to fill multipole boundary
  // conditiosn
  //  for grids at this level > 0 -- this case is not currently supported.
  //  Here we shrink the domain at this level by 1 in any direction which is not
  //  symmetry or periodic, then ask if the grids at this level are contained in
  //  the shrunken domain.  If not, then grids at this level touch the domain
  //  boundary and we must abort.

  const Geometry &geom = parent->Geom(level);

  if (level > 0 && !geom.isAllPeriodic()) {
    Box shrunk_domain(geom.Domain());
    for (int dir = 0; dir < AMREX_SPACEDIM; dir++) {
      if (!geom.isPeriodic(dir)) {
        if (phys_bc->lo(dir) != Symmetry)
          shrunk_domain.growLo(dir, -1);
        if (phys_bc->hi(dir) != Symmetry)
          shrunk_domain.growHi(dir, -1);
      }
    }
    if (!shrunk_domain.contains(grids[level].minimalBox()))
      amrex::Error("Oops -- don't know how to set boundary conditions for "
                   "grids at this level that touch the domain boundary!");
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
    for (int lev = 0; lev < nlevs; ++lev)
      rhs[lev]->plus(-mass_offset, 0, 1, 0);
  }

  for (int lev = 0; lev < nlevs; ++lev) {
    rhs[lev]->mult(Ggravity);
  }

  max_rhs = 0.0;

  for (int lev = 0; lev < nlevs; ++lev)
    max_rhs = std::max(max_rhs, rhs[lev]->max(0));
}

Real Gravity::solve_phi_with_mlmg(int crse_level, int fine_level,
                                  const Vector<MultiFab *> &phi,
                                  const Vector<MultiFab *> &rhs,
                                  const Vector<Vector<MultiFab *>> &grad_phi,
                                  const Vector<MultiFab *> &res, Real time) {
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

Real Gravity::actual_solve_with_mlmg(
    int crse_level, int fine_level, const amrex::Vector<amrex::MultiFab *> &phi,
    const amrex::Vector<const amrex::MultiFab *> &rhs,
    const amrex::Vector<std::array<amrex::MultiFab *, AMREX_SPACEDIM>>
        &grad_phi,
    const amrex::Vector<amrex::MultiFab *> &res,
    const amrex::MultiFab *const crse_bcdata, amrex::Real rel_eps,
    amrex::Real abs_eps) {
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
  info.setAgglomeration(gravity::mlmg_agglomeration);
  info.setConsolidation(gravity::mlmg_consolidation);

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
    if (!gmv[0].isAllPeriodic())
      mlmg.setAlwaysUseBNorm(true);

    mlmg.setNSolve(gravity::mlmg_nsolve);
    final_resnorm = mlmg.solve(phi, rhs, rel_eps, abs_eps);

    mlmg.getGradSolution(grad_phi);
  } else if (!res.empty()) {
    mlmg.compResidual(res, phi, rhs);
  }

  return final_resnorm;
}
