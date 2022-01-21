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
/// \file Gravity_impl.hpp
/// \brief Implements a class for solving the Poisson equation for 3D, Cartesian
/// geometry problems.
///

#include "AMReX_BLassert.H"
#include "Gravity.hpp"

template <typename T> Real Gravity<T>::mass_offset = 0.0;
template <typename T> Real Gravity<T>::Ggravity = 0.;

template <typename T>
Gravity<T>::Gravity(AMRSimulation<T> *_sim, BCRec &_phys_bc,
                    GpuArray<Real, AMREX_SPACEDIM> &_coordCenter, int Density_)
    : sim(_sim), phi_old_(_sim->maxLevel() + 1), phi_new_(_sim->maxLevel() + 1),
      g_old_(_sim->maxLevel() + 1), g_new_(_sim->maxLevel() + 1),
      grad_phi_curr(_sim->maxLevel() + 1), grad_phi_prev(_sim->maxLevel() + 1),
      corr_phi_(_sim->maxLevel() + 1), corr_grad_phi_(_sim->maxLevel() + 1),
      abs_tol(_sim->maxLevel() + 1), rel_tol(_sim->maxLevel() + 1),
      level_solver_resnorm(_sim->maxLevel() + 1), coordCenter(_coordCenter),
      max_lev(_sim->maxLevel()), phys_bc(&_phys_bc) {
  // test whether sim has been properly initialized
  for (int i = 0; i < sim->maxLevel(); ++i) {
    AMREX_ALWAYS_ASSERT(sim->boxArray(i).ixType().cellCentered());
    AMREX_ALWAYS_ASSERT(sim->Geom(i).IsCartesian());
  }

  Density = Density_; // index of density component
  read_params();
  finest_level_allocated = -1;
  make_mg_bc();

  if (gravity::gravity_type == GravityMode::Poisson) {
    init_multipole_grav();
    numpts.resize(max_lev + 1);
  }
  max_rhs = 0.0;
  // possibly uninitialized:
  //   mlmg_lobc, mlmg_hibc -- set by Gravity<T>::make_mg_bc() above
}

template <typename T> void Gravity<T>::read_params() {
  static bool done = false;

  if (!done) {
    static_assert(gravity::gravity_type == GravityMode::Constant ||
                  gravity::gravity_type == GravityMode::Poisson);

    int nlevs = max_lev + 1;

    // Allow run-time input of solver tolerance. If the user
    // provides no value, set a reasonable default value on the
    // coarse level, and then increase it by ref_ratio**2 as the
    // levels get finer to account for the change in the absolute
    // scale of the Laplacian. If the user provides one value, use
    // that on the coarse level, and increase it the same way for
    // the fine levels. If the user provides more than one value,
    // we expect them to provide one for every level, and we do
    // not apply the ref_ratio effect.

    ParmParse pp("gravity");

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
      // in absolute terms the Laplacian increases by that ratio.

      for (int lev = 1; lev < nlevs; ++lev) {
        abs_tol[lev] =
            abs_tol[lev - 1] * std::pow(sim->refRatio(lev - 1)[0], 2);
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

    std::string rel_tol_name = "rel_tol";
    int n_rel_tol = pp.countval(rel_tol_name.c_str());

    if (n_rel_tol <= 1) {
      Real tol = NAN;
      if (n_rel_tol == 1) {
        pp.get(rel_tol_name.c_str(), tol);
      } else {
        tol = 0.0;
      }
      for (int lev = 0; lev < max_lev; ++lev) {
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
    if (gravity::verbose > 1) {
      amrex::Print() << "Getting Gconst from constants: " << C::Gconst
                     << std::endl;
      amrex::Print() << "Using " << Ggravity << " for 4 pi G in Gravity.cpp "
                     << std::endl;
    }

    done = true;
  }
}

template <typename T> void Gravity<T>::set_numpts_in_gravity(int level) {
  AMREX_ASSERT(AMREX_SPACEDIM == 3);

  Box bx(sim->Geom(level).Domain());
  std::int64_t nx = bx.size()[0];
  std::int64_t ny = bx.size()[1];
  std::int64_t nz = bx.size()[2];
  Real ndiagsq = Real(nx * nx + ny * ny + nz * nz);
  numpts[level] = int(sqrt(ndiagsq)) + 2 * sim->nghost();
}

template <typename T> void Gravity<T>::install_level(int level) {
  if (gravity::verbose > 1) {
    amrex::Print() << "Installing Gravity level " << level << '\n';
  }

  // test whether sim has been properly initialized
  AMREX_ASSERT(sim->boxArray(level).ixType().cellCentered());
  AMREX_ASSERT(sim->Geom(level).IsCartesian());

  level_solver_resnorm[level] = 0.0;

  if (gravity::gravity_type == GravityMode::Poisson) {
    set_numpts_in_gravity(level);

    const DistributionMapping &dm = sim->DistributionMap(level);

    grad_phi_prev[level].resize(AMREX_SPACEDIM);
    for (int n = 0; n < AMREX_SPACEDIM; ++n) {
      grad_phi_prev[level][n] = std::make_unique<MultiFab>(
          amrex::convert(sim->boxArray(level), IntVect::TheDimensionVector(n)),
          dm, 1, 0);
      grad_phi_prev[level][n]->setVal(0.);
    }

    grad_phi_curr[level].resize(AMREX_SPACEDIM);
    for (int n = 0; n < AMREX_SPACEDIM; ++n) {
      grad_phi_curr[level][n] = std::make_unique<MultiFab>(
          amrex::convert(sim->boxArray(level), IntVect::TheDimensionVector(n)),
          dm, 1, 0);
      grad_phi_curr[level][n]->setVal(0.);
    }
  }

  finest_level_allocated = level;
}

template <typename T>
auto Gravity<T>::get_gravity_type() -> gravity::GravityMode {
  return gravity::gravity_type;
}

template <typename T> auto Gravity<T>::get_max_solve_level() -> int {
  return gravity::max_solve_level;
}

template <typename T> auto Gravity<T>::NoSync() -> int {
  return gravity::no_sync;
}

template <typename T> auto Gravity<T>::NoComposite() -> int {
  return gravity::no_composite;
}

template <typename T> auto Gravity<T>::DoCompositeCorrection() -> int {
  return gravity::do_composite_phi_correction;
}

// if true, check the residuals manually
template <typename T> auto Gravity<T>::test_results_of_solves() -> int {
  return test_solves;
}

template <typename T>
auto Gravity<T>::get_grad_phi_prev(int level)
    -> Vector<std::unique_ptr<MultiFab>> & {
  return grad_phi_prev[level];
}

template <typename T>
auto Gravity<T>::get_grad_phi_prev_comp(int level, int comp) -> MultiFab * {
  return grad_phi_prev[level][comp].get();
}

template <typename T>
auto Gravity<T>::get_grad_phi_curr(int level)
    -> Vector<std::unique_ptr<MultiFab>> & {
  return grad_phi_curr[level];
}

template <typename T>
void Gravity<T>::plus_grad_phi_curr(int level,
                                    Vector<std::unique_ptr<MultiFab>> &addend) {
  for (int n = 0; n < AMREX_SPACEDIM; n++) {
    grad_phi_curr[level][n]->plus(*addend[n], 0, 1, 0);
  }
}

template <typename T> void Gravity<T>::swapTimeLevels(int level) {
  BL_PROFILE("Gravity<T>::swapTimeLevels()");

  if (gravity::gravity_type == GravityMode::Poisson) {
    for (int n = 0; n < AMREX_SPACEDIM; n++) {
      std::swap(grad_phi_prev[level][n], grad_phi_curr[level][n]);
      grad_phi_curr[level][n]->setVal(1.e50);
    }
  }
}

template <typename T>
void Gravity<T>::solve_for_phi(int level, MultiFab &phi,
                               const Vector<MultiFab *> &grad_phi, int is_new) {
  BL_PROFILE("Gravity<T>::solve_for_phi()");

  if (gravity::verbose > 1) {
    amrex::Print() << " ... solve for phi at level " << level << std::endl;
  }

  const Real strt = amrex::ParallelDescriptor::second();

  if (is_new == 0) {
    sanity_check(level);
  }

  Real time = NAN;
  if (is_new == 1) {
    time = sim->tNew_[level];
  } else {
    time = sim->tOld_[level];
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
    sim->FillCoarsePatch(level, time, phi, phi_old_, phi_new_);
  }

  if (gravity::verbose != 0) {
    const int IOProc = amrex::ParallelDescriptor::IOProcessorNumber();
    Real end = amrex::ParallelDescriptor::second() - strt;
    amrex::ParallelDescriptor::ReduceRealMax(end, IOProc);
    amrex::Print() << "Gravity<T>::solve_for_phi() time = " << end << std::endl
                   << std::endl;
  }
}

// This is used to obtain boundary conditions at coarse-fine interfaces for
// level solves
template <typename T>
void Gravity<T>::GetCrsePhi(int level, MultiFab &phi_crse, Real time) {
  BL_PROFILE("Gravity<T>::GetCrsePhi()");

  AMREX_ASSERT(level != 0);

  const Real t_old = sim->tOld_[level - 1];
  const Real t_new = sim->tNew_[level - 1];
  Real alpha = (time - t_old) / (t_new - t_old);
  AMREX_ASSERT(alpha >= 0.);
  Real omalpha = 1.0 - alpha;

  MultiFab const &phi_old = phi_old_[level - 1];
  MultiFab const &phi_new = phi_new_[level - 1];

  phi_crse.clear();
  phi_crse.define(sim->boxArray(level - 1), sim->DistributionMap(level - 1), 1,
                  1);

  MultiFab::LinComb(phi_crse, alpha, phi_new, 0, omalpha, phi_old, 0, 0, 1, 1);

  const Geometry &geom = sim->Geom(level - 1);
  phi_crse.FillBoundary(geom.periodicity());
}

template <typename T>
void Gravity<T>::multilevel_solve_for_new_phi(int level, int finest_level_in) {
  BL_PROFILE("Gravity<T>::multilevel_solve_for_new_phi()");

  if (gravity::verbose > 1) {
    amrex::Print() << "... multilevel solve for new phi at base level " << level
                   << " to finest level " << finest_level_in << std::endl;
  }

  for (int lev = level; lev <= finest_level_in; lev++) {
    AMREX_ASSERT(grad_phi_curr[lev].size() == AMREX_SPACEDIM);
    for (int n = 0; n < AMREX_SPACEDIM; ++n) {
      grad_phi_curr[lev][n] = std::make_unique<MultiFab>(
          amrex::convert(sim->boxArray(lev), IntVect::TheDimensionVector(n)),
          sim->DistributionMap(lev), 1, 0);
    }
  }

  int is_new = 1;
  actual_multilevel_solve(level, finest_level_in,
                          amrex::GetVecOfVecOfPtrs(grad_phi_curr), is_new);
}

template <typename T>
void Gravity<T>::actual_multilevel_solve(
    int crse_level, int finest_level_in,
    const Vector<Vector<MultiFab *>> &grad_phi, int is_new) {
  BL_PROFILE("Gravity<T>::actual_multilevel_solve()");

  for (int ilev = crse_level; ilev <= finest_level_in; ++ilev) {
    sanity_check(ilev);
  }

  int nlevels = finest_level_in - crse_level + 1;

  Vector<MultiFab *> phi_p(nlevels);
  for (int ilev = 0; ilev < nlevels; ilev++) {
    int amr_lev = ilev + crse_level;
    if (is_new == 1) {
      phi_p[ilev] = &phi_new_[amr_lev];
    } else {
      phi_p[ilev] = &phi_old_[amr_lev];
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
    time = sim->tNew_[crse_level];
  } else {
    time = sim->tOld_[crse_level];
  }

  int fine_level = amrex::min(finest_level_in, gravity::max_solve_level);

  if (fine_level >= crse_level) {

    Vector<MultiFab *> res_null;
    solve_phi_with_mlmg(crse_level, fine_level, phi_p, amrex::GetVecOfPtrs(rhs),
                        grad_phi_p, res_null, time);

    // Average phi from fine to coarse level
    for (int amr_lev = fine_level; amr_lev > crse_level; amr_lev--) {
      const IntVect &ratio = sim->refRatio(amr_lev - 1);
      if (is_new == 1) {
        amrex::average_down(phi_new_[amr_lev], phi_new_[amr_lev - 1], 0, 1,
                            ratio);
      } else if (is_new == 0) {
        amrex::average_down(phi_old_[amr_lev], phi_old_[amr_lev - 1], 0, 1,
                            ratio);
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

      MultiFab &phi = phi_new_[amr_lev];

      sim->FillCoarsePatch(amr_lev, time, phi, phi_old_, phi_new_);

    } else {

      MultiFab &phi = phi_old_[amr_lev];

      sim->FillCoarsePatch(amr_lev, time, phi, phi_old_, phi_new_);
    }

    // Interpolate the grad_phi.

    // Instantiate a bare physical BC function for grad_phi. It doesn't do
    // anything since the fine levels for Poisson gravity do not touch the
    // physical boundary.

    GradPhiPhysBCFunct gp_phys_bc;

    // We need to use a interpolater that works with data on faces.

    amrex::Interpolater *gp_interp = &amrex::face_linear_interp;

    // (Will not do anything because we do not fill on physical boundaries.)

    Vector<BCRec> gp_bcs;
    BCRec dirichlet_bcs;
    for (int i = 0; i < AMREX_SPACEDIM; ++i) {
      dirichlet_bcs.setHi(i, amrex::BCType::ext_dir);
    }
    gp_bcs.push_back(dirichlet_bcs);

    for (int n = 0; n < AMREX_SPACEDIM; ++n) {
      amrex::InterpFromCoarseLevel(
          *grad_phi[amr_lev][n], time, *grad_phi[amr_lev - 1][n], 0, 0, 1,
          sim->Geom(amr_lev - 1), sim->Geom(amr_lev), gp_phys_bc, 0, gp_phys_bc,
          0, sim->refRatio(amr_lev - 1), gp_interp, gp_bcs, 0);
    }
  }
}

template <typename T>
void Gravity<T>::average_fine_ec_onto_crse_ec(int level, int is_new) {
  BL_PROFILE("Gravity<T>::average_fine_ec_onto_crse_ec()");

  // NOTE: this is called with level == the coarser of the two levels involved
  if (level == sim->finestLevel()) {
    return;
  }

  //
  // Coarsen() the fine stuff on processors owning the fine data.
  //
  amrex::BoxArray crse_gphi_fine_BA(sim->boxArray(level + 1).size());

  IntVect fine_ratio = sim->refRatio(level);

  for (int i = 0; i < crse_gphi_fine_BA.size(); ++i) {
    crse_gphi_fine_BA.set(
        i, amrex::coarsen(sim->boxArray(level + 1)[i], fine_ratio));
  }

  Vector<std::unique_ptr<MultiFab>> crse_gphi_fine(AMREX_SPACEDIM);
  for (int n = 0; n < AMREX_SPACEDIM; ++n) {
    amrex::BoxArray eba = crse_gphi_fine_BA;
    eba.surroundingNodes(n);
    crse_gphi_fine[n] =
        std::make_unique<MultiFab>(eba, sim->DistributionMap(level + 1), 1, 0);
  }

  auto &grad_phi = (is_new) != 0 ? grad_phi_curr : grad_phi_prev;

  amrex::average_down_faces(amrex::GetVecOfConstPtrs(grad_phi[level + 1]),
                            amrex::GetVecOfPtrs(crse_gphi_fine), fine_ratio);

  const Geometry &cgeom = sim->Geom(level);

  for (int n = 0; n < AMREX_SPACEDIM; ++n) {
    grad_phi[level][n]->ParallelCopy(*crse_gphi_fine[n], cgeom.periodicity());
  }
}

template <typename T>
auto Gravity<T>::get_rhs(int crse_level, int nlevs, int is_new)
    -> Vector<std::unique_ptr<MultiFab>> {
  Vector<std::unique_ptr<MultiFab>> rhs(nlevs);

  for (int ilev = 0; ilev < nlevs; ++ilev) {
    int amr_lev = ilev + crse_level;
    amrex::Print() << "creating rhs from amr_lev = " << amr_lev << std::endl;

    AMREX_ASSERT(sim->boxArray(amr_lev).ixType().cellCentered());
    rhs[ilev] = std::make_unique<MultiFab>(sim->boxArray(amr_lev),
                                           sim->DistributionMap(amr_lev), 1, 0);

    MultiFab &state = (is_new == 1) ? *(sim->getStateNew(amr_lev))
                                    : *(sim->getStateOld(amr_lev));
    MultiFab::Copy(*rhs[ilev], state, Density, 0, 1, 0);
  }
  return rhs;
}

template <typename T> void Gravity<T>::sanity_check(int level) {
  // This is a sanity check on whether we are trying to fill multipole boundary
  // conditions for grids at this level > 0 -- this case is not currently
  // supported. Here we shrink the domain at this level by 1 in any direction
  // which is not symmetry or periodic, then ask if the grids at this level are
  // contained in the shrunken domain.  If not, then grids at this level touch
  // the domain boundary and we must abort.

  const Geometry &geom = sim->Geom(level);

  if (level > 0 && !geom.isAllPeriodic()) {
    Box shrunk_domain(geom.Domain());
    for (int dir = 0; dir < AMREX_SPACEDIM; dir++) {
      if (!geom.isPeriodic(dir)) {
        shrunk_domain.growLo(dir, -1);
        shrunk_domain.growHi(dir, -1);
      }
    }
    if (!shrunk_domain.contains(sim->boxArray(level).minimalBox())) {
      amrex::Error("Oops -- don't know how to set boundary conditions for "
                   "grids at this level that touch the domain boundary!");
    }
  }
}

template <typename T> void Gravity<T>::update_max_rhs() {
  BL_PROFILE("Gravity<T>::update_max_rhs()");

  // Calculate the maximum value of the RHS over all levels.
  // This should only be called at a synchronization point where
  // all Quokka levels have valid new time data at the same simulation time.
  // (The RHS is the density multiplied by 4*pi*G.)

  int crse_level = 0;
  int nlevs = sim->finestLevel() + 1;
  int is_new = 1;

  const auto &rhs = get_rhs(crse_level, nlevs, is_new);

  const Geometry &geom0 = sim->Geom(0);

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

template <typename T>
auto Gravity<T>::solve_phi_with_mlmg(int crse_level, int fine_level,
                                     const Vector<MultiFab *> &phi,
                                     const Vector<MultiFab *> &rhs,
                                     const Vector<Vector<MultiFab *>> &grad_phi,
                                     const Vector<MultiFab *> &res, Real time)
    -> Real {
  BL_PROFILE("Gravity<T>::solve_phi_with_mlmg()");

  int nlevs = fine_level - crse_level + 1;

  if (crse_level == 0 && !(sim->Geom(0).isAllPeriodic())) {
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
  // density on the domain.

  Real abs_eps = abs_tol[fine_level] * max_rhs;

  Vector<const MultiFab *> crhs{rhs.begin(), rhs.end()};

  Vector<std::array<MultiFab *, AMREX_SPACEDIM>> gp;
  for (const auto &x : grad_phi) {
    gp.push_back({AMREX_D_DECL(x[0], x[1], x[2])});
  }

  return actual_solve_with_mlmg(crse_level, fine_level, phi, crhs, gp, res,
                                crse_bcdata, rel_eps, abs_eps);
}

template <typename T>
auto Gravity<T>::actual_solve_with_mlmg(
    int crse_level, int fine_level, const amrex::Vector<amrex::MultiFab *> &phi,
    const amrex::Vector<const amrex::MultiFab *> &rhs,
    const amrex::Vector<std::array<amrex::MultiFab *, AMREX_SPACEDIM>>
        &grad_phi,
    const amrex::Vector<amrex::MultiFab *> &res,
    const amrex::MultiFab *const crse_bcdata, amrex::Real rel_eps,
    amrex::Real abs_eps) const -> Real {
  BL_PROFILE("Gravity<T>::actual_solve_with_mlmg()");

  Real final_resnorm = -1.0;
  int nlevs = fine_level - crse_level + 1;

  Vector<Geometry> gmv;
  Vector<BoxArray> bav;
  Vector<DistributionMapping> dmv;
  for (int ilev = 0; ilev < nlevs; ++ilev) {
    gmv.push_back(sim->Geom(ilev + crse_level));
    bav.push_back(rhs[ilev]->boxArray());
    dmv.push_back(rhs[ilev]->DistributionMap());
  }

  amrex::LPInfo info;
  info.setAgglomeration(gravity::mlmg_agglomeration != 0);
  info.setConsolidation(gravity::mlmg_consolidation != 0);

  amrex::MLPoisson mlpoisson(gmv, bav, dmv, info);

  // BC
  mlpoisson.setDomainBC(mlmg_lobc, mlmg_hibc);
  if (mlpoisson.needsCoarseDataForBC()) {
    mlpoisson.setCoarseFineBC(crse_bcdata, sim->refRatio(crse_level - 1)[0]);
  }

  for (int ilev = 0; ilev < nlevs; ++ilev) {
    mlpoisson.setLevelBC(ilev, phi[ilev]);
  }

  amrex::MLMG mlmg(mlpoisson);
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

    mlmg.setNSolve(gravity::mlmg_nsolve); // what does this do??
    final_resnorm = mlmg.solve(phi, rhs, rel_eps, abs_eps);

    mlmg.getGradSolution(grad_phi);
  } else if (!res.empty()) {
    mlmg.compResidual(res, phi, rhs);
  }

  return final_resnorm;
}
