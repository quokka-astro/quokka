#ifndef GRAVITY_HPP_
#define GRAVITY_HPP_
//==============================================================================
// Poisson gravity solver, adapted from Castro's gravity module:
//   Commit history:
//   https://github.com/AMReX-Astro/Castro/commits/main/Source/gravity/Gravity.H
// Used under the terms of the open-source license (BSD 3-clause) given here:
//   https://github.com/AMReX-Astro/Castro/blob/main/license.txt
//==============================================================================
/// \file gravity.hpp
/// \brief Defines a class for solving the Poisson equation.
///

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>

#include "AMReX_Array.H"
#include "AMReX_BCRec.H"
#include "AMReX_BC_TYPES.H"
#include "AMReX_BoxArray.H"
#include "AMReX_Config.H"
#include "AMReX_DistributionMapping.H"
#include "AMReX_Geometry.H"
#include "AMReX_GpuContainers.H"
#include "AMReX_IntVect.H"
#include "AMReX_MFIter.H"
#include "AMReX_MultiFabUtil.H"
#include "AMReX_PhysBCFunct.H"
#include "AMReX_REAL.H"
#include <AMReX_FillPatchUtil.H>
#include <AMReX_MLLinOp.H>
#include <AMReX_MLMG.H>
#include <AMReX_MLPoisson.H>
#include <AMReX_ParmParse.H>

#include "simulation.hpp"

//#define GRAVITY_DEBUG

namespace C {
// newton's gravitational constant taken from NIST's 2010 CODATA recommended
// value
constexpr amrex::Real Gconst = 6.67428e-8; // cm^3/g/s^2
} // namespace C

// This vector can be accessed on the GPU.
using RealVector = amrex::Gpu::ManagedVector<amrex::Real>;
using Real = amrex::Real;
using Box = amrex::Box;
using BoxArray = amrex::BoxArray;
using MultiFab = amrex::MultiFab;
using BCRec = amrex::BCRec;
using ParmParse = amrex::ParmParse;
using IntVect = amrex::IntVect;
using Geometry = amrex::Geometry;
using DistributionMapping = amrex::DistributionMapping;
using MFIter = amrex::MFIter;
template <typename T> using Array4 = amrex::Array4<T>;
template <typename T, int N> using GpuArray = amrex::GpuArray<T, N>;
template <typename T> using Vector = amrex::Vector<T>;

///
/// Gravity solve parameters
///

namespace gravity {
enum class GravityMode {
  Poisson,
  Constant,
};

const GravityMode gravity_type = GravityMode::Poisson;
const amrex::Real const_grav = 0.0;
const int lnum = 16;
const int max_solve_level = 10;

const int verbose = 0;
const int no_sync = 1;
const int do_composite_phi_correction = 1;

// multigrid solve parameters (all boolean)
const int mlmg_agglomeration = 1;
const int mlmg_consolidation = 1;
const int mlmg_max_fmg_iter = 0;
} // namespace gravity

///
/// Multipole gravity data
///
namespace multipole {
const int lnum_max = 30;

constexpr amrex::Real volumeFactor = 1.0;
constexpr amrex::Real parityFactor = 1.0;

amrex::Array1D<bool, 0, 2> constexpr doSymmetricAddLo = {false};
amrex::Array1D<bool, 0, 2> constexpr doSymmetricAddHi = {false};
bool constexpr doSymmetricAdd = false;

amrex::Array1D<bool, 0, 2> constexpr doReflectionLo = {false};
amrex::Array1D<bool, 0, 2> constexpr doReflectionHi = {false};

extern AMREX_GPU_MANAGED amrex::Real rmax;

extern AMREX_GPU_MANAGED amrex::Array2D<amrex::Real, 0, lnum_max, 0, lnum_max>
    factArray;
extern AMREX_GPU_MANAGED amrex::Array1D<amrex::Real, 0, lnum_max> parity_q0;
extern AMREX_GPU_MANAGED amrex::Array2D<amrex::Real, 0, lnum_max, 0, lnum_max>
    parity_qC_qS;
} // namespace multipole

///
/// @class Gravity
/// @brief
///
template <typename T> class Gravity {

public:
  ///
  /// Constructor
  ///
  /// @param _Density         index of density component
  ///
  Gravity(AMRSimulation<T> *_sim, amrex::BCRec &phys_bc,
          amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> &_coordCenter,
          int Density_);

  ///
  /// Read gravity-related parameters from parameter file
  ///
  void read_params();

  ///
  /// Set ``Gravity`` object's ``numpts_at_level`` variable.
  ///
  ///
  void set_numpts_in_gravity(int level);

  ///
  /// Setup gravity at level ``level``.
  ///
  /// @param level        integer, level number
  ///
  void install_level(int level);

  ///
  /// Returns ``gravity_type``
  ///
  static auto get_gravity_type() -> gravity::GravityMode;

  ///
  /// Returns ``max_solve_level``
  ///
  static auto get_max_solve_level() -> int;

  ///
  /// Returns ``no_sync``
  ///
  static auto NoSync() -> int;

  ///
  /// Returns ``do_composite_phi_correction``
  ///
  static auto DoCompositeCorrection() -> int;

  ///
  /// Returns ``test_solves``
  ///
  static auto test_results_of_solves() -> int;

  ///
  /// Return ``grad_phi_prev`` at given level
  ///
  /// @param level        level index
  ///
  auto get_grad_phi_prev(int level)
      -> amrex::Vector<std::unique_ptr<amrex::MultiFab>> &;

  ///
  /// Return ``grad_phi_curr`` at given level
  ///
  /// @param level        level index
  ///
  auto get_grad_phi_curr(int level)
      -> amrex::Vector<std::unique_ptr<amrex::MultiFab>> &;

  ///
  /// Return given component of ``grad_phi_prev`` at given level
  ///
  /// @param level        level index
  /// @param comp         component index
  ///
  auto get_grad_phi_prev_comp(int level, int comp) -> amrex::MultiFab *;

  ///
  /// Return ``grad_phi_curr`` at given level plus the given vector
  ///
  /// @param level        level index
  /// @param addend       Vector of MultiFabs to add to grad phi
  ///
  void
  plus_grad_phi_curr(int level,
                     amrex::Vector<std::unique_ptr<amrex::MultiFab>> &addend);

  ///
  /// Swap ``grad_phi_prev`` with ``grad_phi_curr`` at given level at set new
  /// ``grad_phi_curr`` to 1.e50.
  ///
  /// @param level        level index
  ///
  void swapTimeLevels(int level);

  ///
  /// Calculate the maximum value of the RHS over all levels.
  /// This should only be called at a synchronization point where
  /// all Castro levels have valid new time data at the same simulation time.
  /// The RHS we will use is the density multiplied by 4*pi*G and also
  /// multiplied by the metric terms, just as it would be in a real solve.
  ///
  void update_max_rhs();

  void construct_old_gravity(amrex::Real time, int level);

  void construct_new_gravity(amrex::Real time, int level);

  ///
  /// Solve Poisson's equation to find the gravitational potential
  ///
  /// @param level        level index
  /// @param phi          MultiFab to store gravitational potential in
  /// @param grad_phi     Vector of MultiFabs, \f$ \nabla \Phi \f$
  /// @param is_new       do we use state data at current time (1) or old time
  /// (0)?
  ///
  void solve_for_phi(int level, amrex::MultiFab &phi,
                     const amrex::Vector<amrex::MultiFab *> &grad_phi,
                     int is_new);

  ///
  /// Find delta phi
  ///
  /// @param crse_level       index of coarse level
  /// @param fine_level       index of fine level
  /// @param rhs              Vector of MultiFabs with right hand side source
  /// terms
  /// @param delta_phi        Vector of MultiFabs delta phi will be saved to
  /// @param grad_delta_phi   Vector of MultiFabs, gradient of delta phi
  ///
  void solve_for_delta_phi(
      int crse_level, int fine_level,
      const amrex::Vector<amrex::MultiFab *> &rhs,
      const amrex::Vector<amrex::MultiFab *> &delta_phi,
      const amrex::Vector<amrex::Vector<amrex::MultiFab *>> &grad_delta_phi);

  ///
  /// Sync gravity across levels
  ///
  /// @param crse_level       index of coarse level
  /// @param fine_level       index of fine level
  /// @param drho
  /// @param dphi
  ///
  void gravity_sync(int crse_level, int fine_level,
                    const amrex::Vector<amrex::MultiFab *> &drho,
                    const amrex::Vector<amrex::MultiFab *> &dphi);

  ///
  /// Multilevel solve for new phi from base level to finest level
  ///
  /// @param level                        Base level index
  /// @param finest_level                 Fine level index
  ///
  void multilevel_solve_for_new_phi(int level, int finest_level);

  ///
  /// Actually do the multilevel solve for new phi from base level to finest
  /// level
  ///
  /// @param level            Base level index
  /// @param finest_level     Fine level index
  /// @param grad_phi         gradient of phi
  /// @param is_new           Should we use the new state (1) or previous state
  /// (0)?
  ///
  void actual_multilevel_solve(
      int level, int finest_level,
      const amrex::Vector<amrex::Vector<amrex::MultiFab *>> &grad_phi,
      int is_new);

  ///
  /// Compute the difference between level and composite solves
  ///
  /// @param level        level index
  /// @param comp_phi     MultiFab containing computed phi
  /// @param comp_gphi    Vector of MultiFabs containing computed grad phi
  /// @param cml_phi      MultiFab, computed minus level phi
  /// @param cml_gphi     Vector of MultiFabs, computed minus level grad phi
  ///
  void create_comp_minus_level_grad_phi(
      int level, amrex::MultiFab &comp_phi,
      const amrex::Vector<amrex::MultiFab *> &comp_gphi,
      std::unique_ptr<amrex::MultiFab> &comp_minus_level_phi,
      amrex::Vector<std::unique_ptr<amrex::MultiFab>>
          &comp_minus_level_grad_phi);

  ///
  /// Get coarse phi on level ``level``-1
  ///
  /// @param level        level index of fine data
  /// @param phi_crse     MultiFab to contain coarse phi
  /// @param time         Current time
  ///
  void GetCrsePhi(int level, amrex::MultiFab &phi_crse, amrex::Real time);

  ///
  /// Get old gravity vector
  ///
  /// @param level        Level index
  /// @param grav_vector  MultiFab containing gravity vector
  /// @param time         Current time
  ///
  void get_old_grav_vector(int level, amrex::MultiFab &grav_vector,
                           amrex::Real time);

  ///
  /// Get new gravity vector
  ///
  /// @param level        Level index
  /// @param grav_vector  MultiFab containing gravity vector
  /// @param time         Current time
  ///
  void get_new_grav_vector(int level, amrex::MultiFab &grav_vector,
                           amrex::Real time);

  ///
  /// Test whether using the edge-based gradients
  /// to compute Div(Grad(Phi)) satisfies Lap(phi) = RHS
  ///
  /// @param bx          box
  /// @param rhs         right-hand-side
  /// @param ecx         gradients wrt x
  /// @param ecy         gradients wrt y
  /// @param ecz         gradients wrt z

  static void test_residual(const amrex::Box &bx,
                            amrex::Array4<amrex::Real> const &rhs,
                            amrex::Array4<amrex::Real> const &ecx,
                            amrex::Array4<amrex::Real> const &ecy,
                            amrex::Array4<amrex::Real> const &ecz,
                            amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx);

  ///
  /// @param level
  ///
  void test_level_grad_phi_prev(int level);

  ///
  /// @param level
  ///
  void test_level_grad_phi_curr(int level);

  ///
  /// @param level
  ///
  void test_composite_phi(int level);

  ///
  /// @param level    Level index
  /// @param is_new   Use new state data (1) or old state data (0)?
  ///
  void average_fine_ec_onto_crse_ec(int level, int is_new);

  ///
  /// Implement multipole boundary conditions
  ///
  /// @param crse_level
  /// @param fine_level
  /// @param Rhs
  /// @param phi
  ///
  void fill_multipole_BCs(int crse_level, int fine_level,
                          const amrex::Vector<amrex::MultiFab *> &Rhs,
                          amrex::MultiFab &phi);

  ///
  /// Initialize multipole gravity
  ///
  void init_multipole_grav();

  ///
  /// Make multigrid boundary conditions
  ///
  void make_mg_bc();

  ///
  /// Pointers to amr,amrlevel.
  ///
  //  amrex::Amr *parent;
  //  amrex::Vector<amrex::AmrLevel *> LevelData;

  ///
  /// Pointer to AMRSimulation<problem_t>
  ///
  AMRSimulation<T> *sim;

  ///
  /// MultiFabs for potential, acceleration
  ///
  amrex::Vector<amrex::MultiFab> phi_old_;
  amrex::Vector<amrex::MultiFab> phi_new_;
  amrex::Vector<amrex::MultiFab> g_old_;
  amrex::Vector<amrex::MultiFab> g_new_;

  ///
  /// Pointers to grad_phi at previous and current time
  ///
  amrex::Vector<amrex::Vector<std::unique_ptr<amrex::MultiFab>>> grad_phi_curr;
  amrex::Vector<amrex::Vector<std::unique_ptr<amrex::MultiFab>>> grad_phi_prev;

  ///
  /// MultiFabs for composite-level corrections
  ///
  amrex::Vector<std::unique_ptr<amrex::MultiFab>> corr_phi_;
  amrex::Vector<amrex::Vector<std::unique_ptr<amrex::MultiFab>>> corr_grad_phi_;

  ///
  /// BoxArray at each level
  ///

  ///
  /// Absolute tolerance on each level
  ///
  amrex::Vector<amrex::Real> abs_tol;

  ///
  /// Relative tolerance on each level
  ///
  amrex::Vector<amrex::Real> rel_tol;

  ///
  /// Resnorm at each level
  ///
  amrex::Vector<amrex::Real> level_solver_resnorm;

  ///
  /// Maximum value of the RHS (used for obtaining absolute tolerances)
  ///
  amrex::Real max_rhs;

  const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> coordCenter;

  int Density; // index of density component
  int finest_level_allocated;
  int max_lev;

  amrex::BCRec *phys_bc; // used to specify whether any axes are
                         // reflection-symmetric (e.g., in octant symmetry)

  std::array<amrex::MLLinOp::BCType, AMREX_SPACEDIM> mlmg_lobc;
  std::array<amrex::MLLinOp::BCType, AMREX_SPACEDIM> mlmg_hibc;

  amrex::Vector<int> numpts;

  static const int test_solves;
  static amrex::Real mass_offset;
  static amrex::Real Ggravity;
  static int stencil_type;

  ///
  /// Get the rhs
  ///
  /// @param crse_level   Index of coarse level
  /// @param nlevs        Number of levels
  /// @param is_new       Use new (1) or old (0) state data
  ///
  auto get_rhs(int crse_level, int nlevs, int is_new)
      -> amrex::Vector<std::unique_ptr<amrex::MultiFab>>;

  ///
  /// This is a sanity check on whether we are trying to fill multipole boundary
  /// conditions for grids at this level > 0 -- this case is not currently
  /// supported.
  ///  Here we shrink the domain at this level by 1 in any direction which is
  ///  not symmetry or periodic, then ask if the grids at this level are
  ///  contained in the shrunken domain.  If not, then grids at this level touch
  ///  the domain boundary and we must abort.
  ///
  /// @param level    Level index
  ///
  void sanity_check(int level);

  ///
  /// Do multigrid solve
  ///
  /// @param crse_level   Coarse level index
  /// @param fine_level   Fine level index
  /// @param phi          Gravitational potential
  /// @param rhs          Right hand side
  /// @param grad_phi     Grad phi
  /// @param res
  /// @param crse_bcdata
  /// @param rel_eps      Relative tolerance
  /// @param abs_eps      Absolute tolerance
  ///
  auto actual_solve_with_mlmg(
      int crse_level, int fine_level,
      const amrex::Vector<amrex::MultiFab *> &phi,
      const amrex::Vector<const amrex::MultiFab *> &rhs,
      const amrex::Vector<std::array<amrex::MultiFab *, AMREX_SPACEDIM>>
          &grad_phi,
      const amrex::Vector<amrex::MultiFab *> &res,
      const amrex::MultiFab *crse_bcdata, amrex::Real rel_eps,
      amrex::Real abs_eps) const -> amrex::Real;

  ///
  /// Do multigrid solve to find phi
  ///
  /// @param crse_level   Coarse level index
  /// @param fine_level   Fine level index
  /// @param phi          Gravitational potential
  /// @param rhs          Right hand side source term
  /// @param grad_phi     Grad phi
  /// @param res
  /// @param time         Current time
  ///
  auto solve_phi_with_mlmg(
      int crse_level, int fine_level,
      const amrex::Vector<amrex::MultiFab *> &phi,
      const amrex::Vector<amrex::MultiFab *> &rhs,
      const amrex::Vector<amrex::Vector<amrex::MultiFab *>> &grad_phi,
      const amrex::Vector<amrex::MultiFab *> &res, amrex::Real time)
      -> amrex::Real;
};

///
/// @class GradPhiPhysBCFunct
/// @brief A physical boundary condition function for grad phi
///
using GradPhiPhysBCFunct = amrex::PhysBCFunctNoOp;

#ifdef GRAVITY_DEBUG
template <typename T> const int Gravity<T>::test_solves = 1;
#else
template <typename T> const int Gravity<T>::test_solves = 0;
#endif

using GravityMode = gravity::GravityMode;

#include "GravityBC.hpp"
#include "Gravity_impl.hpp"
#include "Gravity_level.hpp"
#include "Gravity_residual_impl.hpp"

#endif // GRAVITY_HPP_
