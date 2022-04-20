#ifndef SIMULATION_HPP_ // NOLINT
#define SIMULATION_HPP_
//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file simulation.cpp
/// \brief Implements classes and functions to organise the overall setup,
/// timestepping, solving, and I/O of a simulation.

// c++ headers
#include <csignal>
#include <iomanip>
#include <limits>
#include <memory>
#include <ostream>
#include <tuple>

// library headers
#include "AMReX.H"
#include "AMReX_AmrCore.H"
#include "AMReX_Array.H"
#include "AMReX_Array4.H"
#include "AMReX_BCRec.H"
#include "AMReX_BC_TYPES.H"
#include "AMReX_BLassert.H"
#include "AMReX_Config.H"
#include "AMReX_DistributionMapping.H"
#include "AMReX_Extension.H"
#include "AMReX_FArrayBox.H"
#include "AMReX_FillPatchUtil.H"
#include "AMReX_FluxRegister.H"
#include "AMReX_GpuQualifiers.H"
#include "AMReX_INT.H"
#include "AMReX_IndexType.H"
#include "AMReX_IntVect.H"
#include "AMReX_LayoutData.H"
#include "AMReX_MFInterpolater.H"
#include "AMReX_MultiFabUtil.H"
#include "AMReX_ParallelDescriptor.H"
#include "AMReX_REAL.H"
#include "AMReX_SPACE.H"
#include "AMReX_Vector.H"
#include "AMReX_VisMF.H"
#include "AMReX_YAFluxRegister.H"
#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParmParse.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_Print.H>
#include <AMReX_Utility.H>
#include <fmt/core.h>

#include "CheckType.hpp"

// internal headers
#include "CheckNaN.hpp"
#include "math_impl.hpp"

#define USE_YAFLUXREGISTER

using array_t = amrex::Array4<amrex::Real> const;
using arrayconst_t = amrex::Array4<const amrex::Real> const;

// Main simulation class; solvers should inherit from this
template <typename problem_t> class AMRSimulation : public amrex::AmrCore {
public:
  amrex::Real maxDt_ =
      std::numeric_limits<double>::max(); // no limit by default
  amrex::Real initDt_ =
      std::numeric_limits<double>::max(); // no limit by default
  amrex::Real constantDt_ = 0.0;
  amrex::Vector<int> istep;         // which step?
  amrex::Vector<int> nsubsteps;     // how many substeps on each level?
  amrex::Vector<amrex::Real> tNew_; // for state_new_
  amrex::Vector<amrex::Real> tOld_; // for state_old_
  amrex::Vector<amrex::Real> dt_;   // timestep for each level
  amrex::Real stopTime_ = 1.0;      // default
  amrex::Real cflNumber_ = 0.3;     // default
  amrex::Long cycleCount_ = 0;
  amrex::Long maxTimesteps_ = 1e4; // default
  int plotfileInterval_ = 10;      // -1 == no output
  int checkpointInterval_ = -1;    // -1 == no output

  // constructors

  AMRSimulation(amrex::Vector<amrex::BCRec> &boundaryConditions,
                const int ncomp, const int ncompPrim)
      : ncomp_(ncomp), ncompPrimitive_(ncompPrim) {
    initialize(boundaryConditions);
  }

  AMRSimulation(amrex::Vector<amrex::BCRec> &boundaryConditions,
                const int ncomp)
      : ncomp_(ncomp), ncompPrimitive_(ncomp) {
    initialize(boundaryConditions);
  }

  void initialize(amrex::Vector<amrex::BCRec> &boundaryConditions);
  void readParameters();
  void setInitialConditions();
  void setInitialConditionsAtLevel(int lev);
  void evolve();
  void computeTimestep();

  virtual void computeMaxSignalLocal(int level) = 0;
  virtual void advanceSingleTimestepAtLevel(int lev, amrex::Real time,
                                            amrex::Real dt_lev, int iteration,
                                            int ncycle) = 0;
  virtual void preCalculateInitialConditions() = 0;
  virtual void setInitialConditionsOnGrid(array_t &state,
                                          const amrex::Box &indexRange,
                                          const amrex::Geometry &geom) = 0;
  virtual void computeAfterTimestep() = 0;
  virtual void computeAfterEvolve(amrex::Vector<amrex::Real> &initSumCons) = 0;

  // compute derived variables
  virtual void ComputeDerivedVar(int lev, std::string const &dname,
                                 amrex::MultiFab &mf, int ncomp) const = 0;

  // fix-up any unphysical states created by AMR operations
  // (e.g., caused by the flux register or from interpolation)
  virtual void FixupState(int level) = 0;

  // tag cells for refinement
  void ErrorEst(int lev, amrex::TagBoxArray &tags, amrex::Real time,
                int ngrow) override = 0;

  // Make a new level using provided BoxArray and DistributionMapping
  void MakeNewLevelFromCoarse(int lev, amrex::Real time,
                              const amrex::BoxArray &ba,
                              const amrex::DistributionMapping &dm) override;

  // Remake an existing level using provided BoxArray and DistributionMapping
  void RemakeLevel(int lev, amrex::Real time, const amrex::BoxArray &ba,
                   const amrex::DistributionMapping &dm) override;

  // Delete level data
  void ClearLevel(int lev) override;

  // Make a new level from scratch using provided BoxArray and
  // DistributionMapping
  void MakeNewLevelFromScratch(int lev, amrex::Real time,
                               const amrex::BoxArray &ba,
                               const amrex::DistributionMapping &dm) override;

  // AMR utility functions
  void fillBoundaryConditions(amrex::MultiFab &S_filled, amrex::MultiFab &state,
                              int lev, amrex::Real time);
  void FillPatchWithData(int lev, amrex::Real time, amrex::MultiFab &mf,
                         amrex::Vector<amrex::MultiFab *> &coarseData,
                         amrex::Vector<amrex::Real> &coarseTime,
                         amrex::Vector<amrex::MultiFab *> &fineData,
                         amrex::Vector<amrex::Real> &fineTime, int icomp,
                         int ncomp);
  void FillPatch(int lev, amrex::Real time, amrex::MultiFab &mf, int icomp,
                 int ncomp);
  void FillCoarsePatch(int lev, amrex::Real time, amrex::MultiFab &mf,
                       int icomp, int ncomp);
  void GetData(int lev, amrex::Real time,
               amrex::Vector<amrex::MultiFab *> &data,
               amrex::Vector<amrex::Real> &datatime);
  void AverageDown();
  void AverageDownTo(int crse_lev);
  void timeStepWithSubcycling(int lev, amrex::Real time, int iteration);
  void doRegridIfNeeded(int step, amrex::Real time);

  void incrementFluxRegisters(
      amrex::MFIter &mfi, amrex::YAFluxRegister *fr_as_crse,
      amrex::YAFluxRegister *fr_as_fine,
      std::array<amrex::FArrayBox, AMREX_SPACEDIM> &fluxArrays, int lev,
      amrex::Real dt_lev);

  // boundary condition
  AMREX_GPU_DEVICE static void setCustomBoundaryConditions(
      const amrex::IntVect &iv, amrex::Array4<amrex::Real> const &dest,
      int dcomp, int numcomp, amrex::GeometryData const &geom, amrex::Real time,
      const amrex::BCRec *bcr, int bcomp,
      int orig_comp); // template specialized by problem generator

  // I/O functions
  [[nodiscard]] auto PlotFileName(int lev) const -> std::string;
  [[nodiscard]] auto PlotFileMF() const -> amrex::Vector<amrex::MultiFab>;
  [[nodiscard]] auto PlotFileMFAtLevel(int lev) const -> amrex::MultiFab;
  void WritePlotFile() const;
  void WriteCheckpointFile() const;
  void ReadCheckpointFile();
  void LoadBalance();
  void ResetCosts();

protected:
  amrex::Vector<amrex::BCRec> boundaryConditions_; // on level 0
  // cell-centred states
  amrex::Vector<amrex::MultiFab> state_new_cc_;
  amrex::Vector<amrex::MultiFab> state_old_cc_;
  // face-centred states
  amrex::Vector<amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>> state_new_fc_;
  amrex::Vector<amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>> state_old_fc_;
  // needed to compute CFL timestep
  amrex::Vector<amrex::MultiFab> max_signal_speed_;

  // flux registers: store fluxes at coarse-fine interface for synchronization
  // this will be sized "nlevs_max+1"
  // NOTE: the flux register associated with flux_reg[lev] is associated with
  // the lev/lev-1 interface (and has grid spacing associated with lev-1)
  // therefore flux_reg[0] and flux_reg[nlevs_max] are never actually used in
  // the reflux operation
  amrex::Vector<std::unique_ptr<amrex::YAFluxRegister>> flux_reg_;

  // Nghost = number of ghost cells for each array
  int nghost_ = 4; // PPM needs nghost >= 3, PPM+flattening needs nghost >= 4
  int ncomp_ = 0; // = number of components (conserved variables) for each array
  int ncompPrimitive_ = 0; // number of primitive variables
  amrex::Vector<std::string> componentNames_;
  amrex::Vector<std::string> derivedNames_;
  bool areInitialConditionsDefined_ = false;

  /// output parameters
  std::string plot_file{"plt"}; // plotfile prefix
  std::string chk_file{"chk"};  // checkpoint prefix
  /// input parameters (if >= 0 we restart from a checkpoint)
  std::string restart_chkfile;

  /// AMR-specific parameters
  int regrid_int = 2;  // regrid interval (number of coarse steps)
  int do_reflux = 1;   // 1 == reflux, 0 == no reflux
  int do_subcycle = 1; // 1 == subcycle, 0 == no subcyle
  int suppress_output =
      0; // 1 == show timestepping, 0 == do not output each timestep

  // performance metrics
  amrex::Long cellUpdates_ = 0;
  amrex::Vector<amrex::Long> cellUpdatesEachLevel_;
};

template <typename problem_t>
void AMRSimulation<problem_t>::initialize(
    amrex::Vector<amrex::BCRec> &boundaryConditions) {
  BL_PROFILE("AMRSimulation::initialize()");

  readParameters();

  // print derived vars
  if (!derivedNames_.empty()) {
    amrex::Print() << "Using derived variables:\n";
    for (auto const &name : derivedNames_) {
      amrex::Print() << "\t" << name << "\n";
    }
    amrex::Print() << "\n";
  }

  int nlevs_max = max_level + 1;
  istep.resize(nlevs_max, 0);
  nsubsteps.resize(nlevs_max, 1);
  if (do_subcycle == 1) {
    for (int lev = 1; lev <= max_level; ++lev) {
      nsubsteps[lev] = MaxRefRatio(lev - 1);
    }
  }

  // vectors to cover the number of grid levels
  amrex::Print() << "\nAMRSimulation::initialize()\n";
  tNew_.resize(nlevs_max, 0.0);
  tOld_.resize(nlevs_max, -1.e100);
  dt_.resize(nlevs_max, 1.e100);
  state_new_cc_.resize(nlevs_max);
  state_old_cc_.resize(nlevs_max);
  state_new_fc_.resize(nlevs_max);
  state_old_fc_.resize(nlevs_max);
  max_signal_speed_.resize(nlevs_max);
  flux_reg_.resize(nlevs_max + 1);
  cellUpdatesEachLevel_.resize(nlevs_max, 0);

  boundaryConditions_ = boundaryConditions;

  // check that grids will be properly nested on each level
  // (this is necessary since FillPatch only fills from non-ghost cells on
  // lev-1)
  auto checkIsProperlyNested = [=](int const lev,
                                   amrex::IntVect const &blockingFactor) {
    return amrex::ProperlyNested(refRatio(lev - 1), blockingFactor, nghost_,
                                 amrex::IndexType::TheCellType(),
                                 &amrex::cell_cons_interp);
  };

  for (int lev = 1; lev <= max_level; ++lev) {
    if (!checkIsProperlyNested(lev, blocking_factor[lev])) {
      // level lev is not properly nested
      amrex::Print()
          << "Blocking factor is too small for proper grid nesting! "
             "Increase blocking factor to >= ceil(nghost,ref_ratio)*ref_ratio."
          << std::endl;
      amrex::Abort("Grids not properly nested!");
    }
  }
}

enum class centering { cc, fc, ec };
enum class direction { na=-1, x=0, y=1, z=2 };

struct grid {
  amrex::Array4<double> &array;
  amrex::Box indexRange;
  amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx;
  amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo;
  amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_hi;
  enum centering cen;
  enum direction dir;
  grid(amrex::Array4<double> &array, amrex::Box indexRange,
       amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx,
       amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo,
       amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_hi,
       centering cen, direction dir)
      : array(array), indexRange(indexRange), dx(dx), prob_lo(prob_lo),
        prob_hi(prob_hi), cen(cen), dir(dir) {}
};

template <typename problem_t>
void AMRSimulation<problem_t>::setInitialConditionsAtLevel(int lev) {
  // iitialise grid-struct for the user to opperate on
  std::vector<grid> grid_vec;
  // perform precalculation
  preCalculateInitialConditions();
  // itterate over the domain
  for (amrex::MFIter iter(state_new_cc_[lev]); iter.isValid(); ++iter) {
    const amrex::Box &indexRange = iter.validbox(); // excludes ghost zones
    auto const &state = state_new_cc_[lev].array(iter);

    // std::cout << "\n" << quokka::type_name<decltype(state_new_cc_[lev].array(iter))>();
    // std::cout << "\n" << quokka::type_name<decltype(iter.validbox())>();
    // std::cout << "\n" << quokka::type_name<decltype(geom[lev].CellSizeArray())>();
    // std::cout << "\n" << quokka::type_name<decltype(geom[lev].ProbLoArray())>();
    // std::cout << "\n" << quokka::type_name<decltype(geom[lev].ProbHiArray())>();
    // std::cout << "\n" << quokka::type_name<decltype(centering::cc)>();
    // std::cout << "\n" << quokka::type_name<decltype(direction::na)>() << "\n";

    // cell-centred states
    grid_vec.emplace_back(grid(
      state_new_cc_[lev].array(iter),
      iter.validbox(),
      geom[lev].CellSizeArray(),
      geom[lev].ProbLoArray(),
      geom[lev].ProbHiArray(),
      centering::cc,
      direction::na
    ));

    // // face-centred states
    // for (int idim = 0; idim < AMREX_SPACEDIM; idim++) {
    //   grid_vec.emplace_back(grid(
    //     state_new_fc_[lev].array(iter),
    //     iter.validbox(),
    //     geom[lev].CellSizeArray(),
    //     geom[lev].ProbLoArray(),
    //     geom[lev].ProbHiArray(),
    //     centering::fc,
    //     direction(idim)
    //   ));
    // }

    // apply initial conditions provided / defined by the user
    setInitialConditionsOnGrid(state, indexRange, geom[lev]);
  }
  // set flag
  areInitialConditionsDefined_ = true;
}

template <typename problem_t> void AMRSimulation<problem_t>::readParameters() {
  BL_PROFILE("AMRSimulation::readParameters()");

  // ParmParse reads inputs from the *.inputs file
  amrex::ParmParse pp;

  // Default nsteps = 1e4
  pp.query("max_timesteps", maxTimesteps_);

  // Default CFL number == 0.3, set to whatever is in the file
  pp.query("cfl", cflNumber_);

  // Default do_reflux = 1
  pp.query("do_reflux", do_reflux);

  // Default do_subcycle = 1
  pp.query("do_subcycle", do_subcycle);

  // Default suppress_output = 0
  pp.query("suppress_output", suppress_output);

  // specify this on the commmand-line in order to restart from a checkpoint
  // file
  pp.query("restartfile", restart_chkfile);

  // Specify derived variables to save to plotfiles
  pp.queryarr("derived_vars", derivedNames_);

  // re-grid interval
  pp.query("regrid_interval", regrid_int);
}

template <typename problem_t>
void AMRSimulation<problem_t>::setInitialConditions() {
  BL_PROFILE("AMRSimulation::setInitialConditions()");

  if (restart_chkfile.empty()) {
    // start simulation from the beginning
    const amrex::Real time = 0.0;
    InitFromScratch(time);
    AverageDown();

    if (checkpointInterval_ > 0) {
      WriteCheckpointFile();
    }
  } else {
    // restart from a checkpoint
    ReadCheckpointFile();
  }

  if (plotfileInterval_ > 0) {
    WritePlotFile();
  }
}

template <typename problem_t> void AMRSimulation<problem_t>::computeTimestep() {
  BL_PROFILE("AMRSimulation::computeTimestep()");

  amrex::Vector<amrex::Real> dt_tmp(finest_level + 1);
  for (int level = 0; level <= finest_level; ++level) {
    computeMaxSignalLocal(level);
    amrex::Real domain_signal_max = max_signal_speed_[level].norminf();
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx =
        geom[level].CellSizeArray();

    amrex::Real dx_min = std::min({AMREX_D_DECL(dx[0], dx[1], dx[2])});
    dt_tmp[level] = cflNumber_ * (dx_min / domain_signal_max);
  }

  constexpr amrex::Real change_max = 1.1;
  amrex::Real dt_0 = dt_tmp[0];
  int n_factor = 1;

  for (int level = 0; level <= finest_level; ++level) {
    dt_tmp[level] = std::min(dt_tmp[level], change_max * dt_[level]);
    n_factor *= nsubsteps[level];
    dt_0 = std::min(dt_0, n_factor * dt_tmp[level]);
    dt_0 = std::min(dt_0, maxDt_); // limit to maxDt_
    if (tNew_[level] == 0.0) {     // first timestep
      dt_0 = std::min(dt_0, initDt_);
    }
    if (constantDt_ > 0.0) { // use constant timestep if set
      dt_0 = constantDt_;
    }
  }

  // Limit dt to avoid overshooting stop_time
  const amrex::Real eps = 1.e-3 * dt_0;

  if (tNew_[0] + dt_0 > stopTime_ - eps) {
    dt_0 = stopTime_ - tNew_[0];
  }

  dt_[0] = dt_0;

  for (int level = 1; level <= finest_level; ++level) {
    dt_[level] = dt_[level - 1] / nsubsteps[level];
  }
}

template <typename problem_t> void AMRSimulation<problem_t>::evolve() {
  BL_PROFILE("AMRSimulation::evolve()");

  AMREX_ALWAYS_ASSERT(areInitialConditionsDefined_);

  amrex::Real cur_time = tNew_[0];
  int last_plot_file_step = 0;

  amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx0 =
      geom[0].CellSizeArray();
  amrex::Real const vol = AMREX_D_TERM(dx0[0], *dx0[1], *dx0[2]);
  amrex::Vector<amrex::Real> init_sum_cons(ncomp_);
  for (int n = 0; n < ncomp_; ++n) {
    const int lev = 0;
    init_sum_cons[n] = state_new_cc_[lev].sum(n) * vol;
  }

  amrex::Real const start_time = amrex::ParallelDescriptor::second();

  // Main time loop
  for (int step = istep[0]; step < maxTimesteps_ && cur_time < stopTime_;
       ++step) {
    if (suppress_output == 0) {
      amrex::Print() << "\nCoarse STEP " << step + 1 << " at t = " << cur_time
                     << " (" << (cur_time / stopTime_) * 100. << "%) starts ..."
                     << std::endl;
    }

    amrex::ParallelDescriptor::Barrier(); // synchronize all MPI ranks
    computeTimestep();

    int lev = 0;
    int iteration = 1;
    timeStepWithSubcycling(lev, cur_time, iteration);

    cur_time += dt_[0];
    ++cycleCount_;
    computeAfterTimestep();

    // sync up time (to avoid roundoff error)
    for (lev = 0; lev <= finest_level; ++lev) {
      tNew_[lev] = cur_time;
    }

    if (plotfileInterval_ > 0 && (step + 1) % plotfileInterval_ == 0) {
      last_plot_file_step = step + 1;
      WritePlotFile();
    }

    if (checkpointInterval_ > 0 && (step + 1) % checkpointInterval_ == 0) {
      WriteCheckpointFile();
    }

    if (cur_time >= stopTime_ - 1.e-6 * dt_[0]) {
      break;
    }
  }

  amrex::Real elapsed_sec = amrex::ParallelDescriptor::second() - start_time;

  // compute reference solution (if it's a test problem)
  computeAfterEvolve(init_sum_cons);

  // compute conservation error
  for (int n = 0; n < ncomp_; ++n) {
    amrex::Real const final_sum = state_new_cc_[0].sum(n) * vol;
    amrex::Real const abs_err = (final_sum - init_sum_cons[n]);
    amrex::Real const rel_err = abs_err / init_sum_cons[n];
    amrex::Print() << "Initial " << componentNames_[n] << " = "
                   << init_sum_cons[n] << std::endl;
    amrex::Print() << "\tabsolute conservation error = " << abs_err
                   << std::endl;
    if (init_sum_cons[n] != 0.0) {
      amrex::Print() << "\trelative conservation error = " << rel_err
                     << std::endl;
    }
    amrex::Print() << std::endl;
  }

  // compute zone-cycles/sec
  const int IOProc = amrex::ParallelDescriptor::IOProcessorNumber();
  amrex::ParallelDescriptor::ReduceRealMax(elapsed_sec, IOProc);
  const double microseconds_per_update = 1.0e6 * elapsed_sec / cellUpdates_;
  const double megaupdates_per_second = 1.0 / microseconds_per_update;
  amrex::Print() << "Performance figure-of-merit: " << microseconds_per_update
                 << " μs/zone-update [" << megaupdates_per_second
                 << " Mupdates/s]\n";
  for (int lev = 0; lev <= max_level; ++lev) {
    amrex::Print() << "Zone-updates on level " << lev << ": "
                   << cellUpdatesEachLevel_[lev] << "\n";
  }
  amrex::Print() << std::endl;

  // write final plotfile
  if (plotfileInterval_ > 0 && istep[0] > last_plot_file_step) {
    WritePlotFile();
  }
}

// N.B.: This function actually works for subcycled or not subcycled, as long as
// nsubsteps[lev] is set correctly.
template <typename problem_t>
void AMRSimulation<problem_t>::timeStepWithSubcycling(int lev, amrex::Real time,
                                                      int iteration) {
  BL_PROFILE("AMRSimulation::timeStepWithSubcycling()");

  if (regrid_int > 0) // regridding is possible
  {
    // help keep track of whether a level was already regridded
    // from a coarser level call to regrid
    static amrex::Vector<int> last_regrid_step(max_level + 1, 0);

    // regrid changes level "lev+1" so we don't regrid on max_level
    // also make sure we don't regrid fine levels again if
    // it was taken care of during a coarser regrid
    if (lev < max_level && istep[lev] > last_regrid_step[lev]) {
      if (istep[lev] % regrid_int == 0) {
        // regrid could add newly refined levels (if finest_level < max_level)
        // so we save the previous finest level index
        int old_finest = finest_level;
        regrid(lev, time);

        // mark that we have regridded this level already
        for (int k = lev; k <= finest_level; ++k) {
          last_regrid_step[k] = istep[k];
        }

        // if there are newly created levels, set the time step
        for (int k = old_finest + 1; k <= finest_level; ++k) {
          if (do_subcycle) {
            dt_[k] = dt_[k - 1] / nsubsteps[k];
          } else {
            dt_[k] = dt_[k - 1];
          }
        }

        // do fix-up on all levels that have been re-gridded
        for (int k = lev; k <= finest_level; ++k) {
          FixupState(k);
        }
      }
    }
  }

  if (Verbose()) {
    amrex::Print() << "[Level " << lev << " step " << istep[lev] + 1 << "] ";
    amrex::Print() << "ADVANCE with time = " << tNew_[lev]
                   << " dt = " << dt_[lev] << std::endl;
  }

  // Advance a single level for a single time step, and update flux registers
  tOld_[lev] = tNew_[lev];
  tNew_[lev] += dt_[lev]; // critical that this is done *before* advanceAtLevel

  advanceSingleTimestepAtLevel(lev, time, dt_[lev], iteration, nsubsteps[lev]);
  ++istep[lev];
  cellUpdates_ += CountCells(lev); // keep track of total number of cell updates
  cellUpdatesEachLevel_[lev] += CountCells(lev);

  if (Verbose()) {
    amrex::Print() << "[Level " << lev << " step " << istep[lev] << "] ";
    amrex::Print() << "Advanced " << CountCells(lev) << " cells" << std::endl;
  }

  if (lev < finest_level) {
    // recursive call for next-finer level
    for (int i = 1; i <= nsubsteps[lev + 1]; ++i) {
      timeStepWithSubcycling(lev + 1, time + (i - 1) * dt_[lev + 1], i);
    }

    if (do_reflux != 0) {
      // update lev based on coarse-fine flux mismatch
      flux_reg_[lev + 1]->Reflux(state_new_cc_[lev]);
    }

    AverageDownTo(lev); // average lev+1 down to lev
    FixupState(lev); // fix any unphysical states created by reflux or averaging
  }
}

template <typename problem_t>
void AMRSimulation<problem_t>::incrementFluxRegisters(
    amrex::MFIter &mfi, amrex::YAFluxRegister *fr_as_crse,
    amrex::YAFluxRegister *fr_as_fine,
    std::array<amrex::FArrayBox, AMREX_SPACEDIM> &fluxArrays, int const lev,
    amrex::Real const dt_lev) {
  BL_PROFILE("AMRSimulation::incrementFluxRegisters()");

  if (fr_as_crse != nullptr) {
    AMREX_ASSERT(lev < finestLevel());
    AMREX_ASSERT(fr_as_crse == flux_reg_[lev + 1].get());
    fr_as_crse->CrseAdd(
        mfi, {AMREX_D_DECL(&fluxArrays[0], &fluxArrays[1], &fluxArrays[2])},
        geom[lev].CellSize(), dt_lev, amrex::RunOn::Gpu);
  }

  if (fr_as_fine != nullptr) {
    AMREX_ASSERT(lev > 0);
    AMREX_ASSERT(fr_as_fine == flux_reg_[lev].get());
    fr_as_fine->FineAdd(
        mfi, {AMREX_D_DECL(&fluxArrays[0], &fluxArrays[1], &fluxArrays[2])},
        geom[lev].CellSize(), dt_lev, amrex::RunOn::Gpu);
  }
}

// Make a new level using provided BoxArray and DistributionMapping and fill
// with interpolated coarse level data. Overrides the pure virtual function in
// AmrCore
template <typename problem_t>
void AMRSimulation<problem_t>::MakeNewLevelFromCoarse(
    int level, amrex::Real time, const amrex::BoxArray &ba,
    const amrex::DistributionMapping &dm) {
  BL_PROFILE("AMRSimulation::MakeNewLevelFromCoarse()");

  const int ncomp = state_new_cc_[level - 1].nComp();
  const int nghost = state_new_cc_[level - 1].nGrow();

  state_new_cc_[level].define(ba, dm, ncomp, nghost);
  state_old_cc_[level].define(ba, dm, ncomp, nghost);

  // This clears the old MultiFab and allocates the new one
  amrex::Print() << "AMRSimulation::MakeNewLevelFromCoarse\n";
  for (int idim = 0; idim < AMREX_SPACEDIM; idim++) {
    state_new_fc_[level][idim] = amrex::MultiFab(
        amrex::convert(ba, amrex::IntVect::TheDimensionVector(idim)), dm, 1, 1);
    state_old_fc_[level][idim] = amrex::MultiFab(
        amrex::convert(ba, amrex::IntVect::TheDimensionVector(idim)), dm, 1, 1);
  }

  max_signal_speed_[level].define(ba, dm, 1, nghost);

  tNew_[level] = time;
  tOld_[level] = time - 1.e200;

  if (level > 0 && (do_reflux != 0)) {
    flux_reg_[level] = std::make_unique<amrex::YAFluxRegister>(
        ba, boxArray(level - 1), dm, DistributionMap(level - 1), Geom(level),
        Geom(level - 1), refRatio(level - 1), level, ncomp);
  }

  // also necessary
  FillCoarsePatch(level, time, state_new_cc_[level], 0, ncomp);
  FillCoarsePatch(level, time, state_old_cc_[level], 0, ncomp);
}

// Remake an existing level using provided BoxArray and DistributionMapping and
// fill with existing fine and coarse data. Overrides the pure virtual function
// in AmrCore
template <typename problem_t>
void AMRSimulation<problem_t>::RemakeLevel(
    int level, amrex::Real time, const amrex::BoxArray &ba,
    const amrex::DistributionMapping &dm) {
  BL_PROFILE("AMRSimulation::RemakeLevel()");

  const int ncomp = state_new_cc_[level].nComp();
  const int nghost = state_new_cc_[level].nGrow();

  amrex::MultiFab new_state(ba, dm, ncomp, nghost);
  amrex::MultiFab old_state(ba, dm, ncomp, nghost);
  amrex::MultiFab max_signal_speed(ba, dm, 1, nghost);

  FillPatch(level, time, new_state, 0, ncomp);
  FillPatch(level, time, old_state, 0, ncomp); // also necessary

  std::swap(new_state, state_new_cc_[level]);
  std::swap(old_state, state_old_cc_[level]);
  std::swap(max_signal_speed, max_signal_speed_[level]);

  tNew_[level] = time;
  tOld_[level] = time - 1.e200;

  // This clears the old MultiFab and allocates the new one
  amrex::Print() << "AMRSimulation::RemakeLevel()\n";
  for (int idim = 0; idim < AMREX_SPACEDIM; idim++) {
    state_new_fc_[level][idim] = amrex::MultiFab(
        amrex::convert(ba, amrex::IntVect::TheDimensionVector(idim)), dm, 1, 1);
    state_old_fc_[level][idim] = amrex::MultiFab(
        amrex::convert(ba, amrex::IntVect::TheDimensionVector(idim)), dm, 1, 1);
  }

  if (level > 0 && (do_reflux != 0)) {
    flux_reg_[level] = std::make_unique<amrex::YAFluxRegister>(
        ba, boxArray(level - 1), dm, DistributionMap(level - 1), Geom(level),
        Geom(level - 1), refRatio(level - 1), level, ncomp);
  }
}

// Delete level data. Overrides the pure virtual function in AmrCore
template <typename problem_t>
void AMRSimulation<problem_t>::ClearLevel(int level) {
  BL_PROFILE("AMRSimulation::ClearLevel()");

  state_new_cc_[level].clear();
  state_old_cc_[level].clear();
  max_signal_speed_[level].clear();
  flux_reg_[level].reset(nullptr);
}

// Make a new level from scratch using provided BoxArray and
// DistributionMapping. Only used during initialization. Overrides the pure
// virtual function in AmrCore
template <typename problem_t>
void AMRSimulation<problem_t>::MakeNewLevelFromScratch(
    int level, amrex::Real time, const amrex::BoxArray &ba,
    const amrex::DistributionMapping &dm) {
  BL_PROFILE("AMRSimulation::MakeNewLevelFromScratch()");

  const int ncomp = ncomp_;
  const int nghost = nghost_;

  // cell-centred quantities
  state_new_cc_[level].define(ba, dm, ncomp, nghost);
  state_old_cc_[level].define(ba, dm, ncomp, nghost);

  // max signal speed
  max_signal_speed_[level].define(ba, dm, 1, nghost);

  tNew_[level] = time;
  tOld_[level] = time - 1.e200;

  // This clears the old MultiFab and allocates the new one
  for (int idim = 0; idim < AMREX_SPACEDIM; idim++) {
    state_new_fc_[level][idim] = amrex::MultiFab(
        amrex::convert(ba, amrex::IntVect::TheDimensionVector(idim)), dm, 1, 1);
    state_old_fc_[level][idim] = amrex::MultiFab(
        amrex::convert(ba, amrex::IntVect::TheDimensionVector(idim)), dm, 1, 1);
  }

  if (level > 0 && (do_reflux != 0)) {
    flux_reg_[level] = std::make_unique<amrex::YAFluxRegister>(
        ba, boxArray(level - 1), dm, DistributionMap(level - 1), Geom(level),
        Geom(level - 1), refRatio(level - 1), level, ncomp);
  }

  // set state_new_[lev] to desired initial condition
  setInitialConditionsAtLevel(level);

  // check that state_new_cc_[lev] is properly filled
  AMREX_ALWAYS_ASSERT(!state_new_cc_[level].contains_nan(0, ncomp));

  // fill ghost zones
  fillBoundaryConditions(state_new_cc_[level], state_new_cc_[level], level,
                         time);

  // copy to state_old_cc_ (including ghost zones)
  state_old_cc_[level].ParallelCopy(state_new_cc_[level], 0, 0, ncomp, nghost,
                                    nghost);
}

template <typename problem_t> struct setBoundaryFunctor {
  AMREX_GPU_DEVICE void
  operator()(const amrex::IntVect &iv, amrex::Array4<amrex::Real> const &dest,
             const int &dcomp, const int &numcomp,
             amrex::GeometryData const &geom, const amrex::Real &time,
             const amrex::BCRec *bcr, int bcomp, const int &orig_comp) const {
    AMRSimulation<problem_t>::setCustomBoundaryConditions(
        iv, dest, dcomp, numcomp, geom, time, bcr, bcomp, orig_comp);
  }
};

template <typename problem_t>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE void
AMRSimulation<problem_t>::setCustomBoundaryConditions(
    const amrex::IntVect &iv, amrex::Array4<amrex::Real> const &dest, int dcomp,
    int numcomp, amrex::GeometryData const &geom, const amrex::Real time,
    const amrex::BCRec *bcr, int bcomp, int orig_comp) {
  // user should implement if needed using template specialization
  // (This is only called when amrex::BCType::ext_dir is set for a given
  // boundary.)

  // set boundary condition for cell 'iv'
}

template <typename problem_t>
void AMRSimulation<problem_t>::fillBoundaryConditions(amrex::MultiFab &S_filled,
                                                      amrex::MultiFab &state,
                                                      int const lev,
                                                      amrex::Real const time) {
  BL_PROFILE("AMRSimulation::fillBoundaryConditions()");

  // On a single level, any periodic boundaries are filled first
  // 	then built-in boundary conditions are filled (with amrex::FilccCell()),
  //	then user-defined Dirichlet boundary conditions are filled.
  // (N.B.: The user-defined boundary function is called for *all* ghost cells.)

  // [NOTE: If user-defined and periodic boundaries are both used
  //  (for different coordinate dimensions), the edge/corner cells *will* be
  //  filled by amrex::FilccCell(). Remember to fill *all* variables in the
  //  MultiFab, both hydro and radiation).

  if (lev > 0) { // refined level
    amrex::Vector<amrex::MultiFab *> fineData{&state};
    amrex::Vector<amrex::Real> fineTime = {time};
    amrex::Vector<amrex::MultiFab *> coarseData;
    amrex::Vector<amrex::Real> coarseTime;

    // returns old state, new state, or both depending on 'time'
    GetData(lev - 1, time, coarseData, coarseTime);
    AMREX_ASSERT(!state.contains_nan(0, state.nComp()));

    for (int i = 0; i < coarseData.size(); ++i) {
      AMREX_ASSERT(!coarseData[i]->contains_nan(0, state.nComp()));
      AMREX_ASSERT(!coarseData[i]->contains_nan()); // check ghost zones
    }

    FillPatchWithData(lev, time, S_filled, coarseData, coarseTime, fineData,
                      fineTime, 0, S_filled.nComp());
  } else { // level 0
    // fill internal and periodic boundaries, ignoring corners (cross=true)
    // (there is no performance benefit for this in practice)
    // state.FillBoundary(geom[lev].periodicity(), true);
    state.FillBoundary(geom[lev].periodicity());

    if (!geom[lev].isAllPeriodic()) {
      amrex::GpuBndryFuncFab<setBoundaryFunctor<problem_t>> boundaryFunctor(
          setBoundaryFunctor<problem_t>{});
      amrex::PhysBCFunct<amrex::GpuBndryFuncFab<setBoundaryFunctor<problem_t>>>
          physicalBoundaryFunctor(geom[lev], boundaryConditions_,
                                  boundaryFunctor);
      // fill physical boundaries
      physicalBoundaryFunctor(state, 0, state.nComp(), state.nGrowVect(), time,
                              0);
    }
  }

  // ensure that there are no NaNs (can happen when domain boundary filling is
  // unimplemented or malfunctioning)
  AMREX_ASSERT(!S_filled.contains_nan(0, S_filled.nComp()));
  AMREX_ASSERT(!S_filled.contains_nan()); // check ghost zones
}

// Compute a new multifab 'mf' by copying in state from given data and filling
// ghost cells
template <typename problem_t>
void AMRSimulation<problem_t>::FillPatchWithData(
    int lev, amrex::Real time, amrex::MultiFab &mf,
    amrex::Vector<amrex::MultiFab *> &coarseData,
    amrex::Vector<amrex::Real> &coarseTime,
    amrex::Vector<amrex::MultiFab *> &fineData,
    amrex::Vector<amrex::Real> &fineTime, int icomp, int ncomp) {
  BL_PROFILE("AMRSimulation::FillPatchWithData()");

  // create functor to fill ghost zones at domain boundaries
  // (note that domain boundaries may be present at any refinement level)
  amrex::GpuBndryFuncFab<setBoundaryFunctor<problem_t>> boundaryFunctor(
      setBoundaryFunctor<problem_t>{});
  amrex::PhysBCFunct<amrex::GpuBndryFuncFab<setBoundaryFunctor<problem_t>>>
      finePhysicalBoundaryFunctor(geom[lev], boundaryConditions_,
                                  boundaryFunctor);

  if (lev == 0) { // NOTE: used by RemakeLevel
    // copies interior zones, fills ghost zones
    amrex::FillPatchSingleLevel(mf, time, fineData, fineTime, 0, icomp, ncomp,
                                geom[lev], finePhysicalBoundaryFunctor, 0);
  } else {
    amrex::PhysBCFunct<amrex::GpuBndryFuncFab<setBoundaryFunctor<problem_t>>>
        coarsePhysicalBoundaryFunctor(geom[lev - 1], boundaryConditions_,
                                      boundaryFunctor);

    // use CellConservativeLinear interpolation onto fine grid
    amrex::Interpolater *mapper = &amrex::cell_cons_interp;
    // amrex::MFInterpolater *mapper = &amrex::mf_cell_cons_interp;
    // amrex::MFInterpolater *mapper = &amrex::mf_pc_interp;

    // copies interior zones, fills ghost zones with space-time interpolated
    // data
    amrex::FillPatchTwoLevels(mf, time, coarseData, coarseTime, fineData,
                              fineTime, 0, icomp, ncomp, geom[lev - 1],
                              geom[lev], coarsePhysicalBoundaryFunctor, 0,
                              finePhysicalBoundaryFunctor, 0, refRatio(lev - 1),
                              mapper, boundaryConditions_, 0);
  }
}

// Compute a new multifab 'mf' by copying in state from valid region and
// filling ghost cells
template <typename problem_t>
void AMRSimulation<problem_t>::FillPatch(int lev, amrex::Real time,
                                         amrex::MultiFab &mf, int icomp,
                                         int ncomp) {
  BL_PROFILE("AMRSimulation::FillPatch()");

  amrex::Vector<amrex::MultiFab *> cmf;
  amrex::Vector<amrex::MultiFab *> fmf;
  amrex::Vector<amrex::Real> ctime;
  amrex::Vector<amrex::Real> ftime;

  if (lev == 0) {
    // in this case, should return either state_new_[lev] or state_old_[lev]
    GetData(lev, time, fmf, ftime);
  } else {
    // in this case, should return either state_new_[lev] or state_old_[lev]
    GetData(lev, time, fmf, ftime);
    // returns old state, new state, or both depending on 'time'
    GetData(lev - 1, time, cmf, ctime);
  }

  FillPatchWithData(lev, time, mf, cmf, ctime, fmf, ftime, icomp, ncomp);
}

// Fill an entire multifab by interpolating from the coarser level
// this comes into play when a new level of refinement appears
template <typename problem_t>
void AMRSimulation<problem_t>::FillCoarsePatch(int lev, amrex::Real time,
                                               amrex::MultiFab &mf, int icomp,
                                               int ncomp) {
  BL_PROFILE("AMRSimulation::FillCoarsePatch()");

  AMREX_ASSERT(lev > 0);

  amrex::Vector<amrex::MultiFab *> cmf;
  amrex::Vector<amrex::Real> ctime;
  GetData(lev - 1, time, cmf, ctime);

  if (cmf.size() != 1) {
    amrex::Abort("FillCoarsePatch: how did this happen?");
  }

  amrex::GpuBndryFuncFab<setBoundaryFunctor<problem_t>> boundaryFunctor(
      setBoundaryFunctor<problem_t>{});
  amrex::PhysBCFunct<amrex::GpuBndryFuncFab<setBoundaryFunctor<problem_t>>>
      finePhysicalBoundaryFunctor(geom[lev], boundaryConditions_,
                                  boundaryFunctor);
  amrex::PhysBCFunct<amrex::GpuBndryFuncFab<setBoundaryFunctor<problem_t>>>
      coarsePhysicalBoundaryFunctor(geom[lev - 1], boundaryConditions_,
                                    boundaryFunctor);

  // use CellConservativeLinear interpolation onto fine grid
  amrex::MFInterpolater *mapper = &amrex::mf_cell_cons_interp;
  // amrex::MFInterpolater *mapper = &amrex::mf_pc_interp;

  amrex::InterpFromCoarseLevel(
      mf, time, *cmf[0], 0, icomp, ncomp, geom[lev - 1], geom[lev],
      coarsePhysicalBoundaryFunctor, 0, finePhysicalBoundaryFunctor, 0,
      refRatio(lev - 1), mapper, boundaryConditions_, 0);
}

// utility to copy in data from state_old_ and/or state_new_ into another
// multifab
template <typename problem_t>
void AMRSimulation<problem_t>::GetData(int lev, amrex::Real time,
                                       amrex::Vector<amrex::MultiFab *> &data,
                                       amrex::Vector<amrex::Real> &datatime) {
  BL_PROFILE("AMRSimulation::GetData()");

  data.clear();
  datatime.clear();

  const amrex::Real teps =
      (tNew_[lev] - tOld_[lev]) * 1.e-3; // generous roundoff error threshold

  if (time > tNew_[lev] - teps &&
      time < tNew_[lev] + teps) { // if time == tNew_[lev] within roundoff
    data.push_back(&state_new_cc_[lev]);
    datatime.push_back(tNew_[lev]);
  } else if (time > tOld_[lev] - teps &&
             time <
                 tOld_[lev] + teps) { // if time == tOld_[lev] within roundoff
    data.push_back(&state_old_cc_[lev]);
    datatime.push_back(tOld_[lev]);
  } else { // otherwise return both old and new states for interpolation
    data.push_back(&state_old_cc_[lev]);
    data.push_back(&state_new_cc_[lev]);
    datatime.push_back(tOld_[lev]);
    datatime.push_back(tNew_[lev]);
  }
}

// average down on all levels
template <typename problem_t> void AMRSimulation<problem_t>::AverageDown() {
  BL_PROFILE("AMRSimulation::AverageDown()");

  for (int lev = finest_level - 1; lev >= 0; --lev) {
    AverageDownTo(lev);
  }
}

// set covered coarse cells to be the average of overlying fine cells
template <typename problem_t>
void AMRSimulation<problem_t>::AverageDownTo(int crse_lev) {
  BL_PROFILE("AMRSimulation::AverageDownTo()");

  amrex::average_down(state_new_cc_[crse_lev + 1], state_new_cc_[crse_lev],
                      geom[crse_lev + 1], geom[crse_lev], 0,
                      state_new_cc_[crse_lev].nComp(), refRatio(crse_lev));
}

// get plotfile name
template <typename problem_t>
auto AMRSimulation<problem_t>::PlotFileName(int lev) const -> std::string {
  return amrex::Concatenate(plot_file, lev, 5);
}

template <typename problem_t>
auto AMRSimulation<problem_t>::PlotFileMFAtLevel(int lev) const
    -> amrex::MultiFab {
  // Combine state_new_cc_[lev] and derived variables in a new MF
  int comp = 0;
  const int nGrow = 0;
  const int nCompState = state_new_cc_[lev].nComp();
  const int nCompDeriv = derivedNames_.size();
  const int nCompPlotMF = nCompState + nCompDeriv;
  amrex::MultiFab plotMF(grids[lev], dmap[lev], nCompPlotMF, nGrow);

  // Copy data from state variables
  for (int i = 0; i < nCompState; i++) {
    amrex::MultiFab::Copy(plotMF, state_new_cc_[lev], i, comp, 1, nGrow);
    comp++;
  }

  // Compute derived vars
  for (auto const &dname : derivedNames_) {
    ComputeDerivedVar(lev, dname, plotMF, comp);
    comp++;
  }

  return plotMF;
}

// put together an array of multifabs for writing
template <typename problem_t>
auto AMRSimulation<problem_t>::PlotFileMF() const
    -> amrex::Vector<amrex::MultiFab> {
  amrex::Vector<amrex::MultiFab> r;
  for (int i = 0; i <= finest_level; ++i) {
    r.push_back(PlotFileMFAtLevel(i));
  }
  return r;
}

// write plotfile to disk
template <typename problem_t>
void AMRSimulation<problem_t>::WritePlotFile() const {
  BL_PROFILE("AMRSimulation::WritePlotFile()");

  const std::string &plotfilename = PlotFileName(istep[0]);
  amrex::Vector<amrex::MultiFab> mf = PlotFileMF();
  amrex::Vector<const amrex::MultiFab *> mf_ptr = amrex::GetVecOfConstPtrs(mf);

  amrex::Vector<std::string> varnames;
  varnames.insert(varnames.end(), componentNames_.begin(),
                  componentNames_.end());
  varnames.insert(varnames.end(), derivedNames_.begin(), derivedNames_.end());

  amrex::Print() << "Writing plotfile " << plotfilename << "\n";

#ifdef AMREX_USE_HDF5
  amrex::WriteMultiLevelPlotfileHDF5(plotfilename, finest_level + 1, mf_ptr,
                                     varnames, Geom(), tNew_[0], istep,
                                     refRatio());
#else
  amrex::WriteMultiLevelPlotfile(plotfilename, finest_level + 1, mf_ptr,
                                 varnames, Geom(), tNew_[0], istep, refRatio());
#endif
}

template <typename problem_t>
void AMRSimulation<problem_t>::WriteCheckpointFile() const {
  BL_PROFILE("AMRSimulation::WriteCheckpointFile()");

  // chk00010            write a checkpoint file with this root directory
  // chk00010/Header     this contains information you need to save (e.g.,
  // finest_level, t_new, etc.) and also
  //                     the BoxArrays at each level
  // chk00010/Level_0/
  // chk00010/Level_1/
  // etc.                these subdirectories will hold the MultiFab data at
  // each level of refinement

  // checkpoint file name, e.g., chk00010
  const std::string &checkpointname = amrex::Concatenate(chk_file, istep[0]);

  amrex::Print() << "Writing checkpoint " << checkpointname << "\n";

  const int nlevels = finest_level + 1;

  // ---- prebuild a hierarchy of directories
  // ---- dirName is built first.  if dirName exists, it is renamed.  then build
  // ---- dirName/subDirPrefix_0 .. dirName/subDirPrefix_nlevels-1
  // ---- if callBarrier is true, call ParallelDescriptor::Barrier()
  // ---- after all directories are built
  // ---- ParallelDescriptor::IOProcessor() creates the directories
  amrex::PreBuildDirectorHierarchy(checkpointname, "Level_", nlevels, true);

  // write Header file
  if (amrex::ParallelDescriptor::IOProcessor()) {

    std::string HeaderFileName(checkpointname + "/Header");
    amrex::VisMF::IO_Buffer io_buffer(amrex::VisMF::IO_Buffer_Size);
    std::ofstream HeaderFile;
    HeaderFile.rdbuf()->pubsetbuf(io_buffer.dataPtr(), io_buffer.size());
    HeaderFile.open(HeaderFileName.c_str(), std::ofstream::out |
                                                std::ofstream::trunc |
                                                std::ofstream::binary);
    if (!HeaderFile.good()) {
      amrex::FileOpenFailed(HeaderFileName);
    }

    HeaderFile.precision(17);

    // write out title line
    HeaderFile << "Checkpoint file for QuokkaCode\n";

    // write out finest_level
    HeaderFile << finest_level << "\n";

    // write out array of istep
    for (int i = 0; i < istep.size(); ++i) {
      HeaderFile << istep[i] << " ";
    }
    HeaderFile << "\n";

    // write out array of dt
    for (int i = 0; i < dt_.size(); ++i) {
      HeaderFile << dt_[i] << " ";
    }
    HeaderFile << "\n";

    // write out array of t_new
    for (int i = 0; i < tNew_.size(); ++i) {
      HeaderFile << tNew_[i] << " ";
    }
    HeaderFile << "\n";

    // write the BoxArray at each level
    for (int lev = 0; lev <= finest_level; ++lev) {
      boxArray(lev).writeOn(HeaderFile);
      HeaderFile << '\n';
    }
  }

  // write the MultiFab data to, e.g., chk00010/Level_0/
  for (int lev = 0; lev <= finest_level; ++lev) {
    amrex::VisMF::Write(
        state_new_cc_[lev],
        amrex::MultiFabFileFullPrefix(lev, checkpointname, "Level_", "Cell"));
  }
}

// utility to skip to next line in Header
inline void GotoNextLine(std::istream &is) {
  constexpr std::streamsize bl_ignore_max{100000};
  is.ignore(bl_ignore_max, '\n');
}

template <typename problem_t>
void AMRSimulation<problem_t>::ReadCheckpointFile() {
  BL_PROFILE("AMRSimulation::ReadCheckpointFile()");

  amrex::Print() << "Restart from checkpoint " << restart_chkfile << "\n";

  // Header
  std::string File(restart_chkfile + "/Header");

  amrex::VisMF::IO_Buffer io_buffer(amrex::VisMF::GetIOBufferSize());

  amrex::Vector<char> fileCharPtr;
  amrex::ParallelDescriptor::ReadAndBcastFile(File, fileCharPtr);
  std::string fileCharPtrString(fileCharPtr.dataPtr());
  std::istringstream is(fileCharPtrString, std::istringstream::in);

  std::string line;
  std::string word;

  // read in title line
  std::getline(is, line);

  // read in finest_level
  is >> finest_level;
  GotoNextLine(is);

  // read in array of istep
  std::getline(is, line);
  {
    std::istringstream lis(line);
    int i = 0;
    while (lis >> word) {
      istep[i++] = std::stoi(word);
    }
  }

  // read in array of dt
  std::getline(is, line);
  {
    std::istringstream lis(line);
    int i = 0;
    while (lis >> word) {
      dt_[i++] = std::stod(word);
    }
  }

  // read in array of t_new
  std::getline(is, line);
  {
    std::istringstream lis(line);
    int i = 0;
    while (lis >> word) {
      tNew_[i++] = std::stod(word);
    }
  }

  for (int lev = 0; lev <= finest_level; ++lev) {

    // read in level 'lev' BoxArray from Header
    amrex::BoxArray ba;
    ba.readFrom(is);
    GotoNextLine(is);

    // create a distribution mapping
    amrex::DistributionMapping dm{ba, amrex::ParallelDescriptor::NProcs()};

    // set BoxArray grids and DistributionMapping dmap in AMReX_AmrMesh.H class
    SetBoxArray(lev, ba);
    SetDistributionMap(lev, dm);

    // build MultiFab and FluxRegister data
    int ncomp = ncomp_;
    int nghost = nghost_;
    state_old_cc_[lev].define(grids[lev], dmap[lev], ncomp, nghost);
    state_new_cc_[lev].define(grids[lev], dmap[lev], ncomp, nghost);
    max_signal_speed_[lev].define(ba, dm, 1, nghost);

    if (lev > 0 && (do_reflux != 0)) {
      flux_reg_[lev] = std::make_unique<amrex::YAFluxRegister>(
          ba, boxArray(lev - 1), dm, DistributionMap(lev - 1), Geom(lev),
          Geom(lev - 1), refRatio(lev - 1), lev, ncomp);
    }
  }

  // read in the MultiFab data
  for (int lev = 0; lev <= finest_level; ++lev) {
    amrex::VisMF::Read(
        state_new_cc_[lev],
        amrex::MultiFabFileFullPrefix(lev, restart_chkfile, "Level_", "Cell"));
  }
  areInitialConditionsDefined_ = true;
}

#endif // SIMULATION_HPP_
