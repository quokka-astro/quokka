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
#include "memory"

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
#include "AMReX_GpuControl.H"
#include "AMReX_GpuQualifiers.H"
#include "AMReX_INT.H"
#include "AMReX_IntVect.H"
#include "AMReX_MultiFabUtil.H"
#include "AMReX_ParallelDescriptor.H"
#include "AMReX_SPACE.H"
#include "AMReX_VisMF.H"
#include "AMReX_YAFluxRegister.H"
#include <AMReX_Geometry.H>
#include <AMReX_Gpu.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParmParse.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_Print.H>
#include <AMReX_Utility.H>
#include <csignal>
#include <iomanip>

// internal headers
#include "CheckNaN.hpp"
#include "math_impl.hpp"

// Main simulation class; solvers should inherit from this
template <typename problem_t> class AMRSimulation : public amrex::AmrCore
{
      public:
	amrex::Real maxDt_ = std::numeric_limits<double>::max(); // no limit by default
	amrex::Vector<int> istep;				 // which step?
	amrex::Vector<int> nsubsteps;				 // how many substeps on each level?
	amrex::Vector<amrex::Real> tNew_;			 // for state_new_
	amrex::Vector<amrex::Real> tOld_;			 // for state_old_
	amrex::Vector<amrex::Real> dt_;				 // timestep for each level
	amrex::Real stopTime_ = 1.0;				 // default
	amrex::Real cflNumber_ = 0.3;				 // default
	amrex::Long cycleCount_ = 0;
	amrex::Long maxTimesteps_ = 1e4; // default
	int plotfileInterval_ = 10;	 // -1 == no output
	int checkpointInterval_ = -1;	 // -1 == no output

	// constructors

	AMRSimulation(amrex::IntVect & /*gridDims*/, amrex::RealBox & /*boxSize*/,
		      amrex::Vector<amrex::BCRec> &boundaryConditions, const int ncomp)
	    : ncomp_(ncomp), ncompPrimitive_(ncomp)
	{
		initialize(boundaryConditions);
	}

	AMRSimulation(amrex::IntVect & /*gridDims*/, amrex::RealBox & /*boxSize*/,
		      amrex::Vector<amrex::BCRec> &boundaryConditions, const int ncomp,
		      const int ncompPrimitive)
	    : ncomp_(ncomp), ncompPrimitive_(ncompPrimitive)
	{
		initialize(boundaryConditions);
	}

	void initialize(amrex::Vector<amrex::BCRec> &boundaryConditions);
	void readParameters();
	void setInitialConditions();
	void evolve();
	void computeTimestep();

	virtual void computeMaxSignalLocal(int level) = 0;
	virtual void setInitialConditionsAtLevel(int level) = 0;
	virtual void advanceSingleTimestepAtLevel(int lev, amrex::Real time, amrex::Real dt_lev,
						  int iteration, int ncycle) = 0;
	virtual void computeAfterTimestep() = 0;

	// tag cells for refinement
	void ErrorEst(int lev, amrex::TagBoxArray &tags, amrex::Real time, int ngrow) override = 0;

	// Make a new level using provided BoxArray and DistributionMapping
	void MakeNewLevelFromCoarse(int lev, amrex::Real time, const amrex::BoxArray &ba,
				    const amrex::DistributionMapping &dm) override;

	// Remake an existing level using provided BoxArray and DistributionMapping
	void RemakeLevel(int lev, amrex::Real time, const amrex::BoxArray &ba,
			 const amrex::DistributionMapping &dm) override;

	// Delete level data
	void ClearLevel(int lev) override;

	// Make a new level from scratch using provided BoxArray and DistributionMapping
	void MakeNewLevelFromScratch(int lev, amrex::Real time, const amrex::BoxArray &ba,
				     const amrex::DistributionMapping &dm) override;

	// AMR utility functions
	void fillBoundaryConditions(amrex::MultiFab &S_filled, amrex::MultiFab &state, int lev,
				    amrex::Real time);
	void FillPatchWithData(int lev, amrex::Real time, amrex::MultiFab &mf,
			       amrex::Vector<amrex::MultiFab *> &coarseData,
			       amrex::Vector<amrex::Real> &coarseTime,
			       amrex::Vector<amrex::MultiFab *> &fineData,
			       amrex::Vector<amrex::Real> &fineTime, int icomp, int ncomp);
	void FillPatch(int lev, amrex::Real time, amrex::MultiFab &mf, int icomp, int ncomp);
	void FillCoarsePatch(int lev, amrex::Real time, amrex::MultiFab &mf, int icomp, int ncomp);
	void GetData(int lev, amrex::Real time, amrex::Vector<amrex::MultiFab *> &data,
		     amrex::Vector<amrex::Real> &datatime);
	void AverageDown();
	void AverageDownTo(int crse_lev);
	void timeStepWithSubcycling(int lev, amrex::Real time, int iteration);

	// boundary condition
	static void
	setCustomBoundaryConditions(const amrex::IntVect &iv,
				    amrex::Array4<amrex::Real> const &dest, int dcomp, int numcomp,
				    amrex::GeometryData const &geom, amrex::Real time,
				    const amrex::BCRec *bcr, int bcomp,
				    int orig_comp); // template specialized by problem generator

	// I/O functions
	[[nodiscard]] auto PlotFileName(int lev) const -> std::string;
	[[nodiscard]] auto PlotFileMF() const -> amrex::Vector<const amrex::MultiFab *>;
	void WritePlotFile() const;
	void WriteCheckpointFile() const;

      protected:
	amrex::Vector<amrex::BCRec> boundaryConditions_; // on level 0
	amrex::Vector<amrex::MultiFab> state_old_;
	amrex::Vector<amrex::MultiFab> state_new_;
	amrex::Vector<amrex::MultiFab> max_signal_speed_; // needed to compute CFL timestep

	// flux registers: store fluxes at coarse-fine interface for synchronization
	// this will be sized "nlevs_max+1"
	// NOTE: the flux register associated with flux_reg[lev] is associated with the lev/lev-1
	// interface (and has grid spacing associated with lev-1) therefore flux_reg[0] and
	// flux_reg[nlevs_max] are never actually used in the reflux operation
	amrex::Vector<std::unique_ptr<amrex::YAFluxRegister>> flux_reg_;

	// Nghost = number of ghost cells for each array
	int nghost_ = 4;	   // PPM needs nghost >= 3, PPM+flattening needs nghost >= 4
	int ncomp_ = NAN;	   // = number of components (conserved variables) for each array
	int ncompPrimitive_ = NAN; // number of primitive variables
	amrex::Vector<std::string> componentNames_;
	bool areInitialConditionsDefined_ = false;

	/// output parameters
	std::string plot_file{"plt"}; // plotfile prefix
	std::string chk_file{"chk"};  // checkpoint prefix

	/// AMR-specific parameters
	int regrid_int = 2; // regrid interval (number of sub-cycles)
	int do_reflux = 0; // 1 == reflux, 0 == no reflux
};

template <typename problem_t>
void AMRSimulation<problem_t>::initialize(amrex::Vector<amrex::BCRec> &boundaryConditions)
{
	int nlevs_max = max_level + 1;
	istep.resize(nlevs_max, 0);
	nsubsteps.resize(nlevs_max, 1);
	for (int lev = 1; lev <= max_level; ++lev) {
		nsubsteps[lev] = MaxRefRatio(lev - 1);
	}

	tNew_.resize(nlevs_max, 0.0);
	tOld_.resize(nlevs_max, -1.e100);
	dt_.resize(nlevs_max, 1.e100);
	state_new_.resize(nlevs_max);
	state_old_.resize(nlevs_max);
	max_signal_speed_.resize(nlevs_max);
	flux_reg_.resize(nlevs_max + 1);

	boundaryConditions_ = boundaryConditions;
}

template <typename problem_t> void AMRSimulation<problem_t>::readParameters()
{
	// ParmParse reads inputs from the *.inputs file
	amrex::ParmParse pp;

	// Default nsteps = 1e4
	pp.query("max_timesteps", maxTimesteps_);

	// Default CFL number == 0.3, set to whatever is in the file
	pp.query("cfl", cflNumber_);
}

template <typename problem_t> void AMRSimulation<problem_t>::setInitialConditions()
{
	// start simulation from the beginning
	const amrex::Real time = 0.0;
	InitFromScratch(time);
	AverageDown();

	if (checkpointInterval_ > 0) {
		WriteCheckpointFile();
	}

	if (plotfileInterval_ > 0) {
		WritePlotFile();
	}
}

template <typename problem_t> void AMRSimulation<problem_t>::computeTimestep()
{
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

template <typename problem_t> void AMRSimulation<problem_t>::evolve()
{
	AMREX_ALWAYS_ASSERT(areInitialConditionsDefined_);

	amrex::Real cur_time = tNew_[0];
	int last_plot_file_step = 0;

	// Main time loop
	for (int step = istep[0]; step < maxTimesteps_ && cur_time < stopTime_; ++step) {
		amrex::Print() << "\nCoarse STEP " << step + 1 << " starts ..." << std::endl;

		computeTimestep();

		int lev = 0;
		int iteration = 1;
		timeStepWithSubcycling(lev, cur_time, iteration);

		cur_time += dt_[0];
		++cycleCount_;
		computeAfterTimestep();

		if (amrex::ParallelDescriptor::IOProcessor()) {
			amrex::Print() << "Coarse STEP " << step + 1 << " ends."
				       << " TIME = " << cur_time << " DT = " << dt_[0] << std::endl;
		}

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

#ifdef AMREX_MEM_PROFILING
		{
			std::ostringstream ss;
			ss << "[STEP " << step + 1 << "]";
			MemProfiler::report(ss.str());
		}
#endif

		if (cur_time >= stopTime_ - 1.e-6 * dt_[0]) {
			break;
		}
	}

	if (plotfileInterval_ > 0 && istep[0] > last_plot_file_step) {
		WritePlotFile();
	}
}

template <typename problem_t>
void AMRSimulation<problem_t>::timeStepWithSubcycling(int lev, amrex::Real time, int iteration)
{
	if (regrid_int > 0) // We may need to regrid
	{

		// help keep track of whether a level was already regridded
		// from a coarser level call to regrid
		static amrex::Vector<int> last_regrid_step(max_level + 1, 0);

		// regrid changes level "lev+1" so we don't regrid on max_level
		// also make sure we don't regrid fine levels again if
		// it was taken care of during a coarser regrid
		if (lev < max_level && istep[lev] > last_regrid_step[lev]) {
			if (istep[lev] % regrid_int == 0) {
				// regrid could add newly refine levels (if finest_level <
				// max_level) so we save the previous finest level index
				int old_finest = finest_level;
				regrid(lev, time);

				// mark that we have regridded this level already
				for (int k = lev; k <= finest_level; ++k) {
					last_regrid_step[k] = istep[k];
				}

				// if there are newly created levels, set the time step
				for (int k = old_finest + 1; k <= finest_level; ++k) {
					dt_[k] = dt_[k - 1] / MaxRefRatio(k - 1);
				}
			}
		}
	}

	if (Verbose() && amrex::ParallelDescriptor::IOProcessor()) {
		amrex::Print() << "[Level " << lev << " step " << istep[lev] + 1 << "] ";
		amrex::Print() << "ADVANCE with time = " << tNew_[lev] << " dt = " << dt_[lev]
			       << std::endl;
	}

	// Advance a single level for a single time step, and update flux registers

	tOld_[lev] = tNew_[lev];
	tNew_[lev] += dt_[lev]; // critical that this is done *before* advanceAtLevel

	advanceSingleTimestepAtLevel(lev, time, dt_[lev], iteration, nsubsteps[lev]);

	++istep[lev];

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
			// do AMR reflux on coarse level (this level)
			amrex::MultiFab &S_coarse = state_new_[lev];
			flux_reg_[lev + 1]->Reflux(S_coarse);
		}

		AverageDownTo(lev); // average lev+1 down to lev
	}
}

// Make a new level using provided BoxArray and DistributionMapping and fill with interpolated
// coarse level data. Overrides the pure virtual function in AmrCore
template <typename problem_t>
void AMRSimulation<problem_t>::MakeNewLevelFromCoarse(int level, amrex::Real time,
						      const amrex::BoxArray &ba,
						      const amrex::DistributionMapping &dm)
{
	const int ncomp = state_new_[level - 1].nComp();
	const int nghost = state_new_[level - 1].nGrow();

	state_new_[level].define(ba, dm, ncomp, nghost);
	state_old_[level].define(ba, dm, ncomp, nghost);
	max_signal_speed_[level].define(ba, dm, 1, nghost);

	tNew_[level] = time;
	tOld_[level] = time - 1.e200;

	if (level > 0) {
		flux_reg_[level] = std::make_unique<amrex::YAFluxRegister>(
		    ba, grids[level - 1], dm, dmap[level - 1], geom[level], geom[level - 1],
		    refRatio(level - 1), level, ncomp);
	}

	FillCoarsePatch(level, time, state_new_[level], 0, ncomp);
}

// Remake an existing level using provided BoxArray and DistributionMapping and fill with existing
// fine and coarse data. Overrides the pure virtual function in AmrCore
template <typename problem_t>
void AMRSimulation<problem_t>::RemakeLevel(int level, amrex::Real time, const amrex::BoxArray &ba,
					   const amrex::DistributionMapping &dm)
{
	const int ncomp = state_new_[level].nComp();
	const int nghost = state_new_[level].nGrow();

	amrex::MultiFab new_state(ba, dm, ncomp, nghost);
	amrex::MultiFab old_state(ba, dm, ncomp, nghost);
	amrex::MultiFab max_signal_speed(ba, dm, 1, nghost);

	FillPatch(level, time, new_state, 0, ncomp);

	std::swap(new_state, state_new_[level]);
	std::swap(old_state, state_old_[level]);
	std::swap(max_signal_speed, max_signal_speed_[level]);

	tNew_[level] = time;
	tOld_[level] = time - 1.e200;

	if (level > 0) {
		flux_reg_[level] = std::make_unique<amrex::YAFluxRegister>(
		    ba, grids[level - 1], dm, dmap[level - 1], geom[level], geom[level - 1],
		    refRatio(level - 1), level, ncomp);
	}
}

// Delete level data. Overrides the pure virtual function in AmrCore
template <typename problem_t> void AMRSimulation<problem_t>::ClearLevel(int level)
{
	state_new_[level].clear();
	state_old_[level].clear();
	max_signal_speed_[level].clear();
	flux_reg_[level].reset(nullptr);
}

// Make a new level from scratch using provided BoxArray and DistributionMapping.
// Only used during initialization. Overrides the pure virtual function in AmrCore
template <typename problem_t>
void AMRSimulation<problem_t>::MakeNewLevelFromScratch(int level, amrex::Real time,
						       const amrex::BoxArray &ba,
						       const amrex::DistributionMapping &dm)
{
	const int ncomp = ncomp_;
	const int nghost = nghost_;

	state_new_[level].define(ba, dm, ncomp, nghost);
	state_old_[level].define(ba, dm, ncomp, nghost);
	max_signal_speed_[level].define(ba, dm, 1, nghost);

	tNew_[level] = time;
	tOld_[level] = time - 1.e200;

	if (level > 0) {
		flux_reg_[level] = std::make_unique<amrex::YAFluxRegister>(
		    ba, grids[level - 1], dm, dmap[level - 1], geom[level], geom[level - 1],
		    refRatio(level - 1), level, ncomp);
	}

	setInitialConditionsAtLevel(level);
}

template <typename problem_t> struct setBoundaryFunctor {
	AMREX_GPU_DEVICE void operator()(const amrex::IntVect &iv,
					 amrex::Array4<amrex::Real> const &dest, const int &dcomp,
					 const int &numcomp, amrex::GeometryData const &geom,
					 const amrex::Real &time, const amrex::BCRec *bcr,
					 int bcomp, const int &orig_comp) const
	{
		AMRSimulation<problem_t>::setCustomBoundaryConditions(
		    iv, dest, dcomp, numcomp, geom, time, bcr, bcomp, orig_comp);
	}
};

template <typename problem_t>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE void AMRSimulation<problem_t>::setCustomBoundaryConditions(
    const amrex::IntVect &iv, amrex::Array4<amrex::Real> const &dest, int dcomp, int numcomp,
    amrex::GeometryData const &geom, const amrex::Real time, const amrex::BCRec *bcr, int bcomp,
    int orig_comp)
{
	// user should implement if needed using template specialization
	// (This is only called when amrex::BCType::ext_dir is set for a given boundary.)

	// set boundary condition for cell 'iv'
}

template <typename problem_t>
void AMRSimulation<problem_t>::fillBoundaryConditions(amrex::MultiFab &S_filled,
						      amrex::MultiFab &state, int lev,
						      amrex::Real time)
{
	amrex::Vector<amrex::MultiFab *> fineData{&state};
	amrex::Vector<amrex::Real> fineTime{time};
	amrex::Vector<amrex::MultiFab *> coarseData;
	amrex::Vector<amrex::Real> coarseTime;

	if (lev > 0) {
		// on coarse level, returns old state, new state, or both depending on 'time'
		GetData(lev - 1, time, coarseData, coarseTime);
	}

	FillPatchWithData(lev, time, S_filled, coarseData, coarseTime, fineData, fineTime, 0,
			  S_filled.nComp());

	// ensure that there are no NaNs (can happen when domain boundary filling is unimplemented
	// or malfunctioning)
	AMREX_ASSERT(!S_filled.contains_nan(0, S_filled.nComp()));
}

// Compute a new multifab 'mf' by copying in state from given data and filling ghost cells
template <typename problem_t>
void AMRSimulation<problem_t>::FillPatchWithData(int lev, amrex::Real time, amrex::MultiFab &mf,
						 amrex::Vector<amrex::MultiFab *> &coarseData,
						 amrex::Vector<amrex::Real> &coarseTime,
						 amrex::Vector<amrex::MultiFab *> &fineData,
						 amrex::Vector<amrex::Real> &fineTime, int icomp,
						 int ncomp)
{
	// create functor to fill ghost zones at domain boundaries
	// (note that domain boundaries may be present at any refinement level)
	amrex::GpuBndryFuncFab<setBoundaryFunctor<problem_t>> boundaryFunctor(
	    setBoundaryFunctor<problem_t>{});
	amrex::PhysBCFunct<amrex::GpuBndryFuncFab<setBoundaryFunctor<problem_t>>>
	    finePhysicalBoundaryFunctor(geom[lev], boundaryConditions_, boundaryFunctor);

	if (lev == 0) {
		// copies interior zones, fills ghost zones
		amrex::FillPatchSingleLevel(mf, time, fineData, fineTime, 0, icomp, ncomp,
					    geom[lev], finePhysicalBoundaryFunctor, 0);
	} else {
		amrex::PhysBCFunct<amrex::GpuBndryFuncFab<setBoundaryFunctor<problem_t>>>
		    coarsePhysicalBoundaryFunctor(geom[lev - 1], boundaryConditions_,
						  boundaryFunctor);

		// use CellConservativeLinear interpolation onto fine grid
		amrex::Interpolater *mapper = &amrex::cell_cons_interp;

		// copies interior zones, fills ghost zones with space-time interpolated data
		amrex::FillPatchTwoLevels(mf, time, coarseData, coarseTime, fineData, fineTime, 0,
					  icomp, ncomp, geom[lev - 1], geom[lev],
					  coarsePhysicalBoundaryFunctor, 0,
					  finePhysicalBoundaryFunctor, 0, refRatio(lev - 1), mapper,
					  boundaryConditions_, 0);
	}
}

// Compute a new multifab 'mf' by copying in state from valid region and filling ghost cells
template <typename problem_t>
void AMRSimulation<problem_t>::FillPatch(int lev, amrex::Real time, amrex::MultiFab &mf, int icomp,
					 int ncomp)
{
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
void AMRSimulation<problem_t>::FillCoarsePatch(int lev, amrex::Real time, amrex::MultiFab &mf,
					       int icomp, int ncomp)
{
	AMREX_ASSERT(lev > 0);

	amrex::Vector<amrex::MultiFab *> cmf;
	amrex::Vector<amrex::Real> ctime;
	GetData(lev - 1, time, cmf, ctime);
	amrex::Interpolater *mapper = &amrex::cell_cons_interp;

	if (cmf.size() != 1) {
		amrex::Abort("FillCoarsePatch: how did this happen?");
	}

	amrex::GpuBndryFuncFab<setBoundaryFunctor<problem_t>> boundaryFunctor(
	    setBoundaryFunctor<problem_t>{});
	amrex::PhysBCFunct<amrex::GpuBndryFuncFab<setBoundaryFunctor<problem_t>>>
	    finePhysicalBoundaryFunctor(geom[lev], boundaryConditions_, boundaryFunctor);
	amrex::PhysBCFunct<amrex::GpuBndryFuncFab<setBoundaryFunctor<problem_t>>>
	    coarsePhysicalBoundaryFunctor(geom[lev - 1], boundaryConditions_, boundaryFunctor);

	amrex::InterpFromCoarseLevel(mf, time, *cmf[0], 0, icomp, ncomp, geom[lev - 1], geom[lev],
				     coarsePhysicalBoundaryFunctor, 0, finePhysicalBoundaryFunctor,
				     0, refRatio(lev - 1), mapper, boundaryConditions_, 0);
}

// utility to copy in data from state_old_ and/or state_new_ into another multifab
template <typename problem_t>
void AMRSimulation<problem_t>::GetData(int lev, amrex::Real time,
				       amrex::Vector<amrex::MultiFab *> &data,
				       amrex::Vector<amrex::Real> &datatime)
{
	data.clear();
	datatime.clear();

	const amrex::Real teps =
	    (tNew_[lev] - tOld_[lev]) * 1.e-3; // generous roundoff error threshold

	if (time > tNew_[lev] - teps &&
	    time < tNew_[lev] + teps) { // if time == tNew_[lev] within roundoff
		data.push_back(&state_new_[lev]);
		datatime.push_back(tNew_[lev]);
	} else if (time > tOld_[lev] - teps &&
		   time < tOld_[lev] + teps) { // if time == tOld_[lev] within roundoff
		data.push_back(&state_old_[lev]);
		datatime.push_back(tOld_[lev]);
	} else { // otherwise return both old and new states for interpolation
		data.push_back(&state_old_[lev]);
		data.push_back(&state_new_[lev]);
		datatime.push_back(tOld_[lev]);
		datatime.push_back(tNew_[lev]);
	}
}

// average down on all levels
template <typename problem_t> void AMRSimulation<problem_t>::AverageDown()
{
	for (int lev = finest_level - 1; lev >= 0; --lev) {
		AverageDownTo(lev);
	}
}

// set covered coarse cells to be the average of overlying fine cells
template <typename problem_t> void AMRSimulation<problem_t>::AverageDownTo(int crse_lev)
{
	amrex::average_down(state_new_[crse_lev + 1], state_new_[crse_lev], geom[crse_lev + 1],
			    geom[crse_lev], 0, state_new_[crse_lev].nComp(), refRatio(crse_lev));
}

// get plotfile name
template <typename problem_t>
auto AMRSimulation<problem_t>::PlotFileName(int lev) const -> std::string
{
	return amrex::Concatenate(plot_file, lev, 5);
}

// put together an array of multifabs for writing
template <typename problem_t>
auto AMRSimulation<problem_t>::PlotFileMF() const -> amrex::Vector<const amrex::MultiFab *>
{
	amrex::Vector<const amrex::MultiFab *> r;
	for (int i = 0; i <= finest_level; ++i) {
		r.push_back(&state_new_[i]);
	}
	return r;
}

// write plotfile to disk
template <typename problem_t> void AMRSimulation<problem_t>::WritePlotFile() const
{
	const std::string &plotfilename = PlotFileName(istep[0]);
	const auto &mf = PlotFileMF();
	const auto &varnames = componentNames_;

	amrex::Print() << "Writing plotfile " << plotfilename << "\n";

	amrex::WriteMultiLevelPlotfile(plotfilename, finest_level + 1, mf, varnames, Geom(),
				       tNew_[0], istep, refRatio());
}

template <typename problem_t> void AMRSimulation<problem_t>::WriteCheckpointFile() const
{

	// chk00010            write a checkpoint file with this root directory
	// chk00010/Header     this contains information you need to save (e.g., finest_level,
	// t_new, etc.) and also
	//                     the BoxArrays at each level
	// chk00010/Level_0/
	// chk00010/Level_1/
	// etc.                these subdirectories will hold the MultiFab data at each level of
	// refinement

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
		HeaderFile.open(HeaderFileName.c_str(),
				std::ofstream::out | std::ofstream::trunc | std::ofstream::binary);
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
		amrex::VisMF::Write(state_new_[lev], amrex::MultiFabFileFullPrefix(
							 lev, checkpointname, "Level_", "Cell"));
	}
}

#endif // SIMULATION_HPP_