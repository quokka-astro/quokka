#ifndef ADVECTION_SIMULATION_HPP_ // NOLINT
#define ADVECTION_SIMULATION_HPP_
//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file AdvectionSimulation.hpp
/// \brief Implements classes and functions to organise the overall setup,
/// timestepping, solving, and I/O of a simulation for linear advection.

#include "AMReX.H"
#include "AMReX_Arena.H"
#include "AMReX_Array.H"
#include "AMReX_Array4.H"
#include "AMReX_BLassert.H"
#include "AMReX_Box.H"
#include "AMReX_DistributionMapping.H"
#include "AMReX_FArrayBox.H"
#include "AMReX_GpuControl.H"
#include "AMReX_GpuLaunchFunctsC.H"
#include "AMReX_GpuUtility.H"
#include "AMReX_IntVect.H"
#include "AMReX_MultiFab.H"
#include "AMReX_REAL.H"
#include "AMReX_SPACE.H"
#include "AMReX_TagBox.H"
#include "AMReX_Utility.H"

#include "AMReX_YAFluxRegister.H"
#include "ArrayView.hpp"
#include "fmt/core.h"
#include "linear_advection.hpp"
#include "simulation.hpp"

// Simulation class should be initialized only once per program (i.e., is a singleton)
template <typename problem_t> class AdvectionSimulation : public AMRSimulation<problem_t>
{
      public:
	using AMRSimulation<problem_t>::state_old_;
	using AMRSimulation<problem_t>::state_new_;
	using AMRSimulation<problem_t>::max_signal_speed_;

	using AMRSimulation<problem_t>::cflNumber_;
	using AMRSimulation<problem_t>::dt_;
	using AMRSimulation<problem_t>::ncomp_;
	using AMRSimulation<problem_t>::nghost_;
	using AMRSimulation<problem_t>::cycleCount_;
	using AMRSimulation<problem_t>::areInitialConditionsDefined_;
	using AMRSimulation<problem_t>::componentNames_;

	using AMRSimulation<problem_t>::fillBoundaryConditions;
	using AMRSimulation<problem_t>::grids;
	using AMRSimulation<problem_t>::dmap;
	using AMRSimulation<problem_t>::geom;
	using AMRSimulation<problem_t>::flux_reg_;
	using AMRSimulation<problem_t>::finest_level;

	AdvectionSimulation(amrex::IntVect &gridDims, amrex::RealBox &boxSize,
			    amrex::Vector<amrex::BCRec> &boundaryConditions, const int ncomp = 1)
	    : AMRSimulation<problem_t>(gridDims, boxSize, boundaryConditions, ncomp)
	{
		componentNames_ = {"density"};
	}

	void computeMaxSignalLocal(int level) override;
	void setInitialConditionsAtLevel(int level) override;
	void advanceSingleTimestepAtLevel(int lev, amrex::Real time, amrex::Real dt_lev,
					  int /*iteration*/, int /*ncycle*/) override;
	void computeAfterTimestep() override;
	// tag cells for refinement
	void ErrorEst(int lev, amrex::TagBoxArray &tags, amrex::Real time, int ngrow) override;

	auto computeFluxes(amrex::Array4<const amrex::Real> const &consVar,
			   const amrex::Box &indexRange, int nvars)
	    -> std::array<amrex::FArrayBox const, AMREX_SPACEDIM>;

	template <FluxDir DIR>
	void fluxFunction(amrex::Array4<const amrex::Real> const &consState,
			  amrex::FArrayBox &x1Flux, const amrex::Box &indexRange, int nvars);

	double advectionVx_ = 1.0;
	double advectionVy_ = 0.0;
	double advectionVz_ = 0.0;
};

template <typename problem_t>
void AdvectionSimulation<problem_t>::computeMaxSignalLocal(int const level)
{
	// loop over local grids, compute CFL timestep
	for (amrex::MFIter iter(state_new_[level]); iter.isValid(); ++iter) {
		const amrex::Box &indexRange = iter.validbox();
		auto const &stateOld = state_old_[level].const_array(iter);
		auto const &maxSignal = max_signal_speed_[level].array(iter);
		LinearAdvectionSystem<problem_t>::ComputeMaxSignalSpeed(stateOld, maxSignal,
									advectionVx_, indexRange);
	}
}

template <typename problem_t>
void AdvectionSimulation<problem_t>::setInitialConditionsAtLevel(int level)
{
	// do nothing -- user should implement using problem-specific template specialization
}

template <typename problem_t> void AdvectionSimulation<problem_t>::computeAfterTimestep()
{
	// do nothing -- user should implement using problem-specific template specialization
}

template <typename problem_t>
void AdvectionSimulation<problem_t>::ErrorEst(int lev, amrex::TagBoxArray &tags, amrex::Real  /*time*/,
					      int  /*ngrow*/)
{
	// tag cells for refinement
	const amrex::MultiFab &state = state_new_[lev];
	const amrex::Vector<amrex::Real> dens_threshold = {0.01, 0.1, 0.2};

	for (amrex::MFIter mfi(state); mfi.isValid(); ++mfi) {
		const amrex::Box &box = mfi.tilebox();
		const auto statearr = state.array(mfi);
		const auto tagarray = tags.array(mfi);

		amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
			amrex::Real threshold = NAN;
			if (lev < dens_threshold.size()) {
				threshold = dens_threshold[lev];
			}
			if (statearr(i, j, k, 0) > threshold) {
				tagarray(i, j, k) = amrex::TagBox::SET;
			}
		});
	}
}

template <typename problem_t>
void AdvectionSimulation<problem_t>::advanceSingleTimestepAtLevel(int lev, amrex::Real time,
								  amrex::Real dt_lev,
								  int /*iteration*/, int /*ncycle*/)
{
	// based on amrex/Tests/EB/CNS/Source/CNS_advance.cpp

	// since we are starting a new timestep, need to swap old and new states on this level
	std::swap(state_old_[lev], state_new_[lev]);

	// check state validity
	AMREX_ASSERT(!state_old_[lev].contains_nan(0, state_old_[lev].nComp()));

	// allocate new MultiFab to hold the boundary-filled state vector
	amrex::MultiFab Sborder(grids[lev], dmap[lev], state_old_[lev].nComp(), nghost_);

	// get geometry (used only for cell sizes)
	auto const &geomLevel = geom[lev];

	// get flux registers
	amrex::YAFluxRegister &fr_as_crse = *flux_reg_[lev + 1];
	amrex::YAFluxRegister &fr_as_fine = *flux_reg_[lev];
	if (lev < finest_level) {
		fr_as_crse.reset(); // set flux register to zero
	}

	// We use the RK2-SSP integrator in a method-of-lines framework. It needs 2 registers: one
	// to store the old timestep, and one to store the intermediate stage and final stage. The
	// intermediate stage and final stage re-use the same register.

	// update ghost zones [w/ old timestep]
	fillBoundaryConditions(Sborder, state_old_[lev], lev, time);

	// advance all grids on local processor (Stage 1 of integrator)
	for (amrex::MFIter iter(state_new_[lev]); iter.isValid(); ++iter) {
		const amrex::Box &indexRange = iter.validbox();
		auto const &stateOld = Sborder.const_array(iter);
		auto const &stateNew = state_new_[lev].array(iter);
		auto const &fluxArrays = computeFluxes(stateOld, indexRange, ncomp_);

		// Stage 1 of RK2-SSP
		LinearAdvectionSystem<problem_t>::PredictStep(
		    stateOld, stateNew,
		    {AMREX_D_DECL(fluxArrays[0].const_array(), fluxArrays[1].const_array(),
				  fluxArrays[2].const_array())},
		    dt_lev, geomLevel.CellSizeArray(), indexRange, ncomp_);

		// increment flux registers
		if (lev < finest_level) {
			fr_as_crse.CrseAdd(
			    iter, {AMREX_D_DECL(&fluxArrays[0], &fluxArrays[1], &fluxArrays[2])},
			    geomLevel.CellSize(), dt_lev, amrex::RunOn::Cpu);
		}
		if (lev != 0) {
			fr_as_fine.FineAdd(
			    iter, {AMREX_D_DECL(&fluxArrays[0], &fluxArrays[1], &fluxArrays[2])},
			    geomLevel.CellSize(), dt_lev, amrex::RunOn::Cpu);
		}
	}

	// update ghost zones [w/ intermediate stage stored in state_new_]
	fillBoundaryConditions(Sborder, state_new_[lev], lev, time + dt_lev);

	// advance all grids on local processor (Stage 2 of integrator)
	for (amrex::MFIter iter(state_new_[lev]); iter.isValid(); ++iter) {
		const amrex::Box &indexRange = iter.validbox();
		auto const &stateInOld = state_old_[lev].const_array(iter);
		auto const &stateInStar = Sborder.const_array(iter);
		auto const &stateOut = state_new_[lev].array(iter);
		auto const &fluxArrays = computeFluxes(stateInStar, indexRange, ncomp_);

		// Stage 2 of RK2-SSP
		LinearAdvectionSystem<problem_t>::AddFluxesRK2(
		    stateOut, stateInOld, stateInStar,
		    {AMREX_D_DECL(fluxArrays[0].const_array(), fluxArrays[1].const_array(),
				  fluxArrays[2].const_array())},
		    dt_lev, geomLevel.CellSizeArray(), indexRange, ncomp_);

		// increment flux registers
		if (lev < finest_level) {
			fr_as_crse.CrseAdd(
			    iter, {AMREX_D_DECL(&fluxArrays[0], &fluxArrays[1], &fluxArrays[2])},
			    geomLevel.CellSize(), dt_lev, amrex::RunOn::Cpu);
		}
		if (lev != 0) {
			fr_as_fine.FineAdd(
			    iter, {AMREX_D_DECL(&fluxArrays[0], &fluxArrays[1], &fluxArrays[2])},
			    geomLevel.CellSize(), dt_lev, amrex::RunOn::Cpu);
		}
	}
}

template <typename problem_t>
auto AdvectionSimulation<problem_t>::computeFluxes(amrex::Array4<const amrex::Real> const &consVar,
						   const amrex::Box &indexRange, const int nvars)
    -> std::array<amrex::FArrayBox const, AMREX_SPACEDIM>
{
	amrex::Box const &x1FluxRange = amrex::surroundingNodes(indexRange, 0);
	amrex::FArrayBox x1Flux(x1FluxRange, nvars, amrex::The_Async_Arena()); // node-centered in x
#if (AMREX_SPACEDIM >= 2)
	amrex::Box const &x2FluxRange = amrex::surroundingNodes(indexRange, 1);
	amrex::FArrayBox x2Flux(x2FluxRange, nvars, amrex::The_Async_Arena()); // node-centered in y
#endif
#if (AMREX_SPACEDIM == 3)
	amrex::Box const &x3FluxRange = amrex::surroundingNodes(indexRange, 2);
	amrex::FArrayBox x3Flux(x3FluxRange, nvars, amrex::The_Async_Arena()); // node-centered in z
#endif

	AMREX_D_TERM(fluxFunction<FluxDir::X1>(consVar, x1Flux, indexRange, nvars);
		     , fluxFunction<FluxDir::X2>(consVar, x2Flux, indexRange, nvars);
		     , fluxFunction<FluxDir::X3>(consVar, x3Flux, indexRange, nvars);)

	return {AMREX_D_DECL(std::move(x1Flux), std::move(x2Flux), std::move(x3Flux))};
}

template <typename problem_t>
template <FluxDir DIR>
void AdvectionSimulation<problem_t>::fluxFunction(amrex::Array4<const amrex::Real> const &consState,
						  amrex::FArrayBox &x1Flux,
						  const amrex::Box &indexRange, const int nvars)
{
	amrex::Real advectionVel = NAN;
	int dim = 0;
	if constexpr (DIR == FluxDir::X1) {
		advectionVel = advectionVx_;
		// [0 == x1 direction]
		dim = 0;
	} else if constexpr (DIR == FluxDir::X2) {
		advectionVel = advectionVy_;
		// [1 == x2 direction]
		dim = 1;
	} else if constexpr (DIR == FluxDir::X3) {
		advectionVel = advectionVz_;
		// [2 == x3 direction]
		dim = 2;
	}

	// extend box to include ghost zones
	amrex::Box const &ghostRange = amrex::grow(indexRange, nghost_);
	amrex::Box const &reconstructRange = amrex::grow(indexRange, 1);
	amrex::Box const &x1ReconstructRange = amrex::surroundingNodes(reconstructRange, dim);
	amrex::FArrayBox primVar(ghostRange, nvars, amrex::The_Async_Arena()); // cell-centered
	amrex::FArrayBox x1LeftState(x1ReconstructRange, nvars, amrex::The_Async_Arena());
	amrex::FArrayBox x1RightState(x1ReconstructRange, nvars, amrex::The_Async_Arena());

	// cell-centered kernel
	LinearAdvectionSystem<problem_t>::ConservedToPrimitive(consState, primVar.array(),
							       ghostRange, nvars);

	// mixed interface/cell-centered kernel
	LinearAdvectionSystem<problem_t>::template ReconstructStatesPPM<DIR>(
	    primVar.array(), x1LeftState.array(), x1RightState.array(), reconstructRange,
	    x1ReconstructRange, nvars);
	// LinearAdvectionSystem<problem_t>::template ReconstructStatesConstant<DIR>(
	//    primVar.array(), x1LeftState.array(), x1RightState.array(), x1ReconstructRange,
	//    nvars);

	// interface-centered kernel
	amrex::Box const &x1FluxRange = amrex::surroundingNodes(indexRange, dim);

	LinearAdvectionSystem<problem_t>::template ComputeFluxes<DIR>(
	    x1Flux.array(), x1LeftState.array(), x1RightState.array(), advectionVel, x1FluxRange,
	    nvars);
}

#endif // ADVECTION_SIMULATION_HPP_