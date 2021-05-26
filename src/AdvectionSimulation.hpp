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

#include "AMReX_Arena.H"
#include "AMReX_Array4.H"
#include "AMReX_BLassert.H"
#include "AMReX_Box.H"
#include "AMReX_FArrayBox.H"
#include "AMReX_IntVect.H"
#include "AMReX_REAL.H"
#include "AMReX_Utility.H"

#include "ArrayView.hpp"
#include "fmt/core.h"
#include "linear_advection.hpp"
#include "simulation.hpp"

// Simulation class should be initialized only once per program (i.e., is a singleton)
template <typename problem_t> class AdvectionSimulation : public SingleLevelSimulation<problem_t>
{
      public:
	using SingleLevelSimulation<problem_t>::simGeometry_;
	using SingleLevelSimulation<problem_t>::state_old_;
	using SingleLevelSimulation<problem_t>::state_new_;
	using SingleLevelSimulation<problem_t>::max_signal_speed_;

	using SingleLevelSimulation<problem_t>::cflNumber_;
	using SingleLevelSimulation<problem_t>::dx_;
	using SingleLevelSimulation<problem_t>::dt_;
	using SingleLevelSimulation<problem_t>::ncomp_;
	using SingleLevelSimulation<problem_t>::nghost_;
	using SingleLevelSimulation<problem_t>::tNow_;
	using SingleLevelSimulation<problem_t>::cycleCount_;
	using SingleLevelSimulation<problem_t>::areInitialConditionsDefined_;

	AdvectionSimulation(amrex::IntVect &gridDims, amrex::RealBox &boxSize,
			    amrex::Vector<amrex::BCRec> &boundaryConditions)
	    : SingleLevelSimulation<problem_t>(gridDims, boxSize, boundaryConditions)
	{
	}

	void computeMaxSignalLocal() override;
	void setInitialConditions() override;
	void advanceSingleTimestep() override;
	void stageOneRK2SSP(amrex::Array4<const amrex::Real> const &consVarOld,
			    amrex::Array4<amrex::Real> const &consVarNew,
			    const amrex::Box &indexRange, int nvars);
	void stageTwoRK2SSP(amrex::Array4<const amrex::Real> const &consVar,
			    amrex::Array4<amrex::Real> const &consVarNew,
			    const amrex::Box &indexRange, int nvars);

	template <FluxDir DIR>
	void fluxFunction(amrex::Array4<const amrex::Real> const &consState,
			  amrex::FArrayBox &x1Flux, const amrex::Box &indexRange, int nvars);

      protected:
	const double advectionVx_ = 1.0;
	const double advectionVy_ = 0.0;
};

template <typename problem_t> void AdvectionSimulation<problem_t>::computeMaxSignalLocal()
{
	// loop over local grids, compute CFL timestep
	for (amrex::MFIter iter(state_new_); iter.isValid(); ++iter) {
		const amrex::Box &indexRange = iter.validbox();
		auto const &stateOld = state_old_.const_array(iter);
		auto const &maxSignal = max_signal_speed_.array(iter);
		LinearAdvectionSystem<problem_t>::ComputeMaxSignalSpeed(stateOld, maxSignal,
									advectionVx_, indexRange);
	}
}

template <typename problem_t> void AdvectionSimulation<problem_t>::setInitialConditions()
{
	// do nothing -- user should implement using problem-specific template specialization
}

template <typename problem_t> void AdvectionSimulation<problem_t>::advanceSingleTimestep()
{
	// We use the RK2-SSP method here. It needs two registers: one to store the old timestep,
	// and another to store the intermediate stage (which is reused for the final stage).

	// update ghost zones [old timestep]
	state_old_.FillBoundary(simGeometry_.periodicity());
	AMREX_ASSERT(!state_old_.contains_nan(0, ncomp_));

	// advance all grids on local processor (Stage 1 of integrator)
	for (amrex::MFIter iter(state_new_); iter.isValid(); ++iter) {
		const amrex::Box &indexRange = iter.validbox(); // 'validbox' == exclude ghost zones
		auto const &stateOld = state_old_.const_array(iter);
		auto const &stateNew = state_new_.array(iter);
		stageOneRK2SSP(stateOld, stateNew, indexRange, ncomp_);
	}

	// update ghost zones [intermediate stage stored in state_new_]
	state_new_.FillBoundary(simGeometry_.periodicity());
	AMREX_ASSERT(!state_new_.contains_nan(0, ncomp_));

	// advance all grids on local processor (Stage 2 of integrator)
	for (amrex::MFIter iter(state_new_); iter.isValid(); ++iter) {
		const amrex::Box &indexRange = iter.validbox(); // 'validbox' == exclude ghost zones
		auto const &stateOld = state_old_.const_array(iter);
		auto const &stateNew = state_new_.array(iter);
		stageTwoRK2SSP(stateOld, stateNew, indexRange, ncomp_);
	}
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
	}

	// extend box to include ghost zones
	amrex::Box const &ghostRange = amrex::grow(indexRange, nghost_);
	// N.B.: A one-zone layer around the cells must be fully reconstructed in order for PPM to
	// work.
	amrex::Box const &reconstructRange = amrex::grow(indexRange, 1);
	amrex::Box const &x1ReconstructRange = amrex::surroundingNodes(reconstructRange, dim);

	// Allocate temporary arrays using CUDA stream async allocator (or equivalent)
	amrex::FArrayBox primVar(ghostRange, nvars, amrex::The_Async_Arena()); // cell-centered
	amrex::FArrayBox x1LeftState(x1ReconstructRange, nvars, amrex::The_Async_Arena());
	amrex::FArrayBox x1RightState(x1ReconstructRange, nvars, amrex::The_Async_Arena());

	// cell-centered kernel
	LinearAdvectionSystem<problem_t>::ConservedToPrimitive(consState, primVar.array(),
							       ghostRange, nvars);
	CheckNaN(primVar, ghostRange, ncomp_);

	// mixed interface/cell-centered kernel
	LinearAdvectionSystem<problem_t>::template ReconstructStatesPPM<DIR>(
	    primVar.array(), x1LeftState.array(), x1RightState.array(), reconstructRange,
	    x1ReconstructRange, nvars);
	CheckNaN(x1LeftState, x1ReconstructRange, ncomp_);
	CheckNaN(x1RightState, x1ReconstructRange, ncomp_);

	// interface-centered kernel
	amrex::Box const &x1FluxRange = amrex::surroundingNodes(indexRange, dim);

	LinearAdvectionSystem<problem_t>::template ComputeFluxes<DIR>(
	    x1Flux.array(), x1LeftState.array(), x1RightState.array(), advectionVel, x1FluxRange,
	    nvars);
	CheckNaN(x1Flux, x1FluxRange, ncomp_);
}

template <typename problem_t>
void AdvectionSimulation<problem_t>::stageOneRK2SSP(
    amrex::Array4<const amrex::Real> const &consVarOld,
    amrex::Array4<amrex::Real> const &consVarNew, const amrex::Box &indexRange, const int nvars)
{
	// Allocate temporary arrays using CUDA stream async allocator (or equivalent)
	amrex::Box const &x1FluxRange = amrex::surroundingNodes(indexRange, 0);
	amrex::FArrayBox x1Flux(x1FluxRange, nvars, amrex::The_Async_Arena()); // node-centered in x

#if (AMREX_SPACEDIM >= 2) // for 2D problems
	amrex::Box const &x2FluxRange = amrex::surroundingNodes(indexRange, 1);
	amrex::FArrayBox x2Flux(x2FluxRange, nvars, amrex::The_Async_Arena()); // node-centered in y
#endif // AMREX_SPACEDIM >= 2

	fluxFunction<FluxDir::X1>(consVarOld, x1Flux, indexRange, nvars);
#if (AMREX_SPACEDIM >= 2) // for 2D problems
	fluxFunction<FluxDir::X2>(consVarOld, x2Flux, indexRange, nvars);
#endif // AMREX_SPACEDIM >= 2

	// Stage 1 of RK2-SSP
#if (AMREX_SPACEDIM == 1)
	amrex::GpuArray<arrayconst_t, AMREX_SPACEDIM> fluxArrays = {x1Flux.const_array()};
#elif (AMREX_SPACEDIM == 2)
	amrex::GpuArray<arrayconst_t, AMREX_SPACEDIM> fluxArrays = {x1Flux.const_array(),
								    x2Flux.const_array()};
#endif

	LinearAdvectionSystem<problem_t>::PredictStep(consVarOld, consVarNew, fluxArrays, dt_, dx_,
						      indexRange, nvars);
}

template <typename problem_t>
void AdvectionSimulation<problem_t>::stageTwoRK2SSP(
    amrex::Array4<const amrex::Real> const &consVarOld,
    amrex::Array4<amrex::Real> const &consVarNew, const amrex::Box &indexRange, const int nvars)
{
	// Allocate temporary arrays using CUDA stream async allocator (or equivalent)
	amrex::Box const &x1FluxRange = amrex::surroundingNodes(indexRange, 0);
	amrex::FArrayBox x1Flux(x1FluxRange, nvars, amrex::The_Async_Arena()); // node-centered in x

#if (AMREX_SPACEDIM >= 2) // for 2D problems
	amrex::Box const &x2FluxRange = amrex::surroundingNodes(indexRange, 1);
	amrex::FArrayBox x2Flux(x2FluxRange, nvars, amrex::The_Async_Arena()); // node-centered in y
#endif // AMREX_SPACEDIM >= 2

	fluxFunction<FluxDir::X1>(consVarNew, x1Flux, indexRange, nvars);
#if (AMREX_SPACEDIM >= 2) // for 2D problems
	fluxFunction<FluxDir::X2>(consVarNew, x2Flux, indexRange, nvars);
#endif // AMREX_SPACEDIM >= 2

#if (AMREX_SPACEDIM == 1)
	amrex::GpuArray<arrayconst_t, AMREX_SPACEDIM> fluxArrays = {x1Flux.const_array()};
#elif (AMREX_SPACEDIM == 2)
	amrex::GpuArray<arrayconst_t, AMREX_SPACEDIM> fluxArrays = {x1Flux.const_array(),
								    x2Flux.const_array()};
#endif

	// Stage 2 of RK2-SSP
	LinearAdvectionSystem<problem_t>::AddFluxesRK2(consVarNew, consVarOld, consVarNew,
						       fluxArrays, dt_, dx_, indexRange, nvars);
}

#endif // ADVECTION_SIMULATION_HPP_