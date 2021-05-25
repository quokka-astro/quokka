#ifndef RADIATION_SIMULATION_HPP_ // NOLINT
#define RADIATION_SIMULATION_HPP_
//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file RadiationSimulation.hpp
/// \brief Implements classes and functions to organise the overall setup,
/// timestepping, solving, and I/O of a simulation for radiation moments.

#include <climits>
#include <limits>
#include <string>
#include <utility>

#include "AMReX_Algorithm.H"
#include "AMReX_Arena.H"
#include "AMReX_Array4.H"
#include "AMReX_BLassert.H"
#include "AMReX_Box.H"
#include "AMReX_FArrayBox.H"
#include "AMReX_IntVect.H"
#include "AMReX_MultiFab.H"
#include "AMReX_MultiFabUtil.H"
#include "AMReX_REAL.H"
#include "AMReX_Utility.H"

#include "radiation_system.hpp"
#include "simulation.hpp"


// Simulation class should be initialized only once per program (i.e., is a singleton)
template <typename problem_t> class RadiationSimulation : public SingleLevelSimulation<problem_t>
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

	// member functions

	explicit RadiationSimulation() = default;

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
};

template <typename problem_t> void RadiationSimulation<problem_t>::computeMaxSignalLocal()
{
	// loop over local grids, compute CFL timestep
	for (amrex::MFIter iter(state_new_); iter.isValid(); ++iter) {
		const amrex::Box &indexRange = iter.validbox();
		auto const &stateOld = state_old_.const_array(iter);
		auto const &maxSignal = max_signal_speed_.array(iter);
		RadSystem<problem_t>::ComputeMaxSignalSpeed(stateOld, maxSignal, indexRange);
	}
}

template <typename problem_t> void RadiationSimulation<problem_t>::setInitialConditions()
{
	// do nothing -- user should implement using problem-specific template specialization
}

template <typename problem_t> void RadiationSimulation<problem_t>::advanceSingleTimestep()
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
void RadiationSimulation<problem_t>::fluxFunction(amrex::Array4<const amrex::Real> const &consState,
					      amrex::FArrayBox &x1Flux,
					      const amrex::Box &indexRange, const int nvars)
{
	int dir = 0;
	if constexpr (DIR == FluxDir::X1) {
		dir = 0;
	} else if constexpr (DIR == FluxDir::X2) {
		dir = 1;
	}

	// extend box to include ghost zones
	amrex::Box const &ghostRange = amrex::grow(indexRange, nghost_);
	// N.B.: A one-zone layer around the cells must be fully reconstructed in order for PPM to
	// work.
	amrex::Box const &reconstructRange = amrex::grow(indexRange, 1);
	amrex::Box const &flatteningRange = amrex::grow(indexRange, 2); // +1 greater than ppmRange
	amrex::Box const &x1ReconstructRange = amrex::surroundingNodes(reconstructRange, dir);

	// Allocate temporary arrays using CUDA stream async allocator (or equivalent)
	amrex::FArrayBox primVar(ghostRange, nvars, amrex::The_Async_Arena()); // cell-centered
	amrex::FArrayBox x1Flat(ghostRange, nvars, amrex::The_Async_Arena());
	amrex::FArrayBox x1LeftState(x1ReconstructRange, nvars, amrex::The_Async_Arena());
	amrex::FArrayBox x1RightState(x1ReconstructRange, nvars, amrex::The_Async_Arena());

	// cell-centered kernel
	RadSystem<problem_t>::ConservedToPrimitive(consState, primVar.array(), ghostRange);
	CheckNaN(primVar, ghostRange, nvars);

	// mixed interface/cell-centered kernel
	RadSystem<problem_t>::template ReconstructStatesPPM<DIR>(
	    primVar.array(), x1LeftState.array(), x1RightState.array(), reconstructRange,
	    x1ReconstructRange, nvars);
	CheckNaN(x1LeftState, x1ReconstructRange, nvars);
	CheckNaN(x1RightState, x1ReconstructRange, nvars);

	// interface-centered kernel
	amrex::Box const &x1FluxRange = amrex::surroundingNodes(indexRange, dir);
	RadSystem<problem_t>::template ComputeFluxes<DIR>(x1Flux.array(), x1LeftState.array(),
						   x1RightState.array(),
						   x1FluxRange); // watch out for argument order!!
	CheckNaN(x1Flux, x1FluxRange, nvars);
}

template <typename problem_t>
void RadiationSimulation<problem_t>::stageOneRK2SSP(amrex::Array4<const amrex::Real> const &consVarOld,
						amrex::Array4<amrex::Real> const &consVarNew,
						const amrex::Box &indexRange, const int nvars)
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

	// Stage 1 of RK2-SSP
	RadSystem<problem_t>::PredictStep(consVarOld, consVarNew, fluxArrays, dt_, dx_,
					    indexRange, nvars);
}

template <typename problem_t>
void RadiationSimulation<problem_t>::stageTwoRK2SSP(amrex::Array4<const amrex::Real> const &consVarOld,
						amrex::Array4<amrex::Real> const &consVarNew,
						const amrex::Box &indexRange, const int nvars)
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
	RadSystem<problem_t>::AddFluxesRK2(consVarNew, consVarOld, consVarNew, fluxArrays, dt_,
					     dx_, indexRange, nvars);
}

#endif // RADIATION_SIMULATION_HPP_