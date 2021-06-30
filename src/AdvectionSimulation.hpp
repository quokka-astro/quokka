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
#include "AMReX_SPACE.H"
#include "AMReX_Utility.H"

#include "ArrayView.hpp"
#include "fmt/core.h"
#include "linear_advection.hpp"
#include "simulation.hpp"

// Simulation class should be initialized only once per program (i.e., is a singleton)
template <typename problem_t> class AdvectionSimulation : public AMRSimulation<problem_t>
{
      public:
	using AMRSimulation<problem_t>::simGeometry_;
	using AMRSimulation<problem_t>::state_old_;
	using AMRSimulation<problem_t>::state_new_;
	using AMRSimulation<problem_t>::max_signal_speed_;

	using AMRSimulation<problem_t>::cflNumber_;
	using AMRSimulation<problem_t>::dx_;
	using AMRSimulation<problem_t>::dt_;
	using AMRSimulation<problem_t>::ncomp_;
	using AMRSimulation<problem_t>::nghost_;
	using AMRSimulation<problem_t>::tNow_;
	using AMRSimulation<problem_t>::cycleCount_;
	using AMRSimulation<problem_t>::areInitialConditionsDefined_;
	using AMRSimulation<problem_t>::componentNames_;

	AdvectionSimulation(amrex::IntVect &gridDims, amrex::RealBox &boxSize,
			    amrex::Vector<amrex::BCRec> &boundaryConditions, const int ncomp = 1)
	    : AMRSimulation<problem_t>(gridDims, boxSize, boundaryConditions, ncomp)
	{
		componentNames_ = {"density"};
	}

	void computeMaxSignalLocal() override;
	void setInitialConditions() override;
	void advanceSingleTimestep() override;
	void computeAfterTimestep() override;

	void stageOneRK2SSP(amrex::Array4<const amrex::Real> const &consVarOld,
			    amrex::Array4<amrex::Real> const &consVarNew,
			    const amrex::Box &indexRange, int nvars);
	void stageTwoRK2SSP(amrex::Array4<const amrex::Real> const &consVar,
			    amrex::Array4<amrex::Real> const &consVarNew,
			    const amrex::Box &indexRange, int nvars);

	template <FluxDir DIR>
	void fluxFunction(amrex::Array4<const amrex::Real> const &consState,
			  amrex::FArrayBox &x1Flux, const amrex::Box &indexRange, int nvars);

	double advectionVx_ = 1.0;
	double advectionVy_ = 0.0;
	double advectionVz_ = 0.0;
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

template <typename problem_t> void AdvectionSimulation<problem_t>::computeAfterTimestep()
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
	} else if constexpr (DIR == FluxDir::X3) {
		advectionVel = advectionVz_;
		// [2 == x3 direction]
		dim = 2;
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
	quokka::CheckNaN<problem_t>(primVar, indexRange, ghostRange, ncomp_, dx_);

	// mixed interface/cell-centered kernel
	LinearAdvectionSystem<problem_t>::template ReconstructStatesPPM<DIR>(
	    primVar.array(), x1LeftState.array(), x1RightState.array(), reconstructRange,
	    x1ReconstructRange, nvars);
	// LinearAdvectionSystem<problem_t>::template ReconstructStatesPLM<DIR>(
	//    primVar.array(), x1LeftState.array(), x1RightState.array(), x1ReconstructRange,
	//    nvars);
	// LinearAdvectionSystem<problem_t>::template ReconstructStatesConstant<DIR>(
	//    primVar.array(), x1LeftState.array(), x1RightState.array(), x1ReconstructRange,
	//    nvars);
	quokka::CheckNaN<problem_t>(x1LeftState, indexRange, x1ReconstructRange, ncomp_, dx_);
	quokka::CheckNaN<problem_t>(x1RightState, indexRange, x1ReconstructRange, ncomp_, dx_);

	// interface-centered kernel
	amrex::Box const &x1FluxRange = amrex::surroundingNodes(indexRange, dim);

	LinearAdvectionSystem<problem_t>::template ComputeFluxes<DIR>(
	    x1Flux.array(), x1LeftState.array(), x1RightState.array(), advectionVel, x1FluxRange,
	    nvars);
	quokka::CheckNaN<problem_t>(x1Flux, indexRange, x1FluxRange, ncomp_, dx_);
}

template <typename problem_t>
void AdvectionSimulation<problem_t>::stageOneRK2SSP(
    amrex::Array4<const amrex::Real> const &consVarOld,
    amrex::Array4<amrex::Real> const &consVarNew, const amrex::Box &indexRange, const int nvars)
{
	// Allocate temporary arrays using CUDA stream async allocator (or equivalent)
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

	AMREX_D_TERM(fluxFunction<FluxDir::X1>(consVarOld, x1Flux, indexRange, nvars);
		     , fluxFunction<FluxDir::X2>(consVarOld, x2Flux, indexRange, nvars);
		     , fluxFunction<FluxDir::X3>(consVarOld, x3Flux, indexRange, nvars);)

	amrex::GpuArray<arrayconst_t, AMREX_SPACEDIM> fluxArrays = {
	    AMREX_D_DECL(x1Flux.const_array(), x2Flux.const_array(), x3Flux.const_array())};

	// if applicable, check that fluxes are symmetric (only for test problems with x/y symmetry)
	quokka::CheckSymmetryFluxes<problem_t>(x1Flux.const_array(), x2Flux.const_array(),
					       indexRange, ncomp_, dx_);

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
#if (AMREX_SPACEDIM >= 2)
	amrex::Box const &x2FluxRange = amrex::surroundingNodes(indexRange, 1);
	amrex::FArrayBox x2Flux(x2FluxRange, nvars, amrex::The_Async_Arena()); // node-centered in y
#endif
#if (AMREX_SPACEDIM == 3)
	amrex::Box const &x3FluxRange = amrex::surroundingNodes(indexRange, 2);
	amrex::FArrayBox x3Flux(x3FluxRange, nvars, amrex::The_Async_Arena()); // node-centered in z
#endif

	AMREX_D_TERM(fluxFunction<FluxDir::X1>(consVarNew, x1Flux, indexRange, nvars);
		     , fluxFunction<FluxDir::X2>(consVarNew, x2Flux, indexRange, nvars);
		     , fluxFunction<FluxDir::X3>(consVarNew, x3Flux, indexRange, nvars);)

	amrex::GpuArray<arrayconst_t, AMREX_SPACEDIM> fluxArrays = {
	    AMREX_D_DECL(x1Flux.const_array(), x2Flux.const_array(), x3Flux.const_array())};

	// if applicable, check that fluxes are symmetric (only for test problems with x/y symmetry)
	quokka::CheckSymmetryFluxes<problem_t>(x1Flux.const_array(), x2Flux.const_array(),
					       indexRange, ncomp_, dx_);

	// Stage 2 of RK2-SSP
	LinearAdvectionSystem<problem_t>::AddFluxesRK2(consVarNew, consVarOld, consVarNew,
						       fluxArrays, dt_, dx_, indexRange, nvars);
}

#endif // ADVECTION_SIMULATION_HPP_