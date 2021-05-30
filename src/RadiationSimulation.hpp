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
#include "AMReX_BCRec.H"
#include "AMReX_BLassert.H"
#include "AMReX_Box.H"
#include "AMReX_FArrayBox.H"
#include "AMReX_IntVect.H"
#include "AMReX_MultiFab.H"
#include "AMReX_MultiFabUtil.H"
#include "AMReX_PhysBCFunct.H"
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
	using SingleLevelSimulation<problem_t>::ncompPrimitive_;
	using SingleLevelSimulation<problem_t>::nghost_;
	using SingleLevelSimulation<problem_t>::tNow_;
	using SingleLevelSimulation<problem_t>::cycleCount_;
	using SingleLevelSimulation<problem_t>::areInitialConditionsDefined_;
	using SingleLevelSimulation<problem_t>::boundaryConditions_;
	using SingleLevelSimulation<problem_t>::componentNames_;

	std::vector<double> t_vec_;
	std::vector<double> Trad_vec_;
	std::vector<double> Tgas_vec_;

	// member functions

	RadiationSimulation(amrex::IntVect &gridDims, amrex::RealBox &boxSize,
			    amrex::Vector<amrex::BCRec> &boundaryConditions,
			    const int ncomp = RadSystem<problem_t>::consVarEnumEnd,
			    const int ncompPrim = RadSystem<problem_t>::primVarEnumEnd)
	    : SingleLevelSimulation<problem_t>(gridDims, boxSize, boundaryConditions, ncomp,
					       ncompPrim)
	{
		componentNames_ = {"radEnergy",	    "x-RadFlux",     "y-RadFlux",
				   "z-RadFlux",	    "gasEnergy",     "gasDensity",
				   "x-GasMomentum", "y-GasMomentum", "z-GasMomentum"};
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
	void operatorSplitSourceTerms(amrex::Array4<amrex::Real> const &stateNew,
				      const amrex::Box &indexRange, const int nvars);

	void fillBoundaryConditions(amrex::MultiFab &state);

	AMREX_GPU_DEVICE AMREX_FORCE_INLINE static void
	setCustomBoundaryConditions(const amrex::IntVect &iv, amrex::Array4<Real> const &dest,
				    int dcomp, int numcomp, amrex::GeometryData const &geom,
				    Real time, const amrex::BCRec *bcr, int bcomp, int orig_comp);

	template <FluxDir DIR>
	void fluxFunction(amrex::Array4<const amrex::Real> const &consState,
			  amrex::FArrayBox &x1Flux, const amrex::Box &indexRange, int nvars);
};

template <typename problem_t> struct setBoundaryFunctor {
	AMREX_GPU_DEVICE void operator()(const amrex::IntVect &iv, amrex::Array4<Real> const &dest,
					 const int &dcomp, const int &numcomp,
					 amrex::GeometryData const &geom, const Real &time,
					 const amrex::BCRec *bcr, int bcomp,
					 const int &orig_comp) const
	{
		RadiationSimulation<problem_t>::setCustomBoundaryConditions(
		    iv, dest, dcomp, numcomp, geom, time, bcr, bcomp, orig_comp);
	}
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

template <typename problem_t> void RadiationSimulation<problem_t>::computeAfterTimestep()
{
	// do nothing -- user should implement if desired
}

template <typename problem_t>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE void
RadiationSimulation<problem_t>::setCustomBoundaryConditions(
    const amrex::IntVect &iv, amrex::Array4<Real> const &dest, int dcomp, int numcomp,
    amrex::GeometryData const &geom, const Real time, const amrex::BCRec *bcr, int bcomp,
    int orig_comp)
{
	// user should implement if needed using template specialization
	// (This is only called when amrex::BCType::ext_dir is set for a given boundary.)

	// set boundary condition for cell 'iv'
}

template <typename problem_t>
void RadiationSimulation<problem_t>::fillBoundaryConditions(amrex::MultiFab &state)
{
	state.FillBoundary(simGeometry_.periodicity());
	if (!simGeometry_.isAllPeriodic()) {
		AMREX_ALWAYS_ASSERT(state.ixType().cellCentered());

		amrex::GpuBndryFuncFab<setBoundaryFunctor<problem_t>> boundaryFunctor(
		    setBoundaryFunctor<problem_t>{});
		amrex::PhysBCFunct<amrex::GpuBndryFuncFab<setBoundaryFunctor<problem_t>>>
		    physicalBoundaryFunctor(simGeometry_, boundaryConditions_, boundaryFunctor);

		physicalBoundaryFunctor(state, 0, state.nComp(), state.nGrowVect(), tNow_, 0);
	}
}

template <typename problem_t> void RadiationSimulation<problem_t>::advanceSingleTimestep()
{
	// We use the RK2-SSP method here. It needs two registers: one to store the old timestep,
	// and another to store the intermediate stage (which is reused for the final stage).

	// update ghost zones [old timestep]
	fillBoundaryConditions(state_old_);
	AMREX_ASSERT(!state_old_.contains_nan(0, ncomp_));

	// advance all grids on local processor (Stage 1 of integrator)
	for (amrex::MFIter iter(state_new_); iter.isValid(); ++iter) {
		const amrex::Box &indexRange = iter.validbox(); // 'validbox' == exclude ghost zones
		auto const &stateOld = state_old_.const_array(iter);
		auto const &stateNew = state_new_.array(iter);
		stageOneRK2SSP(stateOld, stateNew, indexRange, ncompPrimitive_);
		quokka::CheckSymmetryArray<problem_t>(stateNew, indexRange, ncomp_);
	}

	// update ghost zones [intermediate stage stored in state_new_]
	fillBoundaryConditions(state_new_);
	AMREX_ASSERT(!state_new_.contains_nan(0, ncomp_));

	// advance all grids on local processor (Stage 2 of integrator)
	for (amrex::MFIter iter(state_new_); iter.isValid(); ++iter) {
		const amrex::Box &indexRange = iter.validbox(); // 'validbox' == exclude ghost zones
		auto const &stateOld = state_old_.const_array(iter);
		auto const &stateNew = state_new_.array(iter);
		stageTwoRK2SSP(stateOld, stateNew, indexRange, ncompPrimitive_);
		quokka::CheckSymmetryArray<problem_t>(stateNew, indexRange, ncomp_);
	}

	// update ghost zones [intermediate stage stored in state_new_]
	fillBoundaryConditions(state_new_);
	AMREX_ASSERT(!state_new_.contains_nan(0, ncomp_));
	
	// source terms
	for (amrex::MFIter iter(state_new_); iter.isValid(); ++iter) {
		const amrex::Box &indexRange = iter.validbox(); // 'validbox' == exclude ghost zones
		auto const &stateNew = state_new_.array(iter);
		operatorSplitSourceTerms(stateNew, indexRange, ncomp_);
		quokka::CheckSymmetryArray<problem_t>(stateNew, indexRange, ncomp_);
	}
}

template <typename problem_t>
void RadiationSimulation<problem_t>::operatorSplitSourceTerms(
    amrex::Array4<amrex::Real> const &stateNew, const amrex::Box &indexRange, const int nvars)
{
	amrex::FArrayBox radEnergySource(indexRange, 1, amrex::The_Async_Arena()); // cell-centered scalar
	amrex::FArrayBox advectionFluxes(indexRange, 3, amrex::The_Async_Arena()); // cell-centered vector
	radEnergySource.setVal(0.);
	advectionFluxes.setVal(0.);

	// cell-centered radiation energy source (used only in test problems)
	RadSystem<problem_t>::SetRadEnergySource(radEnergySource.array(), indexRange, dx_, tNow_);

	// cell-centered source terms
	RadSystem<problem_t>::AddSourceTerms(stateNew, radEnergySource.const_array(),
					     advectionFluxes.const_array(), indexRange, dt_);
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
	amrex::Box const &x1ReconstructRange = amrex::surroundingNodes(reconstructRange, dir);

	// Allocate temporary arrays using CUDA stream async allocator (or equivalent)
	amrex::FArrayBox primVar(ghostRange, ncompPrimitive_,
				 amrex::The_Async_Arena()); // cell-centered
	amrex::FArrayBox x1LeftState(x1ReconstructRange, ncompPrimitive_, amrex::The_Async_Arena());
	amrex::FArrayBox x1RightState(x1ReconstructRange, ncompPrimitive_,
				      amrex::The_Async_Arena());

	// cell-centered kernel
	RadSystem<problem_t>::ConservedToPrimitive(consState, primVar.array(), ghostRange);
	quokka::CheckNaN<problem_t>(primVar, indexRange, ghostRange, ncompPrimitive_);

	// mixed interface/cell-centered kernel
	RadSystem<problem_t>::template ReconstructStatesPPM<DIR>(
	    primVar.array(), x1LeftState.array(), x1RightState.array(), reconstructRange,
	    x1ReconstructRange, ncompPrimitive_);
	quokka::CheckNaN<problem_t>(x1LeftState, indexRange, x1ReconstructRange, ncompPrimitive_);
	quokka::CheckNaN<problem_t>(x1RightState, indexRange, x1ReconstructRange, ncompPrimitive_);

	// interface-centered kernel
	amrex::Box const &x1FluxRange = amrex::surroundingNodes(indexRange, dir);
	RadSystem<problem_t>::template ComputeFluxes<DIR>(
	    x1Flux.array(), x1LeftState.array(), x1RightState.array(), x1FluxRange, consState,
	    dx_); // watch out for argument order!!
	quokka::CheckNaN<problem_t>(x1Flux, indexRange, x1FluxRange, ncompPrimitive_);
}

template <typename problem_t>
void RadiationSimulation<problem_t>::stageOneRK2SSP(
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

	// Stage 1 of RK2-SSP
	RadSystem<problem_t>::PredictStep(consVarOld, consVarNew, fluxArrays, dt_, dx_, indexRange,
					  nvars);
}

template <typename problem_t>
void RadiationSimulation<problem_t>::stageTwoRK2SSP(
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
	RadSystem<problem_t>::AddFluxesRK2(consVarNew, consVarOld, consVarNew, fluxArrays, dt_, dx_,
					   indexRange, nvars);
}

#endif // RADIATION_SIMULATION_HPP_