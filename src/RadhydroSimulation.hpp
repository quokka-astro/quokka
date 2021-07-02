#ifndef RADIATION_SIMULATION_HPP_ // NOLINT
#define RADIATION_SIMULATION_HPP_
//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file RadhydroSimulation.hpp
/// \brief Implements classes and functions to organise the overall setup,
/// timestepping, solving, and I/O of a simulation for radiation moments.

#include <climits>
#include <limits>
#include <string>
#include <utility>

#include "AMReX_Algorithm.H"
#include "AMReX_Arena.H"
#include "AMReX_Array.H"
#include "AMReX_Array4.H"
#include "AMReX_BCRec.H"
#include "AMReX_BLassert.H"
#include "AMReX_Box.H"
#include "AMReX_FArrayBox.H"
#include "AMReX_GpuControl.H"
#include "AMReX_IntVect.H"
#include "AMReX_MultiFab.H"
#include "AMReX_MultiFabUtil.H"
#include "AMReX_PhysBCFunct.H"
#include "AMReX_REAL.H"
#include "AMReX_Utility.H"

#include "hydro_system.hpp"
#include "radiation_system.hpp"
#include "simulation.hpp"

// Simulation class should be initialized only once per program (i.e., is a singleton)
template <typename problem_t> class RadhydroSimulation : public AMRSimulation<problem_t>
{
      public:
	using AMRSimulation<problem_t>::state_old_;
	using AMRSimulation<problem_t>::state_new_;
	using AMRSimulation<problem_t>::max_signal_speed_;

	using AMRSimulation<problem_t>::cflNumber_;
	using AMRSimulation<problem_t>::dx_;
	using AMRSimulation<problem_t>::dt_; // this is the *hydro* timestep
	using AMRSimulation<problem_t>::tNew_;
	using AMRSimulation<problem_t>::ncomp_;
	using AMRSimulation<problem_t>::nghost_;
	using AMRSimulation<problem_t>::cycleCount_;
	using AMRSimulation<problem_t>::areInitialConditionsDefined_;
	using AMRSimulation<problem_t>::boundaryConditions_;
	using AMRSimulation<problem_t>::componentNames_;

	// from super-superclass
	using AMRSimulation<problem_t>::dmap;
	using AMRSimulation<problem_t>::grids;

	std::vector<double> t_vec_;
	std::vector<double> Trad_vec_;
	std::vector<double> Tgas_vec_;

	static constexpr int nvarTotal_ = RadSystem<problem_t>::nvar_;
	static constexpr int ncompHydro_ = HydroSystem<problem_t>::nvar_; // hydro
	static constexpr int ncompHyperbolic_ = RadSystem<problem_t>::nvarHyperbolic_;
	static constexpr int nstartHyperbolic_ = RadSystem<problem_t>::nstartHyperbolic_;

	amrex::Real dtRadiation_ = NAN; // this is radiation subcycle timestep
	amrex::Real radiationCflNumber_ = 0.3;
	bool is_hydro_enabled_ = false;
	bool is_radiation_enabled_ = true;

	// member functions

	RadhydroSimulation(amrex::IntVect &gridDims, amrex::RealBox &boxSize,
			   amrex::Vector<amrex::BCRec> &boundaryConditions)
	    : AMRSimulation<problem_t>(gridDims, boxSize, boundaryConditions,
				       RadSystem<problem_t>::nvar_, ncompHyperbolic_)
	{
		componentNames_ = {"gasDensity",    "x-GasMomentum", "y-GasMomentum",
				   "z-GasMomentum", "gasEnergy",     "radEnergy",
				   "x-RadFlux",	    "y-RadFlux",     "z-RadFlux"};
	}

	void computeMaxSignalLocal(int level) override;
	void setInitialConditions() override;
	void advanceSingleTimestepAtLevel(int lev, amrex::Real time, amrex::Real dt_lev,
					  int iteration, int ncycle);
	void computeAfterTimestep() override;

	// radiation subcycle
	void advanceSingleTimestepRadiation();
	void subcycleRadiation();

	void stageOneRK2SSP(amrex::Array4<const amrex::Real> const &consVarOld,
			    amrex::Array4<amrex::Real> const &consVarNew,
			    const amrex::Box &indexRange, int nvars);
	void stageTwoRK2SSP(amrex::Array4<const amrex::Real> const &consVarOld,
			    amrex::Array4<amrex::Real> const &consVarNew,
			    const amrex::Box &indexRange, int nvars);
	void operatorSplitSourceTerms(amrex::Array4<amrex::Real> const &stateNew,
				      const amrex::Box &indexRange, int nvars, double dt);

	void hydroStageOneRK2SSP(amrex::Array4<const amrex::Real> const &consVarOld,
				 amrex::Array4<amrex::Real> const &consVarNew,
				 const amrex::Box &indexRange, int nvars);
	void hydroStageTwoRK2SSP(amrex::Array4<const amrex::Real> const &consVarOld,
				 amrex::Array4<amrex::Real> const &consVarNew,
				 const amrex::Box &indexRange, int nvars);

	void fillBoundaryConditions(amrex::MultiFab &state);

	template <FluxDir DIR>
	void fluxFunction(amrex::Array4<const amrex::Real> const &consState,
			  amrex::FArrayBox &x1Flux, amrex::FArrayBox &x1FluxDiffusive,
			  const amrex::Box &indexRange, int nvars);

	template <FluxDir DIR>
	void hydroFluxFunction(amrex::Array4<const amrex::Real> const &consState,
			       amrex::FArrayBox &x1Flux, const amrex::Box &indexRange, int nvars);
};

template <typename problem_t>
void RadhydroSimulation<problem_t>::computeMaxSignalLocal(int const level)
{
	// hydro: loop over local grids, compute hydro CFL timestep
	for (amrex::MFIter iter(state_new_[level]); iter.isValid(); ++iter) {
		const amrex::Box &indexRange = iter.validbox();
		auto const &stateOld = state_old_[level].const_array(iter);
		auto const &maxSignal = max_signal_speed_[level].array(iter);
		HydroSystem<problem_t>::ComputeMaxSignalSpeed(stateOld, maxSignal, indexRange);
	}
}

template <typename problem_t> void RadhydroSimulation<problem_t>::setInitialConditions()
{
	// do nothing -- user should implement using problem-specific template specialization
}

template <typename problem_t> void RadhydroSimulation<problem_t>::computeAfterTimestep()
{
	// do nothing -- user should implement if desired
}

template <typename problem_t>
void RadhydroSimulation<problem_t>::fillBoundaryConditions(amrex::MultiFab &state)
{
	// TODO(ben): should be implemented by base class!
}

template <typename problem_t>
void RadhydroSimulation<problem_t>::advanceSingleTimestepAtLevel(int lev, amrex::Real time,
								 amrex::Real dt_lev,
								 int /*iteration*/, int /*ncycle*/)
{
	// since we are starting a new timestep, need to swap old and new states
	std::swap(state_old_[lev], state_new_[lev]);

	/// advance hydro
	if (is_hydro_enabled_) {
		// update ghost zones [old timestep]
		fillBoundaryConditions(state_old_);
		AMREX_ASSERT(!state_old_.contains_nan(0, ncomp_));

		// advance all grids on local processor (Stage 1 of integrator)
		for (amrex::MFIter iter(state_new_); iter.isValid(); ++iter) {
			const amrex::Box &indexRange =
			    iter.validbox(); // 'validbox' == exclude ghost zones
			auto const &stateOld = state_old_.const_array(iter);
			auto const &stateNew = state_new_.array(iter);
			hydroStageOneRK2SSP(stateOld, stateNew, indexRange, ncompHydro_);
		}

		// update ghost zones [intermediate stage stored in state_new_]
		fillBoundaryConditions(state_new_);
		AMREX_ASSERT(!state_new_.contains_nan(0, ncomp_));

		// advance all grids on local processor (Stage 2 of integrator)
		for (amrex::MFIter iter(state_new_); iter.isValid(); ++iter) {
			const amrex::Box &indexRange =
			    iter.validbox(); // 'validbox' == exclude ghost zones
			auto const &stateOld = state_old_.const_array(iter);
			auto const &stateNew = state_new_.array(iter);
			hydroStageTwoRK2SSP(stateOld, stateNew, indexRange, ncompHydro_);
		}
	}

	// subcycle radiation
	if (is_radiation_enabled_) {
		subcycleRadiation();
	}
}

template <typename problem_t>
template <FluxDir DIR>
void RadhydroSimulation<problem_t>::hydroFluxFunction(
    amrex::Array4<const amrex::Real> const &consState, amrex::FArrayBox &x1Flux,
    const amrex::Box &indexRange, const int nvars)
{
	int dir = 0;
	if constexpr (DIR == FluxDir::X1) {
		dir = 0;
	} else if constexpr (DIR == FluxDir::X2) {
		dir = 1;
	} else if constexpr (DIR == FluxDir::X3) {
		dir = 2;
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
	HydroSystem<problem_t>::ConservedToPrimitive(consState, primVar.array(), ghostRange);
	quokka::CheckNaN<problem_t>(primVar, indexRange, ghostRange, nvars, dx_);

	// mixed interface/cell-centered kernel
	HydroSystem<problem_t>::template ReconstructStatesPPM<DIR>(
	    primVar.array(), x1LeftState.array(), x1RightState.array(), reconstructRange,
	    x1ReconstructRange, nvars);
	quokka::CheckNaN<problem_t>(x1LeftState, indexRange, x1ReconstructRange, nvars, dx_);
	quokka::CheckNaN<problem_t>(x1RightState, indexRange, x1ReconstructRange, nvars, dx_);

	// cell-centered kernel
	HydroSystem<problem_t>::template ComputeFlatteningCoefficients<DIR>(
	    primVar.array(), x1Flat.array(), flatteningRange);
	quokka::CheckNaN<problem_t>(x1LeftState, indexRange, x1ReconstructRange, nvars, dx_);
	quokka::CheckNaN<problem_t>(x1RightState, indexRange, x1ReconstructRange, nvars, dx_);

	// cell-centered kernel
	HydroSystem<problem_t>::template FlattenShocks<DIR>(
	    primVar.array(), x1Flat.array(), x1LeftState.array(), x1RightState.array(),
	    reconstructRange, nvars);
	quokka::CheckNaN<problem_t>(x1LeftState, indexRange, x1ReconstructRange, nvars, dx_);
	quokka::CheckNaN<problem_t>(x1RightState, indexRange, x1ReconstructRange, nvars, dx_);

	// interface-centered kernel
	amrex::Box const &x1FluxRange = amrex::surroundingNodes(indexRange, dir);
	HydroSystem<problem_t>::template ComputeFluxes<DIR>(
	    x1Flux.array(), x1LeftState.array(), x1RightState.array(),
	    x1FluxRange); // watch out for argument order!!
	quokka::CheckNaN<problem_t>(x1Flux, indexRange, x1FluxRange, nvars, dx_);
}

template <typename problem_t>
void RadhydroSimulation<problem_t>::hydroStageOneRK2SSP(
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

	AMREX_D_TERM(hydroFluxFunction<FluxDir::X1>(consVarOld, x1Flux, indexRange, nvars);
		     , hydroFluxFunction<FluxDir::X2>(consVarOld, x2Flux, indexRange, nvars);
		     , hydroFluxFunction<FluxDir::X3>(consVarOld, x3Flux, indexRange, nvars);)

	amrex::GpuArray<arrayconst_t, AMREX_SPACEDIM> fluxArrays = {
	    AMREX_D_DECL(x1Flux.const_array(), x2Flux.const_array(), x3Flux.const_array())};

	// Stage 1 of RK2-SSP
	HydroSystem<problem_t>::PredictStep(consVarOld, consVarNew, fluxArrays, dt_, dx_,
					    indexRange, nvars);
}

template <typename problem_t>
void RadhydroSimulation<problem_t>::hydroStageTwoRK2SSP(
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

	AMREX_D_TERM(hydroFluxFunction<FluxDir::X1>(consVarNew, x1Flux, indexRange, nvars);
		     , hydroFluxFunction<FluxDir::X2>(consVarNew, x2Flux, indexRange, nvars);
		     , hydroFluxFunction<FluxDir::X3>(consVarNew, x3Flux, indexRange, nvars);)

	amrex::GpuArray<arrayconst_t, AMREX_SPACEDIM> fluxArrays = {
	    AMREX_D_DECL(x1Flux.const_array(), x2Flux.const_array(), x3Flux.const_array())};

	// Stage 2 of RK2-SSP
	HydroSystem<problem_t>::AddFluxesRK2(consVarNew, consVarOld, consVarNew, fluxArrays, dt_,
					     dx_, indexRange, nvars);
}

template <typename problem_t> void RadhydroSimulation<problem_t>::subcycleRadiation()
{
	// compute radiation timestep 'dtrad_tmp'
	amrex::Real domain_signal_max = RadSystem<problem_t>::c_hat_;
	amrex::Real dx_min = std::min({AMREX_D_DECL(dx_[0], dx_[1], dx_[2])});
	amrex::Real dtrad_tmp = radiationCflNumber_ * (dx_min / domain_signal_max);

	amrex::Long nsubSteps = 0;
	if (is_hydro_enabled_) {
		// adjust to get integer number of substeps
		amrex::Real dt_hydro = dt_;
		nsubSteps = std::ceil(dt_hydro / dtrad_tmp);
		dtRadiation_ = dt_hydro / static_cast<double>(nsubSteps);
	} else { // no hydro (this is necessary for radiation test problems)
		nsubSteps = 1;
		dtRadiation_ = dtrad_tmp;
		dt_ = dtRadiation_; // adjust global timestep (ok because no hydro was computed)
	}
	AMREX_ALWAYS_ASSERT(nsubSteps >= 1);
	AMREX_ALWAYS_ASSERT(nsubSteps < 1e4);
	AMREX_ALWAYS_ASSERT(dtRadiation_ > 0.0);

	if (amrex::ParallelDescriptor::IOProcessor()) {
		amrex::Print() << "\nRadiation substeps: " << nsubSteps << "\tdt: " << dtRadiation_
			       << "\n";
	}

	// subcycle
	for (int i = 0; i < nsubSteps; ++i) {
		advanceSingleTimestepRadiation(); // using dt_radiation_
	}
}

template <typename problem_t> void RadhydroSimulation<problem_t>::advanceSingleTimestepRadiation()
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
		stageOneRK2SSP(stateOld, stateNew, indexRange, ncompHyperbolic_);
		quokka::CheckSymmetryArray<problem_t>(stateNew, indexRange, ncompHyperbolic_, dx_);
	}

	// update ghost zones [intermediate stage stored in state_new_]
	fillBoundaryConditions(state_new_);
	AMREX_ASSERT(!state_new_.contains_nan(0, ncomp_));

	// advance all grids on local processor (Stage 2 of integrator)
	for (amrex::MFIter iter(state_new_); iter.isValid(); ++iter) {
		const amrex::Box &indexRange = iter.validbox(); // 'validbox' == exclude ghost zones
		auto const &stateOld = state_old_.const_array(iter);
		auto const &stateNew = state_new_.array(iter);
		stageTwoRK2SSP(stateOld, stateNew, indexRange, ncompHyperbolic_);
		quokka::CheckSymmetryArray<problem_t>(stateNew, indexRange, ncompHyperbolic_, dx_);
	}

	// matter-radiation exchange source terms
	for (amrex::MFIter iter(state_new_); iter.isValid(); ++iter) {
		const amrex::Box &indexRange = iter.validbox(); // 'validbox' == exclude ghost zones
		auto const &stateNew = state_new_.array(iter);
		operatorSplitSourceTerms(stateNew, indexRange, ncomp_, dtRadiation_);
		quokka::CheckSymmetryArray<problem_t>(stateNew, indexRange, ncomp_, dx_);
	}
}

template <typename problem_t>
void RadhydroSimulation<problem_t>::operatorSplitSourceTerms(
    amrex::Array4<amrex::Real> const &stateNew, const amrex::Box &indexRange, const int /*nvars*/,
    const double dt)
{
	amrex::FArrayBox radEnergySource(indexRange, 1,
					 amrex::The_Async_Arena()); // cell-centered scalar
	amrex::FArrayBox advectionFluxes(indexRange, 3,
					 amrex::The_Async_Arena()); // cell-centered vector

	radEnergySource.template setVal<amrex::RunOn::Device>(0.);
	advectionFluxes.template setVal<amrex::RunOn::Device>(0.);

	// cell-centered radiation energy source (used only in test problems)
	RadSystem<problem_t>::SetRadEnergySource(radEnergySource.array(), indexRange, dx_, tNew_);

	// cell-centered source terms
	RadSystem<problem_t>::AddSourceTerms(stateNew, radEnergySource.const_array(),
					     advectionFluxes.const_array(), indexRange, dt);
}

template <typename problem_t>
template <FluxDir DIR>
void RadhydroSimulation<problem_t>::fluxFunction(amrex::Array4<const amrex::Real> const &consState,
						 amrex::FArrayBox &x1Flux,
						 amrex::FArrayBox &x1FluxDiffusive,
						 const amrex::Box &indexRange, const int nvars)
{
	int dir = 0;
	if constexpr (DIR == FluxDir::X1) {
		dir = 0;
	} else if constexpr (DIR == FluxDir::X2) {
		dir = 1;
	} else if constexpr (DIR == FluxDir::X3) {
		dir = 2;
	}

	// extend box to include ghost zones
	amrex::Box const &ghostRange = amrex::grow(indexRange, nghost_);
	// N.B.: A one-zone layer around the cells must be fully reconstructed in order for PPM to
	// work.
	amrex::Box const &reconstructRange = amrex::grow(indexRange, 1);
	amrex::Box const &x1ReconstructRange = amrex::surroundingNodes(reconstructRange, dir);

	// Allocate temporary arrays using CUDA stream async allocator (or equivalent)
	amrex::FArrayBox primVar(ghostRange, nvars,
				 amrex::The_Async_Arena()); // cell-centered
	amrex::FArrayBox x1LeftState(x1ReconstructRange, nvars, amrex::The_Async_Arena());
	amrex::FArrayBox x1RightState(x1ReconstructRange, nvars, amrex::The_Async_Arena());

	// cell-centered kernel
	RadSystem<problem_t>::ConservedToPrimitive(consState, primVar.array(), ghostRange);
	quokka::CheckNaN<problem_t>(primVar, indexRange, ghostRange, nvars, dx_);

	// mixed interface/cell-centered kernel
	RadSystem<problem_t>::template ReconstructStatesPPM<DIR>(
	    primVar.array(), x1LeftState.array(), x1RightState.array(), reconstructRange,
	    x1ReconstructRange, nvars);
	// PLM and donor cell are interface-centered kernels
	// RadSystem<problem_t>::template ReconstructStatesConstant<DIR>(
	//     primVar.array(), x1LeftState.array(), x1RightState.array(), x1ReconstructRange,
	//     nvars);
	// RadSystem<problem_t>::template ReconstructStatesPLM<DIR>(
	//     primVar.array(), x1LeftState.array(), x1RightState.array(), x1ReconstructRange,
	//     nvars);
	quokka::CheckNaN<problem_t>(x1LeftState, indexRange, x1ReconstructRange, nvars, dx_);
	quokka::CheckNaN<problem_t>(x1RightState, indexRange, x1ReconstructRange, nvars, dx_);

	// interface-centered kernel
	amrex::Box const &x1FluxRange = amrex::surroundingNodes(indexRange, dir);
	RadSystem<problem_t>::template ComputeFluxes<DIR>(x1Flux.array(), x1FluxDiffusive.array(),
							  x1LeftState.array(), x1RightState.array(),
							  x1FluxRange, consState,
							  dx_); // watch out for argument order!!
	quokka::CheckNaN<problem_t>(x1Flux, indexRange, x1FluxRange, nvars, dx_);
}

template <typename problem_t>
void RadhydroSimulation<problem_t>::stageOneRK2SSP(
    amrex::Array4<const amrex::Real> const &consVarOld,
    amrex::Array4<amrex::Real> const &consVarNew, const amrex::Box &indexRange, const int nvars)
{
	// Allocate temporary arrays using CUDA stream async allocator (or equivalent)
	amrex::Box const &x1FluxRange = amrex::surroundingNodes(indexRange, 0);
	amrex::FArrayBox x1Flux(x1FluxRange, nvars, amrex::The_Async_Arena()); // node-centered in x
	amrex::FArrayBox x1FluxDiffusive(x1FluxRange, nvars, amrex::The_Async_Arena());
#if (AMREX_SPACEDIM >= 2)
	amrex::Box const &x2FluxRange = amrex::surroundingNodes(indexRange, 1);
	amrex::FArrayBox x2Flux(x2FluxRange, nvars, amrex::The_Async_Arena()); // node-centered in y
	amrex::FArrayBox x2FluxDiffusive(x2FluxRange, nvars, amrex::The_Async_Arena());
#endif
#if (AMREX_SPACEDIM == 3)
	amrex::Box const &x3FluxRange = amrex::surroundingNodes(indexRange, 2);
	amrex::FArrayBox x3Flux(x3FluxRange, nvars, amrex::The_Async_Arena()); // node-centered in z
	amrex::FArrayBox x3FluxDiffusive(x3FluxRange, nvars, amrex::The_Async_Arena());
#endif

	AMREX_D_TERM(
	    fluxFunction<FluxDir::X1>(consVarOld, x1Flux, x1FluxDiffusive, indexRange, nvars);
	    , fluxFunction<FluxDir::X2>(consVarOld, x2Flux, x2FluxDiffusive, indexRange, nvars);
	    , fluxFunction<FluxDir::X3>(consVarOld, x3Flux, x3FluxDiffusive, indexRange, nvars);)

	amrex::GpuArray<arrayconst_t, AMREX_SPACEDIM> fluxArrays = {
	    AMREX_D_DECL(x1Flux.const_array(), x2Flux.const_array(), x3Flux.const_array())};
	amrex::GpuArray<arrayconst_t, AMREX_SPACEDIM> fluxDiffusiveArrays{
	    AMREX_D_DECL(x1FluxDiffusive.const_array(), x2FluxDiffusive.const_array(),
			 x3FluxDiffusive.const_array())};

	// Stage 1 of RK2-SSP
	RadSystem<problem_t>::PredictStep(consVarOld, consVarNew, fluxArrays, fluxDiffusiveArrays,
					  dtRadiation_, dx_, indexRange, nvars);
}

template <typename problem_t>
void RadhydroSimulation<problem_t>::stageTwoRK2SSP(
    amrex::Array4<const amrex::Real> const &consVarOld,
    amrex::Array4<amrex::Real> const &consVarNew, const amrex::Box &indexRange, const int nvars)
{
	// Allocate temporary arrays using CUDA stream async allocator (or equivalent)
	amrex::Box const &x1FluxRange = amrex::surroundingNodes(indexRange, 0);
	amrex::FArrayBox x1Flux(x1FluxRange, nvars, amrex::The_Async_Arena()); // node-centered in x
	amrex::FArrayBox x1FluxDiffusive(x1FluxRange, nvars, amrex::The_Async_Arena());
#if (AMREX_SPACEDIM >= 2)
	amrex::Box const &x2FluxRange = amrex::surroundingNodes(indexRange, 1);
	amrex::FArrayBox x2Flux(x2FluxRange, nvars, amrex::The_Async_Arena()); // node-centered in y
	amrex::FArrayBox x2FluxDiffusive(x2FluxRange, nvars, amrex::The_Async_Arena());
#endif
#if (AMREX_SPACEDIM == 3)
	amrex::Box const &x3FluxRange = amrex::surroundingNodes(indexRange, 2);
	amrex::FArrayBox x3Flux(x3FluxRange, nvars, amrex::The_Async_Arena()); // node-centered in z
	amrex::FArrayBox x3FluxDiffusive(x3FluxRange, nvars, amrex::The_Async_Arena());
#endif

	AMREX_D_TERM(
	    fluxFunction<FluxDir::X1>(consVarNew, x1Flux, x1FluxDiffusive, indexRange, nvars);
	    , fluxFunction<FluxDir::X2>(consVarNew, x2Flux, x2FluxDiffusive, indexRange, nvars);
	    , fluxFunction<FluxDir::X3>(consVarNew, x3Flux, x3FluxDiffusive, indexRange, nvars);)

	amrex::GpuArray<arrayconst_t, AMREX_SPACEDIM> fluxArrays = {
	    AMREX_D_DECL(x1Flux.const_array(), x2Flux.const_array(), x3Flux.const_array())};
	amrex::GpuArray<arrayconst_t, AMREX_SPACEDIM> fluxDiffusiveArrays{
	    AMREX_D_DECL(x1FluxDiffusive.const_array(), x2FluxDiffusive.const_array(),
			 x3FluxDiffusive.const_array())};

	// Stage 2 of RK2-SSP
	RadSystem<problem_t>::AddFluxesRK2(consVarNew, consVarOld, consVarNew, fluxArrays,
					   fluxDiffusiveArrays, dtRadiation_, dx_, indexRange,
					   nvars);
}

#endif // RADIATION_SIMULATION_HPP_