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

#include "AMReX_Array4.H"
#include "AMReX_BLassert.H"
#include "AMReX_Box.H"
#include "AMReX_FArrayBox.H"
#include "AMReX_IntVect.H"
#include "AMReX_REAL.H"
#include "linear_advection.hpp"
#include "simulation.hpp"
#include <climits>
#include <limits>
#include <utility>

// Simulation class should be initialized only once per program (i.e., is a singleton)
template <typename problem_t> class AdvectionSimulation : public SingleLevelSimulation<problem_t>
{
      public:
	using SingleLevelSimulation<problem_t>::simGeometry_;
	using SingleLevelSimulation<problem_t>::state_old_;
	using SingleLevelSimulation<problem_t>::state_new_;

	using SingleLevelSimulation<problem_t>::cflNumber_;
	using SingleLevelSimulation<problem_t>::dx_;
	using SingleLevelSimulation<problem_t>::dt_;
	using SingleLevelSimulation<problem_t>::ncomp_;
	using SingleLevelSimulation<problem_t>::nghost_;
	using SingleLevelSimulation<problem_t>::tNow_;
	using SingleLevelSimulation<problem_t>::cycleCount_;

	explicit AdvectionSimulation() = default;

	auto computeTimestepLocal() -> amrex::Real override;
	void setInitialConditions() override;
	void advanceSingleTimestep() override;
	void stageOneRK2SSP(amrex::Array4<const amrex::Real> const &consVarOld,
			    amrex::Array4<amrex::Real> const &consVarNew,
			    const amrex::Box &indexRange, int nvars);
	void stageTwoRK2SSP(amrex::Array4<const amrex::Real> const &consVar,
			    amrex::Array4<amrex::Real> const &consVarNew,
			    const amrex::Box &indexRange, int nvars);

      protected:
	const double advectionVx_ = 1.0;
};

template <typename problem_t>
auto AdvectionSimulation<problem_t>::computeTimestepLocal() -> amrex::Real
{
	// loop over local grids, compute timestep based on linear advection CFL

	AMREX_D_TERM(const Real dxinv = simGeometry_.InvCellSize(0);
		     , const Real dyinv = simGeometry_.InvCellSize(1);
		     , const Real dzinv = simGeometry_.InvCellSize(2););

	const auto dt_max = std::numeric_limits<double>::max();
	amrex::Real dt = 0.0;

	// iterating over multifabs is technically not necessary for linear advection timestep
	// but we do so here in order to illustrate the idea
	for (amrex::MFIter iter(state_new_); iter.isValid(); ++iter) {
		auto thisDt = LinearAdvectionSystem<problem_t>::ComputeTimestep(
		    dt_max, cflNumber_, 1.0 / dxinv, advectionVx_);
		dt = std::max(dt, thisDt);
	}

	return dt;
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

	// advance all grids on local processor (Stage 1 of integrator)
	for (amrex::MFIter iter(state_new_); iter.isValid(); ++iter) {
		const amrex::Box &indexRange = iter.validbox(); // 'validbox' == exclude ghost zones
		auto const &stateOld = state_old_.const_array(iter);
		auto const &stateNew = state_new_.array(iter);
		stageOneRK2SSP(stateOld, stateNew, indexRange,
			       ncomp_); // result saved in state_new_
	}

	const std::string& pltfile = amrex::Concatenate("plt", cycleCount_, 5);
    amrex::WriteSingleLevelPlotfile(pltfile, state_new_, {"density"}, simGeometry_, tNow_, cycleCount_);
	AMREX_ASSERT(!state_new_.contains_nan());
	//amrex::VisMF::Write(state_new_, "state_new_.amrex");

	// update ghost zones [intermediate stage stored in state_new_]
	state_new_.FillBoundary(simGeometry_.periodicity());

	// advance all grids on local processor (Stage 2 of integrator)
	for (amrex::MFIter iter(state_new_); iter.isValid(); ++iter) {
		const amrex::Box &indexRange = iter.validbox(); // 'validbox' == exclude ghost zones
		auto const &stateOld = state_old_.const_array(iter);
		auto const &stateNew = state_new_.array(iter);
		stageTwoRK2SSP(stateOld, stateNew, indexRange,
			       ncomp_); // result saved in state_new_
	}
	AMREX_ASSERT(!state_new_.contains_nan());
}

template <typename problem_t>
void AdvectionSimulation<problem_t>::stageOneRK2SSP(
    amrex::Array4<const amrex::Real> const &consVarOld,
    amrex::Array4<amrex::Real> const &consVarNew, const amrex::Box &indexRange, const int nvars)
{
	// extend box to include ghost zones (PPM requires at least 2 ghost zones)
	amrex::Box const &ghostRange = amrex::grow(indexRange, nghost_);
	amrex::Box const &ppmRange = amrex::grow(indexRange, 1);

	// Allocate temporary arrays
	amrex::FArrayBox primVar(ghostRange, nvars);
	amrex::FArrayBox x1LeftState(ghostRange, nvars);
	amrex::FArrayBox x1RightState(ghostRange, nvars);
	amrex::FArrayBox x1Flux(ghostRange, nvars);

	// Stage 1 of RK2-SSP
	{
		LinearAdvectionSystem<problem_t>::ConservedToPrimitive(consVarOld, primVar.array(),
								       ghostRange, nvars);
		LinearAdvectionSystem<problem_t>::ReconstructStatesPPM(
		    primVar.array(), x1LeftState.array(), x1RightState.array(), ppmRange, nvars);

		LinearAdvectionSystem<problem_t>::ComputeFluxes(x1Flux.array(), x1LeftState.array(),
								x1RightState.array(), advectionVx_,
								indexRange, nvars);
		LinearAdvectionSystem<problem_t>::PredictStep(
		    consVarOld, consVarNew, x1Flux.array(), dt_, dx_[0], indexRange, nvars);
	}
}

template <typename problem_t>
void AdvectionSimulation<problem_t>::stageTwoRK2SSP(
    amrex::Array4<const amrex::Real> const &consVarOld,
    amrex::Array4<amrex::Real> const &consVarNew, const amrex::Box &indexRange, const int nvars)
{
	// extend box to include ghost zones (PPM requires at least 2 ghost zones)
	amrex::Box const &ghostRange = amrex::grow(indexRange, nghost_);
	amrex::Box const &ppmRange = amrex::grow(indexRange, 1);

	// Allocate temporary arrays
	amrex::FArrayBox primVar(ghostRange, nvars);
	amrex::FArrayBox x1LeftState(ghostRange, nvars);
	amrex::FArrayBox x1RightState(ghostRange, nvars);
	amrex::FArrayBox x1Flux(ghostRange, nvars);

	// Stage 2 of RK2-SSP
	{
		LinearAdvectionSystem<problem_t>::ConservedToPrimitive(consVarNew, primVar.array(),
								       ghostRange, nvars);
		LinearAdvectionSystem<problem_t>::ReconstructStatesPPM(
		    primVar.array(), x1LeftState.array(), x1RightState.array(), ppmRange, nvars);

		LinearAdvectionSystem<problem_t>::ComputeFluxes(x1Flux.array(), x1LeftState.array(),
								x1RightState.array(), advectionVx_,
								indexRange, nvars);
		LinearAdvectionSystem<problem_t>::AddFluxesRK2(consVarNew, consVarOld, consVarNew,
							       x1Flux.array(), dt_, dx_[0],
							       indexRange, nvars);
	}
}

#endif // ADVECTION_SIMULATION_HPP_