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
#include "AMReX_REAL.H"
#include "linear_advection.hpp"
#include "simulation.hpp"
#include <limits>
#include <utility>

// Simulation class should be initialized only once per program (i.e., is a singleton)
template <typename problem_t> class AdvectionSimulation : public SingleLevelSimulation<problem_t>
{
	using SingleLevelSimulation<problem_t>::simGeometry_;
	using SingleLevelSimulation<problem_t>::state_old_;
	using SingleLevelSimulation<problem_t>::state_new_;

	using SingleLevelSimulation<problem_t>::cflNumber_;
	using SingleLevelSimulation<problem_t>::dx_;
	using SingleLevelSimulation<problem_t>::dt_;
	using SingleLevelSimulation<problem_t>::ncomp_;

      public:
	explicit AdvectionSimulation() = default;

	auto computeTimestepLocal() -> amrex::Real override;
	void setInitialConditions() override;
	void advanceSingleTimestep() override;
	void stageOneRK2SSP(double dt, array_t &consVar, const amrex::Box &indexRange, int nvars);
	void stageTwoRK2SSP(double dt, array_t &consVar, const amrex::Box &indexRange, int nvars);

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
	for (amrex::MFIter iter(state_new_); iter.isValid(); ++iter) {
		auto thisDt = LinearAdvectionSystem<problem_t>::ComputeTimestep(
		    dt_max, cflNumber_, 1.0 / dxinv, advectionVx_);
		dt = std::max(dt, thisDt);
	}

	return dt;
}

template <typename problem_t> void AdvectionSimulation<problem_t>::setInitialConditions()
{
	// do nothing -- user should implement using template override
}

template <typename problem_t> void AdvectionSimulation<problem_t>::advanceSingleTimestep()
{
	// update ghost zones
	// TODO(ben): implement

	// advance all grids on local processor (Stage 1 of integrator)
	for (amrex::MFIter iter(state_new_); iter.isValid(); ++iter) {
		const amrex::Box &indexRange = iter.validbox(); // 'validbox' == exclude ghost zones
		amrex::Array4<amrex::Real> const &stateNew = state_new_.array(iter);
		stageOneRK2SSP(dt_, stateNew, indexRange, ncomp_);
	}

	// update ghost zones (again)
	// TODO(ben): implement

	// advance all grids on local processor (Stage 2 of integrator)
	for (amrex::MFIter iter(state_new_); iter.isValid(); ++iter) {
		const amrex::Box &indexRange = iter.validbox(); // 'validbox' == exclude ghost zones
		amrex::Array4<amrex::Real> const &stateNew = state_new_.array(iter);
		stageTwoRK2SSP(dt_, stateNew, indexRange, ncomp_);
	}
}

template <typename problem_t>
void AdvectionSimulation<problem_t>::stageOneRK2SSP(const double dt, array_t &consVar,
						    const amrex::Box &indexRange, const int nvars)
{
	// convert indexRange to cell_range (std::pair<int,int> along x-direction)
	const auto lowerIndex = indexRange.smallEnd();
	const auto upperIndex = indexRange.bigEnd();
	const auto cell_range = std::make_pair(lowerIndex[0], upperIndex[0]);

	const auto ppm_range = std::make_pair(-1 + cell_range.first, 1 + cell_range.second);

	// Allocate temporary arrays
	const auto [dim1, dim2, dim3] = amrex::length(consVar.arr_);
	array_t primVar(nvars, dim1);
	array_t x1LeftState(nvars, dim1);
	array_t x1RightState(nvars, dim1);
	array_t x1Fluxes(nvars, dim1);
	array_t consVarPredictStep(nvars, dim1);

	// Stage 1 of RK2-SSP
	{
		// FillGhostZones(consVar);
		LinearAdvectionSystem<problem_t>::ConservedToPrimitive(consVar,
								       std::make_pair(0, dim1));
		LinearAdvectionSystem<problem_t>::ReconstructStatesPPM(
		    primVar, x1LeftState, x1RightState, ppm_range, nvars);
		LinearAdvectionSystem<problem_t>::ComputeFluxes(cell_range);
		LinearAdvectionSystem<problem_t>::PredictStep(cell_range);
	}

	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
	    LinearAdvectionSystem<problem_t>::CheckStatesValid(consVarPredictStep, cell_range),
	    "[stage 1] Non-realizable states produced. This should not happen!");
}

template <typename problem_t>
void AdvectionSimulation<problem_t>::stageTwoRK2SSP(const double dt, array_t &consVar,
						    const amrex::Box &indexRange, const int nvars)
{
	// convert indexRange to cell_range (std::pair<int,int> along x-direction)
	const auto lowerIndex = indexRange.smallEnd();
	const auto upperIndex = indexRange.bigEnd();
	const auto cell_range = std::make_pair(lowerIndex[0], upperIndex[0]);

	const auto ppm_range = std::make_pair(-1 + cell_range.first, 1 + cell_range.second);

	// Allocate temporary arrays
	const auto [dim1, dim2, dim3] = amrex::length(consVar.arr_);
	array_t primVar(nvars, dim1);
	array_t x1LeftState(nvars, dim1);
	array_t x1RightState(nvars, dim1);
	array_t x1Fluxes(nvars, dim1);
	array_t consVarPredictStep(nvars, dim1);

	// Stage 2 of RK2-SSP
	{
		// FillGhostZones(consVarPredictStep);
		LinearAdvectionSystem<problem_t>::ConservedToPrimitive(consVarPredictStep,
								       std::make_pair(0, dim1));
		LinearAdvectionSystem<problem_t>::ReconstructStatesPPM(
		    primVar, x1LeftState, x1RightState, ppm_range, nvars);
		LinearAdvectionSystem<problem_t>::ComputeFluxes(cell_range);
		LinearAdvectionSystem<problem_t>::AddFluxesRK2(consVar, consVarPredictStep);
	}

	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
	    LinearAdvectionSystem<problem_t>::CheckStatesValid(consVar, cell_range),
	    "[stage 2] Non-realizable states produced. This should not happen!");
}

#endif // ADVECTION_SIMULATION_HPP_