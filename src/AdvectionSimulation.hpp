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

// Simulation class should be initialized only once per program (i.e., is a singleton)
template <typename problem_t>
class AdvectionSimulation : public SingleLevelSimulation<problem_t>
{
    using SingleLevelSimulation<problem_t>::state_old_;
    using SingleLevelSimulation<problem_t>::state_new_;
    using SingleLevelSimulation<problem_t>::cflNumber_;
    using SingleLevelSimulation<problem_t>::simGeometry_;

      public:
    LinearAdvectionSystem<problem_t> advectionSystem_;

    auto computeTimestepLocal() -> amrex::Real override;
	void setInitialConditions() override;
	void advanceSingleTimestep() override;

    protected:
    const double advectionVx_ = 1.0;
};

template <typename problem_t>
auto AdvectionSimulation<problem_t>::computeTimestepLocal() -> amrex::Real
{
    // loop over local grids, compute timestep based on linear advection CFL

    AMREX_D_TERM(const Real dxinv = simGeometry_.InvCellSize(0);,
                 const Real dyinv = simGeometry_.InvCellSize(1);,
                 const Real dzinv = simGeometry_.InvCellSize(2););
    
    const auto dt_max = std::numeric_limits<double>::max();
    amrex::Real dt = 0.0;

    // MFIter = MultiFab Iterator
    for ( amrex::MFIter iter(state_new_); iter.isValid(); ++iter )
    {
        const amrex::Box& indexRange = iter.validbox(); // 'validbox' == exclude ghost zones
        amrex::Array4<amrex::Real> const& stateNew = state_new_.array(iter);
        auto thisDt = advectionSystem_.ComputeTimestep(dt_max, cflNumber_, 1.0/dxinv, advectionVx_);
        dt = std::max(dt, thisDt);
    }
}

template <typename problem_t>
void AdvectionSimulation<problem_t>::setInitialConditions()
{
    // do nothing -- user should implement using template override
}

template <typename problem_t>
void AdvectionSimulation<problem_t>::advanceSingleTimestep()
{
    // update ghost zones
    // TODO(ben): implement

    // advance all grids on local processor with timestep dt_ (already computed)
    for ( amrex::MFIter iter(state_new_); iter.isValid(); ++iter )
    {
        const amrex::Box& indexRange = iter.validbox(); // 'validbox' == exclude ghost zones
        amrex::Array4<amrex::Real> const& stateNew = state_new_.array(iter);
        
        amrex::ParallelFor(indexRange,
        [=] AMREX_GPU_DEVICE(int i, int j, int k)
        {
            // compute zone update for cell (i,j,k)
        });
    }
}

#endif // ADVECTION_SIMULATION_HPP_