//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file testStromgrenSphereTempIndependentRecombination.cpp
/// \brief The static Stromgren sphere problem of testStromgrenSphere.cpp, built as its own executable for
/// inputs/StromgrenSphereTempIndependentRecombination.toml.
///

#include "StromgrenSphere.hpp"
#include "microphysics/microphysics_traits.hpp"

struct StromgrenSphereTempIndependentRecombination {};

template <> struct Microphysics_Traits<StromgrenSphereTempIndependentRecombination> : DefaultMicrophysicsTraits {
	static constexpr bool integrate_rad_flux = false; // built with SKIP_PHOTOCHEMFLUX
};

template <> struct quokka::EOS_Traits<StromgrenSphereTempIndependentRecombination> : stromgren_sphere::EOSTraits {};
template <> struct Physics_Traits<StromgrenSphereTempIndependentRecombination> : stromgren_sphere::PhysicsTraits {};
template <> struct RadSystem_Traits<StromgrenSphereTempIndependentRecombination> : stromgren_sphere::RadTraits {};
template <> struct SimulationData<StromgrenSphereTempIndependentRecombination> : stromgren_sphere::Data {};

template <>
void RadSystem<StromgrenSphereTempIndependentRecombination>::AddRadSource(array_t &radEnergy, array_t & /*reducedFluxSource*/, const amrex::Box &indexRange,
									  amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx,
									  amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_lo,
									  amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_hi, amrex::Real /*time*/)
{
	stromgren_sphere::AddRadSource<StromgrenSphereTempIndependentRecombination>(radEnergy, indexRange, dx, prob_lo, prob_hi);
}

template <>
AMREX_GPU_HOST_DEVICE auto RadSystem<StromgrenSphereTempIndependentRecombination>::ComputePlanckOpacity(const double /*rho*/, const double /*Tgas*/)
    -> amrex::Real
{
	return stromgren_sphere::opacity;
}

template <>
AMREX_GPU_HOST_DEVICE auto RadSystem<StromgrenSphereTempIndependentRecombination>::ComputeFluxMeanOpacity(const double /*rho*/, const double /*Tgas*/)
    -> amrex::Real
{
	return stromgren_sphere::opacity;
}

template <> void QuokkaSimulation<StromgrenSphereTempIndependentRecombination>::preCalculateInitialConditions()
{
	stromgren_sphere::PreCalculateInitialConditions(*this);
}

template <> void QuokkaSimulation<StromgrenSphereTempIndependentRecombination>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	stromgren_sphere::SetInitialConditionsOnGrid(*this, grid_elem);
}

template <> void QuokkaSimulation<StromgrenSphereTempIndependentRecombination>::computeAfterTimestep() { stromgren_sphere::ComputeAfterTimestep(*this); }

auto problem_main() -> int { return stromgren_sphere::ProblemMain<StromgrenSphereTempIndependentRecombination>(); }
