//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file testStromgrenSphere.cpp
/// \brief Defines a test problem for the static Stromgren sphere with no temperature dependence.
///

#include "StromgrenSphere.hpp"
#include "microphysics/microphysics_traits.hpp"

struct StromgrenSphere {};

template <> struct Microphysics_Traits<StromgrenSphere> : DefaultMicrophysicsTraits {
	using Integrator = quokka::microphysics::Rosenbrock<quokka::microphysics::ROS2S>;
	static constexpr bool photochemistry = true;	  // built with PHOTOCHEMISTRY
	static constexpr bool integrate_rad_flux = false; // built with SKIP_PHOTOCHEMFLUX
};

template <> struct quokka::EOS_Traits<StromgrenSphere> : stromgren_sphere::EOSTraits {};
template <> struct Physics_Traits<StromgrenSphere> : stromgren_sphere::PhysicsTraits {};
template <> struct RadSystem_Traits<StromgrenSphere> : stromgren_sphere::RadTraits {};
template <> struct SimulationData<StromgrenSphere> : stromgren_sphere::Data {};

template <>
void RadSystem<StromgrenSphere>::AddRadSource(array_t &radEnergy, array_t & /*reducedFluxSource*/, const amrex::Box &indexRange,
					      amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx,
					      amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_lo,
					      amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_hi, amrex::Real /*time*/)
{
	stromgren_sphere::AddRadSource<StromgrenSphere>(radEnergy, indexRange, dx, prob_lo, prob_hi);
}

template <> AMREX_GPU_HOST_DEVICE auto RadSystem<StromgrenSphere>::ComputePlanckOpacity(const double /*rho*/, const double /*Tgas*/) -> amrex::Real
{
	return stromgren_sphere::opacity;
}

template <> AMREX_GPU_HOST_DEVICE auto RadSystem<StromgrenSphere>::ComputeFluxMeanOpacity(const double /*rho*/, const double /*Tgas*/) -> amrex::Real
{
	return stromgren_sphere::opacity;
}

template <> void QuokkaSimulation<StromgrenSphere>::preCalculateInitialConditions() { stromgren_sphere::PreCalculateInitialConditions(*this); }

template <> void QuokkaSimulation<StromgrenSphere>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	stromgren_sphere::SetInitialConditionsOnGrid(*this, grid_elem);
}

template <> void QuokkaSimulation<StromgrenSphere>::computeAfterTimestep() { stromgren_sphere::ComputeAfterTimestep(*this); }

auto problem_main() -> int { return stromgren_sphere::ProblemMain<StromgrenSphere>(); }
