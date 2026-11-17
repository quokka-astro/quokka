// SPDX-FileCopyrightText: © The Quokka Authors
// SPDX-License-Identifier: MIT

/// \file testStromgrenSphere.cpp
/// \brief Defines a test problem for the static Stromgren sphere with no temperature dependence.
///

#include "StromgrenSphere.hpp"
#include "microphysics/microphysics_traits.hpp"
#include "networks/photoionization_H_caseB_handwritten/network.hpp"

struct StromgrenSphere {};

template <> struct Microphysics_Traits<StromgrenSphere> : DefaultMicrophysicsTraits {
	using Network = quokka::networks::PhotoionizationHCaseB;
	using Integrator = quokka::microphysics::Rosenbrock<quokka::microphysics::ROS2S>;
	static constexpr bool photochemistry = true;
	static constexpr bool integrate_rad_flux = false;
};

template <> struct quokka::EOS_Traits<StromgrenSphere> : stromgren_sphere::EOSTraits {};
template <> struct Physics_Traits<StromgrenSphere> : stromgren_sphere::PhysicsTraits<StromgrenSphere> {};
template <> struct RadSystem_Traits<StromgrenSphere> : stromgren_sphere::RadTraits<StromgrenSphere> {};
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
