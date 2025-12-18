#ifndef DENSITY_FLOOR_HPP_ // NOLINT
#define DENSITY_FLOOR_HPP_
//==============================================================================
// Quokka - a GPU-accelerated astrophysical simulation code built on AMReX.
//==============================================================================
/// \file density_floor.hpp
/// \brief Per-problem customization point for a spatially varying density floor.

#include "AMReX_Extension.H"
#include "AMReX_GpuQualifiers.H"
#include "AMReX_REAL.H"
#include "AMReX_SPACE.H"

namespace quokka
{

// Specialize this in the problem generator (e.g. `src/problems/.../*.cpp`) to
// set a spatially varying density floor.
//
// The default implementation returns `base_density_floor` (i.e. constant floor).
template <typename problem_t> struct DensityFloor {
	AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE static auto value(amrex::Real x, amrex::Real y, amrex::Real z,
								  amrex::Real base_density_floor) -> amrex::Real
	{
		amrex::ignore_unused(x, y, z);
		return base_density_floor;
	}
};

} // namespace quokka

#endif // DENSITY_FLOOR_HPP_
