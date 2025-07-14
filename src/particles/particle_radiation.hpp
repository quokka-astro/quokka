#ifndef PARTICLE_RADIATION_HPP_
#define PARTICLE_RADIATION_HPP_

#include "AMReX_Array.H"
#include "AMReX_Array4.H"
#include "AMReX_Extension.H"
#include "AMReX_ParticleInterpolators.H"
#include "hydro/hydro_system.hpp"
// Forward declaration for particle types
#include "particle_types.hpp"
#include "physics_info.hpp"

namespace quokka
{
// Traits class for specializing particle property update behavior
template <ParticleType particleType>
struct ParticlePropertyUpdateTraits {
	// Default implementation - does nothing
	template <typename problem_t, typename ParticleType>
	AMREX_GPU_DEVICE AMREX_FORCE_INLINE static void updateProperties(ParticleType &p, int mass_idx, int lum_idx, int birth_time_idx,
									 amrex::Real current_time) noexcept
	{
		// Default implementation does nothing
		amrex::ignore_unused(p, mass_idx, lum_idx, birth_time_idx, current_time);
	}
};
} // namespace quokka

#endif