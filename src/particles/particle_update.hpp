#ifndef PARTICLE_UPDATE_HPP_
#define PARTICLE_UPDATE_HPP_

#include "AMReX_Extension.H"
#include "particle_radiation.hpp"
#include "particle_types.hpp"

#if AMREX_SPACEDIM == 3

namespace quokka
{

// Traits class for specializing particle property update behavior
template <ParticleType particleType> struct ParticlePropertyUpdateTraits {
	// Default implementation - does nothing
	template <typename problem_t, typename ParticleType>
	AMREX_GPU_DEVICE AMREX_FORCE_INLINE static void updateProperties(ParticleType & /*p*/, amrex::Real /*current_time*/) noexcept
	{
		// Default implementation does nothing
	}
};

// Specialization for StochasticStellarPop particles with stellar evolution
// This uses table interpolation and can be overridden in the problem generator
template <> struct ParticlePropertyUpdateTraits<ParticleType::StochasticStellarPop> {
	template <typename problem_t, typename ParticleType>
	AMREX_GPU_DEVICE AMREX_FORCE_INLINE static void updateProperties(ParticleType &p, amrex::Real current_time) noexcept
	{
		// Update luminosity using the LuminosityUpdate class
		LuminosityUpdate::updateLuminosity<problem_t>(p, current_time);
	}
};

} // namespace quokka

#endif // AMREX_SPACEDIM == 3

#endif // PARTICLE_UPDATE_HPP_

