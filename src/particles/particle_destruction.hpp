#ifndef PARTICLE_DESTRUCTION_HPP_
#define PARTICLE_DESTRUCTION_HPP_

#include "particle_types.hpp"

namespace quokka
{

// Helper namespace with implementation details for particle destruction
namespace ParticleDestructionImpl
{
// Common implementation of particle destruction logic
template <typename problem_t, typename ContainerType, template <typename> class CheckerType>
static void destroyParticlesImpl(ContainerType *container, int mass_idx, int lev, amrex::Real current_time, amrex::Real dt)
{
	if (container != nullptr) {
		if (mass_idx >= 0) {
			// Use the provided ParticleChecker type to determine which particles to destroy
			CheckerType<problem_t> particle_checker;

			// Iterate through all particles at this level
			for (typename ContainerType::ParIterType pti(*container, lev); pti.isValid(); ++pti) {
				auto& particles = pti.GetArrayOfStructs();
				const int np = particles.numParticles();
				
				// Skip if no particles in this tile
				if (np == 0) {
					continue;
				}
				
				// Process particles on the device
				auto* parray = particles().data();
				
				amrex::ParallelFor(np, [=] AMREX_GPU_DEVICE(int i) {
					auto& p = parray[i]; // NOLINT
					
					// Check if this particle should be destroyed
					if (particle_checker(p, mass_idx, current_time, dt)) {
						// Mark particle as invalid by negating its ID
						// This is the AMReX way to mark particles for removal
						// They will be removed during the next Redistribute call
						p.id() = -1;
					}
				});
			}
			
			// Redistribute particles to actually remove the invalid particles
			container->Redistribute(lev);
		}
	}
}
} // namespace ParticleDestructionImpl

// Traits class for specializing particle destruction behavior
template <ParticleType particleType> struct ParticleDestructionTraits {
	// Default nested ParticleChecker - determines if a particle should be destroyed
	template <typename problem_t> struct ParticleChecker {
		AMREX_GPU_HOST_DEVICE ParticleChecker() = default;

		template <typename ParticleType>
		AMREX_GPU_DEVICE auto operator()(ParticleType& p, int mass_idx, amrex::Real current_time, amrex::Real dt) const -> bool
		{
			// Default implementation: destroy particles with mass < 1.0
			amrex::ignore_unused(current_time, dt);
			
			// Check if mass is below threshold
			return (p.rdata(mass_idx) < 1.0); // Destroy this particle
		}
	};

	// Main method to destroy particles - uses the helper implementation
	template <typename problem_t, typename ContainerType>
	static void destroyParticles(ContainerType *container, int mass_idx, int lev, amrex::Real current_time, amrex::Real dt)
	{
		// Use the common implementation with our checker type
		ParticleDestructionImpl::destroyParticlesImpl<problem_t, ContainerType, 
			ParticleDestructionTraits<particleType>::template ParticleChecker>(
				container, mass_idx, lev, current_time, dt);
	}
};

} // namespace quokka

#endif // PARTICLE_DESTRUCTION_HPP_
