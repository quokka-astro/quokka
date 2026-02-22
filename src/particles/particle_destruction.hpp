#ifndef PARTICLE_DESTRUCTION_HPP_
#define PARTICLE_DESTRUCTION_HPP_

#include "AMReX_BLProfiler.H"
#include "particle_types.hpp"
#include <format>

namespace quokka
{

#if AMREX_SPACEDIM == 3

// Helper namespace with implementation details for particle destruction
namespace ParticleDestructionImpl
{
// Common implementation of particle destruction logic
template <typename problem_t, typename ContainerType, template <typename> class CheckerType>
static void destroyParticlesImpl(ContainerType *container, int mass_idx, int lev_min, amrex::Real current_time, amrex::Real dt, int birth_time_index,
				 int evolution_stage_index)
{
	const BL_PROFILE("ParticleDestructionImpl::destroyParticlesImpl()");
	if (container != nullptr) {
		if (mass_idx >= 0) {
			// Counter for total particles destroyed at this time step
			amrex::Long total_particles_destroyed = 0;

			// Use the provided ParticleChecker type to determine which particles to destroy
			CheckerType<problem_t> particle_checker(birth_time_index, evolution_stage_index);

			for (int lev = lev_min; lev <= container->finestLevel(); ++lev) {
				// Iterate through all particles at this level
				for (typename ContainerType::ParIterType pti(*container, lev); pti.isValid(); ++pti) {
					auto &particles = pti.GetArrayOfStructs();
					const int np = particles.numParticles();

					// Skip if no particles in this tile
					if (np == 0) {
						continue;
					}

					// Process particles on the device
					auto *parray = particles().data();

					// Create temporary array for reduction
					int destroyed_count = 0;
					amrex::Gpu::AsyncArray<int> h_count(&destroyed_count, 1);
					int *p_count = h_count.data();

					amrex::ParallelFor(np, [=] AMREX_GPU_DEVICE(int i) {
						auto &p = parray[i]; // NOLINT

						// Check if this particle should be destroyed
						if (particle_checker(p, mass_idx, current_time, dt)) {
							// Mark particle as invalid by negating its ID
							// This is the AMReX way to mark particles for removal
							// They will be removed during the next Redistribute call
							p.id() = -1;
							amrex::Gpu::Atomic::Add(p_count, 1);
						}
					});

					// Get the count from device memory
					// amrex::Gpu::streamSynchronize();
					h_count.copyToHost(&destroyed_count, 1);
					total_particles_destroyed += destroyed_count;
				}

				// Redistribute particles at this level to actually remove the invalid particles
				// TODO(cch): This won't work when AMR subcycling is enabled. When a particle moves into the ghost cells in the first step of
				// the subcycle, it may be moved from that level into a lower level. Then, in the second step of the subcycle, it will not be
				// drifted.
			}

			// Redistribute particles at lev and above to actually remove the invalid particles
			container->Redistribute(lev_min);

			// Sum up total particles destroyed across all processors
			amrex::Long global_total_destroyed = total_particles_destroyed;
			amrex::ParallelDescriptor::ReduceLongSum(global_total_destroyed);

			// Print the total number of particles destroyed at this time step
			if (amrex::ParallelDescriptor::IOProcessor()) {
				if (particle_verbose > 0) {
					amrex::Print()
					    << std::format("[PARTICLES] Particle destruction: Time: {} - Destroyed {} particles (from level {} and above)\n",
							   current_time, global_total_destroyed, lev_min);
				}
			}
		}
	}
}
} // namespace ParticleDestructionImpl

// Traits class for specializing particle destruction behavior
template <ParticleType particleType> struct ParticleDestructionTraits {
	// Default nested ParticleChecker - determines if a particle should be destroyed
	template <typename problem_t> struct ParticleChecker {
		int birth_time_index;
		int evolution_stage_index;

		AMREX_GPU_HOST_DEVICE explicit ParticleChecker(int birth_time_index, int evolution_stage_index)
		    : birth_time_index(birth_time_index), evolution_stage_index(evolution_stage_index)
		{
		}

		template <typename ParticleType>
		AMREX_GPU_DEVICE auto operator()(ParticleType &p, int mass_idx, amrex::Real current_time, amrex::Real dt) const -> bool
		{
			amrex::ignore_unused(mass_idx, current_time, dt);
			// Remove particles with evolution stage Removed
			const bool will_be_removed = (p.idata(evolution_stage_index) == static_cast<int>(StellarEvolutionStage::Removed));
			return will_be_removed;
		}
	};

	// Main method to destroy particles - uses the helper implementation
	template <typename problem_t, typename ContainerType>
	static void destroyParticles(ContainerType *container, int mass_idx, int lev_min, amrex::Real current_time, amrex::Real dt, int birth_time_index,
				     int evolution_stage_index)
	{
		const BL_PROFILE("ParticleDestructionTraits::destroyParticles()");
		// Use the common implementation with our checker type
		ParticleDestructionImpl::destroyParticlesImpl<problem_t, ContainerType, ParticleDestructionTraits<particleType>::template ParticleChecker>(
		    container, mass_idx, lev_min, current_time, dt, birth_time_index, evolution_stage_index);
	}
};

#endif // AMREX_SPACEDIM == 3

} // namespace quokka

#endif // PARTICLE_DESTRUCTION_HPP_
