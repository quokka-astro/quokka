#ifndef PARTICLE_CREATION_HPP_
#define PARTICLE_CREATION_HPP_

#include "particle_types.hpp"

namespace quokka
{

// Helper namespace with implementation details for particle creation
namespace ParticleCreationImpl
{
// Common implementation of particle creation logic
template <typename problem_t, typename ContainerType, template <typename> class CheckerType, template <typename> class CreatorType>
static void createParticlesImpl(ContainerType *container, int mass_idx, amrex::MultiFab &state, int lev, amrex::Real current_time, amrex::Real dt,
				int evolution_stage_index = -1, int birth_time_index = -1)
{
	if (container != nullptr) {
		if (mass_idx >= 0) {
			// Use the provided ParticleChecker type with global particle parameters
			CheckerType<problem_t> particle_checker(current_time, dt);

			for (amrex::MFIter mfi = container->MakeMFIter(lev); mfi.isValid(); ++mfi) {
				const auto &box = mfi.validbox();
				const auto &state_arr = state.array(mfi);
				const auto &geom = container->Geom(lev);
				const auto dx = geom.CellSizeArray();
				const auto plo = geom.ProbLoArray();

				// Count particles to be created in this box
				amrex::Gpu::DeviceVector<unsigned int> counts(box.numPts()); // 1 if cell creates particle, 0 if not
				amrex::Gpu::DeviceVector<unsigned int> offset(box.numPts()); // Will store starting index for each cell's particle
				auto *pcounts = counts.data();

				// Count potential particles per cell
				amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
					const amrex::IntVect iv(AMREX_D_DECL(i, j, k));
					const auto index = box.index(iv);
					// Check if we should create a particle at this location and time
					pcounts[index] = particle_checker(state_arr, i, j, k, dx); // NOLINT
				});

				// Calculate exclusive prefix sum to get unique position for each particle
				// Example: counts  = [1, 0, 1, 0, 1]
				//         offset  = [0, 1, 1, 2, 2]
				const unsigned int max_new_particles = amrex::Scan::ExclusiveSum(counts.size(), counts.data(), offset.data());

				// Update NextID to include particles that will be created
				const amrex::Long pid = ContainerType::ParticleType::NextID();
				ContainerType::ParticleType::NextID(pid + max_new_particles);

				// Get the particle tile and prepare for new particles
				auto &particle_tile = container->DefineAndReturnParticleTile(lev, mfi);
				auto &aos = particle_tile.GetArrayOfStructs();
				const int old_size = aos.size();
				aos.resize(old_size + max_new_particles);

				// Create the particles
				auto *poffset = offset.data();
				auto *pdata = aos.data() + old_size;
				const int cpu_id = amrex::ParallelDescriptor::MyProc();

				// Initialize particle creator functor using the provided ParticleCreator type
				CreatorType<problem_t> particle_creator(mass_idx, birth_time_index, cpu_id, pid, evolution_stage_index, current_time);

				amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
					const amrex::IntVect iv(AMREX_D_DECL(i, j, k));
					const auto index = box.index(iv);

					if (pcounts[index] > 0) {									 // NOLINT
						const int num_particles = pcounts[index];						 // NOLINT
						auto *particles = &pdata[poffset[index]];						 // NOLINT
						particle_creator(particles, num_particles, state_arr, i, j, k, dx, plo, poffset[index]); // NOLINT
					}
				});
			}
		}
	}
}
} // namespace ParticleCreationImpl

// Traits class for specializing particle creation behavior
template <ParticleType particleType> struct ParticleCreationTraits {
	// Default nested ParticleChecker - determines if a particle should be created at a location
	template <typename problem_t> struct ParticleChecker {
		amrex::Real current_time;
		amrex::Real dt;

		AMREX_GPU_HOST_DEVICE ParticleChecker(amrex::Real current_time, amrex::Real dt) : current_time(current_time), dt(dt) {}

		AMREX_GPU_DEVICE auto operator()(amrex::Array4<const amrex::Real> const &state_arr, int i, int j, int k,
						 amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx) const -> int
		{
			// Default implementation creates no particles
			amrex::ignore_unused(state_arr, i, j, k, dx);
			return 0;
		}
	};

	// Default nested ParticleCreator - initializes a particle's properties
	template <typename problem_t> struct ParticleCreator {
		int mass_idx;
		int birth_time_index;
		int evolution_stage_index;
		int cpu_id;
		amrex::Long pid_start;
		amrex::Real current_time;

		AMREX_GPU_HOST_DEVICE
		ParticleCreator(int mass_index, int birth_time_index, int processor_id, amrex::Long particle_id_start, int evolution_stage_index,
				amrex::Real current_time)
		    : mass_idx(mass_index), birth_time_index(birth_time_index), evolution_stage_index(evolution_stage_index), cpu_id(processor_id),
		      pid_start(particle_id_start), current_time(current_time)
		{
		}

		template <typename ParticleType, typename StateArray>
		AMREX_GPU_DEVICE void operator()(ParticleType *particles, int num_particles, StateArray const &state_arr, int i, int j, int k,
						 amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx,
						 amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &plo, amrex::Long base_offset) const
		{
			// Default implementation does nothing
			amrex::ignore_unused(particles, num_particles, state_arr, i, j, k, dx, plo, base_offset);
		}
	};

	// Main method to create particles - uses the helper implementation
	template <typename problem_t, typename ContainerType>
	static void createParticles(ContainerType *container, int mass_idx, amrex::MultiFab &state, int lev, amrex::Real current_time, amrex::Real dt,
				    int evolution_stage_index = -1, int birth_time_index = -1)
	{
		// Use the common implementation with our checker and creator types
		ParticleCreationImpl::createParticlesImpl<problem_t, ContainerType, ParticleCreationTraits<particleType>::template ParticleChecker,
							  ParticleCreationTraits<particleType>::template ParticleCreator>(
		    container, mass_idx, state, lev, current_time, dt, evolution_stage_index, birth_time_index);
	}
};

} // namespace quokka

#endif // PARTICLE_CREATION_HPP_