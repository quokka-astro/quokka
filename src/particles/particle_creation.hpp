#ifndef PARTICLE_CREATION_HPP_
#define PARTICLE_CREATION_HPP_

#include "hydro/hydro_system.hpp"
#include "particle_types.hpp"

namespace quokka
{

// Helper namespace with implementation details for particle creation
namespace ParticleCreationImpl
{
// Common implementation of particle creation logic
template <typename problem_t, typename ContainerType, template <typename> class CheckerType, template <typename> class CreatorType>
static void createParticlesImpl(ContainerType *container, int mass_idx, amrex::MultiFab &state, int lev, amrex::Real current_time, amrex::Real dt)
{
	if (container != nullptr) {
		if (mass_idx >= 0 && mass_idx + 3 < ContainerType::ParticleType::NReal) {
			// Use the provided ParticleChecker type with global particle parameters
			CheckerType<problem_t> particle_checker;

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
					pcounts[index] = particle_checker(state_arr, i, j, k, dx, current_time, dt) ? 1 : 0; // NOLINT
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
				CreatorType<problem_t> particle_creator(mass_idx, cpu_id, pid);

				amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
					const amrex::IntVect iv(AMREX_D_DECL(i, j, k));
					const auto index = box.index(iv);

					if (pcounts[index] > 0) {						  // NOLINT
						auto &p = pdata[poffset[index]];				  // NOLINT
						particle_creator(p, state_arr, i, j, k, dx, plo, poffset[index]); // NOLINT
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
		AMREX_GPU_HOST_DEVICE ParticleChecker() = default;

		AMREX_GPU_DEVICE auto operator()(amrex::Array4<const amrex::Real> const &state_arr, int i, int j, int k,
						 amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx, amrex::Real current_time, amrex::Real dt) const -> bool
		{
			// Default implementation creates no particles
			amrex::ignore_unused(state_arr, i, j, k, dx, current_time, dt);
			return false;
		}
	};

	// Default nested ParticleCreator - initializes a particle's properties
	template <typename problem_t> struct ParticleCreator {
		int mass_idx;
		int cpu_id;
		amrex::Long pid_start;

		AMREX_GPU_HOST_DEVICE
		ParticleCreator(int mass_index, int processor_id, amrex::Long particle_id_start)
		    : mass_idx(mass_index), cpu_id(processor_id), pid_start(particle_id_start)
		{
		}

		template <typename ParticleType, typename StateArray>
		AMREX_GPU_DEVICE void operator()(ParticleType &p, StateArray const &state_arr, int i, int j, int k,
						 amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx,
						 amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &plo, amrex::Long particle_offset) const
		{
			// Default implementation does nothing
			amrex::ignore_unused(p, state_arr, i, j, k, dx, plo, particle_offset);
		}
	};

	// Main method to create particles - uses the helper implementation
	template <typename problem_t, typename ContainerType>
	static void createParticles(ContainerType *container, int mass_idx, amrex::MultiFab &state, int lev, amrex::Real current_time, amrex::Real dt)
	{
		// Use the common implementation with our checker and creator types
		ParticleCreationImpl::createParticlesImpl<problem_t, ContainerType, ParticleCreationTraits<particleType>::template ParticleChecker,
							  ParticleCreationTraits<particleType>::template ParticleCreator>(container, mass_idx, state, lev,
															  current_time, dt);
	}
};

// Specialization for CIC particles
template <> struct ParticleCreationTraits<ParticleType::CIC> {
	// Specialized nested ParticleChecker for CIC particles
	template <typename problem_t> struct ParticleChecker {
		amrex::Real param1 = particle_param1;
		amrex::Real param2 = particle_param2;

		AMREX_GPU_DEVICE auto operator()(amrex::Array4<const amrex::Real> const &state_arr, int i, int j, int k,
						 amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx, amrex::Real current_time, amrex::Real dt) const -> bool
		{
			// A simple demonstration of particle creation
			// Could check density threshold or other state-based conditions
			amrex::ignore_unused(state_arr, dx);
			const int spacing = 16;
			const bool is_create_particle_1 = current_time <= param1 && current_time + dt > param1;
			const bool is_create_particle_2 = current_time <= param2 && current_time + dt > param2;
			return (is_create_particle_1 || is_create_particle_2) && (i != 0 && i % spacing == 0) && (j != 0 && j % spacing == 0) &&
			       (k != 0 && k % spacing == 0);
		}
	};

	// Specialized nested ParticleCreator for CIC particles
	template <typename problem_t> struct ParticleCreator {
		int mass_idx;
		int cpu_id;
		amrex::Long pid_start;
		amrex::Real param1 = particle_param1;
		amrex::Real param2 = particle_param2;

		AMREX_GPU_HOST_DEVICE
		ParticleCreator(int mass_index, int processor_id, amrex::Long particle_id_start)
		    : mass_idx(mass_index), cpu_id(processor_id), pid_start(particle_id_start)
		{
		}

		template <typename ParticleType, typename StateArray>
		AMREX_GPU_DEVICE void operator()(ParticleType &p, StateArray const &state_arr, int i, int j, int k,
						 amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx,
						 amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &plo, amrex::Long particle_offset) const
		{
			// A simple demonstration of particle creation
			if (mass_idx + 3 < ParticleType::NReal) {
				p.pos(0) = plo[0] + (i + 0.5) * dx[0];
				p.pos(1) = plo[1] + (j + 0.5) * dx[1];
				p.pos(2) = plo[2] + (k + 0.5) * dx[2];

				// Set particle ID and CPU
				p.id() = pid_start + particle_offset;
				p.cpu() = cpu_id;

				// Set particle mass and velocities
				const amrex::Real cell_volume = AMREX_D_TERM(dx[0], *dx[1], *dx[2]);
				const amrex::Real cell_density = state_arr(i, j, k, HydroSystem<problem_t>::density_index);
				const amrex::Real cell_mass = cell_density * cell_volume;

				// Initialize particle properties
				p.rdata(mass_idx) = 0.5 * cell_mass;
				p.rdata(mass_idx + 1) = state_arr(i, j, k, HydroSystem<problem_t>::x1Momentum_index) / cell_density;
				p.rdata(mass_idx + 2) = state_arr(i, j, k, HydroSystem<problem_t>::x2Momentum_index) / cell_density;
				p.rdata(mass_idx + 3) = state_arr(i, j, k, HydroSystem<problem_t>::x3Momentum_index) / cell_density;

				// Update cell density (remove mass that was given to particle)
				state_arr(i, j, k, HydroSystem<problem_t>::density_index) = 0.5 * cell_density;
			}
		}
	};

	// Main method to create particles - uses the helper implementation
	template <typename problem_t, typename ContainerType>
	static void createParticles(ContainerType *container, int mass_idx, amrex::MultiFab &state, int lev, amrex::Real current_time, amrex::Real dt)
	{
		// Use the common implementation with our checker and creator types
		ParticleCreationImpl::createParticlesImpl<problem_t, ContainerType, ParticleCreationTraits<ParticleType::CIC>::template ParticleChecker,
							  ParticleCreationTraits<ParticleType::CIC>::template ParticleCreator>(container, mass_idx, state, lev,
															       current_time, dt);
	}
};

} // namespace quokka

#endif // PARTICLE_CREATION_HPP_