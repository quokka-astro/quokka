#ifndef PARTICLE_UPDATE_HPP_
#define PARTICLE_UPDATE_HPP_

#include "AMReX_BLProfiler.H"
#include "AMReX_Extension.H"
#include "particle_radiation.hpp"
#include "particle_types.hpp"
#include "physics_info.hpp"

#if AMREX_SPACEDIM == 3

namespace quokka
{

// Traits class for specializing particle property update behavior
template <ParticleType particleType> struct ParticlePropertyUpdateTraits {
	// Default per-particle implementation - does nothing
	template <typename problem_t, typename ParticleType, int Nout>
	AMREX_GPU_DEVICE AMREX_FORCE_INLINE static void updateProperties(ParticleType & /*p*/, amrex::Real /*current_time*/,
									 LuminosityGpuConstTables<Nout> const & /*gpu_tables*/) noexcept
	{
		// Default implementation does nothing
	}

	// Default container-level update - does nothing
	template <typename problem_t, typename ContainerType>
	static void updateParticleProperties(ContainerType * /*container*/, amrex::Real /*current_time*/) noexcept
	{
		// Default implementation does nothing
	}
};

// Specialization for StochasticStellarPop particles with stellar evolution
// This uses table interpolation and can be overridden in the problem generator
// Currently, the default is updateLuminosity. In the future, we can add more properties to update
template <> struct ParticlePropertyUpdateTraits<ParticleType::StochasticStellarPop> {
	template <typename problem_t, typename ParticleType, int Nout>
	AMREX_GPU_DEVICE AMREX_FORCE_INLINE static void updateProperties(ParticleType &p, amrex::Real current_time,
									 LuminosityGpuConstTables<Nout> const &gpu_tables) noexcept
	{
		// Update luminosity using the LuminosityUpdate class
		LuminosityUpdate::updateLuminosity<problem_t>(p, current_time, gpu_tables);
	}

	// Container-level update for StochasticStellarPop particles
	template <typename problem_t, typename ContainerType> static void updateParticleProperties(ContainerType *container, amrex::Real current_time)
	{
		const BL_PROFILE("ParticlePropertyUpdateTraits<StochasticStellarPop>::updateParticleProperties()");
		if (container != nullptr) {
			constexpr int nGroups = Physics_Traits<problem_t>::nGroups;
			auto *host_tables_ptr = quokka::g_luminosity_tables_ptr<nGroups>;

			// Only proceed if tables are initialized
			if (host_tables_ptr != nullptr && host_tables_ptr->is_initialized()) {
				// Create GPU const tables by value to pass to device
				auto const gpu_tables = host_tables_ptr->const_tables();

				// Apply the updater to all particles across all levels
				for (int lev = 0; lev <= container->finestLevel(); ++lev) {
					for (typename ContainerType::ParIterType pIter(*container, lev); pIter.isValid(); ++pIter) {
						auto &particles = pIter.GetArrayOfStructs();
						auto *pData = particles().data();
						const amrex::Long np = pIter.numParticles();

						amrex::ParallelFor(np, [=] AMREX_GPU_DEVICE(int64_t idx) {
							auto &p = pData[idx]; // NOLINT
							ParticlePropertyUpdateTraits<ParticleType::StochasticStellarPop>::template updateProperties<
							    problem_t, typename ContainerType::ParticleType, nGroups>(p, current_time, gpu_tables);
						});
					}
				}
			}
		}
	}
};

// // Specialization for StochasticStellarPop particles from a simple analytical formula
// // This is kept for debugging purpose.
// template <> struct ParticlePropertyUpdateTraits<ParticleType::StochasticStellarPop> {
// 	static constexpr double star_lum_per_M_solar = 4.0e33;

// 	template <typename problem_t, typename ParticleType>
// 	AMREX_GPU_DEVICE AMREX_FORCE_INLINE static void updateProperties(ParticleType &p, amrex::Real current_time) noexcept
// 	{
// 		const int mass_idx = StochasticStellarPopParticleMassIdx;
// 		const int birth_time_idx = StochasticStellarPopParticleBirthTimeIdx;
// 		const int lum_idx = StochasticStellarPopParticleLumIdx;
// 		const amrex::Real age = current_time - p.rdata(birth_time_idx);
// 		const amrex::Real mass = p.rdata(mass_idx);

// 		// A simple luminosity function for testing purpose. Keep it linear function of mass for easy answer
// 		// validation. L/(M / M_sun) = L_sun = 4e33 erg/s
// 		const double is_on = age < 1.0e14 ? 1.0 : 0.0; // 3 Myr

// 		// Update luminosity components (they are stored consecutively starting at lum_idx)
// 		for (int g = 0; g < Physics_Traits<problem_t>::nGroups; ++g) {
// 			const amrex::Real luminosity = star_lum_per_M_solar * (mass / C::M_solar) * (g + 1) * is_on; // erg / s
// 			p.rdata(lum_idx + g) = luminosity;
// 		}
// 	}
// };

} // namespace quokka

#endif // AMREX_SPACEDIM == 3

#endif // PARTICLE_UPDATE_HPP_
