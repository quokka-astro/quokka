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

// Forward declaration required so ParticlePropertyUpdateBase can call the (potentially specialized)
// updateProperties without needing the full definition at the point of its own definition.
template <ParticleType particleType> struct ParticlePropertyUpdateTraits;

// Base class that holds the single shared container-level loop.
// Calls ParticlePropertyUpdateTraits<particleType>::updateProperties per particle, which resolves
// to whichever specialization (or the default no-op) is appropriate for particleType.
// Specializations that require luminosity tables should override updateParticleProperties to
// load the tables before calling applyUpdate.
template <ParticleType particleType> struct ParticlePropertyUpdateBase {
	template <typename problem_t, typename ContainerType>
	static void updateParticleProperties(ContainerType *container, amrex::Real current_time, amrex::Real dt) noexcept
	{
		const BL_PROFILE("ParticlePropertyUpdateTraits::updateParticleProperties()");
		if (container == nullptr) {
			return;
		}

		constexpr int nGroups = Physics_Traits<problem_t>::nGroups;
		// Default: pass empty tables (unused by per-particle functions that don't need them)
		LuminosityGpuConstTables<nGroups> const gpu_tables{};
		applyUpdate<problem_t, ContainerType>(container, current_time, dt, gpu_tables);
	}

      public:
	template <typename problem_t, typename ContainerType>
	static void applyUpdate(ContainerType *container, amrex::Real current_time, amrex::Real dt,
				LuminosityGpuConstTables<Physics_Traits<problem_t>::nGroups> const &gpu_tables) noexcept
	{
		constexpr int nGroups = Physics_Traits<problem_t>::nGroups;
		// Apply the updater to all particles across all levels
		for (int lev = 0; lev <= container->finestLevel(); ++lev) {
			for (typename ContainerType::ParIterType pIter(*container, lev); pIter.isValid(); ++pIter) {
				auto &particles = pIter.GetArrayOfStructs();
				auto *pData = particles().data();
				const amrex::Long np = pIter.numParticles();

				amrex::ParallelFor(np, [=] AMREX_GPU_DEVICE(int64_t idx) {
					auto &p = pData[idx]; // NOLINT
					ParticlePropertyUpdateTraits<particleType>::template updateProperties<problem_t, typename ContainerType::ParticleType,
													      nGroups>(p, current_time, dt, gpu_tables);
				});
			}
		}
	}
};

// Traits class for specializing the per-particle update. Inherits updateParticleProperties from the base.
// Specializations only need to override updateProperties.
template <ParticleType particleType> struct ParticlePropertyUpdateTraits : ParticlePropertyUpdateBase<particleType> {
	// Default per-particle implementation - does nothing
	template <typename problem_t, typename ParticleType, int Nout>
	AMREX_GPU_DEVICE AMREX_FORCE_INLINE static void updateProperties(ParticleType & /*p*/, amrex::Real /*current_time*/, amrex::Real /*dt*/,
									 LuminosityGpuConstTables<Nout> const & /*gpu_tables*/) noexcept
	{
		// Default implementation does nothing
	}

	// Default container-level update - does nothing
	template <typename problem_t, typename ContainerType>
	static void updateParticleProperties(ContainerType * /*container*/, amrex::Real /*current_time*/, amrex::Real /*dt*/) noexcept
	{
		// Default implementation does nothing
	}
};

// Specialization for StochasticStellarPop particles: updates luminosity via table interpolation.
// Overrides updateParticleProperties to gate the update on luminosity tables being initialized.
template <> struct ParticlePropertyUpdateTraits<ParticleType::StochasticStellarPop> : ParticlePropertyUpdateBase<ParticleType::StochasticStellarPop> {
	template <typename problem_t, typename ContainerType>
	static void updateParticleProperties(ContainerType *container, amrex::Real current_time, amrex::Real dt) noexcept
	{
		const BL_PROFILE("ParticlePropertyUpdateTraits::updateParticleProperties()");
		if (container == nullptr) {
			return;
		}

		constexpr int nGroups = Physics_Traits<problem_t>::nGroups;
		auto *host_tables_ptr = quokka::g_luminosity_tables_ptr<nGroups>;

		// Only proceed if tables are initialized
		if (host_tables_ptr != nullptr && host_tables_ptr->is_initialized()) {
			auto const gpu_tables = host_tables_ptr->const_tables();
			applyUpdate<problem_t, ContainerType>(container, current_time, dt, gpu_tables);
		}
	}

	template <typename problem_t, typename ParticleType, int Nout>
	AMREX_GPU_DEVICE AMREX_FORCE_INLINE static void updateProperties(ParticleType &p, amrex::Real current_time, amrex::Real /*dt*/,
									 LuminosityGpuConstTables<Nout> const &gpu_tables) noexcept
	{
		LuminosityUpdate::updateLuminosity<problem_t>(p, current_time, gpu_tables);
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
