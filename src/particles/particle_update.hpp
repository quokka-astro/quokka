#ifndef PARTICLE_UPDATE_HPP_
#define PARTICLE_UPDATE_HPP_

#include "AMReX_BLProfiler.H"
#include "AMReX_Extension.H"
#include "particle_radiation.hpp"
#include "particle_types.hpp"
#include "physics_info.hpp"
#include "starparticle_radiation.hpp"

namespace quokka
{

// Forward declaration required so ParticlePropertyUpdateBase can call the (potentially specialized)
// updateProperties without needing the full definition at the point of its own definition.
template <ParticleType particleType> struct ParticlePropertyUpdateTraits;

// Base class that holds the single shared container-level loop.
// Calls ParticlePropertyUpdateTraits<particleType>::updateProperties per particle, which resolves
// to whichever specialization (or the default no-op) is appropriate for particleType.
// Specializations that need GPU tables set them in a global before calling applyUpdate.
template <ParticleType particleType> struct ParticlePropertyUpdateBase {
	template <typename problem_t, typename ContainerType>
	static void updateParticleProperties(ContainerType *container, amrex::Real current_time, amrex::Real dt) noexcept
	{
		const BL_PROFILE("ParticlePropertyUpdateTraits::updateParticleProperties()");
		if (container == nullptr) {
			return;
		}
		applyUpdate<problem_t, ContainerType>(container, current_time, dt);
	}

      public:
	template <typename problem_t, typename ContainerType>
	static void applyUpdate(ContainerType *container, amrex::Real current_time, amrex::Real dt) noexcept
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
													      nGroups>(p, current_time, dt);
				});
			}
		}
	}
};

// Traits class for specializing the per-particle update. Inherits updateParticleProperties from the base.
// Specializations only need to override updateProperties (and, for particle types that use luminosity
// tables, override updateParticleProperties to load the tables into g_device_luminosity_tables first).
template <ParticleType particleType> struct ParticlePropertyUpdateTraits : ParticlePropertyUpdateBase<particleType> {
	// Default per-particle implementation - does nothing
	template <typename problem_t, typename ParticleType, int Nout>
	AMREX_GPU_DEVICE AMREX_FORCE_INLINE static void updateProperties(ParticleType & /*p*/, amrex::Real /*current_time*/, amrex::Real /*dt*/) noexcept
	{
		// Default implementation does nothing
	}

	// Default container-level update - does nothing (overrides base class active default)
	template <typename problem_t, typename ContainerType>
	static void updateParticleProperties(ContainerType * /*container*/, amrex::Real /*current_time*/, amrex::Real /*dt*/) noexcept
	{
		// Default implementation does nothing
	}
};

// Specialization for StochasticStellarPop particles: updates luminosity via table interpolation.
// Overrides updateParticleProperties to gate the update on luminosity tables being initialized,
// and to load the tables into g_device_luminosity_tables before launching the GPU kernel.
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
			g_device_luminosity_tables<nGroups> = host_tables_ptr->const_tables();
			applyUpdate<problem_t, ContainerType>(container, current_time, dt);
		}
	}

	template <typename problem_t, typename ParticleType, int Nout>
	AMREX_GPU_DEVICE AMREX_FORCE_INLINE static void updateProperties(ParticleType &p, amrex::Real current_time, amrex::Real /*dt*/) noexcept
	{
		LuminosityUpdate::updateLuminosity<problem_t, ParticleType, Nout>(p, current_time);
	}
};

#if AMREX_SPACEDIM == 3
// Specialization for Star particles: dispatches to the modular stellar-evolution framework.
// Stellar models own their internal tables, so no gpu_tables plumbing is needed.
// Inherits updateParticleProperties from the base class (which calls applyUpdate).
template <> struct ParticlePropertyUpdateTraits<ParticleType::Star> : ParticlePropertyUpdateBase<ParticleType::Star> {
	template <typename problem_t, typename ParticleType, int Nout>
	AMREX_GPU_DEVICE AMREX_FORCE_INLINE static void updateProperties(ParticleType &p, amrex::Real current_time, amrex::Real dt) noexcept
	{
		StellarUpdate::updateStellarProperties<problem_t, ParticleType, Nout>(p, current_time, dt);
	}
};
#endif // AMREX_SPACEDIM == 3

} // namespace quokka

#endif // PARTICLE_UPDATE_HPP_
