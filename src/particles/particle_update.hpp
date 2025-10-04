#ifndef PARTICLE_UPDATE_HPP_
#define PARTICLE_UPDATE_HPP_

#include "AMReX_Extension.H"
#include "particle_radiation.hpp"
#include "particle_types.hpp"

namespace quokka
{

#if AMREX_SPACEDIM == 3

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
// This can be overridden by redefining the specialization in the problem generator
template <> struct ParticlePropertyUpdateTraits<ParticleType::StochasticStellarPop> {
	static constexpr double star_lum_per_M_solar = 4.0e33; // Default luminosity per solar mass (erg/s)

	template <typename problem_t, typename ParticleType>
	AMREX_GPU_DEVICE AMREX_FORCE_INLINE static void updateProperties(ParticleType &p, amrex::Real current_time) noexcept
	{
		const int mass_idx = StochasticStellarPopParticleMassIdx;
		const int birth_time_idx = StochasticStellarPopParticleBirthTimeIdx;
		const int lum_idx = StochasticStellarPopParticleLumIdx;
		const amrex::Real age = current_time - p.rdata(birth_time_idx);
		const amrex::Real mass = p.rdata(mass_idx);

		// Check if luminosity tables are initialized
		if (g_luminosity_tables_ptr != nullptr && g_luminosity_tables_ptr->is_initialized()) {
			// Use table interpolation: (mass_in_solar_masses, age) -> luminosity
			auto const tables = g_luminosity_tables_ptr->const_tables();
			const amrex::Real mass_in_solar_masses = mass / C::M_solar;
			std::array<amrex::Real, 2> const point = {mass_in_solar_masses, age};

			// Interpolate luminosity from table (with automatic clamping to table bounds)
			const amrex::Real luminosity = tables.luminosity.interpolate_single(point);

			// Update luminosity components (they are stored consecutively starting at lum_idx)
			for (int g = 0; g < Physics_Traits<problem_t>::nGroups; ++g) {
				p.rdata(lum_idx + g) = luminosity * (g + 1); // Scale by group index for multi-group
			}
		} else {
			// Fallback to analytical formula if table is not initialized
			// This simple luminosity function is for testing purposes.
			// Keep it linear in mass for easy answer validation: L/(M / M_sun) = L_sun = 4e33 erg/s
			const double is_on = age < 1.0e14 ? 1.0 : 0.0; // Turn off after 3 Myr (~1e14 s)

			// Update luminosity components (they are stored consecutively starting at lum_idx)
			for (int g = 0; g < Physics_Traits<problem_t>::nGroups; ++g) {
				const amrex::Real luminosity = star_lum_per_M_solar * (mass / C::M_solar) * (g + 1) * is_on; // erg / s
				p.rdata(lum_idx + g) = luminosity;
			}
		}
	}
};

#endif // AMREX_SPACEDIM == 3

} // namespace quokka

#endif // PARTICLE_UPDATE_HPP_

