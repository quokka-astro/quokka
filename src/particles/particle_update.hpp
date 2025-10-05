#ifndef PARTICLE_UPDATE_HPP_
#define PARTICLE_UPDATE_HPP_

#include "AMReX_Extension.H"
#include "particle_radiation.hpp"
#include "particle_types.hpp"

#if AMREX_SPACEDIM == 3

namespace quokka
{

constexpr amrex::Real seconds_per_year = 3.15576e+07;

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
		// Use table interpolation: (age_in_seconds, mass_in_solar_masses) -> luminosity
		// Table format: rows = ages, columns = masses
		// Requires that g_luminosity_tables_ptr is initialized
		if (g_luminosity_tables_ptr != nullptr && g_luminosity_tables_ptr->is_initialized()) {
			const int mass_idx = StochasticStellarPopParticleMassIdx;
			const int birth_time_idx = StochasticStellarPopParticleBirthTimeIdx;
			const int lum_idx = StochasticStellarPopParticleLumIdx;
			const amrex::Real age_in_seconds = current_time - p.rdata(birth_time_idx);
			const amrex::Real mass = p.rdata(mass_idx);

			auto const tables = g_luminosity_tables_ptr->const_tables();
			const amrex::Real mass_in_solar_masses = mass / C::M_solar;
			const amrex::Real age_in_years = age_in_seconds / seconds_per_year;
			// Table coordinates: (age, mass) since rows=ages, columns=masses
			std::array<amrex::Real, 2> const point = {age_in_years, mass_in_solar_masses};

			// Interpolate luminosity from table (with automatic clamping to table bounds)
			const amrex::Real luminosity = tables.luminosity.interpolate_single(point);

			printf("mass: %e, age: %e, luminosity: %e\n", mass_in_solar_masses, age_in_years, luminosity);

			// Update luminosity components (they are stored consecutively starting at lum_idx)
			for (int g = 0; g < Physics_Traits<problem_t>::nGroups; ++g) {
				p.rdata(lum_idx + g) = luminosity * (g + 1); // Scale by group index for multi-group
			}
		}
		// If table is not initialized, luminosity values remain unchanged
	}
};

} // namespace quokka

#endif // AMREX_SPACEDIM == 3

#endif // PARTICLE_UPDATE_HPP_

