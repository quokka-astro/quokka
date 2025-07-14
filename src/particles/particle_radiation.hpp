#ifndef PARTICLE_RADIATION_HPP_
#define PARTICLE_RADIATION_HPP_

#include "AMReX_Array.H"
#include "AMReX_Array4.H"
#include "AMReX_Extension.H"
#include "AMReX_ParticleInterpolators.H"
#include "hydro/hydro_system.hpp"
// Forward declaration for particle types
#include "particle_types.hpp"
#include "physics_info.hpp"

namespace quokka
{
// Traits class for specializing particle property update behavior
template <ParticleType particleType>
struct ParticlePropertyUpdateTraits {
	// Default implementation - does nothing
	template <typename problem_t, typename ParticleType>
	AMREX_GPU_DEVICE AMREX_FORCE_INLINE static void updateProperties(ParticleType &p, int mass_idx, int lum_idx, int birth_time_idx,
									 amrex::Real current_time) noexcept
	{
		// Default implementation does nothing
		amrex::ignore_unused(p, mass_idx, lum_idx, birth_time_idx, current_time);
	}
};

// Specialization for star particles with stellar evolution
template <>
struct ParticlePropertyUpdateTraits<ParticleType::StochasticStellarPop> {
	static constexpr double star_lum_per_M_solar = 4.0e33;

	template <typename problem_t, typename ParticleType>
	AMREX_GPU_DEVICE AMREX_FORCE_INLINE static void updateProperties(ParticleType &p, int mass_idx, int lum_idx, int birth_time_idx,
									 amrex::Real current_time) noexcept
	{
		if (mass_idx >= 0 && birth_time_idx >= 0 && lum_idx >= 0) {
			const amrex::Real age = current_time - p.rdata(birth_time_idx);
			const amrex::Real mass = p.rdata(mass_idx);

			// A simple luminosity function for testing purpose. Keep it linear function of mass for easy answer
			// validation. L/(M / M_sun) = L_sun = 4e33 erg/s
			const double is_on = age < 1.0e14 ? 1.0 : 0.0; // 3 Myr
			
			// Update luminosity components (they are stored consecutively starting at lum_idx)
			for (int g = 0; g < Physics_Traits<problem_t>::nGroups; ++g) {
				const amrex::Real luminosity = star_lum_per_M_solar * (mass / C::M_solar) * (g + 1) * is_on; // erg / s
				p.rdata(lum_idx + g) = luminosity;
			}
		}
	}
};

// Example specialization for Test particles - demonstrates how users can customize behavior
template <>
struct ParticlePropertyUpdateTraits<ParticleType::Test> {
	static constexpr double test_lum_scale = 1.0e32;

	template <typename problem_t, typename ParticleType>
	AMREX_GPU_DEVICE AMREX_FORCE_INLINE static void updateProperties(ParticleType &p, int mass_idx, int lum_idx, int birth_time_idx,
									 amrex::Real current_time) noexcept
	{
		amrex::ignore_unused(birth_time_idx, current_time);
		if (mass_idx >= 0 && lum_idx >= 0) {
			const amrex::Real mass = p.rdata(mass_idx);
			
			// Simple test luminosity function - can be overridden by users
			for (int g = 0; g < Physics_Traits<problem_t>::nGroups; ++g) {
				const amrex::Real test_luminosity = test_lum_scale * mass * (g + 1);
				p.rdata(lum_idx + g) = test_luminosity;
			}
		}
	}
};

} // namespace quokka

#endif