#ifndef PARTICLE_UPDATE_HPP_
#define PARTICLE_UPDATE_HPP_

#include "AMReX_Extension.H"
#include "particle_radiation.hpp"
#include "particle_types.hpp"

#if AMREX_SPACEDIM == 3

namespace quokka
{

// Traits class for specializing particle property update behavior
template <ParticleType particleType> struct ParticlePropertyUpdateTraits {
	// Default implementation - does nothing
	template <typename problem_t, typename PTDType, int Nout>
	AMREX_GPU_DEVICE AMREX_FORCE_INLINE static void updateProperties(PTDType & /*ptd*/, int /*i*/, amrex::Real /*current_time*/,
									 LuminosityGpuConstTables<Nout> const & /*gpu_tables*/) noexcept
	{
		// Default implementation does nothing
	}
};

// Specialization for StochasticStellarPop particles with stellar evolution
// This uses table interpolation and can be overridden in the problem generator
// Currently, the default is updateLuminosity. In the future, we can add more properties to update
template <> struct ParticlePropertyUpdateTraits<ParticleType::StochasticStellarPop> {
	template <typename problem_t, typename PTDType, int Nout>
	AMREX_GPU_DEVICE AMREX_FORCE_INLINE static void updateProperties(PTDType &ptd, int i, amrex::Real current_time,
									 LuminosityGpuConstTables<Nout> const &gpu_tables) noexcept
	{
		// Update luminosity using the LuminosityUpdate class
		LuminosityUpdate::updateLuminosity<problem_t>(ptd, i, current_time, gpu_tables);
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
