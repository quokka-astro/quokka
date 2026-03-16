#ifndef PARTICLE_RADIATION_HPP_
#define PARTICLE_RADIATION_HPP_

#include "AMReX_Extension.H"
#include "fundamental_constants.H"
#include "particle_types.hpp"
#include "util/DataTable.hpp"

namespace quokka
{

constexpr amrex::Real seconds_per_year = 3.15576e+07;

// GPU-friendly const table access for luminosity tables
// Nout should match nGroups in the problem
template <int Nout = 1, quokka::OutOfBounds oob_policy = quokka::OutOfBounds::clamp> struct LuminosityGpuConstTables {
	quokka::DataTableGpuConst<2, Nout, oob_policy> luminosity; // 2D table: (age, mass) -> luminosity per group
};

// Host-side luminosity table storage
template <int Nout = 1, quokka::OutOfBounds oob_policy = quokka::OutOfBounds::clamp> class LuminosityTables
{
      public:
	quokka::DataTable<2, Nout, oob_policy> luminosity; // 2D table: (age, mass) -> luminosity per group

	[[nodiscard]] auto const_tables() const -> LuminosityGpuConstTables<Nout, oob_policy>
	{
		LuminosityGpuConstTables<Nout, oob_policy> tables{luminosity.const_tables()};
		return tables;
	}

	[[nodiscard]] auto is_initialized() const -> bool { return luminosity.is_initialized(); }
};

// Static pointer to the current simulation's luminosity tables (set during initialization)
// Default to single output (Nout=1) and clamp out-of-bounds for backward compatibility
template <int Nout = 1, quokka::OutOfBounds oob_policy = quokka::OutOfBounds::clamp>
inline LuminosityTables<Nout, oob_policy> *g_luminosity_tables_ptr = nullptr; // NOLINT

// Class to handle luminosity updates for stellar particles
class LuminosityUpdate
{
      public:
	template <typename problem_t, typename ParticleType, int Nout, quokka::OutOfBounds oob_policy = quokka::OutOfBounds::clamp>
	AMREX_GPU_DEVICE AMREX_FORCE_INLINE static void updateLuminosity(ParticleType &p, amrex::Real current_time,
									 LuminosityGpuConstTables<Nout, oob_policy> const &gpu_tables) noexcept
	{
		constexpr int nGroups = Physics_Traits<problem_t>::nGroups;
		static_assert(nGroups == Nout, "Number of groups must match table outputs");

		// Use table interpolation: (age, mass) -> luminosity per group
		const int mass_idx = StochasticStellarPopParticleMassIdx;
		const int birth_time_idx = StochasticStellarPopParticleBirthTimeIdx;
		const int lum_idx = StochasticStellarPopParticleLumIdx;
		const amrex::Real age_in_seconds = current_time - p.rdata(birth_time_idx);
		const amrex::Real mass = p.rdata(mass_idx);

		const amrex::Real mass_in_solar_masses = mass / C::M_solar;
		amrex::Real age_in_years = age_in_seconds / seconds_per_year;
		age_in_years = std::max(age_in_years, 1.0e-30); // age = 0 is allowed
		// Table coordinates: (age, mass) as specified in CSV input_names
		std::array<amrex::Real, 2> const point = {age_in_years, mass_in_solar_masses};

		// Interpolate luminosity from table (returns array with nGroups elements)
		// Conversion from log space is handled automatically by DataTable::interpolate()
		auto const luminosities = gpu_tables.luminosity.interpolate(point);

		// Update luminosity components (they are stored consecutively starting at lum_idx)
		if (lum_idx + nGroups <= ParticleType::NReal) {
			for (int g = 0; g < nGroups; ++g) {
				p.rdata(lum_idx + g) = luminosities[g];
			}
		}
	}
};

} // namespace quokka

#endif // PARTICLE_RADIATION_HPP_