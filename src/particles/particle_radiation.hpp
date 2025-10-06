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
template <int Nout = 1> struct LuminosityGpuConstTables {
	quokka::DataTableGpuConst<2, Nout> luminosity; // 2D table: (age, mass) -> luminosity per group
};

// Host-side luminosity table storage
template <int Nout = 1> class LuminosityTables
{
      public:
	quokka::DataTable<2, Nout> luminosity; // 2D table: (age, mass) -> luminosity per group

	[[nodiscard]] auto const_tables() const -> LuminosityGpuConstTables<Nout>
	{
		LuminosityGpuConstTables<Nout> tables{luminosity.const_tables()};
		return tables;
	}

	[[nodiscard]] auto is_initialized() const -> bool { return luminosity.is_initialized(); }
};

// Static pointer to the current simulation's luminosity tables (set during initialization)
// Default to single output (Nout=1) for backward compatibility
template <int Nout = 1> inline LuminosityTables<Nout> *g_luminosity_tables_ptr = nullptr; // NOLINT

// Class to handle luminosity updates for stellar particles
class LuminosityUpdate
{
      public:
	template <typename problem_t, typename ParticleType>
	AMREX_GPU_DEVICE AMREX_FORCE_INLINE static void updateLuminosity(ParticleType &p, amrex::Real current_time) noexcept
	{
		constexpr int nGroups = Physics_Traits<problem_t>::nGroups;

		// Get pointer for the appropriate number of outputs
		auto *tables_ptr = g_luminosity_tables_ptr<nGroups>;

		// Use table interpolation: (age, mass) -> luminosity per group
		// Requires that g_luminosity_tables_ptr is initialized
		if (tables_ptr != nullptr && tables_ptr->is_initialized()) {
			const int mass_idx = StochasticStellarPopParticleMassIdx;
			const int birth_time_idx = StochasticStellarPopParticleBirthTimeIdx;
			const int lum_idx = StochasticStellarPopParticleLumIdx;
			const amrex::Real age_in_seconds = current_time - p.rdata(birth_time_idx);
			const amrex::Real mass = p.rdata(mass_idx);

			auto const tables = tables_ptr->const_tables();
			const amrex::Real mass_in_solar_masses = mass / C::M_solar;
			amrex::Real age_in_years = age_in_seconds / seconds_per_year;
			age_in_years = std::max(age_in_years, 1.0e-10); // age = 0 is allowed
			// Table coordinates: (age, mass) as specified in CSV input_names
			std::array<amrex::Real, 2> const point = {age_in_years, mass_in_solar_masses};

			// Interpolate luminosity from table (returns array with nGroups elements)
			// Conversion from log space is handled automatically by DataTable::interpolate()
			auto const luminosities = tables.luminosity.interpolate(point);

			printf("age: %f, mass: %f, luminosities: %e %e\n", age_in_years, mass_in_solar_masses, luminosities[0], luminosities[1]);

			// Update luminosity components (they are stored consecutively starting at lum_idx)
			for (int g = 0; g < nGroups; ++g) {
				p.rdata(lum_idx + g) = luminosities[g];
			}
		}
		// If table is not initialized, luminosity values remain unchanged
	}
};

} // namespace quokka

#endif // PARTICLE_RADIATION_HPP_