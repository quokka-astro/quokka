#ifndef PARTICLE_RADIATION_HPP_
#define PARTICLE_RADIATION_HPP_

#include "AMReX_Extension.H"
#include "particle_types.hpp"
#include "util/DataTable.hpp"

namespace quokka
{

// GPU-friendly const table access for luminosity tables
struct LuminosityGpuConstTables {
	quokka::DataTableGpuConst<2, 1> luminosity; // 2D table: (mass, age) -> luminosity
};

// Host-side luminosity table storage
class LuminosityTables
{
      public:
	quokka::DataTable<2, 1> luminosity; // 2D table: (mass, age) -> luminosity

	[[nodiscard]] auto const_tables() const -> LuminosityGpuConstTables
	{
		LuminosityGpuConstTables tables{luminosity.const_tables()};
		return tables;
	}

	[[nodiscard]] auto is_initialized() const -> bool { return luminosity.is_initialized(); }
};

// Global luminosity tables instance (will be initialized in problem setup)
inline LuminosityTables g_luminosity_tables; // NOLINT

// Traits class for specializing particle property update behavior
template <ParticleType particleType> struct ParticlePropertyUpdateTraits {
	// Default implementation - does nothing
	template <typename problem_t, typename ParticleType>
	AMREX_GPU_DEVICE AMREX_FORCE_INLINE static void updateProperties(ParticleType & /*p*/, amrex::Real /*current_time*/) noexcept
	{
		// Default implementation does nothing
	}
};
} // namespace quokka

#endif