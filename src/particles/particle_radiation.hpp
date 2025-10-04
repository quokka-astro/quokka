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

// Static pointer to the current simulation's luminosity tables (set during initialization)
inline LuminosityTables *g_luminosity_tables_ptr = nullptr; // NOLINT

} // namespace quokka

#endif // PARTICLE_RADIATION_HPP_