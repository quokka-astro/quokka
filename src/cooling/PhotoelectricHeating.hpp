#ifndef PHOTOELECTRIC_HEATING_HPP_
#define PHOTOELECTRIC_HEATING_HPP_

#include "AMReX_REAL.H"
#include "fundamental_constants.H"
#include "util/DataTable.hpp"
#include <array>
#include <tuple>
#include <vector>

namespace quokka
{

// Forward declare seconds_per_year (defined in particle_radiation.hpp)
// We can't include particle_radiation.hpp here to avoid circular dependencies
namespace detail
{
constexpr amrex::Real seconds_per_year_local = 3.15576e+07;
}

// GPU-friendly const table access for PE heating table
template <quokka::OutOfBounds oob_policy = quokka::OutOfBounds::clamp> struct PeHeatingGpuConstTables {
	quokka::DataTableGpuConst<1, 1, oob_policy> pe_heating; // 1D table: (age) -> PE heating rate per unit mass
};

// Host-side PE heating table storage
template <quokka::OutOfBounds oob_policy = quokka::OutOfBounds::clamp> class PeHeatingTables
{
      public:
	quokka::DataTable<1, 1, oob_policy> pe_heating; // 1D table: (age) -> PE heating rate per unit mass

	[[nodiscard]] auto const_tables() const -> PeHeatingGpuConstTables<oob_policy>
	{
		PeHeatingGpuConstTables<oob_policy> tables{pe_heating.const_tables()};
		return tables;
	}

	[[nodiscard]] auto is_initialized() const -> bool { return pe_heating.is_initialized(); }
};

// Static pointer to the current simulation's PE heating tables (set during initialization)
template <quokka::OutOfBounds oob_policy = quokka::OutOfBounds::clamp> inline PeHeatingTables<oob_policy> *g_pe_heating_tables_ptr = nullptr; // NOLINT

// Function to compute PE heating rate from star formation history
// sfh_data: vector of tuples (nstep, time, mass)
// sf_area_kpc2: area of the star formation history region in kpc^2
// Returns: PE heating rate, unit: erg/s/H
template <quokka::OutOfBounds oob_policy = quokka::OutOfBounds::clamp>
auto PeHeatingFromSfh(const std::vector<std::tuple<int, amrex::Real, amrex::Real>> &sfh_data, amrex::Real current_time,
		      PeHeatingGpuConstTables<oob_policy> const &gpu_tables, amrex::Real sf_area_kpc2) -> amrex::Real
{
	amrex::Real heating_rate = 0.0;

	// Use real star formation history
	amrex::Real mass_old = 0.0;
	for (const auto &entry : sfh_data) {
		const auto time = std::get<1>(entry);
		const auto mass = std::get<2>(entry);

		const auto mass_in_Msun = mass / C::M_solar;

		// Compute age of this stellar population (in years)
		const amrex::Real age_in_seconds = current_time - time;
		amrex::Real age_in_years = age_in_seconds / detail::seconds_per_year_local;
		age_in_years = std::max(age_in_years, 1.0e-30); // age = 0 is allowed, but avoid exact zero for table lookup

		// Table coordinate: (age) as specified in CSV input_names
		std::array<amrex::Real, 1> const point = {age_in_years};

		// Interpolate PE heating rate per unit mass from table
		// Conversion from log space is handled automatically by DataTable::interpolate()
		auto const pe_heating_per_mass = gpu_tables.pe_heating.interpolate(point);

		// Accumulate contribution: (mass increment) * (PE heating rate per unit mass)
		// The table is given in units of erg/s/H/Msun, and it expects the input to be SFR per kpc^2, so we need to divide the mass increment by the
		// area.
		const amrex::Real delta_mass_per_area = (mass_in_Msun - mass_old) / sf_area_kpc2;
		heating_rate += delta_mass_per_area * pe_heating_per_mass[0];
		mass_old = mass_in_Msun;
	}

	return heating_rate;
}

// Function to compute PE heating rate from constant star formation rate
template <quokka::OutOfBounds oob_policy = quokka::OutOfBounds::clamp>
auto PeHeatingFromConstSfr(amrex::Real const_sfr_Msun_per_year_per_kpc2, PeHeatingGpuConstTables<oob_policy> const &gpu_tables) -> amrex::Real
{
	// compute PE heating rate from constant star formation rate. Use intervals of 1 Myr from 0 to 100 Myr
	const amrex::Real age_interval_in_years = 1.0e6;			   // 1 Myr
	const int num_intervals = static_cast<int>(1.0e9 / age_interval_in_years); // 1 Gyr
	amrex::Real heating_rate = 0.0;
	for (int i = 0; i < num_intervals; ++i) {
		const amrex::Real age_in_years = i * age_interval_in_years;
		const amrex::Real delta_mass = const_sfr_Msun_per_year_per_kpc2 * age_interval_in_years;

		std::array<amrex::Real, 1> const point = {age_in_years};
		auto const pe_heating_per_mass = gpu_tables.pe_heating.interpolate(point);
		heating_rate += delta_mass * pe_heating_per_mass[0];
	}
	return heating_rate;
}

} // namespace quokka

#endif // PHOTOELECTRIC_HEATING_HPP_