#ifndef TIME_UNITS_HPP_
#define TIME_UNITS_HPP_
/// \file time_units.hpp
/// \brief Registers physical time unit constants (yr, kyr, Myr, Gyr) in AMReX ParmParse
/// so that time-valued parameters can use math expressions in input files, e.g.:
///   quokka.plt.time_int = "1.0*Myr"
///   stop_time = "2.5*Myr + 500*kyr"
/// Call registerTimeUnitConstants() once before any queryWithParser call for time params.

#include "AMReX_ParmParse.H"

namespace quokka
{

/// Time unit conversion factors to CGS seconds (Julian year = 365.25 days).
inline constexpr double yr_in_s = 3.15576e7;
inline constexpr double kyr_in_s = 3.15576e10;
inline constexpr double Myr_in_s = 3.15576e13;
inline constexpr double Gyr_in_s = 3.15576e16;

/// \brief Register yr/kyr/Myr/Gyr as named constants in ParmParse and set the global
/// parser prefix so that queryWithParser can resolve them in time expressions.
///
/// Must be called once before any pp.queryWithParser() call for time-valued parameters.
/// Safe to call multiple times (ParmParse::add silently overwrites existing entries).
inline void registerTimeUnitConstants()
{
	amrex::ParmParse pp_tc("quokka_time_units");
	pp_tc.add("yr", yr_in_s);
	pp_tc.add("kyr", kyr_in_s);
	pp_tc.add("Myr", Myr_in_s);
	pp_tc.add("Gyr", Gyr_in_s);
	amrex::ParmParse::SetParserPrefix("quokka_time_units");
}

} // namespace quokka

#endif // TIME_UNITS_HPP_
