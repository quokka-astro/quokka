#ifndef TIME_UNITS_HPP_
#define TIME_UNITS_HPP_
/// \file time_units.hpp
/// \brief Utility for parsing time-valued ParmParse entries with optional physical unit suffixes.
/// Supported suffixes: _yr, _kyr, _Myr, _Gyr (case-sensitive).
/// Examples: "1.0_Myr", "500_kyr", "1.3_Gyr", or plain "3.15576e13" (CGS seconds).

#include <string>
#include <unordered_map>

#include "AMReX_ParmParse.H"
#include "AMReX_Print.H"
#include "AMReX_REAL.H"

namespace quokka
{

/// Time unit conversion factors to CGS seconds (Julian year = 365.25 days).
inline constexpr double yr_in_s = 3.15576e7;
inline constexpr double kyr_in_s = 3.15576e10;
inline constexpr double Myr_in_s = 3.15576e13;
inline constexpr double Gyr_in_s = 3.15576e16;

/// \brief Parse a time string that may carry a physical unit suffix.
///
/// Accepted formats:
///   "3.15576e13"   — plain CGS seconds, returned as-is
///   "1.0_Myr"      — converted: 1.0 * 3.15576e13
///   "500_kyr"      — converted: 500 * 3.15576e10
///
/// Supported units: yr, kyr, Myr, Gyr (case-sensitive).
/// Aborts with a descriptive message on unrecognised unit.
///
/// \param s    the string to parse
/// \param name parameter name used in error messages
/// \return value in CGS seconds
inline auto parseTimeString(const std::string &s, const std::string &name) -> amrex::Real
{
	static const std::unordered_map<std::string, double> unitMap = {
	    {"yr", yr_in_s}, {"kyr", kyr_in_s}, {"Myr", Myr_in_s}, {"Gyr", Gyr_in_s}};

	const auto pos = s.rfind('_');
	if (pos != std::string::npos) {
		const std::string numStr = s.substr(0, pos);
		const std::string unit = s.substr(pos + 1);
		const auto it = unitMap.find(unit);
		if (it == unitMap.end()) {
			amrex::Abort("queryTime: unrecognised time unit '" + unit + "' for parameter '" + name +
				     "'. Supported units: yr, kyr, Myr, Gyr.");
		}
		return static_cast<amrex::Real>(std::stod(numStr) * it->second);
	}
	return static_cast<amrex::Real>(std::stod(s));
}

/// \brief Drop-in replacement for pp.query() for time-valued parameters.
///
/// Reads the parameter as a string (works for both .in and .toml formats),
/// then calls parseTimeString. If the parameter is absent, \p val is unchanged.
///
/// \param pp   ParmParse instance (any prefix)
/// \param name parameter name
/// \param val  output value in CGS seconds; unchanged if parameter not found
/// \return true if the parameter was found, false otherwise
inline auto queryTime(const amrex::ParmParse &pp, const std::string &name, amrex::Real &val) -> bool
{
	std::string str;
	if (pp.query(name.c_str(), str) == 0) {
		return false;
	}
	val = parseTimeString(str, name);
	return true;
}

} // namespace quokka

#endif // TIME_UNITS_HPP_
