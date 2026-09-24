#include "util/ParmParseOptionRegistry.hpp"
#include <sstream>
#include <string>

static_assert(quokka::isRegisteredOption<"hydro", "rk_integrator_order">());
static_assert(!quokka::isRegisteredOption<"hydro", "rk_integrator_orderr">());
static_assert(quokka::isRegisteredOption<"quokka_time_units", "Myr">());

auto main() -> int
{
	for (auto const &option : quokka::parmParseOptions) {
		if (option.description.empty() || option.description.starts_with("Option read in ")) {
			return 1;
		}
	}
	std::ostringstream out;
	quokka::printParmParseOptions(out, "HydroContact");
	std::string const help = out.str();
	return (help.find("hydro.rk_integrator_order") != std::string::npos && help.find("quokka_time_units.Myr") != std::string::npos &&
		help.find("Mach_shock") == std::string::npos)
		   ? 0
		   : 1;
}
