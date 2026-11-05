#include "QuokkaSimulation.hpp"
#include <cfenv>

struct MetadataProblem {};
template <> struct Physics_Traits<MetadataProblem> : DefaultPhysicsTraits {
	static constexpr bool is_hydro_enabled = true;
};

auto problem_main() -> int
{
	QuokkaSimulation<MetadataProblem> sim;
	amrex::ParmParse pp("metadata_test");
	std::string directory;
	bool expect_exception = false;
	pp.get("directory", directory);
	pp.get("expect_exception", expect_exception);

	// Hold the caller's traps before raising a sticky exception for the restoration check.
	std::fenv_t original{};
	AMREX_ALWAYS_ASSERT(std::feholdexcept(&original) == 0);
	AMREX_ALWAYS_ASSERT(std::fesetround(FE_DOWNWARD) == 0);
	AMREX_ALWAYS_ASSERT(std::feraiseexcept(FE_DIVBYZERO) == 0);
	amrex::setFPExcept(amrex::FPExcept::invalid);
	const auto traps_before = amrex::getFPExcept();
	const int flags_before = std::fetestexcept(FE_ALL_EXCEPT);
	const int rounding_before = std::fegetround();
	bool caught = false;
	try {
		sim.ReadMetadataFile(directory);
	} catch (const YAML::Exception &) {
		caught = true;
	}
	const auto traps_after = amrex::getFPExcept();
	const int flags_after = std::fetestexcept(FE_ALL_EXCEPT);
	const int rounding_after = std::fegetround();
	AMREX_ALWAYS_ASSERT(std::fesetenv(&original) == 0);

	const bool restored = traps_before == traps_after && flags_before == flags_after && rounding_before == rounding_after;
	amrex::Print() << "Metadata exception expected/caught: " << expect_exception << '/' << caught << "; floating-point environment restored: " << restored
		       << '\n';
#if !(defined(__linux__) && defined(__GLIBC__))
	amrex::Print() << "Trap-mask controls unavailable on this platform; checked exception flags and rounding mode.\n";
#endif
	return caught == expect_exception && restored ? 0 : 1;
}
