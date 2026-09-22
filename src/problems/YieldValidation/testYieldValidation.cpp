#include "problems/YieldValidation.hpp"

auto problem_main() -> int
{
	const volatile amrex::Real zero_yield = 0.0;
	quokka::testing::assertYieldClose("zero yield", zero_yield, zero_yield);
	quokka::testing::assertYieldClose("nonzero yield", 1.0, 1.0);
	quokka::testing::assertYieldClose("within tolerance", 1.0 + 1.0e-12, 1.0);
	return 0;
}
