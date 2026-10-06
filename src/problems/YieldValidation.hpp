#ifndef YIELD_VALIDATION_HPP_
#define YIELD_VALIDATION_HPP_

#include "AMReX_BLassert.H"
#include "AMReX_Print.H"
#include "AMReX_REAL.H"

#include <cmath>
#include <format>
#include <string>

namespace quokka::testing
{
inline void assertYieldClose(const std::string &label, amrex::Real simulated, amrex::Real expected, amrex::Real tolerance = 1.0e-10)
{
	const amrex::Real error = std::abs(simulated - expected);
	const amrex::Real allowed_error = tolerance * std::abs(expected);
	amrex::Print() << label << ": simulated=" << simulated << " expected=" << expected << " absolute_error=" << error << "\n";
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(error <= allowed_error, std::format("{} failed: error={} > {}", label, error, allowed_error).c_str());
}
} // namespace quokka::testing

#endif
