/// \file network.cpp
/// \brief Definitions and reader of the runtime parameters of a jaff-generated
/// network.

#include "network.hpp"

#include "AMReX_ParmParse.H"

// $$JAFF SUB label
namespace quokka::networks::photoionization_H_caseB
{
// $$JAFF END

using amrex::literals::operator""_rt;

AMREX_GPU_MANAGED amrex::Real small_x = 1.0e-30_rt; // NOLINT

void Network::ReadParameters()
{
	const amrex::ParmParse pp("network");
	pp.query("small_x", small_x);
}

// $$JAFF SUB label
} // namespace quokka::networks::photoionization_H_caseB
// $$JAFF END
