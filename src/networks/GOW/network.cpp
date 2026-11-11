/// \file network.cpp
/// \brief Definitions and reader of the runtime parameters of a jaff-generated
/// network.

#include "network.hpp"

#include "AMReX_ParmParse.H"

// $$JAFF SUB label
namespace quokka::networks::GOW
{
// $$JAFF END

using amrex::literals::operator""_rt;

AMREX_GPU_MANAGED amrex::Real small_x = 1.0e-30_rt; // NOLINT
// $$JAFF REPEAT idx, parameter IN parameters
AMREX_GPU_MANAGED amrex::Real Zd = 0.0_rt;    // NOLINT
AMREX_GPU_MANAGED amrex::Real crate = 0.0_rt; // NOLINT
AMREX_GPU_MANAGED amrex::Real d2g = 0.0_rt;   // NOLINT
AMREX_GPU_MANAGED amrex::Real gradv = 0.0_rt; // NOLINT
AMREX_GPU_MANAGED amrex::Real vdisp = 0.0_rt; // NOLINT
// $$JAFF END

void Network::ReadParameters()
{
	const amrex::ParmParse pp("network");
	pp.query("small_x", small_x);
	// $$JAFF REPEAT idx, parameter IN parameters
	pp.get("Zd", Zd);
	pp.get("crate", crate);
	pp.get("d2g", d2g);
	pp.get("gradv", gradv);
	pp.get("vdisp", vdisp);
	// $$JAFF END
	// $$JAFF SUB ninterp
#if 9 > 0
	LoadTables();
#endif
	// $$JAFF END
}

// $$JAFF SUB label
} // namespace quokka::networks::GOW
// $$JAFF END
