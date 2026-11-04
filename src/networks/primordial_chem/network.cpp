/// \file network.cpp
/// \brief Definitions and reader of the runtime parameters of the primordial_chem network.

#include "networks/primordial_chem/network.hpp"

#include "AMReX_ParmParse.H"

namespace quokka::networks
{

namespace primordial_chem
{
using amrex::literals::operator""_rt;

AMREX_GPU_MANAGED amrex::Real redshift = 30e0_rt;    // NOLINT
AMREX_GPU_MANAGED amrex::Real small_x = 1.0e-100_rt; // NOLINT
} // namespace primordial_chem

void PrimordialChem::ReadParameters()
{
	namespace params = primordial_chem;
	const amrex::ParmParse pp("network");
	pp.query("redshift", params::redshift);
	pp.query("small_x", params::small_x);
}

} // namespace quokka::networks
