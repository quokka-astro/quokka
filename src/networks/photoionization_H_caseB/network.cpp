/// \file network.cpp
/// \brief Definitions and reader of the runtime parameters of a jaff-generated
/// network.

#include "network.hpp"

#include "AMReX_ParmParse.H"

namespace quokka::networks
{

namespace jaff_network
{
using amrex::literals::operator""_rt;

AMREX_GPU_MANAGED amrex::Real small_x = 1.0e-30_rt; // NOLINT
} // namespace jaff_network

void JaffNetwork::ReadParameters()
{
	namespace params = jaff_network;
	const amrex::ParmParse pp("network");
	pp.query("small_x", params::small_x);
}

} // namespace quokka::networks
