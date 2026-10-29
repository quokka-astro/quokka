#ifndef EOS_SHIM_THERMOPHOTOCHEMISTRY_HANDWRITTEN_NETWORK_PROPERTIES_H_
#define EOS_SHIM_THERMOPHOTOCHEMISTRY_HANDWRITTEN_NETWORK_PROPERTIES_H_
/// \file network_properties.H
/// \brief The species table of the thermophotochemistry_handwritten network, with the values that Microphysics'
/// generator writes for it. The vendored EOS includes this file by this name.
/// Transitional: part of the EOS shim, which goes when the EOS is ported to Quokka.

#include "../species_table.H"
#include "networks/thermophotochemistry_handwritten/network.hpp"

constexpr int NumSpec = quokka::networks::Thermophotochemistry::NumSpec;

namespace
{
[[maybe_unused]] EOS_SHIM_HIP_CONSTEXPR AMREX_GPU_MANAGED amrex::Real aion[NumSpec] = {1.0, 1.0, 5.4858e-4};
[[maybe_unused]] EOS_SHIM_HIP_CONSTEXPR AMREX_GPU_MANAGED amrex::Real aion_inv[NumSpec] = {1.0, 1.0, 1822.89};
[[maybe_unused]] EOS_SHIM_HIP_CONSTEXPR AMREX_GPU_MANAGED amrex::Real zion[NumSpec] = {1.0, 1.0, 1.0};
[[maybe_unused]] const std::vector<std::string> short_spec_names_cxx = {"H", "H+", "e-"};
[[maybe_unused]] const std::vector<std::string> spec_names_cxx = {"H", "H_p", "e"};
} // namespace

// the table lists the species in the network's order
static_assert(quokka::networks::Thermophotochemistry::H == 0 && quokka::networks::Thermophotochemistry::H_p == 1 &&
	      quokka::networks::Thermophotochemistry::e == 2);

#endif // EOS_SHIM_THERMOPHOTOCHEMISTRY_HANDWRITTEN_NETWORK_PROPERTIES_H_
