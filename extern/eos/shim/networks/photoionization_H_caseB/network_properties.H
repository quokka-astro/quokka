#ifndef EOS_SHIM_PHOTOIONIZATION_H_CASEB_NETWORK_PROPERTIES_H_
#define EOS_SHIM_PHOTOIONIZATION_H_CASEB_NETWORK_PROPERTIES_H_
/// \file network_properties.H
/// \brief The species table of the jaff network photoionization_H_caseB, with the values that Microphysics' generator
/// writes for it. The vendored EOS includes this file by this name. A frozen copy: regenerating the network does not
/// update it.
/// Transitional: part of the EOS shim, which goes when the EOS is ported to Quokka.

#include "../species_table.H"
#include "networks/photoionization_H_caseB/network.hpp"

constexpr int NumSpec = quokka::networks::JaffNetwork::NumSpec;

namespace
{
[[maybe_unused]] EOS_SHIM_HIP_CONSTEXPR AMREX_GPU_MANAGED amrex::Real aion[NumSpec] = {1.007970e+00, 1.007421e+00, 5.485799e-04};
[[maybe_unused]] EOS_SHIM_HIP_CONSTEXPR AMREX_GPU_MANAGED amrex::Real aion_inv[NumSpec] = {9.920934e-01, 9.926336e-01, 1.822888e+03};
[[maybe_unused]] EOS_SHIM_HIP_CONSTEXPR AMREX_GPU_MANAGED amrex::Real zion[NumSpec] = {1.000000e+00, 1.000000e+00, 0.000000e+00};
[[maybe_unused]] const std::vector<std::string> short_spec_names_cxx = {"H", "H+", "e-"};
[[maybe_unused]] const std::vector<std::string> spec_names_cxx = {"H", "H_p", "e"};
} // namespace

// the table lists the species in the network's order
static_assert(quokka::networks::JaffNetwork::h == 0 && quokka::networks::JaffNetwork::hj == 1 && quokka::networks::JaffNetwork::ek == 2);

#endif // EOS_SHIM_PHOTOIONIZATION_H_CASEB_NETWORK_PROPERTIES_H_
