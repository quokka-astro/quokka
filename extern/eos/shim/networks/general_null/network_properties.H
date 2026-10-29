#ifndef EOS_SHIM_GENERAL_NULL_NETWORK_PROPERTIES_H_
#define EOS_SHIM_GENERAL_NULL_NETWORK_PROPERTIES_H_
/// \file network_properties.H
/// \brief The species table of the problems without a network: one placeholder species, with the values that
/// Microphysics' generator writes for general_null. The vendored EOS includes this file by this name.
/// Transitional: part of the EOS shim, which goes when the EOS is ported to Quokka.

#include "../species_table.H"

constexpr int NumSpec = 1;

namespace
{
[[maybe_unused]] EOS_SHIM_HIP_CONSTEXPR AMREX_GPU_MANAGED amrex::Real aion[NumSpec] = {1.0};
[[maybe_unused]] EOS_SHIM_HIP_CONSTEXPR AMREX_GPU_MANAGED amrex::Real aion_inv[NumSpec] = {1.0};
[[maybe_unused]] EOS_SHIM_HIP_CONSTEXPR AMREX_GPU_MANAGED amrex::Real zion[NumSpec] = {1.0};
[[maybe_unused]] const std::vector<std::string> short_spec_names_cxx = {"X"};
[[maybe_unused]] const std::vector<std::string> spec_names_cxx = {"X"};
[[maybe_unused]] const std::vector<std::string> short_aux_names_cxx = {};
[[maybe_unused]] const std::vector<std::string> aux_names_cxx = {};
} // namespace

#endif // EOS_SHIM_GENERAL_NULL_NETWORK_PROPERTIES_H_
