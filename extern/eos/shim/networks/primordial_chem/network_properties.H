#ifndef EOS_SHIM_PRIMORDIAL_CHEM_NETWORK_PROPERTIES_H_
#define EOS_SHIM_PRIMORDIAL_CHEM_NETWORK_PROPERTIES_H_
/// \file network_properties.H
/// \brief The species table of the primordial_chem network, with the values that Microphysics' generator writes for it.
/// The vendored EOS includes this file by this name.
/// Transitional: part of the EOS shim, which goes when the EOS is ported to Quokka.

#include "../species_table.H"
#include "networks/primordial_chem/network.hpp"

constexpr int NumSpec = quokka::networks::PrimordialChem::NumSpec;

namespace
{
[[maybe_unused]] EOS_SHIM_HIP_CONSTEXPR AMREX_GPU_MANAGED amrex::Real aion[NumSpec] = {1.0,  4.0,  12.0, 13.0, 13.0, 14.0, 15.0,
										       14.0, 15.0, 16.0, 17.0, 17.0, 18.0, 4.0};
[[maybe_unused]] EOS_SHIM_HIP_CONSTEXPR AMREX_GPU_MANAGED amrex::Real aion_inv[NumSpec] = {1.0,	       1.0 / 4.0,  1.0 / 12.0, 1.0 / 13.0, 1.0 / 13.0,
											   1.0 / 14.0, 1.0 / 15.0, 1.0 / 14.0, 1.0 / 15.0, 1.0 / 16.0,
											   1.0 / 17.0, 1.0 / 17.0, 1.0 / 18.0, 1.0 / 4.0};
[[maybe_unused]] EOS_SHIM_HIP_CONSTEXPR AMREX_GPU_MANAGED amrex::Real zion[NumSpec] = {1.0, 2.0, 6.0, 6.0, 7.0, 7.0, 7.0, 8.0, 8.0, 8.0, 8.0, 9.0, 9.0, 2.0};
[[maybe_unused]] const std::vector<std::string> short_spec_names_cxx = {"E", "Hp", "H", "Hm", "Dp", "D", "H2p", "Dm", "H2", "HDp", "HD", "HEpp", "HEp", "HE"};
[[maybe_unused]] const std::vector<std::string> spec_names_cxx = {"elec", "hp", "h", "hm", "dp", "d", "h2p", "dm", "h2", "hdp", "hd", "hepp", "hep", "he"};
[[maybe_unused]] const std::vector<std::string> short_aux_names_cxx = {"Tdust"};
[[maybe_unused]] const std::vector<std::string> aux_names_cxx = {"Tdust"};
} // namespace

// the table lists the species in the network's order
static_assert(quokka::networks::PrimordialChem::E == 0 && quokka::networks::PrimordialChem::Hp == 1 && quokka::networks::PrimordialChem::H == 2 &&
	      quokka::networks::PrimordialChem::Hm == 3 && quokka::networks::PrimordialChem::Dp == 4 && quokka::networks::PrimordialChem::D == 5 &&
	      quokka::networks::PrimordialChem::H2p == 6 && quokka::networks::PrimordialChem::Dm == 7 && quokka::networks::PrimordialChem::H2 == 8 &&
	      quokka::networks::PrimordialChem::Hdp == 9 && quokka::networks::PrimordialChem::Hd == 10 && quokka::networks::PrimordialChem::Hepp == 11 &&
	      quokka::networks::PrimordialChem::Hep == 12 && quokka::networks::PrimordialChem::He == 13);

#endif // EOS_SHIM_PRIMORDIAL_CHEM_NETWORK_PROPERTIES_H_
