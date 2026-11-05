#ifndef EOS_SHIM_PHOTOIONIZATION_EXTERN_PARAMETERS_H_
#define EOS_SHIM_PHOTOIONIZATION_EXTERN_PARAMETERS_H_
/// \file extern_parameters.H
/// \brief The runtime parameters of the photoionization EOS, with the names, defaults and input keys that Microphysics'
/// generator gives them. The vendored EOS includes this file by this name.
/// Transitional: part of the EOS shim, which goes when the EOS is ported to Quokka.

#include <string>

#include "AMReX_GpuQualifiers.H"
#include "AMReX_REAL.H"

namespace eos_rp
{
extern std::string species_1_name;		      // NOLINT
extern AMREX_GPU_MANAGED amrex::Real species_1_gamma; // NOLINT
extern AMREX_GPU_MANAGED amrex::Real species_1_mass;  // NOLINT
extern std::string species_2_name;		      // NOLINT
extern AMREX_GPU_MANAGED amrex::Real species_2_gamma; // NOLINT
extern AMREX_GPU_MANAGED amrex::Real species_2_mass;  // NOLINT
extern std::string species_3_name;		      // NOLINT
extern AMREX_GPU_MANAGED amrex::Real species_3_gamma; // NOLINT
extern AMREX_GPU_MANAGED amrex::Real species_3_mass;  // NOLINT
extern std::string species_4_name;		      // NOLINT
extern AMREX_GPU_MANAGED amrex::Real species_4_gamma; // NOLINT
extern AMREX_GPU_MANAGED amrex::Real species_4_mass;  // NOLINT
extern std::string species_5_name;		      // NOLINT
extern AMREX_GPU_MANAGED amrex::Real species_5_gamma; // NOLINT
extern AMREX_GPU_MANAGED amrex::Real species_5_mass;  // NOLINT
extern std::string species_6_name;		      // NOLINT
extern AMREX_GPU_MANAGED amrex::Real species_6_gamma; // NOLINT
extern AMREX_GPU_MANAGED amrex::Real species_6_mass;  // NOLINT

/// Read the parameters from the `eos` prefix.
void ReadParameters();
} // namespace eos_rp

#endif // EOS_SHIM_PHOTOIONIZATION_EXTERN_PARAMETERS_H_
