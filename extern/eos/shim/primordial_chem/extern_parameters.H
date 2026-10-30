#ifndef EOS_SHIM_PRIMORDIAL_CHEM_EXTERN_PARAMETERS_H_
#define EOS_SHIM_PRIMORDIAL_CHEM_EXTERN_PARAMETERS_H_
/// \file extern_parameters.H
/// \brief The runtime parameters of the primordial_chem EOS, with the names, defaults and input keys that Microphysics'
/// generator gives them for the primordial_chem network. The vendored EOS includes this file by this name.
/// Transitional: part of the EOS shim, which goes when the EOS is ported to Quokka.

#include <string>

#include "AMReX_GpuQualifiers.H"
#include "AMReX_REAL.H"

namespace eos_rp
{
extern AMREX_GPU_MANAGED amrex::Real eos_gamma_default; // NOLINT
extern std::string species_1_name;			// NOLINT
extern AMREX_GPU_MANAGED amrex::Real species_1_gamma;	// NOLINT
extern AMREX_GPU_MANAGED amrex::Real species_1_mass;	// NOLINT
extern std::string species_2_name;			// NOLINT
extern AMREX_GPU_MANAGED amrex::Real species_2_gamma;	// NOLINT
extern AMREX_GPU_MANAGED amrex::Real species_2_mass;	// NOLINT
extern std::string species_3_name;			// NOLINT
extern AMREX_GPU_MANAGED amrex::Real species_3_gamma;	// NOLINT
extern AMREX_GPU_MANAGED amrex::Real species_3_mass;	// NOLINT
extern std::string species_4_name;			// NOLINT
extern AMREX_GPU_MANAGED amrex::Real species_4_gamma;	// NOLINT
extern AMREX_GPU_MANAGED amrex::Real species_4_mass;	// NOLINT
extern std::string species_5_name;			// NOLINT
extern AMREX_GPU_MANAGED amrex::Real species_5_gamma;	// NOLINT
extern AMREX_GPU_MANAGED amrex::Real species_5_mass;	// NOLINT
extern std::string species_6_name;			// NOLINT
extern AMREX_GPU_MANAGED amrex::Real species_6_gamma;	// NOLINT
extern AMREX_GPU_MANAGED amrex::Real species_6_mass;	// NOLINT
extern std::string species_7_name;			// NOLINT
extern AMREX_GPU_MANAGED amrex::Real species_7_gamma;	// NOLINT
extern AMREX_GPU_MANAGED amrex::Real species_7_mass;	// NOLINT
extern std::string species_8_name;			// NOLINT
extern AMREX_GPU_MANAGED amrex::Real species_8_gamma;	// NOLINT
extern AMREX_GPU_MANAGED amrex::Real species_8_mass;	// NOLINT
extern std::string species_9_name;			// NOLINT
extern AMREX_GPU_MANAGED amrex::Real species_9_gamma;	// NOLINT
extern AMREX_GPU_MANAGED amrex::Real species_9_mass;	// NOLINT
extern std::string species_10_name;			// NOLINT
extern AMREX_GPU_MANAGED amrex::Real species_10_gamma;	// NOLINT
extern AMREX_GPU_MANAGED amrex::Real species_10_mass;	// NOLINT
extern std::string species_11_name;			// NOLINT
extern AMREX_GPU_MANAGED amrex::Real species_11_gamma;	// NOLINT
extern AMREX_GPU_MANAGED amrex::Real species_11_mass;	// NOLINT
extern std::string species_12_name;			// NOLINT
extern AMREX_GPU_MANAGED amrex::Real species_12_gamma;	// NOLINT
extern AMREX_GPU_MANAGED amrex::Real species_12_mass;	// NOLINT
extern std::string species_13_name;			// NOLINT
extern AMREX_GPU_MANAGED amrex::Real species_13_gamma;	// NOLINT
extern AMREX_GPU_MANAGED amrex::Real species_13_mass;	// NOLINT
extern std::string species_14_name;			// NOLINT
extern AMREX_GPU_MANAGED amrex::Real species_14_gamma;	// NOLINT
extern AMREX_GPU_MANAGED amrex::Real species_14_mass;	// NOLINT

/// Read the parameters from the `eos` prefix.
void ReadParameters();
} // namespace eos_rp

#endif // EOS_SHIM_PRIMORDIAL_CHEM_EXTERN_PARAMETERS_H_
