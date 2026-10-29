#ifndef EOS_SHIM_GAMMA_LAW_EXTERN_PARAMETERS_H_
#define EOS_SHIM_GAMMA_LAW_EXTERN_PARAMETERS_H_
/// \file extern_parameters.H
/// \brief The runtime parameters of the gamma_law EOS, with the names, defaults and input keys that Microphysics'
/// generator gives them. The vendored EOS includes this file by this name.
/// Transitional: part of the EOS shim, which goes when the EOS is ported to Quokka.

#include "AMReX_GpuQualifiers.H"
#include "AMReX_REAL.H"

namespace eos_rp
{
extern AMREX_GPU_MANAGED amrex::Real eos_gamma;	  // NOLINT
extern AMREX_GPU_MANAGED bool eos_assume_neutral; // NOLINT

/// Read the parameters from the `eos` prefix.
void ReadParameters();
} // namespace eos_rp

#endif // EOS_SHIM_GAMMA_LAW_EXTERN_PARAMETERS_H_
