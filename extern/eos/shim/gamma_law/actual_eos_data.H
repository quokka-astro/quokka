#ifndef EOS_SHIM_GAMMA_LAW_ACTUAL_EOS_DATA_H_
#define EOS_SHIM_GAMMA_LAW_ACTUAL_EOS_DATA_H_
/// \file actual_eos_data.H
/// \brief A placeholder spmasses for the gamma_law EOS, which has none. Quokka's burn code and EOSMicrophysics name
/// spmasses, which the photoionization and primordial_chem EOSs declare in their actual_eos_data.H; with this, they
/// compile in every build without an #ifdef. Nothing reads it: a problem with chemistry is built with a chemistry EOS,
/// and its network's NumSpec check fails against this table otherwise.
/// Transitional: part of the EOS shim, which goes when the EOS is ported to Quokka.

#include "AMReX_GpuQualifiers.H"
#include "AMReX_REAL.H"

#include <network_properties.H>

extern AMREX_GPU_MANAGED amrex::Real spmasses[NumSpec]; // NOLINT

#endif // EOS_SHIM_GAMMA_LAW_ACTUAL_EOS_DATA_H_
