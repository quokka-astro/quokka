/// \file actual_eos_data.cpp
/// \brief The definition of the gamma_law EOS's placeholder spmasses (see actual_eos_data.H). It is defined here, not
/// inline in the header: HIP cannot register inline managed variables.
/// Transitional: part of the EOS shim, which goes when the EOS is ported to Quokka.

#include "actual_eos_data.H"

AMREX_GPU_MANAGED amrex::Real spmasses[NumSpec]; // NOLINT
