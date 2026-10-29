#ifndef EOS_SHIM_SPECIES_TABLE_H_
#define EOS_SHIM_SPECIES_TABLE_H_
/// \file species_table.H
/// \brief What every species table (networks/<network>/network_properties.H) needs: the qualifiers that Microphysics'
/// generator gives the species arrays, so that the device code does not change.
/// Transitional: part of the EOS shim, which goes when the EOS is ported to Quokka.

#include <string>
#include <vector>

#include "AMReX_GpuQualifiers.H"
#include "AMReX_REAL.H"

// The number of auxiliary variables, as Microphysics generates it: the build defines NAUX_NET (= 1) until the submodule
// goes, so the EOS gives eos_t an aux member.
constexpr int NumAux = NAUX_NET;

// The arrays are constexpr on HIP only, as Microphysics generates them.
#ifdef AMREX_USE_HIP
#define EOS_SHIM_HIP_CONSTEXPR constexpr
#else
#define EOS_SHIM_HIP_CONSTEXPR
#endif

#endif // EOS_SHIM_SPECIES_TABLE_H_
