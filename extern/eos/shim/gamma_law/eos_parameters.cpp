/// \file eos_parameters.cpp
/// \brief Definitions and reader of the runtime parameters of the gamma_law EOS.
/// Transitional: part of the EOS shim, which goes when the EOS is ported to Quokka.

#include "extern_parameters.H"

#include "AMReX_ParmParse.H"

namespace eos_rp
{
using amrex::literals::operator""_rt;

AMREX_GPU_MANAGED amrex::Real eos_gamma = 5.e0 / 3.e0_rt; // NOLINT
AMREX_GPU_MANAGED bool eos_assume_neutral = true;	  // NOLINT

void ReadParameters()
{
	const amrex::ParmParse pp("eos");
	pp.query("eos_gamma", eos_gamma);
	pp.query("eos_assume_neutral", eos_assume_neutral);
}
} // namespace eos_rp
