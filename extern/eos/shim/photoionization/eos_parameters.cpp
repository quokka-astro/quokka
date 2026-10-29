/// \file eos_parameters.cpp
/// \brief Definitions and reader of the runtime parameters of the photoionization EOS.
/// Transitional: part of the EOS shim, which goes when the EOS is ported to Quokka.

#include "extern_parameters.H"

#include "AMReX_ParmParse.H"

namespace eos_rp
{
using amrex::literals::operator""_rt;

std::string species_1_name = "e-";					  // NOLINT
AMREX_GPU_MANAGED amrex::Real species_1_gamma = 5. / 3._rt;		  // NOLINT
AMREX_GPU_MANAGED amrex::Real species_1_mass = 9.10938371e-28_rt;	  // NOLINT
std::string species_2_name = "H";					  // NOLINT
AMREX_GPU_MANAGED amrex::Real species_2_gamma = 5. / 3._rt;		  // NOLINT
AMREX_GPU_MANAGED amrex::Real species_2_mass = 1.673773e-24_rt;		  // NOLINT
std::string species_3_name = "H+";					  // NOLINT
AMREX_GPU_MANAGED amrex::Real species_3_gamma = 5. / 3._rt;		  // NOLINT
AMREX_GPU_MANAGED amrex::Real species_3_mass = 1.6728620616289998e-24_rt; // NOLINT
std::string species_4_name = "He";					  // NOLINT
AMREX_GPU_MANAGED amrex::Real species_4_gamma = 5. / 3._rt;		  // NOLINT
AMREX_GPU_MANAGED amrex::Real species_4_mass = 6.646473e-24_rt;		  // NOLINT
std::string species_5_name = "He+";					  // NOLINT
AMREX_GPU_MANAGED amrex::Real species_5_gamma = 5. / 3._rt;		  // NOLINT
AMREX_GPU_MANAGED amrex::Real species_5_mass = 6.645562061629e-24_rt;	  // NOLINT
std::string species_6_name = "He++";					  // NOLINT
AMREX_GPU_MANAGED amrex::Real species_6_gamma = 5. / 3._rt;		  // NOLINT
AMREX_GPU_MANAGED amrex::Real species_6_mass = 6.644651123258e-24_rt;	  // NOLINT

void ReadParameters()
{
	const amrex::ParmParse pp("eos");
	pp.query("species_1_name", species_1_name);
	pp.query("species_1_gamma", species_1_gamma);
	pp.query("species_1_mass", species_1_mass);
	pp.query("species_2_name", species_2_name);
	pp.query("species_2_gamma", species_2_gamma);
	pp.query("species_2_mass", species_2_mass);
	pp.query("species_3_name", species_3_name);
	pp.query("species_3_gamma", species_3_gamma);
	pp.query("species_3_mass", species_3_mass);
	pp.query("species_4_name", species_4_name);
	pp.query("species_4_gamma", species_4_gamma);
	pp.query("species_4_mass", species_4_mass);
	pp.query("species_5_name", species_5_name);
	pp.query("species_5_gamma", species_5_gamma);
	pp.query("species_5_mass", species_5_mass);
	pp.query("species_6_name", species_6_name);
	pp.query("species_6_gamma", species_6_gamma);
	pp.query("species_6_mass", species_6_mass);
}
} // namespace eos_rp
