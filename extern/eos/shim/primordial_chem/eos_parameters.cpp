/// \file eos_parameters.cpp
/// \brief Definitions and reader of the runtime parameters of the primordial_chem EOS.
/// Transitional: part of the EOS shim, which goes when the EOS is ported to Quokka.

#include "extern_parameters.H"

#include "AMReX_ParmParse.H"

namespace eos_rp
{
using amrex::literals::operator""_rt;

AMREX_GPU_MANAGED amrex::Real eos_gamma_default = 1.4_rt;	      // NOLINT
std::string species_1_name = "elec";				      // NOLINT
AMREX_GPU_MANAGED amrex::Real species_1_gamma = 5. / 3._rt;	      // NOLINT
AMREX_GPU_MANAGED amrex::Real species_1_mass = 9.10938188e-28_rt;     // NOLINT
std::string species_2_name = "hp";				      // NOLINT
AMREX_GPU_MANAGED amrex::Real species_2_gamma = 5. / 3._rt;	      // NOLINT
AMREX_GPU_MANAGED amrex::Real species_2_mass = 1.67262158e-24_rt;     // NOLINT
std::string species_3_name = "h";				      // NOLINT
AMREX_GPU_MANAGED amrex::Real species_3_gamma = 5. / 3._rt;	      // NOLINT
AMREX_GPU_MANAGED amrex::Real species_3_mass = 1.67353251819e-24_rt;  // NOLINT
std::string species_4_name = "hm";				      // NOLINT
AMREX_GPU_MANAGED amrex::Real species_4_gamma = 5. / 3._rt;	      // NOLINT
AMREX_GPU_MANAGED amrex::Real species_4_mass = 1.67444345638e-24_rt;  // NOLINT
std::string species_5_name = "dp";				      // NOLINT
AMREX_GPU_MANAGED amrex::Real species_5_gamma = 5. / 3._rt;	      // NOLINT
AMREX_GPU_MANAGED amrex::Real species_5_mass = 3.34512158e-24_rt;     // NOLINT
std::string species_6_name = "d";				      // NOLINT
AMREX_GPU_MANAGED amrex::Real species_6_gamma = 5. / 3._rt;	      // NOLINT
AMREX_GPU_MANAGED amrex::Real species_6_mass = 3.34603251819e-24_rt;  // NOLINT
std::string species_7_name = "h2p";				      // NOLINT
AMREX_GPU_MANAGED amrex::Real species_7_gamma = 1.4_rt;		      // NOLINT
AMREX_GPU_MANAGED amrex::Real species_7_mass = 3.34615409819e-24_rt;  // NOLINT
std::string species_8_name = "dm";				      // NOLINT
AMREX_GPU_MANAGED amrex::Real species_8_gamma = 5. / 3._rt;	      // NOLINT
AMREX_GPU_MANAGED amrex::Real species_8_mass = 3.34694345638e-24_rt;  // NOLINT
std::string species_9_name = "h2";				      // NOLINT
AMREX_GPU_MANAGED amrex::Real species_9_gamma = 1.4_rt;		      // NOLINT
AMREX_GPU_MANAGED amrex::Real species_9_mass = 3.34706503638e-24_rt;  // NOLINT
std::string species_10_name = "hdp";				      // NOLINT
AMREX_GPU_MANAGED amrex::Real species_10_gamma = 1.4_rt;	      // NOLINT
AMREX_GPU_MANAGED amrex::Real species_10_mass = 5.01865409819e-24_rt; // NOLINT
std::string species_11_name = "hd";				      // NOLINT
AMREX_GPU_MANAGED amrex::Real species_11_gamma = 1.4_rt;	      // NOLINT
AMREX_GPU_MANAGED amrex::Real species_11_mass = 5.01956503638e-24_rt; // NOLINT
std::string species_12_name = "hepp";				      // NOLINT
AMREX_GPU_MANAGED amrex::Real species_12_gamma = 5. / 3._rt;	      // NOLINT
AMREX_GPU_MANAGED amrex::Real species_12_mass = 6.69024316e-24_rt;    // NOLINT
std::string species_13_name = "hep";				      // NOLINT
AMREX_GPU_MANAGED amrex::Real species_13_gamma = 5. / 3._rt;	      // NOLINT
AMREX_GPU_MANAGED amrex::Real species_13_mass = 6.69115409819e-24_rt; // NOLINT
std::string species_14_name = "he";				      // NOLINT
AMREX_GPU_MANAGED amrex::Real species_14_gamma = 5. / 3._rt;	      // NOLINT
AMREX_GPU_MANAGED amrex::Real species_14_mass = 6.69206503638e-24_rt; // NOLINT

void ReadParameters()
{
	const amrex::ParmParse pp("eos");
	pp.query("eos_gamma_default", eos_gamma_default);
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
	pp.query("species_7_name", species_7_name);
	pp.query("species_7_gamma", species_7_gamma);
	pp.query("species_7_mass", species_7_mass);
	pp.query("species_8_name", species_8_name);
	pp.query("species_8_gamma", species_8_gamma);
	pp.query("species_8_mass", species_8_mass);
	pp.query("species_9_name", species_9_name);
	pp.query("species_9_gamma", species_9_gamma);
	pp.query("species_9_mass", species_9_mass);
	pp.query("species_10_name", species_10_name);
	pp.query("species_10_gamma", species_10_gamma);
	pp.query("species_10_mass", species_10_mass);
	pp.query("species_11_name", species_11_name);
	pp.query("species_11_gamma", species_11_gamma);
	pp.query("species_11_mass", species_11_mass);
	pp.query("species_12_name", species_12_name);
	pp.query("species_12_gamma", species_12_gamma);
	pp.query("species_12_mass", species_12_mass);
	pp.query("species_13_name", species_13_name);
	pp.query("species_13_gamma", species_13_gamma);
	pp.query("species_13_mass", species_13_mass);
	pp.query("species_14_name", species_14_name);
	pp.query("species_14_gamma", species_14_gamma);
	pp.query("species_14_mass", species_14_mass);
}
} // namespace eos_rp
