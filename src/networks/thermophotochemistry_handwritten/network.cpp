/// \file network.cpp
/// \brief Definitions and reader of the runtime parameters of the thermophotochemistry_handwritten network.

#include "networks/thermophotochemistry_handwritten/network.hpp"

#include "AMReX_ParmParse.H"

namespace quokka::networks
{

namespace thermophotochemistry
{
using amrex::literals::operator""_rt;

AMREX_GPU_MANAGED int recombination_switch = 1;		       // NOLINT
AMREX_GPU_MANAGED int recombination_temperature_dependent = 1; // NOLINT
AMREX_GPU_MANAGED int collisional_ionization_switch = 1;       // NOLINT
AMREX_GPU_MANAGED int energy_switch = 1;		       // NOLINT
AMREX_GPU_MANAGED int photoheating_switch = 1;		       // NOLINT
AMREX_GPU_MANAGED int KI_heating_switch = 1;		       // NOLINT
AMREX_GPU_MANAGED int recombination_cooling_switch = 1;	       // NOLINT
AMREX_GPU_MANAGED int KI_cooling_switch = 1;		       // NOLINT
AMREX_GPU_MANAGED int ion_ff_cooling_switch = 1;	       // NOLINT
AMREX_GPU_MANAGED int allow_mixed_cell_cooling = 1;	       // NOLINT
AMREX_GPU_MANAGED amrex::Real mixed_cell_lb = 0.01_rt;	       // NOLINT
AMREX_GPU_MANAGED amrex::Real mixed_cell_ub = 0.99_rt;	       // NOLINT
AMREX_GPU_MANAGED amrex::Real dust_kappa = 0.0_rt;	       // NOLINT
AMREX_GPU_MANAGED amrex::Real small_x = 1.0e-30_rt;	       // NOLINT
} // namespace thermophotochemistry

void Thermophotochemistry::ReadParameters()
{
	namespace params = thermophotochemistry;
	const amrex::ParmParse pp("network");
	pp.query("recombination_switch", params::recombination_switch);
	pp.query("recombination_temperature_dependent", params::recombination_temperature_dependent);
	pp.query("collisional_ionization_switch", params::collisional_ionization_switch);
	pp.query("energy_switch", params::energy_switch);
	pp.query("photoheating_switch", params::photoheating_switch);
	pp.query("KI_heating_switch", params::KI_heating_switch);
	pp.query("recombination_cooling_switch", params::recombination_cooling_switch);
	pp.query("KI_cooling_switch", params::KI_cooling_switch);
	pp.query("ion_ff_cooling_switch", params::ion_ff_cooling_switch);
	pp.query("allow_mixed_cell_cooling", params::allow_mixed_cell_cooling);
	pp.query("mixed_cell_lb", params::mixed_cell_lb);
	pp.query("mixed_cell_ub", params::mixed_cell_ub);
	pp.query("dust_kappa", params::dust_kappa);
	pp.query("small_x", params::small_x);
}

} // namespace quokka::networks
