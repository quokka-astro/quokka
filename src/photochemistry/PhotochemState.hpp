#ifndef PHOTOCHEMSTATE_HPP_
#define PHOTOCHEMSTATE_HPP_
/// \file PhotochemState.hpp
/// \brief The state of one cell during the photochemistry solve (replaces Microphysics's burn_t).

#include "AMReX_Array.H"
#include "AMReX_REAL.H"

namespace quokka::photochemistry
{

/// \tparam NSpec number of species
/// \tparam NChemRadEqs number of chemical-band radiation variables (per band: photon number density, then the flux factor
///         unless SKIP_PHOTOCHEMFLUX)
/// \tparam NThermal number of thermal bands coupled to the chemistry
template <int NSpec, int NChemRadEqs, int NThermal> struct State {
	amrex::Real rho{};							  ///< mass density [g cm^-3]
	amrex::Real e{};							  ///< specific internal energy [erg g^-1]
	amrex::Real T{};							  ///< temperature [K], from the EOS
	amrex::Real dedT{};							  ///< de/dT at constant composition, from the EOS
	amrex::Real c_hat{};							  ///< reduced speed of light [cm s^-1]
	amrex::GpuArray<amrex::Real, NSpec> xn{};				  ///< species number densities [cm^-3]
	amrex::GpuArray<amrex::Real, (NChemRadEqs > 0) ? NChemRadEqs : 1> rn{};	  ///< chemical-band radiation variables
	amrex::GpuArray<amrex::Real, (NThermal > 0) ? NThermal : 1> re_thermal{}; ///< thermal-band radiation energy densities
	amrex::Real dust_kappa{};						  ///< dust opacity (DUST_CHEMICAL_BAND_ABSORPTION only)
	amrex::Real e_dust_absorbed{}; ///< chemical-band energy absorbed by dust (DUST_CHEMICAL_BAND_ABSORPTION only)
};

} // namespace quokka::photochemistry

#endif // PHOTOCHEMSTATE_HPP_
