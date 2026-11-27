// SPDX-FileCopyrightText: © The Quokka Authors
// SPDX-License-Identifier: MIT

#ifndef MICROPHYSICS_BURN_STATE_HPP_
#define MICROPHYSICS_BURN_STATE_HPP_
/// \file burn_state.hpp
/// \brief The state of one cell during a chemistry or photochemistry burn. Replaces Microphysics' burn_t, which carried
/// the radiation members in every photochemistry build: here only the states that need them carry them.
///  - BurnState: the chemistry (density, temperature, energy, species);
///  - PhotochemBurnState: adds the chemical- and thermal-band radiation;
///  - DustyPhotochemBurnState: adds the absorption of chemical-band photons by dust.

#include "AMReX_Array.H"
#include "AMReX_REAL.H"

#include "math/rosenbrock/rosenbrock.hpp"

namespace quokka::microphysics
{

/// The chemistry of one cell, with NumSpec species (number densities).
template <int NumSpec> struct BurnState {
	static constexpr int num_spec = NumSpec;
	static constexpr bool photochemistry = false;
	static constexpr bool dust_absorption = false;

	amrex::Real rho{};
	amrex::Real T{};
	amrex::Real e{};    ///< specific internal energy
	amrex::Real dedT{}; ///< de/dT at constant composition, from the EOS; read by the networks' Jacobians
	amrex::GpuArray<amrex::Real, NumSpec> xn{};

	amrex::Real time{}; ///< the burn's time, for the failure report

	// the cell, for the failure report
	int i{};
	int j{};
	int k{};

	// written by the burn
	int n_rhs{};
	int n_jac{};
	int n_step{};
	bool success{};
	math::rosenbrock::IntegratorStatus integrator_status{};
};

/// A burn with chemical-band photons, of which each of the NumChemBands bands carries VarsPerGroup variables (the photon
/// number density and, with 2, a flux attenuation factor), and NumThermalBands thermal bands.
template <int NumSpec, int NumChemBands, int VarsPerGroup, int NumThermalBands> struct PhotochemBurnState : BurnState<NumSpec> {
	static constexpr bool photochemistry = true;
	static constexpr int num_chem_bands = NumChemBands;
	static constexpr int vars_per_group = VarsPerGroup;
	static constexpr int num_chem_rad_eqs = NumChemBands * VarsPerGroup;
	static constexpr int num_thermal_bands = NumThermalBands;
	static_assert(VarsPerGroup == 1 || VarsPerGroup == 2, "a chemical band carries its photon number and optionally a flux attenuation factor");

	/// per chemical band: the photon number density, then (with VarsPerGroup = 2) the flux attenuation factor
	amrex::GpuArray<amrex::Real, num_chem_rad_eqs> rn{};
	amrex::GpuArray<amrex::Real, NumThermalBands> re_thermal{}; ///< thermal-band energy densities
	amrex::Real c_hat{};					    ///< reduced speed of light
};

/// A photochemistry burn in which dust also absorbs the chemical-band photons.
template <int NumSpec, int NumChemBands, int VarsPerGroup, int NumThermalBands>
struct DustyPhotochemBurnState : PhotochemBurnState<NumSpec, NumChemBands, VarsPerGroup, NumThermalBands> {
	static constexpr bool dust_absorption = true;

	amrex::Real dust_kappa{};      ///< dust opacity to chemical-band photons
	amrex::Real e_dust_absorbed{}; ///< energy density [erg cm^-3] of the chemical-band photons absorbed by dust during the burn
};

} // namespace quokka::microphysics

#endif // MICROPHYSICS_BURN_STATE_HPP_
