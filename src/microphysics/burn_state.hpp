#ifndef MICROPHYSICS_BURN_STATE_HPP_
#define MICROPHYSICS_BURN_STATE_HPP_
/// \file burn_state.hpp
/// \brief BurnState<problem_t>, the state of one cell during a photochemistry burn. Replaces Microphysics' burn_t for
/// the photochemistry problems: its members follow the problem's Microphysics_Traits instead of build macros.

#include <cstdint>

#include "AMReX_Array.H"
#include "AMReX_REAL.H"

#include "microphysics/microphysics_traits.hpp"

// still from extern/Microphysics: NumSpec, NumChemRadEqs and NumThermalBands
#include "network_properties.H"

namespace quokka::microphysics
{

namespace detail
{
/// The members for the dust absorption of chemical-band photons (Q_dust), present only with
/// Microphysics_Traits::dust_chemical_band_absorption.
template <bool dust_chemical_band_absorption> struct DustChemicalBandAbsorptionState {};

template <> struct DustChemicalBandAbsorptionState<true> {
	amrex::Real dust_kappa{};      ///< dust opacity to chemical-band photons
	amrex::Real e_dust_absorbed{}; ///< energy density [erg cm^-3] of the chemical-band photons absorbed by dust during the burn
};
} // namespace detail

template <typename problem_t> struct BurnState : detail::DustChemicalBandAbsorptionState<Microphysics_Traits<problem_t>::dust_chemical_band_absorption> {
	using problem_type = problem_t;
	using traits = Microphysics_Traits<problem_t>;
	static_assert(traits::photochemistry, "BurnState is used by the photochemistry problems only, for now");

	amrex::Real rho{};
	amrex::Real T{};
	amrex::Real e{}; ///< specific internal energy
	amrex::GpuArray<amrex::Real, NumSpec> xn{};
	/// per chemical band: the photon number density, and the flux-attenuation factor (if integrate_rad_flux is true)
	amrex::GpuArray<amrex::Real, NumChemRadEqs> rn{};
	amrex::GpuArray<amrex::Real, NumThermalBands> re_thermal{}; /// thermal-band energy densities
	amrex::Real c_hat{};					    /// reduced speed of light

	amrex::Real dedT{}; /// read by the networks' Jacobians

	// Written by Microphysics' eos(); they go when the EOS is ported.
	amrex::Real dedr{};
	amrex::Real mu{};
	amrex::Real mu_e{};
	amrex::Real y_e{};
	amrex::Real abar{};
	amrex::Real zbar{};

	amrex::Real time{}; /// written by the integrator; nothing reads it

	// the cell, for the failure output
	int i{};
	int j{};
	int k{};

	int n_rhs{};
	int n_jac{};
	int n_step{};
	bool success{};
	std::int16_t error_code{};
};

} // namespace quokka::microphysics

#endif // MICROPHYSICS_BURN_STATE_HPP_
