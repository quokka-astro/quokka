#ifndef MICROPHYSICS_BURN_STATE_HPP_
#define MICROPHYSICS_BURN_STATE_HPP_
/// \file burn_state.hpp
/// \brief BurnState<problem_t>, the state of one cell during a chemistry or photochemistry burn. Replaces Microphysics'
/// burn_t: its members follow the problem's Microphysics_Traits instead of build macros.

#include <cstdint>
#include <type_traits>

#include "AMReX_Array.H"
#include "AMReX_REAL.H"

#include "microphysics/microphysics_traits.hpp"

namespace quokka::microphysics
{

namespace detail
{
/// The members for the chemical and thermal bands, present only with Microphysics_Traits::photochemistry.
template <typename problem_t, bool photochemistry = Microphysics_Traits<problem_t>::photochemistry> struct PhotochemistryState {};

template <typename problem_t> struct PhotochemistryState<problem_t, true> {
	using traits = Microphysics_Traits<problem_t>;
	using Network = typename traits::Network;

	/// per chemical band: the photon number density, and the flux-attenuation factor (if integrate_rad_flux is true)
	amrex::GpuArray<amrex::Real, (traits::integrate_rad_flux ? 2 : 1) * Network::NumChemBands> rn{};
	amrex::GpuArray<amrex::Real, Network::NumThermalBands> re_thermal{}; /// thermal-band energy densities
	amrex::Real c_hat{};						     /// reduced speed of light
};

/// The members for the dust absorption of chemical-band photons (Q_dust), present only with
/// Microphysics_Traits::dust_chemical_band_absorption. They extend the chemical-band members.
template <typename problem_t, bool dust_chemical_band_absorption = Microphysics_Traits<problem_t>::dust_chemical_band_absorption>
struct DustChemicalBandAbsorptionState : PhotochemistryState<problem_t> {};

template <typename problem_t> struct DustChemicalBandAbsorptionState<problem_t, true> : PhotochemistryState<problem_t> {
	amrex::Real dust_kappa{};      ///< dust opacity to chemical-band photons
	amrex::Real e_dust_absorbed{}; ///< energy density [erg cm^-3] of the chemical-band photons absorbed by dust during the burn
};
} // namespace detail

template <typename problem_t> struct BurnState : detail::DustChemicalBandAbsorptionState<problem_t> {
	using problem_type = problem_t;
	using traits = Microphysics_Traits<problem_t>;
	using Network = typename traits::Network;
	static_assert(traits::chemistry || traits::photochemistry, "BurnState is used by the chemistry and photochemistry problems only");
	static_assert(!std::is_same_v<Network, quokka::networks::None>, "a chemistry or photochemistry problem must set Microphysics_Traits::Network");

	amrex::Real rho{};
	amrex::Real T{};
	amrex::Real e{}; ///< specific internal energy
	amrex::GpuArray<amrex::Real, Network::NumSpec> xn{};

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
