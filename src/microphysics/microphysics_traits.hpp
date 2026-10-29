#ifndef MICROPHYSICS_TRAITS_HPP_
#define MICROPHYSICS_TRAITS_HPP_
/// \file microphysics_traits.hpp
/// \brief Compile-time microphysics settings, chosen per problem through Microphysics_Traits.

#include "AMReX_REAL.H"

#include "microphysics/rosenbrock/tableaus.hpp"

namespace quokka::microphysics
{
template <typename Tableau> struct Rosenbrock;
} // namespace quokka::microphysics

namespace quokka::networks
{
/// The default network type: the problem names no network.
struct None {};
} // namespace quokka::networks

/// Default values for all Microphysics_Traits fields.
struct DefaultMicrophysicsTraits {
	/// The species network: a type from src/networks.
	using Network = quokka::networks::None;
	using Integrator = quokka::microphysics::Rosenbrock<quokka::microphysics::Rodas5P>;
	/// Retry a failed burn. Replaces the runtime parameter integrator.use_burn_retry. The retry itself is not
	/// implemented yet, so with this set a failed burn aborts.
	static constexpr bool burn_retry = false;
	/// Enable chemistry
	static constexpr bool chemistry = false;
	/// Enable photochemistry.
	static constexpr bool photochemistry = false;
	/// Dust absorption in chemical bands.
	static constexpr bool dust_chemical_band_absorption = false;
	/// Whether to integrate the photon flux equations or determine them algebraically.
	static constexpr bool integrate_rad_flux = true;

	/// Use the network's analytic Jacobian.
	static constexpr bool analytic_jacobian = true;
	/// With analytic_jacobian, replace an analytic Jacobian that has a non-finite entry by finite differences. With
	/// false, the finite-difference Jacobian is not compiled and such a burn fails with IERR_JACOBIAN_NOT_FINITE.
	static constexpr bool numerical_jacobian_fallback = true;
	/// LU decomposition with partial pivoting.
	static constexpr bool linalg_do_pivoting = true;
	/// Call the EOS in the right-hand side to get T from e.
	static constexpr bool call_eos_in_rhs = true;
	/// Integrate the energy; with false it is frozen.
	static constexpr bool integrate_energy = true;
	/// Clip the species to [SMALL_X_SAFE, 1] before each right-hand side.
	static constexpr bool do_species_clip = false;
	/// Renormalize the species to sum to 1 before each right-hand side.
	static constexpr bool renormalize_abundances = false;
	/// The right-hand side and the Jacobian are zero at or above this temperature.
	static constexpr amrex::Real max_temp = 1.0e11;
	/// The largest internal timestep.
	static constexpr amrex::Real ode_max_dt = 1.e30;
	/// The H211b step-size controller of the Rosenbrock integrator.
	static constexpr amrex::Real h211b_b = 4.0;
	static constexpr amrex::Real h211b_k = 2.5;
	static constexpr amrex::Real h211b_fac_min = 0.2;
	static constexpr amrex::Real h211b_fac_max = 6.0;
	static constexpr amrex::Real h211b_reduction_fac = 0.5;
};

template <typename problem_t> struct Microphysics_Traits : DefaultMicrophysicsTraits {};

#endif // MICROPHYSICS_TRAITS_HPP_
