// SPDX-FileCopyrightText: © The Quokka Authors
// SPDX-License-Identifier: MIT

#ifndef MICROPHYSICS_TRAITS_HPP_
#define MICROPHYSICS_TRAITS_HPP_
/// \file microphysics_traits.hpp
/// \brief Compile-time microphysics settings, chosen per problem through Microphysics_Traits.

#include "AMReX_REAL.H"

#include "math/rosenbrock/tableaus.hpp"

namespace quokka::networks
{
/// The network of a problem without chemistry.
struct None {};
} // namespace quokka::networks

/// Default values for all Microphysics_Traits fields. A problem derives its Microphysics_Traits from this struct and
/// overrides the fields it needs.
struct DefaultMicrophysicsTraits {
	/// The reaction network: a type from src/networks (e.g. quokka::networks::PrimordialChem). It gives the species, the
	/// radiation bands, the runtime parameters, the right-hand side and the Jacobian.
	using Network = quokka::networks::None;
	/// Burn the network as a chemistry source term (QuokkaSimulation's chemistry step).
	static constexpr bool chemistry = false;
	/// Burn the network with the chemical-band radiation (photochemistry, in the radiation update).
	static constexpr bool photochemistry = false;

	/// The Rosenbrock method of the burn (math/rosenbrock/tableaus.hpp).
	using Tableau = quokka::math::rosenbrock::ROS2S;

	/// Use an analytic Jacobian.
	static constexpr bool analytic_jacobian = true;
	/// With analytic_jacobian, replace an analytic Jacobian that has a non-finite entry by finite differences. With
	/// false, such a burn fails with IntegratorStatus::JacobianNotFinite.
	static constexpr bool numerical_jacobian_fallback = false;
	/// LU decomposition with partial pivoting.
	static constexpr bool pivoting = true;

	/// Call the EOS in the right-hand side to get T from e.
	static constexpr bool call_eos_in_rhs = true;
	/// Integrate the energy; with false it is frozen.
	static constexpr bool integrate_energy = true;
	/// The right-hand side and the Jacobian are zero at or above this temperature.
	static constexpr amrex::Real max_temp = 1.0e11;

	/// The largest internal timestep.
	static constexpr amrex::Real ode_max_dt = 1.e30;
};

template <typename problem_t> struct Microphysics_Traits : DefaultMicrophysicsTraits {};

#endif // MICROPHYSICS_TRAITS_HPP_
