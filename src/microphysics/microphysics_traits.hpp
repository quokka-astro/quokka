#ifndef MICROPHYSICS_TRAITS_HPP_
#define MICROPHYSICS_TRAITS_HPP_
/// \file microphysics_traits.hpp
/// \brief Compile-time microphysics settings, chosen per problem through Microphysics_Traits.

/// Default values for all Microphysics_Traits fields.
struct DefaultMicrophysicsTraits {
	/// Retry a failed burn. Replaces the runtime parameter integrator.use_burn_retry. The retry itself is not
	/// implemented yet, so with this set a failed burn aborts.
	static constexpr bool burn_retry = false;
};

template <typename problem_t> struct Microphysics_Traits : DefaultMicrophysicsTraits {};

#endif // MICROPHYSICS_TRAITS_HPP_
