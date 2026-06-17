#ifndef STELLAR_MODELS_HPP_
#define STELLAR_MODELS_HPP_

#include "AMReX_Extension.H"
#include "AMReX_GpuQualifiers.H"
#include "AMReX_REAL.H"
#include "fundamental_constants.H"
#include <cmath>

namespace quokka
{

// Toy stellar-evolution model: stateless analytic laws.
//   R(M)      = R_sun * (M / M_sun)^0.4
//   L_star(M) = L_sun * (M / M_sun)^3.5
//   L_acc     = G * M * mdot / R
// All functions are pure (no particle, no field indices), so this header has no particle
// dependency and can be included by particle_types.hpp without a circular include.
struct ToyStellarModel {
	// Extra per-particle components this model needs beyond the base Star layout.
	// The toy model is stateless, so it needs none.
	static constexpr int nExtraReal = 0;
	static constexpr int nExtraInt = 0;

	static constexpr amrex::Real L_solar = 3.828e33; // erg/s (CODATA 2022)
	static constexpr amrex::Real radius_exponent = 0.4;
	static constexpr amrex::Real luminosity_exponent = 3.5;

	AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE static auto radius(amrex::Real mass) -> amrex::Real
	{
		if (mass <= 0.0) {
			return 0.0;
		}
		return C::R_solar * std::pow(mass / C::M_solar, radius_exponent);
	}

	AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE static auto luminosityStar(amrex::Real mass) -> amrex::Real
	{
		if (mass <= 0.0) {
			return 0.0;
		}
		return L_solar * std::pow(mass / C::M_solar, luminosity_exponent);
	}

	AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE static auto luminosityAcc(amrex::Real mass, amrex::Real mdot, amrex::Real radius_val) -> amrex::Real
	{
		if (radius_val <= 0.0 || mdot <= 0.0) {
			return 0.0;
		}
		return C::Gconst * mass * mdot / radius_val;
	}

	// Stellar-evolution step.
	//   mass:       [in/out] current stellar mass (model may modify, e.g. wind mass loss)
	//   mdot:       [in]     mass accretion rate onto the star (set by accretion module)
	//   radius_val: [in/out] current stellar radius (model may read and/or modify)
	//   lum:        [out]    luminosity array of size n_groups — model writes all groups
	//   n_groups:   [in]     number of radiation groups (= length of lum array)
	//   dt:         [in]     timestep
	// The toy model is stateless, so it ignores the incoming radius/lum values and dt.
	AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE static void evolve(amrex::Real &mass, amrex::Real mdot, amrex::Real &radius_val,
								    amrex::Real *lum, [[maybe_unused]] int n_groups,
								    [[maybe_unused]] amrex::Real dt)
	{
		radius_val = radius(mass);
		if (n_groups > 0) {
			lum[0] = luminosityStar(mass) + luminosityAcc(mass, mdot, radius_val);
			for (int g = 1; g < n_groups; ++g) {
				lum[g] = 0.0;
			}
		}
	}
};

} // namespace quokka

#endif // STELLAR_MODELS_HPP_
