#ifndef STELLAR_MODELS_HPP_
#define STELLAR_MODELS_HPP_

#include "AMReX_Extension.H"
#include "AMReX_GpuQualifiers.H"
#include "AMReX_REAL.H"
#include "fundamental_constants.H"
#include "particles/star_particle_indices.H"
#include <cmath>

namespace quokka
{

// Toy stellar-evolution model: stateless analytic laws.
//   R(M)      = R_sun * (M / M_sun)^0.4
//   L_star(M) = L_sun * (M / M_sun)^3.5
//   L_acc     = G * M * mdot / R
struct ToyStellarModel {
	// Extra per-particle components this model needs beyond the base Star layout.
	// One extra real holds the ionizing photon rate Q, which is assigned once at birth
	// (see ionizingPhotonRate below) and thereafter held fixed, so it cannot be recomputed
	// on the fly from the current mass.
	static constexpr int nExtraReal = 1;
	static constexpr int nExtraInt = 0;

	// Offset of the ionizing photon rate within this model's extra-real block. The absolute
	// particle component index is StarParticleLumIdx + n_groups + QIonExtraOffset.
	static constexpr int QIonExtraOffset = 0;

	static constexpr amrex::Real L_solar = 3.828e33; // erg/s (CODATA 2022)
	static constexpr amrex::Real radius_exponent = 0.4;
	static constexpr amrex::Real luminosity_exponent = 3.5;

	// Hydrogen-ionizing photon rate from the Sternberg, Hoffmann & Pauldrach (2003) grid, as
	// recalibrated by Martins, Schaerer & Hillier (2005). The coefficients below are a least-squares
	// fit to all twelve luminosity-class-V rows of their Table 1 (theoretical Teff scale), regressing
	// the tabulated log10 Q0 on x = log10(M_spec / M_sun):
	//
	//   log10 Q0 [1/s] = c0 + c1 x + c2 x^2
	//
	// The fit spans O9.5V (16.46 M_sun, log Q0 = 47.56) to O3V (58.34 M_sun, log Q0 = 49.63), with
	// an rms residual of 0.036 dex and a maximum residual of 0.064 dex.
	//
	// A quadratic is used rather than a cubic deliberately. Fitting a cubic to the same twelve points
	// lowers the rms only from 0.036 to 0.035 dex -- no real gain -- but its leading coefficient comes
	// out negative, so it peaks at 71 M_sun and then falls: it would assign a 150 M_sun star less
	// ionizing flux than a 37 M_sun one. The quadratic has the same qualitative flaw but only beyond
	// 88 M_sun, well outside the tabulated range, and the clamp below removes it entirely.
	static constexpr amrex::Real Q_ion_c0 = 34.314866;
	static constexpr amrex::Real Q_ion_c1 = 15.893125;
	static constexpr amrex::Real Q_ion_c2 = -4.082487;

	// Mass range of the Martins et al. table, in solar masses.
	//
	// Below the lower edge the star is later than O9.5V and its ionizing output falls off a cliff, so
	// Q is set to zero rather than extrapolating: the fit continued down to 5 M_sun would return
	// ~1e43 photons/s against a true value near 1e38, and low-mass stars outnumber O stars by orders
	// of magnitude in a sampled IMF. The early return also keeps log10 away from zero mass, where it
	// would raise a floating-point trap.
	//
	// Above the upper edge the mass is clamped, so Q saturates at the O3V value instead of running off
	// into the fit's unphysical turnover. This under-predicts genuinely very massive stars (a 100
	// M_sun star gets the 58 M_sun rate, low by roughly a factor of two), which is the conservative
	// direction for a feedback module and is preferable to extrapolating a polynomial past its data.
	static constexpr amrex::Real Q_ion_min_mass = 16.46;
	static constexpr amrex::Real Q_ion_max_mass = 58.34;

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

	// Hydrogen-ionizing photon rate (1/s) for a star of the given mass. See the coefficients above.
	AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE static auto ionizingPhotonRate(amrex::Real mass) -> amrex::Real
	{
		const amrex::Real mass_in_solar = mass / C::M_solar;
		if (mass_in_solar < Q_ion_min_mass) {
			return 0.0;
		}
		const amrex::Real m_eff = (mass_in_solar > Q_ion_max_mass) ? Q_ion_max_mass : mass_in_solar;
		const amrex::Real x = std::log10(m_eff);
		const amrex::Real log_Q = Q_ion_c0 + (x * (Q_ion_c1 + (x * Q_ion_c2)));
		return std::pow(10.0, log_Q);
	}

	AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE static auto luminosityAcc(amrex::Real mass, amrex::Real mdot, amrex::Real radius_val) -> amrex::Real
	{
		if (radius_val <= 0.0 || mdot <= 0.0) {
			return 0.0;
		}
		return C::Gconst * mass * mdot / radius_val;
	}

	// Stellar-evolution step — takes the full particle real- and integer-data arrays
	// so the model can read and modify any component (mass, radius, luminosity groups,
	// integer state, etc.).
	//   rdata:    [in/out] particle real-component array (layout defined by StarParticleDataIdx)
	//   idata:    [in/out] particle integer-component array (may be nullptr if NInt == 0)
	//   n_groups: [in]     number of radiation groups (= number of lum slots after index 7)
	//   dt:       [in]     timestep
	// The toy model is stateless, so it only reads mass/mdot and writes radius/lum.
	AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE static void evolve(amrex::Real *rdata, [[maybe_unused]] int *idata, int n_groups,
								    [[maybe_unused]] amrex::Real dt)
	{
		const amrex::Real mass = rdata[StarParticleMassIdx];
		const amrex::Real mdot = rdata[StarParticleMdotIdx];
		rdata[StarParticleRadiusIdx] = radius(mass);
		if (n_groups > 0) {
			rdata[StarParticleLumIdx] = luminosityStar(mass) + luminosityAcc(mass, mdot, rdata[StarParticleRadiusIdx]);
			for (int g = 1; g < n_groups; ++g) {
				rdata[StarParticleLumIdx + g] = 0.0;
			}
		}

		// Assign the ionizing photon rate once, at birth: a non-positive value marks a slot that has
		// not been set yet. Q is deliberately NOT refreshed once it is positive, because it is fixed
		// at the birth mass and must not drift as the particle accretes.
		//
		// A star born below Q_ion_min_mass gets Q = 0 and is therefore re-evaluated on later calls.
		// That is intended: such a star contributes no ionizing photons until accretion carries it
		// into the O-star range, at which point Q is assigned from the mass it has then and frozen.
		//
		// IMPORTANT: this sentinel requires the slot to be zero before the first call. Particle real
		// components are NOT zero-initialized by amrex::ParticleContainer::InitFromAsciiFile, which
		// fills only the components present in the file and leaves the rest indeterminate. Any
		// problem that creates Star particles that way must explicitly zero the components it does
		// not set -- see src/problems/StromgrenVolumeFeedback/ and src/problems/ParticleStarEvolution/.
		const int q_ion_idx = StarParticleLumIdx + n_groups + QIonExtraOffset;
		if (rdata[q_ion_idx] <= 0.0) {
			rdata[q_ion_idx] = ionizingPhotonRate(mass);
		}
	}
};

} // namespace quokka

#endif // STELLAR_MODELS_HPP_
