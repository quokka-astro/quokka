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
	// recalibrated by Martins, Schaerer & Hillier (2005). Over the O-star main sequence
	// (roughly 15-60 M_sun at Z ~ Z_sun), log10 Q0 is well represented by a cubic in
	// x = log10(m / M_sun):
	//
	//   log10 Q0 [1/s] = c0 + c1 x + c2 x^2 + c3 x^3
	//
	// The coefficients below reproduce the published Martins et al. anchor points exactly:
	//   20 M_sun -> log Q0 = 48.5,  30 -> 49.0,  50 -> 49.5,  60 -> 49.65.
	// The cubic is monotonic in mass everywhere (its derivative has no real roots), so it cannot
	// produce a non-physical inversion where a more massive star ionizes less.
	static constexpr amrex::Real Q_ion_c0 = 40.076466;
	static constexpr amrex::Real Q_ion_c1 = 10.795187;
	static constexpr amrex::Real Q_ion_c2 = -4.078483;
	static constexpr amrex::Real Q_ion_c3 = 0.582245;

	// Lower edge of the fit range, in solar masses. Stars below this are later than about B0 and
	// their ionizing output is negligible, so Q is set to zero rather than extrapolating the cubic
	// downward -- extrapolation would credit a 5 M_sun star with ~1e46 photons/s against a true
	// value near 1e38, and low-mass stars vastly outnumber O stars in a sampled IMF. Returning
	// early here also keeps log10 away from zero mass, where it would raise a floating-point trap.
	static constexpr amrex::Real Q_ion_min_mass = 15.0;

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
		const amrex::Real x = std::log10(mass_in_solar);
		const amrex::Real log_Q = Q_ion_c0 + (x * (Q_ion_c1 + (x * (Q_ion_c2 + (x * Q_ion_c3)))));
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
