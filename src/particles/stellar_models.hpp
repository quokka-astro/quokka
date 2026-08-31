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

	// Vacca, Garmany & Shull (1996) fitting formula for the hydrogen-ionizing photon rate:
	//   log10 Q [1/s] = 49 + log10(3.12e-8) + 4.91 * log10(m / M_sun)
	// The fit was derived for 20 <= m/M_sun <= 30; we apply it at all masses as a toy model,
	// which is adequate because the ionizing budget is dominated by the most massive stars.
	static constexpr amrex::Real Q_ion_coeff = 3.12e-8 * 1.0e49; // 1/s, Q at m = 1 M_sun
	static constexpr amrex::Real Q_ion_exponent = 4.91;

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

	// Hydrogen-ionizing photon rate (1/s) for a star of the given mass. See Q_ion_coeff above.
	AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE static auto ionizingPhotonRate(amrex::Real mass) -> amrex::Real
	{
		if (mass <= 0.0) {
			return 0.0;
		}
		return Q_ion_coeff * std::pow(mass / C::M_solar, Q_ion_exponent);
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
		// not been set yet. Q is deliberately NOT refreshed on later calls, because it is fixed at
		// the birth mass and must not drift as the particle accretes.
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
