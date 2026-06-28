#ifndef PARTICLE_UPDATE_HPP_
#define PARTICLE_UPDATE_HPP_

#include <algorithm>

#include "AMReX_BLProfiler.H"
#include "AMReX_Extension.H"
#include "AMReX_MultiFab.H"
#include "AMReX_REAL.H"
#include "fundamental_constants.H"
#include "hydro/hydro_system.hpp"
#include "math/quadrature.hpp"
#include "particle_chemical_yield.hpp"
#include "particle_deposition.hpp"
#include "particle_radiation.hpp"
#include "particle_types.hpp"
#include "particle_utils.hpp"
#include "physics_info.hpp"
#include "starparticle_radiation.hpp"

namespace quokka
{

// Forward declaration required so ParticlePropertyUpdateBase can call the (potentially specialized)
// updateProperties without needing the full definition at the point of its own definition.
template <ParticleType particleType> struct ParticlePropertyUpdateTraits;

// Base class that holds the single shared container-level loop.
// Calls ParticlePropertyUpdateTraits<particleType>::updateProperties per particle, which resolves
// to whichever specialization (or the default no-op) is appropriate for particleType.
// Specializations that need GPU tables set them in a global before calling applyUpdate.
template <ParticleType particleType> struct ParticlePropertyUpdateBase {
	template <typename problem_t, typename ContainerType>
	static void updateParticleProperties(ContainerType *container, amrex::Real current_time, amrex::Real dt) noexcept
	{
		const BL_PROFILE("ParticlePropertyUpdateTraits::updateParticleProperties()");
		if (container == nullptr) {
			return;
		}
		applyUpdate<problem_t, ContainerType>(container, current_time, dt);
	}

      public:
	template <typename problem_t, typename ContainerType>
	static void applyUpdate(ContainerType *container, amrex::Real current_time, amrex::Real dt) noexcept
	{
		constexpr int nGroups = Physics_Traits<problem_t>::nGroups;
		// Apply the updater to all particles across all levels
		for (int lev = 0; lev <= container->finestLevel(); ++lev) {
			for (typename ContainerType::ParIterType pIter(*container, lev); pIter.isValid(); ++pIter) {
				auto &particles = pIter.GetArrayOfStructs();
				auto *pData = particles().data();
				const amrex::Long np = pIter.numParticles();

				amrex::ParallelFor(np, [=] AMREX_GPU_DEVICE(int64_t idx) {
					auto &p = pData[idx]; // NOLINT
					ParticlePropertyUpdateTraits<particleType>::template updateProperties<problem_t, typename ContainerType::ParticleType,
													      nGroups>(p, current_time, dt);
				});
			}
		}
	}
};

// Traits class for specializing the per-particle update. Inherits updateParticleProperties from the base.
// Specializations only need to override updateProperties (and, for particle types that use luminosity
// tables, override updateParticleProperties to load the tables into g_device_luminosity_tables first).
template <ParticleType particleType> struct ParticlePropertyUpdateTraits : ParticlePropertyUpdateBase<particleType> {
	// Default per-particle implementation - does nothing
	template <typename problem_t, typename ParticleType, int Nout>
	AMREX_GPU_DEVICE AMREX_FORCE_INLINE static void updateProperties(ParticleType & /*p*/, amrex::Real /*current_time*/, amrex::Real /*dt*/) noexcept
	{
		// Default implementation does nothing
	}

	// Default container-level update - does nothing (overrides base class active default)
	template <typename problem_t, typename ContainerType>
	static void updateParticleProperties(ContainerType * /*container*/, amrex::Real /*current_time*/, amrex::Real /*dt*/) noexcept
	{
		// Default implementation does nothing
	}

	template <typename problem_t, typename ContainerType>
	static void updateChemicalFeedback(ContainerType * /*container*/, amrex::MultiFab & /*state*/, int /*lev*/, amrex::Real /*current_time*/,
					   amrex::Real /*dt*/) noexcept
	{
		// Default implementation does nothing
	}
};

// Specialization for StochasticStellarPop particles: updates luminosity via table interpolation.
// Overrides updateParticleProperties to gate the update on luminosity tables being initialized,
// and to load the tables into g_device_luminosity_tables before launching the GPU kernel.
template <> struct ParticlePropertyUpdateTraits<ParticleType::StochasticStellarPop> : ParticlePropertyUpdateBase<ParticleType::StochasticStellarPop> {
	template <typename problem_t, typename ContainerType>
	static void updateParticleProperties(ContainerType *container, amrex::Real current_time, amrex::Real dt) noexcept
	{
		const BL_PROFILE("ParticlePropertyUpdateTraits::updateParticleProperties()");
		if (container == nullptr) {
			return;
		}

		constexpr int nGroups = Physics_Traits<problem_t>::nGroups;
		auto *host_tables_ptr = quokka::g_luminosity_tables_ptr<nGroups>;

		// Only proceed if tables are initialized
		if (host_tables_ptr != nullptr && host_tables_ptr->is_initialized()) {
			g_device_luminosity_tables<nGroups> = host_tables_ptr->const_tables();
			applyUpdate<problem_t, ContainerType>(container, current_time, dt);
		}
	}

	template <typename problem_t, typename ParticleType, int Nout>
	AMREX_GPU_DEVICE AMREX_FORCE_INLINE static void updateProperties(ParticleType &p, amrex::Real current_time, amrex::Real /*dt*/) noexcept
	{
		LuminosityUpdate::updateLuminosity<problem_t, ParticleType, Nout>(p, current_time);
	}

	template <typename problem_t, typename ContainerType>
	static void updateChemicalFeedback(ContainerType *container, amrex::MultiFab &state, int lev, amrex::Real time, amrex::Real dt) noexcept
	{
		const BL_PROFILE("ParticlePropertyUpdateTraits::updateChemicalFeedback()");
		if (container == nullptr) {
			return;
		}

		if constexpr (Physics_Traits<problem_t>::numPassiveScalars <= 0) {
			return;
		}

		if (!enable_chemical_feedback) {
			return;
		}

		const int nPassive = Physics_Traits<problem_t>::numPassiveScalars;
		const int scalar_offset = std::max(0, chemical_scalar_offset);
		const int nchem = std::max(0, std::min(chemical_num_scalars, nPassive - scalar_offset));
		if (nchem <= 0) {
			return;
		}
		const bool use_chemical_tables = use_table_driven_chemical_yield && ChemicalYieldLookup::isLoaded();
		ChemicalYieldLookup::ChemicalYieldGpuConstTables yield_tables{};
		if (use_chemical_tables) {
			yield_tables = ChemicalYieldLookup::constTables();
		}

		amrex::MultiFab state_buffer(state.boxArray(), state.DistributionMap(), state.nComp() + 1, state.nGrow());
		state_buffer.setVal(0.0);

		for (typename ContainerType::ParIterType pti(*container, lev); pti.isValid(); ++pti) {
			auto &particles = pti.GetArrayOfStructs();
			auto *pData = particles().data();
			const amrex::Long np = pti.numParticles();

			const auto &local_state = state_buffer.array(pti);
			const auto &geom = container->Geom(lev);
			const auto plo = geom.ProbLoArray();
			const auto dxi = geom.InvCellSizeArray();
			const amrex::Real vol_inverse = AMREX_D_TERM(dxi[0], *dxi[1], *dxi[2]);
			const int chem_base = StochasticStellarPopParticleChemistryBaseIdx<problem_t>();

			constexpr int W_stencil_N = 2;
			constexpr int W_stencil_width = 2 * W_stencil_N + 1;
			constexpr amrex::Real W_cutoff_r2 = static_cast<amrex::Real>(W_stencil_N * W_stencil_N);
			constexpr amrex::Real W_inv_N = 1.0 / static_cast<amrex::Real>(W_stencil_N);

			amrex::ParallelFor(np, [=] AMREX_GPU_DEVICE(int64_t idx) {
				auto &p = pData[idx]; // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)

				const amrex::Real mass_birth = std::max<amrex::Real>(0.0, p.rdata(StochasticStellarPopParticleMassAtBirthIdx));
				const amrex::Real mass_birth_msun = mass_birth / C::M_solar;
				const amrex::Real age = time - p.rdata(StochasticStellarPopParticleBirthTimeIdx);
				const int stage = p.idata(StochasticStellarPopParticleStageIdx);
				amrex::ParticleInterpolator::WendlandC2<W_stencil_N> interp(p, plo, dxi);

				for (int n = 0; n < nchem; ++n) {
					const amrex::Real birth_iso_abundance = std::max<amrex::Real>(0.0, p.rdata(chem_base + n));
					const amrex::Real z_lookup = std::max<amrex::Real>(1.0e-12, stellar_metallicity_fraction);

					amrex::Real y_wr = 0.0;
					if (enable_WR_metal) {
						const bool wr_stage = ((stage == static_cast<int>(StellarEvolutionStage::SNProgenitor)) ||
								       (stage == static_cast<int>(StellarEvolutionStage::HighMassNonExploding))) &&
								      (mass_birth_msun >= 9.0);
						const amrex::Real wr_lifetime =
						    p.rdata(StochasticStellarPopParticleDeathTimeIdx) - p.rdata(StochasticStellarPopParticleBirthTimeIdx);
						const bool wr_active = (age < wr_lifetime) && (wr_lifetime > 0.0);
						if (wr_stage && wr_active) {
							if (use_chemical_tables) {
								const amrex::Real wr_total_frac =
								    ChemicalYieldLookup::queryYieldFraction(yield_tables, 1, n, mass_birth_msun, z_lookup);
								const amrex::Real age_begin = std::max<amrex::Real>(0.0, age);
								const amrex::Real age_end = std::min<amrex::Real>(wr_lifetime, age_begin + dt);
								const amrex::Real f_begin = ChemicalYieldLookup::queryWRMassLossCumulativeFraction(
								    yield_tables, age_begin, mass_birth_msun);
								const amrex::Real f_end = ChemicalYieldLookup::queryWRMassLossCumulativeFraction(
								    yield_tables, age_end, mass_birth_msun);
								const amrex::Real delta_fraction = std::max<amrex::Real>(0.0, f_end - f_begin);
								y_wr = std::max<amrex::Real>(0.0, wr_total_frac * mass_birth * delta_fraction);
							} else {
								const amrex::Real wr_window = std::max<amrex::Real>(0.0, wr_lifetime - wr_age_start);
								if (age >= wr_age_start && wr_window > 0.0) {
									const amrex::Real baseline_wr_rate_per_mass = birth_iso_abundance / wr_window;
									y_wr = std::max<amrex::Real>(
									    0.0, (baseline_wr_rate_per_mass + wr_metal_yield_rate_per_mass) * mass_birth * dt);
								}
							}
						}
					}

					const amrex::Real total_mass = y_wr;
					if (total_mass <= 0.0) {
						continue;
					}

					const int total_comp = HydroSystem<problem_t>::scalar0_index + scalar_offset + n;
					const int nz_loop = (AMREX_SPACEDIM >= 3) ? W_stencil_width : 1;
					const int ny_loop = (AMREX_SPACEDIM >= 2) ? W_stencil_width : 1;

					for (int kk = 0; kk < nz_loop; ++kk) {
						const amrex::Real dz =
						    (AMREX_SPACEDIM >= 3) ? static_cast<amrex::Real>(kk - W_stencil_N) + 0.5 - interp.frac[2] : 0.0;
						for (int jj = 0; jj < ny_loop; ++jj) {
							const amrex::Real dy =
							    (AMREX_SPACEDIM >= 2) ? static_cast<amrex::Real>(jj - W_stencil_N) + 0.5 - interp.frac[1] : 0.0;
							for (int ii = 0; ii < W_stencil_width; ++ii) {
								const amrex::Real dx = static_cast<amrex::Real>(ii - W_stencil_N) + 0.5 - interp.frac[0];
								const amrex::Real r2 = AMREX_D_TERM(dx * dx, +dy * dy, +dz * dz);
								if (r2 <= W_cutoff_r2) {
									const amrex::Real wt = kernel_wendland_c2(std::sqrt(r2) * W_inv_N) * interp.inv_norm;
									const amrex::Real total_val = wt * total_mass * vol_inverse;
									// WR metals are injected with the stellar-wind kernel, matching the continuous WR
									// feedback channel.
									amrex::Gpu::Atomic::AddNoRet(&local_state(interp.index[0] + ii, interp.index[1] + jj,
														  interp.index[2] + kk, total_comp),
												     total_val);

									if (store_channel_fields && enable_WR_metal && y_wr > 0.0) {
										const int wr_comp =
										    HydroSystem<problem_t>::scalar0_index + scalar_offset + 2 * nchem + n;
										if (wr_comp < HydroSystem<problem_t>::scalar0_index + nPassive) {
											amrex::Gpu::Atomic::AddNoRet(
											    &local_state(interp.index[0] + ii, interp.index[1] + jj,
													 interp.index[2] + kk, wr_comp),
											    wt * y_wr * vol_inverse);
										}
									}
								}
							}
						}
					}
				}
			});
		}

		state_buffer.SumBoundary(container->Geom(lev).periodicity());
		ParticleUtils::roundoffMultiFab(state_buffer);
		state.plus(state_buffer, 0, state.nComp(), 0);
	}
};

#if AMREX_SPACEDIM == 3
// Specialization for Star particles: dispatches to the modular stellar-evolution framework.
// Stellar models own their internal tables, so no gpu_tables plumbing is needed.
// Inherits updateParticleProperties from the base class (which calls applyUpdate).
template <> struct ParticlePropertyUpdateTraits<ParticleType::Star> : ParticlePropertyUpdateBase<ParticleType::Star> {
	template <typename problem_t, typename ParticleType, int Nout>
	AMREX_GPU_DEVICE AMREX_FORCE_INLINE static void updateProperties(ParticleType &p, amrex::Real current_time, amrex::Real dt) noexcept
	{
		StellarUpdate::updateStellarProperties<problem_t, ParticleType, Nout>(p, current_time, dt);
	}
};
#endif // AMREX_SPACEDIM == 3

} // namespace quokka

#endif // PARTICLE_UPDATE_HPP_
