#ifndef PARTICLE_UPDATE_HPP_
#define PARTICLE_UPDATE_HPP_

#include <algorithm>

#include "AMReX_BLProfiler.H"
#include "AMReX_Extension.H"
#include "AMReX_MultiFab.H"
#include "AMReX_REAL.H"
#include "fundamental_constants.H"
#include "hydro/hydro_system.hpp"
#include "particle_chemical_yield.hpp"
#include "particle_radiation.hpp"
#include "particle_types.hpp"
#include "physics_info.hpp"

namespace quokka
{

// Forward declaration required so ParticlePropertyUpdateBase can call the (potentially specialized)
// updateProperties without needing the full definition at the point of its own definition.
template <ParticleType particleType> struct ParticlePropertyUpdateTraits;

// Base class that holds the single shared container-level loop.
// Calls ParticlePropertyUpdateTraits<particleType>::updateProperties per particle, which resolves
// to whichever specialization (or the default no-op) is appropriate for particleType.
// Specializations that require luminosity tables should override updateParticleProperties to
// load the tables before calling applyUpdate.
template <ParticleType particleType> struct ParticlePropertyUpdateBase {
	template <typename problem_t, typename ContainerType>
	static void updateParticleProperties(ContainerType *container, amrex::Real current_time, amrex::Real dt) noexcept
	{
		const BL_PROFILE("ParticlePropertyUpdateTraits::updateParticleProperties()");
		if (container == nullptr) {
			return;
		}

		constexpr int nGroups = Physics_Traits<problem_t>::nGroups;
		// Default: pass empty tables (unused by per-particle functions that don't need them)
		LuminosityGpuConstTables<nGroups> const gpu_tables{};
		applyUpdate<problem_t, ContainerType>(container, current_time, dt, gpu_tables);
	}

	template <typename problem_t, typename ContainerType>
	static void updateChemicalFeedback(ContainerType * /*container*/, amrex::MultiFab & /*state*/, int /*lev*/, amrex::Real /*current_time*/, amrex::Real /*dt*/) noexcept
	{
		// Default implementation does nothing.
	}

      public:
	template <typename problem_t, typename ContainerType>
	static void applyUpdate(ContainerType *container, amrex::Real current_time, amrex::Real dt,
				LuminosityGpuConstTables<Physics_Traits<problem_t>::nGroups> const &gpu_tables) noexcept
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
													      nGroups>(p, current_time, dt, gpu_tables);
				});
			}
		}
	}
};

// Traits class for specializing the per-particle update. Inherits updateParticleProperties from the base.
// Specializations only need to override updateProperties.
template <ParticleType particleType> struct ParticlePropertyUpdateTraits : ParticlePropertyUpdateBase<particleType> {
	// Default per-particle implementation - does nothing
	template <typename problem_t, typename ParticleType, int Nout>
	AMREX_GPU_DEVICE AMREX_FORCE_INLINE static void updateProperties(ParticleType & /*p*/, amrex::Real /*current_time*/, amrex::Real /*dt*/,
									 LuminosityGpuConstTables<Nout> const & /*gpu_tables*/) noexcept
	{
		// Default implementation does nothing
	}

	// Default container-level update - does nothing
	template <typename problem_t, typename ContainerType>
	static void updateParticleProperties(ContainerType * /*container*/, amrex::Real /*current_time*/, amrex::Real /*dt*/) noexcept
	{
		// Default implementation does nothing
	}

	template <typename problem_t, typename ContainerType>
	static void updateChemicalFeedback(ContainerType * /*container*/, amrex::MultiFab & /*state*/, int /*lev*/, amrex::Real /*current_time*/, amrex::Real /*dt*/) noexcept
	{
		// Default implementation does nothing
	}
};

// Specialization for StochasticStellarPop particles: updates luminosity via table interpolation.
// Overrides updateParticleProperties to gate the update on luminosity tables being initialized.
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
			auto const gpu_tables = host_tables_ptr->const_tables();
			applyUpdate<problem_t, ContainerType>(container, current_time, dt, gpu_tables);
		}
	}

	template <typename problem_t, typename ParticleType, int Nout>
	AMREX_GPU_DEVICE AMREX_FORCE_INLINE static void updateProperties(ParticleType &p, amrex::Real current_time, amrex::Real /*dt*/,
									 LuminosityGpuConstTables<Nout> const &gpu_tables) noexcept
	{
		LuminosityUpdate::updateLuminosity<problem_t>(p, current_time, gpu_tables);
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

		for (typename ContainerType::ParIterType pti(*container, lev); pti.isValid(); ++pti) {
			auto &particles = pti.GetArrayOfStructs();
			auto *pData = particles().data();
			const amrex::Long np = pti.numParticles();

			const auto &local_state = state.array(pti);
			const auto &geom = container->Geom(lev);
			const auto plo = geom.ProbLoArray();
			const auto dxi = geom.InvCellSizeArray();
			const amrex::Real vol_inverse = AMREX_D_TERM(dxi[0], *dxi[1], *dxi[2]);
			const int chem_block_size = StochasticStellarPopParticleChemistryBlockSize<problem_t>();
			const int chem_base = StochasticStellarPopParticleChemistryBaseIdx<problem_t>();

			amrex::ParallelFor(np, [=] AMREX_GPU_DEVICE(int64_t idx) {
				auto &p = pData[idx]; // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)

				const amrex::Real mass_birth = std::max<amrex::Real>(0.0, p.rdata(StochasticStellarPopParticleMassAtBirthIdx));
				const amrex::Real mass_birth_msun = mass_birth / C::M_solar;
				const amrex::Real age = time - p.rdata(StochasticStellarPopParticleBirthTimeIdx);
				const int stage = p.idata(StochasticStellarPopParticleStageIdx);

				const int ix = static_cast<int>(amrex::Math::floor((p.pos(0) - plo[0]) * dxi[0]));
				const int iy = static_cast<int>(amrex::Math::floor((p.pos(1) - plo[1]) * dxi[1]));
				const int iz = static_cast<int>(amrex::Math::floor((p.pos(2) - plo[2]) * dxi[2]));

				for (int n = 0; n < nchem; ++n) {
					const amrex::Real birth_iso_abundance = std::max<amrex::Real>(0.0, p.rdata(chem_base + n));
					const amrex::Real z_wr_lookup = store_channel_fields ? std::max<amrex::Real>(1.0e-12, p.rdata(chem_base + 2 * chem_block_size + n))
					                                                   : std::max<amrex::Real>(1.0e-12, birth_iso_abundance);
					const amrex::Real z_agb_lookup = store_channel_fields ? std::max<amrex::Real>(1.0e-12, p.rdata(chem_base + 3 * chem_block_size + n))
					                                                     : std::max<amrex::Real>(1.0e-12, birth_iso_abundance);

					amrex::Real y_wr = 0.0;
					if (enable_WR_metal) {
						const bool wr_stage = (stage == static_cast<int>(StellarEvolutionStage::SNProgenitor)) ||
								      (stage == static_cast<int>(StellarEvolutionStage::HighMassNonExploding));
						const bool wr_active = (age >= wr_age_start) && (age < wr_age_end);
						if (wr_stage && wr_active) {
							const amrex::Real wr_window = std::max<amrex::Real>(0.0, wr_age_end - wr_age_start);
							amrex::Real wr_rate_per_mass = wr_metal_yield_rate_per_mass;
							if (use_table_driven_chemical_yield && ChemicalYieldLookup::isLoaded() && wr_window > 0.0) {
								wr_rate_per_mass = std::max<amrex::Real>(0.0, ChemicalYieldLookup::queryYieldFraction(1, n, mass_birth_msun, z_wr_lookup)) / wr_window;
							}
							const amrex::Real baseline_wr_rate_per_mass = (wr_window > 0.0) ? (birth_iso_abundance / wr_window) : 0.0;
							y_wr = std::max<amrex::Real>(0.0, (baseline_wr_rate_per_mass + wr_rate_per_mass) * mass_birth * dt);
						}
					}

					amrex::Real y_agb = 0.0;
					if (enable_AGB_metal) {
						const bool agb_stage = (stage == static_cast<int>(StellarEvolutionStage::LowMassComposite));
						const bool agb_active = (age >= agb_age_start) && (age < agb_age_end);
						if (agb_stage && agb_active) {
							const amrex::Real agb_window = std::max<amrex::Real>(0.0, agb_age_end - agb_age_start);
							amrex::Real agb_rate_per_mass = agb_metal_yield_rate_per_mass;
							if (use_table_driven_chemical_yield && ChemicalYieldLookup::isLoaded() && agb_window > 0.0) {
								agb_rate_per_mass = std::max<amrex::Real>(0.0, ChemicalYieldLookup::queryYieldFraction(2, n, mass_birth_msun, z_agb_lookup)) / agb_window;
							}
							const amrex::Real baseline_agb_rate_per_mass = (agb_window > 0.0) ? (birth_iso_abundance / agb_window) : 0.0;
							y_agb = std::max<amrex::Real>(0.0, (baseline_agb_rate_per_mass + agb_rate_per_mass) * mass_birth * dt);
						}
					}

					const amrex::Real total_density_injection = (y_wr + y_agb) * vol_inverse;
					const int total_comp = HydroSystem<problem_t>::scalar0_index + scalar_offset + n;
					amrex::Gpu::Atomic::AddNoRet(&local_state(ix, iy, iz, total_comp), total_density_injection);

					if (store_channel_fields) {
						if (enable_WR_metal) {
							const int wr_comp = HydroSystem<problem_t>::scalar0_index + scalar_offset + 2 * nchem + n;
							if (wr_comp < HydroSystem<problem_t>::scalar0_index + nPassive) {
								amrex::Gpu::Atomic::AddNoRet(&local_state(ix, iy, iz, wr_comp), y_wr * vol_inverse);
							}
						}
						if (enable_AGB_metal) {
							const int agb_comp = HydroSystem<problem_t>::scalar0_index + scalar_offset + 3 * nchem + n;
							if (agb_comp < HydroSystem<problem_t>::scalar0_index + nPassive) {
								amrex::Gpu::Atomic::AddNoRet(&local_state(ix, iy, iz, agb_comp), y_agb * vol_inverse);
							}
						}
					}
				}
			});
		}
	}
};

// // Specialization for StochasticStellarPop particles from a simple analytical formula
// // This is kept for debugging purpose.
// template <> struct ParticlePropertyUpdateTraits<ParticleType::StochasticStellarPop> {
// 	static constexpr double star_lum_per_M_solar = 4.0e33;

// 	template <typename problem_t, typename ParticleType>
// 	AMREX_GPU_DEVICE AMREX_FORCE_INLINE static void updateProperties(ParticleType &p, amrex::Real current_time) noexcept
// 	{
// 		const int mass_idx = StochasticStellarPopParticleMassIdx;
// 		const int birth_time_idx = StochasticStellarPopParticleBirthTimeIdx;
// 		const int lum_idx = StochasticStellarPopParticleLumIdx;
// 		const amrex::Real age = current_time - p.rdata(birth_time_idx);
// 		const amrex::Real mass = p.rdata(mass_idx);

// 		// A simple luminosity function for testing purpose. Keep it linear function of mass for easy answer
// 		// validation. L/(M / M_sun) = L_sun = 4e33 erg/s
// 		const double is_on = age < 1.0e14 ? 1.0 : 0.0; // 3 Myr

// 		// Update luminosity components (they are stored consecutively starting at lum_idx)
// 		for (int g = 0; g < Physics_Traits<problem_t>::nGroups; ++g) {
// 			const amrex::Real luminosity = star_lum_per_M_solar * (mass / C::M_solar) * (g + 1) * is_on; // erg / s
// 			p.rdata(lum_idx + g) = luminosity;
// 		}
// 	}
// };

} // namespace quokka

#endif // PARTICLE_UPDATE_HPP_
