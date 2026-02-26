#ifndef PARTICLE_PHOTOIONIZATION_HPP_
#define PARTICLE_PHOTOIONIZATION_HPP_

#include <algorithm>
#include <array>
#include <cmath>

#include "AMReX_Array.H"
#include "AMReX_Geometry.H"
#include "AMReX_GpuDevice.H"
#include "AMReX_GpuQualifiers.H"
#include "AMReX_Math.H"
#include "AMReX_MultiFab.H"

#include "fundamental_constants.H"
#include "hydro/hydro_system.hpp"
#include "particles/particle_types.hpp"
#include "util/DataTable.hpp"

namespace quokka::photoionization
{

template <typename problem_t, quokka::OutOfBounds oob_policy>
void FillTemperatureFloorFromStromgrenVolumes(quokka::StochasticStellarPopParticleContainer<problem_t> *stellar_particles, int lev, amrex::Real time,
					      amrex::BoxArray const &ba_lev, amrex::DistributionMapping const &dm_lev, amrex::Geometry const &geom_lev,
					      amrex::MultiFab const &state_cc, amrex::MultiFab &temp_floor_cc,
					      quokka::DataTableGpuConst<2, 1, oob_policy> const &qh0_table,
					      amrex::Real const mass_to_table_units = 1.0 / C::M_solar, amrex::Real const age_to_table_units = 1.0 / 3.15576e7,
					      bool const table_axes_are_mass_age = true, amrex::Real const alphaB = 2.6e-13,
					      amrex::Real const mean_particle_mass_mu = 1.27, amrex::Real const mH = 1.67e-24,
					      amrex::Real const ionized_temperature = 1.0e4, int max_neighbor_hops = 1,
					      amrex::Real const photon_luminosity_tolerance = 1.0)
{
#if AMREX_SPACEDIM != 3
	amrex::ignore_unused(stellar_particles, lev, time, ba_lev, dm_lev, geom_lev, state_cc, temp_floor_cc, qh0_table, mass_to_table_units,
			     age_to_table_units, table_axes_are_mass_age, alphaB, mean_particle_mass_mu, mH, ionized_temperature, max_neighbor_hops,
			     photon_luminosity_tolerance);
	return;
#else
	if (stellar_particles == nullptr) {
		return;
	}

	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(state_cc.boxArray() == temp_floor_cc.boxArray(), "state_cc and temp_floor_cc must have the same BoxArray.");
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(state_cc.DistributionMap() == temp_floor_cc.DistributionMap(),
					 "state_cc and temp_floor_cc must have the same DistributionMap.");
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(temp_floor_cc.nComp() >= 1, "temp_floor_cc must have at least one component.");

	auto const dx = geom_lev.CellSizeArray();
	auto const plo = geom_lev.ProbLoArray();
	auto const dxi = geom_lev.InvCellSizeArray();
	amrex::Real const cell_volume = dx[0] * dx[1] * dx[2];
	amrex::Real const n_to_rho = mean_particle_mass_mu * mH;

	// Hard cutoff on propagation radius: one iteration == one 26-neighbor cell hop.
	// Negative values are treated as zero to avoid accidental whole-domain sweeps.
	max_neighbor_hops = std::max(0, max_neighbor_hops);

	amrex::MultiFab photons_curr(ba_lev, dm_lev, 1, 0);
	amrex::MultiFab photons_esc(ba_lev, dm_lev, 1, 1);
	amrex::MultiFab photons_next(ba_lev, dm_lev, 1, 0);
	photons_curr.setVal(0.0);
	photons_esc.setVal(0.0);
	photons_next.setVal(0.0);

	// Deposit ionizing photon luminosity (photons/s) from individually sampled stars only.
	// We exclude LowMassComposite, SNRemnant, and Removed particles.
	for (quokka::StochasticStellarPopParticleIterator<problem_t> pti(*stellar_particles, lev); pti.isValid(); ++pti) {
		auto &particles = pti.GetArrayOfStructs();
		auto *pData = particles().data();
		auto const np = pti.numParticles();
		auto const src = photons_curr.array(pti);
		auto const box = pti.validbox();
		auto const lo = amrex::lbound(box);
		auto const hi = amrex::ubound(box);

		amrex::ParallelFor(np, [=] AMREX_GPU_DEVICE(int64_t idx) noexcept {
			auto const &p = pData[idx]; // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
			int const stage = p.idata(quokka::StochasticStellarPopParticleStageIdx);
			bool const is_individual_ionizing_star = (stage == static_cast<int>(quokka::StellarEvolutionStage::HighMassNonExploding)) ||
								 (stage == static_cast<int>(quokka::StellarEvolutionStage::SNProgenitor));
			if (!is_individual_ionizing_star) {
				return;
			}

			amrex::Real const age = time - p.rdata(quokka::StochasticStellarPopParticleBirthTimeIdx);
			if (age <= 0.0) {
				return;
			}

			amrex::Real const zams_mass = p.rdata(quokka::StochasticStellarPopParticleMassAtBirthIdx);
			if (zams_mass <= 0.0) {
				return;
			}

			amrex::Real const mass_coord = zams_mass * mass_to_table_units;
			amrex::Real const age_coord = age * age_to_table_units;
			std::array<amrex::Real, 2> point{};
			if (table_axes_are_mass_age) {
				point = {mass_coord, age_coord};
			} else {
				point = {age_coord, mass_coord};
			}

			amrex::Real const S = qh0_table.interpolate_single(point, 0);
			if (!(S > 0.0) || !std::isfinite(S)) {
				return;
			}

			int const i = static_cast<int>(amrex::Math::floor((p.pos(0) - plo[0]) * dxi[0]));
			int const j = static_cast<int>(amrex::Math::floor((p.pos(1) - plo[1]) * dxi[1]));
			int const k = static_cast<int>(amrex::Math::floor((p.pos(2) - plo[2]) * dxi[2]));

			if ((i < lo.x) || (i > hi.x) || (j < lo.y) || (j > hi.y) || (k < lo.z) || (k > hi.z)) {
				return;
			}

			amrex::Gpu::Atomic::AddNoRet(&src(i, j, k, 0), S);
		});
	}

	auto const state = state_cc.const_arrays();
	auto temp_floor = temp_floor_cc.arrays();

	// Overlapping H II regions are handled by superposition: all incoming photon luminosities
	// are summed into S_in for each cell before computing heating and escaped luminosity.
	for (int iter = 0; iter <= max_neighbor_hops; ++iter) {
		amrex::Real const max_incoming = photons_curr.norm0(0, 0, false);
		if (max_incoming <= photon_luminosity_tolerance) {
			break;
		}

		auto const photons_curr_arr = photons_curr.const_arrays();
		auto photons_esc_arr = photons_esc.arrays();

		amrex::ParallelFor(photons_curr, [=] AMREX_GPU_DEVICE(int nbx, int i, int j, int k) noexcept {
			amrex::Real const S_in = photons_curr_arr[nbx](i, j, k, 0);
			if (S_in <= 0.0) {
				photons_esc_arr[nbx](i, j, k, 0) = 0.0;
				return;
			}

			amrex::Real const rho = state[nbx](i, j, k, HydroSystem<problem_t>::density_index);
			amrex::Real const n = (rho > 0.0) ? (rho / n_to_rho) : 0.0;
			amrex::Real const S_absorb_cell = alphaB * n * n * cell_volume;

			amrex::Real heated_temp = ionized_temperature;
			amrex::Real S_esc = 0.0;
			if (S_absorb_cell > 0.0) {
				amrex::Real const frac = S_in / S_absorb_cell;
				heated_temp = ionized_temperature * amrex::min<amrex::Real>(1.0, frac);
				S_esc = amrex::max<amrex::Real>(0.0, S_in - S_absorb_cell);
			} else {
				S_esc = S_in;
			}

			temp_floor[nbx](i, j, k, 0) = amrex::max(temp_floor[nbx](i, j, k, 0), heated_temp);
			photons_esc_arr[nbx](i, j, k, 0) = S_esc;
		});

		photons_esc.FillBoundary(geom_lev.periodicity());

		auto const photons_esc_const = photons_esc.const_arrays();
		auto photons_next_arr = photons_next.arrays();
		constexpr amrex::Real inv_nnbr = 1.0 / 26.0;

		amrex::ParallelFor(photons_next, [=] AMREX_GPU_DEVICE(int nbx, int i, int j, int k) noexcept {
			amrex::Real S_next = 0.0;
			for (int kk = -1; kk <= 1; ++kk) {
				for (int jj = -1; jj <= 1; ++jj) {
					for (int ii = -1; ii <= 1; ++ii) {
						if ((ii == 0) && (jj == 0) && (kk == 0)) {
							continue;
						}
						S_next += photons_esc_const[nbx](i + ii, j + jj, k + kk, 0) * inv_nnbr;
					}
				}
			}
			photons_next_arr[nbx](i, j, k, 0) = S_next;
		});

		std::swap(photons_curr, photons_next);
		photons_next.setVal(0.0);
	}
#endif
}

} // namespace quokka::photoionization

#endif // PARTICLE_PHOTOIONIZATION_HPP_
