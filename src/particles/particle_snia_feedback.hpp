#ifndef PARTICLE_SNIA_FEEDBACK_HPP_
#define PARTICLE_SNIA_FEEDBACK_HPP_

#include "AMReX_GpuContainers.H"
#include "AMReX_ParallelDescriptor.H"
#include "particles/particle_deposition.hpp"

#include <format>
#include <limits>
#include <utility>
#include <vector>

namespace quokka::SNFeedbackUtils
{
struct SNFeedbackEvent {
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> position{};
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> velocity{};
};

inline auto replicatedEventCount(const std::vector<SNFeedbackEvent> &events, const amrex::Geometry &geom) -> int
{
	AMREX_ALWAYS_ASSERT(events.size() <= static_cast<std::size_t>(std::numeric_limits<int>::max()));
	const int count = static_cast<int>(events.size());
	int min_count = count;
	int max_count = count;
	amrex::ParallelDescriptor::ReduceIntMin(min_count);
	amrex::ParallelDescriptor::ReduceIntMax(max_count);
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(min_count == max_count, "Pass the same SNIa event list on every MPI rank");
	const auto lower = geom.ProbLoArray();
	const auto upper = geom.ProbHiArray();
	for (const auto &event : events) {
		for (int direction = 0; direction < AMREX_SPACEDIM; ++direction) {
			AMREX_ALWAYS_ASSERT(event.position[direction] >= lower[direction] && event.position[direction] < upper[direction]);
		}
	}
	return count;
}

template <typename problem_t>
auto depositSNFeedbackEvents(amrex::MultiFab &state, std::array<amrex::MultiFab, AMREX_SPACEDIM> const *state_fc, amrex::Geometry const &geom,
			     std::vector<SNFeedbackEvent> const &events, amrex::Real ejecta_mass, amrex::Real blast_energy) -> std::pair<int, amrex::Real>
{
	const BL_PROFILE("SNFeedbackUtils::depositSNFeedbackEvents()");
	static_assert(AMREX_SPACEDIM == 3);
	static_assert(SN_stencil_size <= 3);
	const int num_events = replicatedEventCount(events, geom);
	if (num_events == 0) {
		return {0, 0.0};
	}

	amrex::Gpu::DeviceVector<SNFeedbackEvent> events_d(events.size());
	amrex::Gpu::copy(amrex::Gpu::hostToDevice, events.begin(), events.end(), events_d.begin());
	const auto *events_ptr = events_d.data();
	AMREX_ALWAYS_ASSERT(state.nGrow() >= SN_stencil_size);
	AMREX_ALWAYS_ASSERT(ejecta_mass > 0.0 && blast_energy >= 0.0);

	amrex::MultiFab state_buffer(state.boxArray(), state.DistributionMap(), state.nComp() + 1, state.nGrow());
	state_buffer.setVal(0.0);

	const SNScheme SN_scheme_d = SN_scheme;
	const bool SN_smooth_gas_velocity_d = SN_smooth_gas_velocity;
	const amrex::Real p_term_exponent = quokka::SN_p_term_exponent;
	const bool enable_chemical_feedback_d = enable_chemical_feedback;
	const amrex::Real p_snr_0 = quokka::SN_p_term_Msunkmps * C::M_solar * 1.0e5;
	const amrex::Real scalar_yield_per_SN_d = scalar_yield_per_SN;

	amrex::Gpu::Buffer<amrex::Real> max_velocity_buffer({0.0});
	amrex::Real *p_max_velocity = max_velocity_buffer.data();
	amrex::Gpu::Buffer<int> sn_count_buffer({0});
	int *p_sn_count = sn_count_buffer.data();

	const auto plo = geom.ProbLoArray();
	const auto dxi = geom.InvCellSizeArray();
	const auto dx = geom.CellSizeArray();
	const amrex::Real vol = AMREX_D_TERM(dx[0], *dx[1], *dx[2]);
	const amrex::Real vol_inverse = 1.0 / vol;
	constexpr int stencil_width = 2 * SN_stencil_size + 1;
	constexpr int stencil_volume = stencil_width * stencil_width * stencil_width;

	for (amrex::MFIter mfi(state); mfi.isValid(); ++mfi) {
		const amrex::Box &box = mfi.validbox();
		auto const &local_state = state.array(mfi);
		auto const &local_buffer = state_buffer.array(mfi);

		amrex::ParallelFor(num_events, [=] AMREX_GPU_DEVICE(int event_index) {
			auto const &event = events_ptr[event_index]; // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
			const int ix = static_cast<int>(amrex::Math::floor((event.position[0] - plo[0]) * dxi[0]));
			const int iy = static_cast<int>(amrex::Math::floor((event.position[1] - plo[1]) * dxi[1]));
			const int iz = static_cast<int>(amrex::Math::floor((event.position[2] - plo[2]) * dxi[2]));
			if (!box.contains(amrex::IntVect(AMREX_D_DECL(ix, iy, iz)))) {
				return;
			}

			amrex::Real avg_density = 0.0;
			for (int ii = ix - SN_stencil_size; ii <= ix + SN_stencil_size; ++ii) {
				for (int jj = iy - SN_stencil_size; jj <= iy + SN_stencil_size; ++jj) {
					for (int kk = iz - SN_stencil_size; kk <= iz + SN_stencil_size; ++kk) {
						const int iii = std::abs(ii - ix);
						const int jjj = std::abs(jj - iy);
						const int kkk = std::abs(kk - iz);
						avg_density +=
						    stencil_weights_gpu[iii][jjj][kkk] * local_state(ii, jj, kk, HydroSystem<problem_t>::density_index);
					}
				}
			}

			const amrex::Real vx = event.velocity[0];
			const amrex::Real vy = event.velocity[1];
			const amrex::Real vz = event.velocity[2];
			if (SN_scheme_d == SNScheme::SN_thermal_only) {
				const amrex::Real SN_kin_energy = 0.5 * ejecta_mass * (vx * vx + vy * vy + vz * vz);
				depositThermalSNR<problem_t>(local_buffer, ix, iy, iz, ejecta_mass, blast_energy, SN_kin_energy, vx, vy, vz, vol_inverse,
							     stencil_weights_gpu, scalar_yield_per_SN_d, enable_chemical_feedback_d);
			} else {
				depositThermalKineticMomentumSNR<problem_t>(
				    local_state, local_buffer, ix, iy, iz, stencil_volume, event.position[0], event.position[1], event.position[2], ejecta_mass,
				    blast_energy, p_snr_0, p_term_exponent, vol_inverse, stencil_weights_gpu, avg_density, vol, dx, plo, SN_scheme_d, vx, vy,
				    vz, SN_smooth_gas_velocity_d, scalar_yield_per_SN_d, enable_chemical_feedback_d);
			}
			amrex::Gpu::Atomic::AddNoRet(p_sn_count, 1);
		});
	}

	state_buffer.SumBoundary(geom.periodicity());
	ParticleUtils::roundoffMultiFab(state_buffer);
	addBufferToState<problem_t>(state, state_fc, state_buffer, SN_scheme_d, p_max_velocity);

	auto *h_max_velocity = max_velocity_buffer.copyToHost();
	amrex::Real max_velocity = h_max_velocity[0];
	amrex::ParallelDescriptor::ReduceRealMax(max_velocity);
	auto *h_sn_count = sn_count_buffer.copyToHost();
	int sn_count = h_sn_count[0];
	amrex::ParallelDescriptor::ReduceIntSum(sn_count);
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(sn_count == num_events, "SNIa events must each belong to exactly one valid grid cell");

	return {sn_count, max_velocity};
}
} // namespace quokka::SNFeedbackUtils

namespace quokka::ChemicalFeedbackUtils
{
template <typename problem_t>
void depositSNIaChemicalFeedback(amrex::MultiFab &state, amrex::Geometry const &geom, std::vector<SNFeedbackUtils::SNFeedbackEvent> const &events,
				 amrex::Real ejecta_mass)
{
	const BL_PROFILE("ChemicalFeedbackUtils::depositSNIaChemicalFeedback()");

	if (!enable_chemical_feedback || !enable_SNIa_metal) {
		return;
	}
	if constexpr (Physics_Traits<problem_t>::numPassiveScalars <= 0) {
		return;
	}

	static_assert(AMREX_SPACEDIM == 3);
	const int num_events = SNFeedbackUtils::replicatedEventCount(events, geom);
	if (num_events == 0) {
		return;
	}
	AMREX_ALWAYS_ASSERT(state.nGrow() >= SN_stencil_size);
	AMREX_ALWAYS_ASSERT(ejecta_mass > 0.0);
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(use_table_driven_chemical_yield && ChemicalYieldLookup::isLoaded(), "SNIa feedback requires loaded yield tables");
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(ChemicalYieldLookup::channel_enabled[ChemicalYieldLookup::snia_channel_index] != 0,
					 "SNIa must be included in chemical_tracked_channels");

	const int nPassive = Physics_Traits<problem_t>::numPassiveScalars;
	const int scalar_offset = std::max(0, chemical_scalar_offset);
	const int nchem = chemical_num_scalars;
	AMREX_ALWAYS_ASSERT(nchem == ChemicalYieldLookup::num_tracked_isotopes);
	AMREX_ALWAYS_ASSERT(scalar_offset + nchem <= nPassive);
	if (nchem <= 0) {
		return;
	}
	if (store_channel_fields) {
		const int required_scalars = scalar_offset + (ChemicalYieldLookup::max_tracked_channels + 1) * nchem;
		AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
		    required_scalars <= nPassive,
		    std::format("SNIa channel fields require at least {} passive scalars, but this problem has {}", required_scalars, nPassive));
	}

	const auto yield_tables = ChemicalYieldLookup::constTables();
	amrex::Gpu::DeviceVector<SNFeedbackUtils::SNFeedbackEvent> events_d(events.size());
	amrex::Gpu::copy(amrex::Gpu::hostToDevice, events.begin(), events.end(), events_d.begin());
	const auto *events_ptr = events_d.data();
	AMREX_ALWAYS_ASSERT(num_events <= std::numeric_limits<int>::max() / nchem);

	amrex::MultiFab state_buffer(state.boxArray(), state.DistributionMap(), state.nComp() + 1, state.nGrow());
	state_buffer.setVal(0.0);

	const auto plo = geom.ProbLoArray();
	const auto dxi = geom.InvCellSizeArray();
	const auto dx = geom.CellSizeArray();
	const amrex::Real vol = AMREX_D_TERM(dx[0], *dx[1], *dx[2]);
	const amrex::Real vol_inverse = 1.0 / vol;
	const amrex::Real ejecta_mass_msun = ejecta_mass / C::M_solar;
	const amrex::Real metallicity_lookup = std::max<amrex::Real>(1.0e-12, stellar_metallicity_fraction);
	const bool store_channel_fields_local = store_channel_fields;

	for (amrex::MFIter mfi(state); mfi.isValid(); ++mfi) {
		const amrex::Box &box = mfi.validbox();
		auto const &local_buffer = state_buffer.array(mfi);

		amrex::ParallelFor(num_events * nchem, [=] AMREX_GPU_DEVICE(int linear_index) {
			const int event_index = linear_index / nchem;
			const int n = linear_index - event_index * nchem;
			auto const &event = events_ptr[event_index]; // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
			const int ix = static_cast<int>(amrex::Math::floor((event.position[0] - plo[0]) * dxi[0]));
			const int iy = static_cast<int>(amrex::Math::floor((event.position[1] - plo[1]) * dxi[1]));
			const int iz = static_cast<int>(amrex::Math::floor((event.position[2] - plo[2]) * dxi[2]));
			if (!box.contains(amrex::IntVect(AMREX_D_DECL(ix, iy, iz)))) {
				return;
			}

			const amrex::Real yield_fraction = ChemicalYieldLookup::queryYieldFraction(yield_tables, ChemicalYieldLookup::snia_channel_index, n,
												   ejecta_mass_msun, metallicity_lookup);
			const amrex::Real yield_mass = std::max<amrex::Real>(0.0, yield_fraction * ejecta_mass);
			if (yield_mass <= 0.0) {
				return;
			}
			const int total_comp = HydroSystem<problem_t>::scalar0_index + scalar_offset + n;
			depositSNStencil(local_buffer, ix, iy, iz, total_comp, yield_mass, vol_inverse);

			if (store_channel_fields_local) {
				const int snia_comp =
				    HydroSystem<problem_t>::scalar0_index + scalar_offset + (ChemicalYieldLookup::snia_channel_index + 1) * nchem + n;
				if (snia_comp < HydroSystem<problem_t>::scalar0_index + nPassive) {
					depositSNStencil(local_buffer, ix, iy, iz, snia_comp, yield_mass, vol_inverse);
				}
			}
		});
	}

	state_buffer.SumBoundary(geom.periodicity());
	ParticleUtils::roundoffMultiFab(state_buffer);
	state.plus(state_buffer, 0, state.nComp(), 0);
}
} // namespace quokka::ChemicalFeedbackUtils

#endif
