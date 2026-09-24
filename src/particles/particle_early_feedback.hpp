#ifndef PARTICLE_EARLY_FEEDBACK_HPP_
#define PARTICLE_EARLY_FEEDBACK_HPP_

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>

#include "AMReX_Array.H"
#include "AMReX_Array4.H"
#include "AMReX_BLProfiler.H"
#include "AMReX_BLassert.H"
#include "AMReX_Box.H"
#include "AMReX_Gpu.H"
#include "AMReX_GpuContainers.H"
#include "AMReX_Math.H"
#include "AMReX_MultiFab.H"
#include "AMReX_ParallelDescriptor.H"
#include "AMReX_REAL.H"

#include "hydro/hydro_system.hpp"
#include "particles/particle_types.hpp"
#include "particles/particle_utils.hpp"
#include "physics_info.hpp"

namespace quokka
{

struct EarlyFeedbackStats {
	int active_particles = 0;
	int clipped_cells = 0;
	amrex::Real scalar_momentum = 0.0;
	amrex::Real min_velocity_scale = 1.0;
	amrex::Real max_velocity = 0.0;
};

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto earlyFeedbackMomentumIncrement(amrex::Real step_time, amrex::Real dt, amrex::Real birth_time,
									     amrex::Real birth_mass, amrex::Real p0, amrex::Real t_fb,
									     amrex::Real alpha) noexcept -> amrex::Real
{
	if (!(birth_mass > 0.0) || !(dt > 0.0) || !(p0 > 0.0) || !(t_fb > 0.0) || !(alpha > 0.25) || !amrex::Math::isfinite(step_time) ||
	    !amrex::Math::isfinite(dt) || !amrex::Math::isfinite(birth_time) || !amrex::Math::isfinite(birth_mass) || !amrex::Math::isfinite(p0) ||
	    !amrex::Math::isfinite(t_fb) || !amrex::Math::isfinite(alpha)) {
		return 0.0;
	}

	const amrex::Real age_begin = step_time - birth_time;
	const amrex::Real age_end = step_time + dt - birth_time;
	if (age_end <= 0.0 || age_begin >= t_fb) {
		return 0.0;
	}

	const amrex::Real x0 = amrex::min(amrex::max(age_begin / t_fb, static_cast<amrex::Real>(0.0)), static_cast<amrex::Real>(1.0));
	const amrex::Real x1 = amrex::min(amrex::max(age_end / t_fb, static_cast<amrex::Real>(0.0)), static_cast<amrex::Real>(1.0));
	const amrex::Real exponent = (4.0 * alpha) - 1.0;
	const amrex::Real increment = alpha * p0 * birth_mass * (std::pow(x1, exponent) - std::pow(x0, exponent));
	return std::max(increment, static_cast<amrex::Real>(0.0));
}

namespace EarlyFeedbackUtils
{

constexpr int stencil_size = ParticleUtils::stencil_size;

template <typename ContainerType, typename problem_t>
void depositToBuffer(ContainerType *container, amrex::MultiFab &state, amrex::MultiFab &state_buffer, int lev, amrex::Real time, amrex::Real dt,
		     int birth_time_index, int mass_at_birth_index, amrex::Real p0, amrex::Real t_fb, amrex::Real alpha, int *active_particle_count,
		     amrex::Real *scalar_momentum, int *invalid_source_count)
{
	const BL_PROFILE("EarlyFeedbackUtils::depositToBuffer()");
	static_assert(AMREX_SPACEDIM == 3, "Empirically motivated early feedback is currently implemented only in 3D.");
	constexpr auto stencil_weights = ParticleUtils::kernel_spherical_3_weights_normalized;

	for (typename ContainerType::ParIterType pti(*container, lev); pti.isValid(); ++pti) {
		auto &particles = pti.GetArrayOfStructs();
		auto *particle_data = particles().data();
		const amrex::Long num_particles = pti.numParticles();
		const auto local_state = state.const_array(pti);
		const auto local_buffer = state_buffer.array(pti);

		const auto &geometry = container->Geom(lev);
		const auto prob_lo = geometry.ProbLoArray();
		const auto dx = geometry.CellSizeArray();
		const auto inv_dx = geometry.InvCellSizeArray();
		const amrex::Real inverse_volume = AMREX_D_TERM(inv_dx[0], *inv_dx[1], *inv_dx[2]);
		const amrex::Dim3 domain_lo = amrex::lbound(geometry.Domain());
		const amrex::Dim3 domain_hi = amrex::ubound(geometry.Domain());
		const amrex::Dim3 fab_lo = amrex::lbound(state_buffer[pti].box());
		const amrex::Dim3 fab_hi = amrex::ubound(state_buffer[pti].box());
		const amrex::GpuArray<int, AMREX_SPACEDIM> is_periodic = {
		    AMREX_D_DECL(static_cast<int>(geometry.isPeriodic(0)), static_cast<int>(geometry.isPeriodic(1)), static_cast<int>(geometry.isPeriodic(2)))};
		const int count_component = state_buffer.nComp() - 1;

		amrex::ParallelFor(num_particles, [=] AMREX_GPU_DEVICE(int64_t particle_index) noexcept {
			const auto &particle = particle_data[particle_index]; // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
			if (particle.id() <= 0) {
				return;
			}

			const amrex::Real requested_momentum =
			    earlyFeedbackMomentumIncrement(time, dt, particle.rdata(birth_time_index), particle.rdata(mass_at_birth_index), p0, t_fb, alpha);
			if (!(requested_momentum > 0.0)) {
				return;
			}

			const amrex::Real pos_x = particle.pos(0);
			const amrex::Real pos_y = particle.pos(1);
			const amrex::Real pos_z = particle.pos(2);
			const int host_i = static_cast<int>(amrex::Math::floor((pos_x - prob_lo[0]) * inv_dx[0]));
			const int host_j = static_cast<int>(amrex::Math::floor((pos_y - prob_lo[1]) * inv_dx[1]));
			const int host_k = static_cast<int>(amrex::Math::floor((pos_z - prob_lo[2]) * inv_dx[2]));

			const bool stencil_outside_fab = host_i - stencil_size < fab_lo.x || host_i + stencil_size > fab_hi.x ||
							 host_j - stencil_size < fab_lo.y || host_j + stencil_size > fab_hi.y ||
							 host_k - stencil_size < fab_lo.z || host_k + stencil_size > fab_hi.z;
			const bool stencil_outside_domain = (!is_periodic[0] && (host_i - stencil_size < domain_lo.x || host_i + stencil_size > domain_hi.x)) ||
							    (!is_periodic[1] && (host_j - stencil_size < domain_lo.y || host_j + stencil_size > domain_hi.y)) ||
							    (!is_periodic[2] && (host_k - stencil_size < domain_lo.z || host_k + stencil_size > domain_hi.z));
			if (stencil_outside_fab || stencil_outside_domain) {
				amrex::Gpu::Atomic::AddNoRet(invalid_source_count, 1);
				return;
			}

			amrex::Real weight_sum = 0.0;
			amrex::Real weighted_direction_x = 0.0;
			amrex::Real weighted_direction_y = 0.0;
			amrex::Real weighted_direction_z = 0.0;
			bool valid_gas_state = true;
			for (int offset_i = -stencil_size; offset_i <= stencil_size; ++offset_i) {
				for (int offset_j = -stencil_size; offset_j <= stencil_size; ++offset_j) {
					for (int offset_k = -stencil_size; offset_k <= stencil_size; ++offset_k) {
						const amrex::Real weight = stencil_weights[std::abs(offset_i)][std::abs(offset_j)][std::abs(offset_k)];
						if (!(weight > 0.0)) {
							continue;
						}
						const int i = host_i + offset_i;
						const int j = host_j + offset_j;
						const int k = host_k + offset_k;
						const amrex::Real rho = local_state(i, j, k, HydroSystem<problem_t>::density_index);
						valid_gas_state = valid_gas_state && rho > 0.0 && amrex::Math::isfinite(rho);

						const amrex::Real delta_x = (static_cast<amrex::Real>(i) + 0.5) * dx[0] + prob_lo[0] - pos_x;
						const amrex::Real delta_y = (static_cast<amrex::Real>(j) + 0.5) * dx[1] + prob_lo[1] - pos_y;
						const amrex::Real delta_z = (static_cast<amrex::Real>(k) + 0.5) * dx[2] + prob_lo[2] - pos_z;
						const amrex::Real radius = std::sqrt((delta_x * delta_x) + (delta_y * delta_y) + (delta_z * delta_z));
						const amrex::Real inverse_radius = (radius > 0.0) ? 1.0 / radius : 0.0;
						weight_sum += weight;
						weighted_direction_x += weight * delta_x * inverse_radius;
						weighted_direction_y += weight * delta_y * inverse_radius;
						weighted_direction_z += weight * delta_z * inverse_radius;
					}
				}
			}

			if (!valid_gas_state || !(weight_sum > 0.0)) {
				amrex::Gpu::Atomic::AddNoRet(invalid_source_count, 1);
				return;
			}

			const amrex::Real mean_direction_x = weighted_direction_x / weight_sum;
			const amrex::Real mean_direction_y = weighted_direction_y / weight_sum;
			const amrex::Real mean_direction_z = weighted_direction_z / weight_sum;
			amrex::Real corrected_norm = 0.0;
			for (int offset_i = -stencil_size; offset_i <= stencil_size; ++offset_i) {
				for (int offset_j = -stencil_size; offset_j <= stencil_size; ++offset_j) {
					for (int offset_k = -stencil_size; offset_k <= stencil_size; ++offset_k) {
						const amrex::Real weight = stencil_weights[std::abs(offset_i)][std::abs(offset_j)][std::abs(offset_k)];
						if (!(weight > 0.0)) {
							continue;
						}
						const int i = host_i + offset_i;
						const int j = host_j + offset_j;
						const int k = host_k + offset_k;
						const amrex::Real delta_x = (static_cast<amrex::Real>(i) + 0.5) * dx[0] + prob_lo[0] - pos_x;
						const amrex::Real delta_y = (static_cast<amrex::Real>(j) + 0.5) * dx[1] + prob_lo[1] - pos_y;
						const amrex::Real delta_z = (static_cast<amrex::Real>(k) + 0.5) * dx[2] + prob_lo[2] - pos_z;
						const amrex::Real radius = std::sqrt((delta_x * delta_x) + (delta_y * delta_y) + (delta_z * delta_z));
						const amrex::Real inverse_radius = (radius > 0.0) ? 1.0 / radius : 0.0;
						const amrex::Real q_x = weight * ((delta_x * inverse_radius) - mean_direction_x);
						const amrex::Real q_y = weight * ((delta_y * inverse_radius) - mean_direction_y);
						const amrex::Real q_z = weight * ((delta_z * inverse_radius) - mean_direction_z);
						corrected_norm += std::sqrt((q_x * q_x) + (q_y * q_y) + (q_z * q_z));
					}
				}
			}

			if (!(corrected_norm > 0.0) || !amrex::Math::isfinite(corrected_norm)) {
				amrex::Gpu::Atomic::AddNoRet(invalid_source_count, 1);
				return;
			}

			amrex::Real work_density = 0.0;
			const amrex::Real momentum_normalization = requested_momentum * inverse_volume / corrected_norm;
			for (int offset_i = -stencil_size; offset_i <= stencil_size; ++offset_i) {
				for (int offset_j = -stencil_size; offset_j <= stencil_size; ++offset_j) {
					for (int offset_k = -stencil_size; offset_k <= stencil_size; ++offset_k) {
						const amrex::Real weight = stencil_weights[std::abs(offset_i)][std::abs(offset_j)][std::abs(offset_k)];
						if (!(weight > 0.0)) {
							continue;
						}
						const int i = host_i + offset_i;
						const int j = host_j + offset_j;
						const int k = host_k + offset_k;
						const amrex::Real delta_x = (static_cast<amrex::Real>(i) + 0.5) * dx[0] + prob_lo[0] - pos_x;
						const amrex::Real delta_y = (static_cast<amrex::Real>(j) + 0.5) * dx[1] + prob_lo[1] - pos_y;
						const amrex::Real delta_z = (static_cast<amrex::Real>(k) + 0.5) * dx[2] + prob_lo[2] - pos_z;
						const amrex::Real radius = std::sqrt((delta_x * delta_x) + (delta_y * delta_y) + (delta_z * delta_z));
						const amrex::Real inverse_radius = (radius > 0.0) ? 1.0 / radius : 0.0;
						const amrex::Real delta_px = momentum_normalization * weight * ((delta_x * inverse_radius) - mean_direction_x);
						const amrex::Real delta_py = momentum_normalization * weight * ((delta_y * inverse_radius) - mean_direction_y);
						const amrex::Real delta_pz = momentum_normalization * weight * ((delta_z * inverse_radius) - mean_direction_z);
						const amrex::Real rho = local_state(i, j, k, HydroSystem<problem_t>::density_index);
						const amrex::Real velocity_x = local_state(i, j, k, HydroSystem<problem_t>::x1Momentum_index) / rho;
						const amrex::Real velocity_y = local_state(i, j, k, HydroSystem<problem_t>::x2Momentum_index) / rho;
						const amrex::Real velocity_z = local_state(i, j, k, HydroSystem<problem_t>::x3Momentum_index) / rho;
						work_density += (velocity_x * delta_px) + (velocity_y * delta_py) + (velocity_z * delta_pz);

						amrex::Gpu::Atomic::AddNoRet(&local_buffer(i, j, k, HydroSystem<problem_t>::x1Momentum_index), delta_px);
						amrex::Gpu::Atomic::AddNoRet(&local_buffer(i, j, k, HydroSystem<problem_t>::x2Momentum_index), delta_py);
						amrex::Gpu::Atomic::AddNoRet(&local_buffer(i, j, k, HydroSystem<problem_t>::x3Momentum_index), delta_pz);
						amrex::Gpu::Atomic::AddNoRet(&local_buffer(i, j, k, count_component), 1.0);
					}
				}
			}

			if (work_density < 0.0) {
				amrex::Gpu::Atomic::AddNoRet(&local_buffer(host_i, host_j, host_k, HydroSystem<problem_t>::energy_index), -work_density);
			}
			amrex::Gpu::Atomic::AddNoRet(active_particle_count, 1);
			amrex::Gpu::Atomic::AddNoRet(scalar_momentum, requested_momentum);
		});
	}
}

template <typename problem_t>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE void addBufferToState(amrex::Array4<amrex::Real> const &state, amrex::Array4<const amrex::Real> const &buffer, int i, int j,
							  int k, int count_component) noexcept
{
	if (!(buffer(i, j, k, count_component) > 0.0)) {
		return;
	}

	const amrex::Real rho = state(i, j, k, HydroSystem<problem_t>::density_index);
	const amrex::Real px = state(i, j, k, HydroSystem<problem_t>::x1Momentum_index);
	const amrex::Real py = state(i, j, k, HydroSystem<problem_t>::x2Momentum_index);
	const amrex::Real pz = state(i, j, k, HydroSystem<problem_t>::x3Momentum_index);
	const amrex::Real px_deposited = px + buffer(i, j, k, HydroSystem<problem_t>::x1Momentum_index);
	const amrex::Real py_deposited = py + buffer(i, j, k, HydroSystem<problem_t>::x2Momentum_index);
	const amrex::Real pz_deposited = pz + buffer(i, j, k, HydroSystem<problem_t>::x3Momentum_index);
	const amrex::Real thermalized_energy = std::max(buffer(i, j, k, HydroSystem<problem_t>::energy_index), static_cast<amrex::Real>(0.0));
	const amrex::Real kinetic_energy_old = 0.5 * ((px * px) + (py * py) + (pz * pz)) / rho;
	const amrex::Real kinetic_energy_new = 0.5 * ((px_deposited * px_deposited) + (py_deposited * py_deposited) + (pz_deposited * pz_deposited)) / rho;

	state(i, j, k, HydroSystem<problem_t>::x1Momentum_index) = px_deposited;
	state(i, j, k, HydroSystem<problem_t>::x2Momentum_index) = py_deposited;
	state(i, j, k, HydroSystem<problem_t>::x3Momentum_index) = pz_deposited;
	state(i, j, k, HydroSystem<problem_t>::internalEnergy_index) += thermalized_energy;
	state(i, j, k, HydroSystem<problem_t>::energy_index) += (kinetic_energy_new - kinetic_energy_old) + thermalized_energy;
}

template <typename problem_t>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE void limitVelocity(amrex::Array4<amrex::Real> const &state, amrex::Array4<const amrex::Real> const &buffer, int i, int j,
						       int k, int count_component, amrex::Real velocity_limit, int *clipped_cell_count,
						       amrex::Real *min_velocity_scale, amrex::Real *max_velocity) noexcept
{
	if (!(buffer(i, j, k, count_component) > 0.0)) {
		return;
	}

	const amrex::Real rho = state(i, j, k, HydroSystem<problem_t>::density_index);
	const amrex::Real px = state(i, j, k, HydroSystem<problem_t>::x1Momentum_index);
	const amrex::Real py = state(i, j, k, HydroSystem<problem_t>::x2Momentum_index);
	const amrex::Real pz = state(i, j, k, HydroSystem<problem_t>::x3Momentum_index);
	const amrex::Real momentum = std::sqrt((px * px) + (py * py) + (pz * pz));
	const amrex::Real velocity = momentum / rho;
	const amrex::Real velocity_scale = (velocity > velocity_limit) ? velocity_limit / velocity : 1.0;
	if (velocity_scale < 1.0) {
		amrex::Gpu::Atomic::AddNoRet(clipped_cell_count, 1);
		amrex::Gpu::Atomic::Min(min_velocity_scale, velocity_scale);
		const amrex::Real kinetic_energy_old = 0.5 * momentum * momentum / rho;
		const amrex::Real kinetic_energy_new = velocity_scale * velocity_scale * kinetic_energy_old;
		state(i, j, k, HydroSystem<problem_t>::x1Momentum_index) = velocity_scale * px;
		state(i, j, k, HydroSystem<problem_t>::x2Momentum_index) = velocity_scale * py;
		state(i, j, k, HydroSystem<problem_t>::x3Momentum_index) = velocity_scale * pz;
		state(i, j, k, HydroSystem<problem_t>::energy_index) += kinetic_energy_new - kinetic_energy_old;
	}

	amrex::Gpu::Atomic::Max(max_velocity, velocity_scale * velocity);
}

template <typename problem_t>
void applyBuffer(amrex::MultiFab &state, amrex::MultiFab const &state_buffer, amrex::Real velocity_limit, int *clipped_cell_count,
		 amrex::Real *min_velocity_scale, amrex::Real *max_velocity)
{
	const BL_PROFILE("EarlyFeedbackUtils::applyBuffer()");
	const int count_component = state_buffer.nComp() - 1;
	for (amrex::MFIter mfi(state); mfi.isValid(); ++mfi) {
		const amrex::Box &box = mfi.validbox();
		const auto local_state = state.array(mfi);
		const auto local_buffer = state_buffer.const_array(mfi);
		amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
			addBufferToState<problem_t>(local_state, local_buffer, i, j, k, count_component);
		});
		amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
			limitVelocity<problem_t>(local_state, local_buffer, i, j, k, count_component, velocity_limit, clipped_cell_count, min_velocity_scale,
						 max_velocity);
		});
	}
}

} // namespace EarlyFeedbackUtils

template <typename ContainerType, typename problem_t>
auto EarlyFeedbackDeposition(ContainerType *container, amrex::MultiFab &state, std::array<amrex::MultiFab, AMREX_SPACEDIM> const * /*state_fc*/, int lev,
			     amrex::Real time, amrex::Real dt, int birth_time_index, int mass_at_birth_index) -> EarlyFeedbackStats
{
	const BL_PROFILE("[particle_early_feedback] EarlyFeedbackDeposition()");
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(state.nGrowVect().allGE(4), "Early feedback requires at least four cell-centered ghost cells.");
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(birth_time_index >= 0 && mass_at_birth_index >= 0,
					 "Early feedback requires valid birth_time and mass_at_birth particle components.");

	amrex::MultiFab state_buffer(state.boxArray(), state.DistributionMap(), state.nComp() + 1, state.nGrow());
	state_buffer.setVal(0.0);
	amrex::Gpu::Buffer<int> active_particle_buffer({0});
	amrex::Gpu::Buffer<amrex::Real> scalar_momentum_buffer({0.0});
	amrex::Gpu::Buffer<int> invalid_source_buffer({0});
	amrex::Gpu::Buffer<int> clipped_cell_buffer({0});
	amrex::Gpu::Buffer<amrex::Real> min_velocity_scale_buffer({1.0});
	amrex::Gpu::Buffer<amrex::Real> max_velocity_buffer({0.0});

	EarlyFeedbackUtils::depositToBuffer<ContainerType, problem_t>(container, state, state_buffer, lev, time, dt, birth_time_index, mass_at_birth_index,
								      EMF_p0, EMF_tFB, EMF_alpha, active_particle_buffer.data(), scalar_momentum_buffer.data(),
								      invalid_source_buffer.data());

	int invalid_sources = invalid_source_buffer.copyToHost()[0];
	amrex::ParallelDescriptor::ReduceIntSum(invalid_sources);
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
	    invalid_sources == 0,
	    "Early feedback found an active particle whose full three-cell stencil is not representable on its particle grid and inside the physical domain.");

	state_buffer.SumBoundary(container->Geom(lev).periodicity());
	ParticleUtils::roundoffMultiFab(state_buffer);
	EarlyFeedbackUtils::applyBuffer<problem_t>(state, state_buffer, EMF_max_velocity, clipped_cell_buffer.data(), min_velocity_scale_buffer.data(),
						   max_velocity_buffer.data());

	EarlyFeedbackStats stats;
	stats.active_particles = active_particle_buffer.copyToHost()[0];
	stats.clipped_cells = clipped_cell_buffer.copyToHost()[0];
	stats.scalar_momentum = scalar_momentum_buffer.copyToHost()[0];
	stats.min_velocity_scale = min_velocity_scale_buffer.copyToHost()[0];
	stats.max_velocity = max_velocity_buffer.copyToHost()[0];
	amrex::ParallelDescriptor::ReduceIntSum(stats.active_particles);
	amrex::ParallelDescriptor::ReduceIntSum(stats.clipped_cells);
	amrex::ParallelDescriptor::ReduceRealSum(stats.scalar_momentum);
	amrex::ParallelDescriptor::ReduceRealMin(stats.min_velocity_scale);
	amrex::ParallelDescriptor::ReduceRealMax(stats.max_velocity);
	return stats;
}

} // namespace quokka

#endif // PARTICLE_EARLY_FEEDBACK_HPP_
