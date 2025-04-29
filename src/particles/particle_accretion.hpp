#ifndef PARTICLE_ACCRETION_HPP_
#define PARTICLE_ACCRETION_HPP_

#include "AMReX_Array4.H"
#include "AMReX_MultiFab.H"
#include "AMReX_REAL.H"
#include "hydro/hydro_system.hpp"
#include "particles/particle_types.hpp"
#include "particles/particle_utils.hpp"

namespace quokka
{

enum class AccretionScheme { Threshold = 0 };

// manually set the accretion scheme
constexpr AccretionScheme accretion_scheme = AccretionScheme::Threshold;

#if AMREX_SPACEDIM == 3

namespace SinkAccretionUtils
{

constexpr int stencil_size = quokka::ParticleUtils::stencil_size;

static constexpr ParticleUtils::kernel_weights_array_t kernel_weights = []() constexpr {
	if constexpr (accretion_scheme == AccretionScheme::Threshold) {
		return ParticleUtils::kernel_spherical_uniform_3_weights;
	} else {
		return ParticleUtils::kernel_spherical_3_weights;
	}
}();

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto get_delta_rho(double rho, double rho_sink) -> double { return -0.5 * (rho - rho_sink) / rho; }

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto compute_rho_sink(const amrex::Array4<const amrex::Real> & /*state*/, int /*i*/, int /*j*/, int /*k*/) -> double
{
	// Jeans criterion, density threshold, etc.

	// A single density threshold for testing
	return 0.2 * C::m_p;
}

// Function to compute accretion rate for particles in a box, including the ParallelFor call
template <typename ContainerType, typename problem_t>
void ComputeAccretionRateInBox(const typename ContainerType::ParIterType &pti, const amrex::Array4<const amrex::Real> &local_state,
			       const amrex::Array4<amrex::Real> &local_accretion_rate, const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> &plo,
			       const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> &dxi, amrex::Real /*time*/, amrex::Real /*dt*/,
			       int /*mass_index*/)
{
	// Get the particle array of structs
	auto &particles = pti.GetArrayOfStructs();
	auto *pData = particles().data();
	const amrex::Long np = pti.numParticles();

	// make a copy of kernel_weights for device
	const auto kernel_weights_d = kernel_weights;

	// Deposit particle data into the local buffer
	amrex::ParallelFor(np, [=] AMREX_GPU_DEVICE(int64_t idx) {
		auto &p = pData[idx]; // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)

		// Find the cell containing the particle
		int ix = static_cast<int>(amrex::Math::floor((p.pos(0) - plo[0]) * dxi[0]));
		int iy = static_cast<int>(amrex::Math::floor((p.pos(1) - plo[1]) * dxi[1]));
		int iz = static_cast<int>(amrex::Math::floor((p.pos(2) - plo[2]) * dxi[2]));

		const double rho_sink = compute_rho_sink(local_state, ix, iy, iz);

		// set accreted mass and momentum in the buffer
		for (int ii = -stencil_size; ii <= stencil_size; ++ii) {
			for (int jj = -stencil_size; jj <= stencil_size; ++jj) {
				for (int kk = -stencil_size; kk <= stencil_size; ++kk) {
					const double weight = kernel_weights_d[std::abs(ii)][std::abs(jj)][std::abs(kk)];
					const double rho = local_state(ix + ii, iy + jj, iz + kk, HydroSystem<problem_t>::density_index);
					if (rho > rho_sink) {
						const double delta_rho = get_delta_rho(rho, rho_sink) * weight;
						// use atomic operation to avoid race conditions
						amrex::Gpu::Atomic::AddNoRet(&local_accretion_rate(ix + ii, iy + jj, iz + kk), delta_rho);
					}
				}
			}
		}
	});
}

// Compute the scale down factor for the accretion rate. This is used to prevent accretion rates from exceeding 100% of the available mass.
// Current implementation: the maximum allowed relative accretion rate is 90% (gas density cannot drop more than 90% in one time step)
// TODO(cch): compute a local accretion_rate_floor
template <typename problem_t> void ComputeScaleDown(amrex::MultiFab &accretion_rate, amrex::MultiFab &scale_down, const amrex::Periodicity &periodicity)
{
	const auto &local_accretion_rate_arr = accretion_rate.arrays();
	const auto &scale_down_arr = scale_down.arrays();

	amrex::ParallelFor(accretion_rate, [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept {
		const double accretion_rate_cell = local_accretion_rate_arr[bx](i, j, k);
		const double accretion_rate_floor = -0.9;
		if (accretion_rate_cell < accretion_rate_floor) {
			// scale down the accretion rate to the maximum allowed value
			scale_down_arr[bx](i, j, k) = accretion_rate_floor / accretion_rate_cell;

			// update the accretion rate
			local_accretion_rate_arr[bx](i, j, k) = accretion_rate_floor;
		}
		AMREX_ASSERT(local_accretion_rate_arr[bx](i, j, k) <= 0.0);
		AMREX_ASSERT(local_accretion_rate_arr[bx](i, j, k) > -1.0);
	});

	// synchronize scale_down
	scale_down.FillBoundary(periodicity);
}

// Function to update particle mass and momentum for particles in a box, including the ParallelFor call
template <typename ContainerType, typename problem_t>
void UpdateParticleMassAndMomentumInBox(const typename ContainerType::ParIterType &pti, const amrex::Array4<const amrex::Real> &local_state,
					const amrex::Array4<const amrex::Real> &local_scale_down, const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> &plo,
					const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> &dxi, int mass_index, amrex::Real /*time*/, amrex::Real /*dt*/, amrex::Real vol)
{
	// Get the particle array of structs
	auto &particles = pti.GetArrayOfStructs();
	auto *pData = particles().data();
	const amrex::Long np = pti.numParticles();

	// make a copy of kernel_weights for device
	const auto kernel_weights_d = kernel_weights;

	amrex::ParallelFor(np, [=] AMREX_GPU_DEVICE(int64_t idx) {
		auto &p = pData[idx]; // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)

		// Find the cell containing the particle
		int ix = static_cast<int>(amrex::Math::floor((p.pos(0) - plo[0]) * dxi[0]));
		int iy = static_cast<int>(amrex::Math::floor((p.pos(1) - plo[1]) * dxi[1]));
		int iz = static_cast<int>(amrex::Math::floor((p.pos(2) - plo[2]) * dxi[2]));

		const double rho_sink = compute_rho_sink(local_state, ix, iy, iz);

		// set accreted mass and momentum in the buffer
		double accreted_mass = 0.0;
		double accreted_momentum_x = 0.0;
		double accreted_momentum_y = 0.0;
		double accreted_momentum_z = 0.0;
		for (int ii = ix - stencil_size; ii <= ix + stencil_size; ++ii) {
			for (int jj = iy - stencil_size; jj <= iy + stencil_size; ++jj) {
				for (int kk = iz - stencil_size; kk <= iz + stencil_size; ++kk) {
					const double weight = kernel_weights_d[std::abs(ii - ix)][std::abs(jj - iy)][std::abs(kk - iz)];
					const double rho = local_state(ii, jj, kk, HydroSystem<problem_t>::density_index);
					const double px = local_state(ii, jj, kk, HydroSystem<problem_t>::x1Momentum_index);
					const double py = local_state(ii, jj, kk, HydroSystem<problem_t>::x2Momentum_index);
					const double pz = local_state(ii, jj, kk, HydroSystem<problem_t>::x3Momentum_index);
					if (rho > rho_sink) {
						// the original accretion rate
						const double delta_rho = get_delta_rho(rho, rho_sink) * weight;
						// the scaled accretion rate
						const double actual_delta_rho = local_scale_down(ii, jj, kk) * delta_rho;
						// sum up the accreted mass and momentum
						accreted_mass += actual_delta_rho * rho * vol;
						accreted_momentum_x += actual_delta_rho * px * vol;
						accreted_momentum_y += actual_delta_rho * py * vol;
						accreted_momentum_z += actual_delta_rho * pz * vol;
					}
				}
			}
		}
		// the accretion rates are negative, so we 'subtract' them
		p.rdata(mass_index) -= accreted_mass;
		p.rdata(mass_index + 1) -= accreted_momentum_x;
		p.rdata(mass_index + 2) -= accreted_momentum_y;
		p.rdata(mass_index + 3) -= accreted_momentum_z;
	});
}

template <typename ContainerType, typename problem_t>
void UpdateParticleMassAndMomentum(ContainerType *container, amrex::MultiFab &state, amrex::MultiFab &scale_down, int lev, int mass_index,
				   amrex::Real time, amrex::Real dt)
{
	for (typename ContainerType::ParIterType pti(*container, lev); pti.isValid(); ++pti) {
		// Get the local deposit array for this box
		const auto &local_state = state.array(pti);
		const auto &local_scale_down = scale_down.array(pti);

		// Get geometry information for this level
		const auto &geom = container->Geom(lev);
		const auto plo = geom.ProbLoArray();
		const auto dxi = geom.InvCellSizeArray();
		const auto dx = geom.CellSizeArray();

		// Calculate cell volume
		const amrex::Real vol = AMREX_D_TERM(dx[0], *dx[1], *dx[2]);

		// Process particles in this box
		UpdateParticleMassAndMomentumInBox<ContainerType, problem_t>(pti, local_state, local_scale_down, plo, dxi, mass_index, time, dt, vol);
	}
}

template <typename problem_t> void UpdateHydroState(amrex::MultiFab &state, amrex::MultiFab &accretion_rate)
{
	const auto &local_accretion_rate_arr = accretion_rate.arrays();
	const auto &state_arr = state.arrays();

	amrex::ParallelFor(state, [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept {
		const double accretion_rate_cell = local_accretion_rate_arr[bx](i, j, k);
		AMREX_ASSERT(accretion_rate_cell <= 0.0);
		AMREX_ASSERT(accretion_rate_cell > -1.0);
		const double accretion_down_factor = 1.0 + accretion_rate_cell;
		state_arr[bx](i, j, k, HydroSystem<problem_t>::density_index) *= accretion_down_factor;
		state_arr[bx](i, j, k, HydroSystem<problem_t>::x1Momentum_index) *= accretion_down_factor;
		state_arr[bx](i, j, k, HydroSystem<problem_t>::x2Momentum_index) *= accretion_down_factor;
		state_arr[bx](i, j, k, HydroSystem<problem_t>::x3Momentum_index) *= accretion_down_factor;
	});
}

template <typename ContainerType, typename problem_t>
void computeAccretion(ContainerType *container, amrex::MultiFab &state, amrex::MultiFab &accretion_rate, int lev, amrex::Real time, amrex::Real dt,
		      int mass_index)
{
	for (typename ContainerType::ParIterType pti(*container, lev); pti.isValid(); ++pti) {
		// Get the local deposit array for this box
		const auto &local_state = state.array(pti);
		const auto &local_accretion_rate = accretion_rate.array(pti);

		// Get geometry information for this level
		const auto &geom = container->Geom(lev);
		const auto plo = geom.ProbLoArray();
		const auto dxi = geom.InvCellSizeArray();

		// Process particles in this box
		ComputeAccretionRateInBox<ContainerType, problem_t>(pti, local_state, local_accretion_rate, plo, dxi, time, dt, mass_index);
	}

	// Sum boundary cell values to real cells
	accretion_rate.SumBoundary(container->Geom(lev).periodicity());
}

// Functor for accreting mass and momentum from gas onto particles.
// For testing purposes, we implement a simplified version of the threshold scheme from Federrath et al. (2010).
// For every cell near the particle, we accrete an amount of mass given by
// $ \Delta m = \max(0, 0.5 (rho - rho_sink) * dx^3) $
// in one time step. rho_sink is a constant threshold density.
// The accreted mass and momentum are added to the particle's mass and momentum.
template <typename ContainerType, typename problem_t>
void applyAccretion(ContainerType *container, amrex::MultiFab &state, amrex::MultiFab &state_accretion_rate, int lev, amrex::Real time, amrex::Real dt,
		    int mass_index)
{
	// Step 2: Compute the scale_down factor. We scale down the accretion rate to prevent accretion rates from exceeding 100%
	// of the available mass.
	amrex::MultiFab scale_down(state.boxArray(), state.DistributionMap(), 1, state.nGrow());
	scale_down.setVal(1.0);
	// Update accretion_rate and compute scale_down
	ComputeScaleDown<problem_t>(state_accretion_rate, scale_down, container->Geom(lev).periodicity());

	// Step 3: Update particle mass and momentum
	UpdateParticleMassAndMomentum<ContainerType, problem_t>(container, state, scale_down, lev, mass_index, time, dt);

	// Step 4: Update the hydro state. We do this at last because the original state is needed for updating particles in step 3.
	UpdateHydroState<problem_t>(state, state_accretion_rate);
}

} // namespace SinkAccretionUtils

#endif // AMREX_SPACEDIM == 3

} // namespace quokka

#endif // PARTICLE_ACCRETION_HPP_
