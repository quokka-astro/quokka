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

enum class AccretionScheme {
	Threshold = 0
};

#if AMREX_SPACEDIM == 3

//-------------------- Mass accretion --------------------

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
auto get_delta_rho(double rho, double rho_sink) -> double
{
	return -0.5 * (rho - rho_sink) / rho;
	// return -0.6 * (rho - rho_sink) / rho;
}

// Functor for accreting mass and momentum from gas onto particles.
// For testing purposes, we implement a simplified version of the threshold scheme from Federrath et al. (2010).
// For every cell near the particle, we accrete an amount of mass given by
// $ \Delta m = \max(0, 0.5 (rho - rho_sink) * dx^3) $
// in one time step. rho_sink is a constant threshold density.
// The accreted mass and momentum are added to the particle's mass and momentum.
template <typename ContainerType, typename problem_t>
void MassAccretion(ContainerType *container, amrex::MultiFab &state, amrex::MultiFab &/*state_buffer*/, int lev, amrex::Real time, amrex::Real dt, int mass_index,
		  int evolutionStageIndex)
{
	const AccretionScheme accretion_scheme = AccretionScheme::Threshold;
	const double rho_sink = 0.1 * C::m_u;

	constexpr int stencil_size = 3;
	static_assert(stencil_size <= 3, "stencil_size must be <= 3");

	constexpr const ParticleUtils::kernel_weights_array_t &kernel_weights = ParticleUtils::kernel_spherical_3_weights;
	static_assert(stencil_size == ParticleUtils::stencil_size, "stencil_size must be equal to ParticleUtils::stencil_size");

	// copy host variables to device
	const amrex::Real step_end_time = time + dt;

	// Accretion rate state. This state stores the *fractional* change in density or momentum. 
	amrex::MultiFab accretion_rate(state.boxArray(), state.DistributionMap(), 1, state.nGrow());
	accretion_rate.setVal(0.0);

	// Step 1: Local deposition within each box
	for (typename ContainerType::ParIterType pti(*container, lev); pti.isValid(); ++pti) {
		// Get the particle array of structs
		auto &particles = pti.GetArrayOfStructs();
		auto *pData = particles().data();
		const amrex::Long np = pti.numParticles();

		// Get the local deposit array for this box
		const auto &local_state = state.array(pti);
		const auto &local_accretion_rate = accretion_rate.array(pti);
		
		// Get geometry information for this level
		const auto &geom = container->Geom(lev);
		const auto plo = geom.ProbLoArray();
		const auto dxi = geom.InvCellSizeArray();
		const auto dx = geom.CellSizeArray();

		// Calculate inverse cell volume
		const amrex::Real vol_inverse = AMREX_D_TERM(dxi[0], *dxi[1], *dxi[2]);
		const amrex::Real vol = AMREX_D_TERM(dx[0], *dx[1], *dx[2]);

		// Step 1: Deposit particle data into the local buffer
		amrex::ParallelFor(np, [=] AMREX_GPU_DEVICE(int64_t idx) {
			auto &p = pData[idx]; // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)

			// Check if this is a supernova progenitor
			const bool is_accreting = (p.idata(evolutionStageIndex) == static_cast<int>(StellarEvolutionStage::LowMassStar) ||
						    p.idata(evolutionStageIndex) == static_cast<int>(StellarEvolutionStage::SNProgenitor));

			if (!is_accreting) {
				return;
			}

			// Find the cell containing the particle
			int ix = static_cast<int>(amrex::Math::floor((p.pos(0) - plo[0]) * dxi[0]));
			int iy = static_cast<int>(amrex::Math::floor((p.pos(1) - plo[1]) * dxi[1]));
			int iz = static_cast<int>(amrex::Math::floor((p.pos(2) - plo[2]) * dxi[2]));

			// set accreted mass and momentum in the buffer
			for (int ii = -stencil_size; ii <= stencil_size; ++ii) {
				for (int jj = -stencil_size; jj <= stencil_size; ++jj) {
					for (int kk = -stencil_size; kk <= stencil_size; ++kk) {
						const double weight = kernel_weights[std::abs(ii)][std::abs(jj)][std::abs(kk)];
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

	// Step 2: Sum boundary cell values to real cells
	accretion_rate.SumBoundary(container->Geom(lev).periodicity());

	// Step 3: Compute the scale_down factor. We scale down the accretion rate to prevent accretion rates from exceeding 100%
	// of the available mass.
	amrex::MultiFab scale_down(state.boxArray(), state.DistributionMap(), 1, state.nGrow());
	scale_down.setVal(1.0);

	// const double accretion_rate_floor = -0.9;

	// We have to MFIter over the hydro state instead of over particles because a particle can accrete from ghost cells, which is then
	// passed to real cells in a neighboring box.
	const auto &local_accretion_rate_arr = accretion_rate.arrays();
	const auto &scale_down_arr = scale_down.arrays();
	const auto &state_arr = state.arrays();
	amrex::ParallelFor(state, [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept {
		const double accretion_rate_cell = local_accretion_rate_arr[bx](i, j, k);
		const double accretion_rate_floor = -(1.0 - rho_sink / state_arr[bx](i, j, k, HydroSystem<problem_t>::density_index));
		if (accretion_rate_cell < accretion_rate_floor) {
			// scale down the accretion rate to the maximum allowed value
			scale_down_arr[bx](i, j, k) = accretion_rate_floor / accretion_rate_cell;

			// update the accretion rate
			local_accretion_rate_arr[bx](i, j, k) = accretion_rate_floor;
			// or, equivalently,
			// local_accretion_rate_arr[bx](i, j, k) *= scale_down_arr[bx](i, j, k);
		}
	});

	// Step 4: Update particle mass and momentum: Re-compute accretion rate from each particle and apply the acrreted mass and momentum
	// to the particles, subject to scale_down.
	for (typename ContainerType::ParIterType pti(*container, lev); pti.isValid(); ++pti) {
		// Get the particle array of structs
		auto &particles = pti.GetArrayOfStructs();
		auto *pData = particles().data();
		const amrex::Long np = pti.numParticles();

		// Get the local deposit array for this box
		const auto &local_state = state.array(pti);
		const auto &local_accretion_rate = accretion_rate.array(pti);
		const auto &local_scale_down = scale_down.array(pti);
		
		// Get geometry information for this level
		const auto &geom = container->Geom(lev);
		const auto plo = geom.ProbLoArray();
		const auto dxi = geom.InvCellSizeArray();
		const auto dx = geom.CellSizeArray();

		// Calculate inverse cell volume
		const amrex::Real vol_inverse = AMREX_D_TERM(dxi[0], *dxi[1], *dxi[2]);
		const amrex::Real vol = AMREX_D_TERM(dx[0], *dx[1], *dx[2]);

		amrex::ParallelFor(np, [=] AMREX_GPU_DEVICE(int64_t idx) {
			auto &p = pData[idx]; // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)

			// Check if this is a supernova progenitor
			const bool is_accreting = (p.idata(evolutionStageIndex) == static_cast<int>(StellarEvolutionStage::LowMassStar) ||
						    p.idata(evolutionStageIndex) == static_cast<int>(StellarEvolutionStage::SNProgenitor));

			if (!is_accreting) {
				return;
			}

			// Find the cell containing the particle
			int ix = static_cast<int>(amrex::Math::floor((p.pos(0) - plo[0]) * dxi[0]));
			int iy = static_cast<int>(amrex::Math::floor((p.pos(1) - plo[1]) * dxi[1]));
			int iz = static_cast<int>(amrex::Math::floor((p.pos(2) - plo[2]) * dxi[2]));

			// set accreted mass and momentum in the buffer
			double accreted_mass = 0.0;
			double accreted_momentum_x = 0.0;
			double accreted_momentum_y = 0.0;
			double accreted_momentum_z = 0.0;
			for (int ii = ix - stencil_size; ii <= ix + stencil_size; ++ii) {
				for (int jj = iy - stencil_size; jj <= iy + stencil_size; ++jj) {
					for (int kk = iz - stencil_size; kk <= iz + stencil_size; ++kk) {
						const double weight = kernel_weights[std::abs(ii - ix)][std::abs(jj - iy)][std::abs(kk - iz)];
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
	
	// Step 5: Update the hydro state. We do this at last because the original state is needed for updating particles in step 4.
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
#endif // AMREX_SPACEDIM == 3

} // namespace quokka

#endif // PARTICLE_ACCRETION_HPP_
