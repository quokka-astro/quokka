#ifndef PARTICLE_ACCRETION_HPP_
#define PARTICLE_ACCRETION_HPP_

#include "AMReX_Array4.H"
#include "AMReX_BLProfiler.H"
#include "AMReX_MultiFab.H"
#include "AMReX_ParallelDescriptor.H"
#include "AMReX_Print.H"
#include "AMReX_REAL.H"
#include "AMReX_Reduce.H"
#include "gcem.hpp"
#include "hydro/hydro_system.hpp"
#include "particles/particle_types.hpp"
#include "particles/particle_utils.hpp"
#include <algorithm>
#include <limits>

namespace quokka
{

enum class AccretionScheme { Threshold = 0, BondiHoyle = 1 };

// manually set the accretion scheme
constexpr AccretionScheme accretion_scheme = AccretionScheme::BondiHoyle;

namespace SinkAccretionUtils
{

// constexpr int stencil_size = quokka::ParticleUtils::stencil_size;
constexpr int stencil_size = 3;
constexpr int rho_infty_stencil_size = stencil_size; // 0: use the cell that the particle is in

constexpr double r_acc_tolerance = 1.0001;

// Enable/disable sink accretion mass-conservation diagnostic.
// Set this to false after debugging.
constexpr bool sink_accretion_mass_check = true;

// -----------------------------------------------------------------------------
// Fixed hard density stop for sink accretion.
//
// The accretion stop density is fixed in mass density units:
//
//     rho_stop = sink_accretion_rho_stop_density
//              = 500.0 * C::m_p
//
// This means:
//   * if rho <= rho_stop, this cell will not be accreted;
//   * if an accretion update would drive rho below rho_stop, the accretion
//     rate is clipped so that rho_new >= rho_stop.
//
// IMPORTANT:
//   The stop criterion is applied inside ComputeScaleDown(), not separately
//   in the gas update or particle update. This guarantees that the accepted
//   gas removal and particle mass increase use the same final accretion rate.
// -----------------------------------------------------------------------------
constexpr double sink_accretion_rho_stop_density = 500.0 * C::m_p;

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto get_delta_rho(double rho, double rho_sink) -> double
{
	return -0.5 * (rho - rho_sink) / rho;
}

template <typename problem_t>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto compute_Mdot_and_r_K(const amrex::Array4<const amrex::Real> &local_state, int ix, int iy, int iz,
								   double par_mass, double par_x, double par_y, double par_z,
								   double par_vx, double par_vy, double par_vz,
								   const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> &plo,
								   const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> &dx,
								   std::array<amrex::Array4<const amrex::Real>, AMREX_SPACEDIM> const *fab_fc = nullptr)
    -> std::tuple<double, double>
{
	const double dx_max = std::max({dx[0], dx[1], dx[2]});

	// compute the average density, momentum, and sound speed in the accretion zone
	int n_cells = 0;
	double sum_rho = 0.0;
	double sum_px = 0.0;
	double sum_py = 0.0;
	double sum_pz = 0.0;
	double sum_cs = 0.0;
	double sum_magnetic_energy = 0.0;
	double sum_pressure = 0.0;

	for (int ii = ix - rho_infty_stencil_size; ii <= ix + rho_infty_stencil_size; ++ii) {
		for (int jj = iy - rho_infty_stencil_size; jj <= iy + rho_infty_stencil_size; ++jj) {
			for (int kk = iz - rho_infty_stencil_size; kk <= iz + rho_infty_stencil_size; ++kk) {
				const double x = par_x - plo[0] - (ii + static_cast<amrex::Real>(0.5)) * dx[0];
				const double y = par_y - plo[1] - (jj + static_cast<amrex::Real>(0.5)) * dx[1];
				const double z = par_z - plo[2] - (kk + static_cast<amrex::Real>(0.5)) * dx[2];
				const double r_sqr = x * x + y * y + z * z;
				const double r_acc_sqr = stencil_size * stencil_size * dx_max * dx_max;

				// allow a small tolerance to avoid numerical issues when the particle is exactly at the cell center
				if (r_sqr > r_acc_sqr * r_acc_tolerance) {
					continue;
				}

				const double rho = local_state(ii, jj, kk, HydroSystem<problem_t>::density_index);
				const double px = local_state(ii, jj, kk, HydroSystem<problem_t>::x1Momentum_index);
				const double py = local_state(ii, jj, kk, HydroSystem<problem_t>::x2Momentum_index);
				const double pz = local_state(ii, jj, kk, HydroSystem<problem_t>::x3Momentum_index);

				const double cs = HydroSystem<problem_t>::ComputeIsothermalSoundSpeed(local_state, ii, jj, kk, fab_fc);

				sum_rho += rho;
				sum_px += px;
				sum_py += py;
				sum_pz += pz;
				sum_cs += cs * rho;
				n_cells += 1;

				if constexpr (Physics_Traits<problem_t>::is_mhd_enabled) {
					sum_magnetic_energy += HydroSystem<problem_t>::ComputeMagneticEnergy(ii, jj, kk, fab_fc);
					sum_pressure += HydroSystem<problem_t>::ComputePressure(local_state, ii, jj, kk, fab_fc);
				}
			}
		}
	}

	const double rho_infty = sum_rho / n_cells;
	const double vx_grid = sum_px / sum_rho;
	const double vy_grid = sum_py / sum_rho;
	const double vz_grid = sum_pz / sum_rho;

	// Transform velocities to the particle frame to ensure Galilean invariance
	const double vx_infty = vx_grid - par_vx;
	const double vy_infty = vy_grid - par_vy;
	const double vz_infty = vz_grid - par_vz;
	const double cs_infty = sum_cs / sum_rho;

	AMREX_ASSERT(!std::isnan(rho_infty));
	AMREX_ASSERT(rho_infty > 0.0);
	AMREX_ASSERT(!std::isnan(cs_infty));
	AMREX_ASSERT(cs_infty > 0.0);

	// Compute average plasma beta in the accretion zone
	double mean_plasma_beta = std::numeric_limits<double>::max();
	if constexpr (Physics_Traits<problem_t>::is_mhd_enabled) {
		mean_plasma_beta = ParticleUtils::computePlasmaBeta(sum_pressure, sum_magnetic_energy);
	}

	// Compute MHD-aware effective fast magnetosonic speed:
	//   cf^2 = cs^2 + vA^2 = cs^2 * (1 + 2/beta)
	// For non-MHD: cf = cs.
	const double cf_infty_sqr = cs_infty * cs_infty * (1.0 + 2.0 / mean_plasma_beta);

	// Compute Bondi-Hoyle accretion radius:
	//   r_BH = G M / (v^2 + cf^2)
	const double v_infty_sqr = vx_infty * vx_infty + vy_infty * vy_infty + vz_infty * vz_infty;
	const double r_BH = C::Gconst * par_mass / (v_infty_sqr + cf_infty_sqr);

	// Compute the accretion rate in the accretion zone:
	//   M_dot = 4 pi rho_infty r_BH^2 sqrt(v_infty^2 + lambda^2 cf^2)
	constexpr double lambda = gcem::exp(1.5) / 4.0;
	AMREX_ASSERT(rho_infty > 0.0);

	const double M_dot = 4.0 * M_PI * rho_infty * r_BH * r_BH * std::sqrt(v_infty_sqr + lambda * lambda * cf_infty_sqr);
	AMREX_ASSERT(M_dot >= 0.0);

	// Compute accretion kernel radius:
	// r_K = dx / 4, if r_BH < dx / 4
	//       r_BH, if dx/4 <= r_BH <= stencil_size * dx / 2
	//       stencil_size * dx / 2, if r_BH > stencil_size * dx / 2
	const double r_acc = stencil_size * dx_max;
	double r_K = NAN;

	if (r_BH < dx_max / 4.0) {
		r_K = dx_max / 4.0;
	} else if (r_BH <= r_acc / 2.0) {
		r_K = r_BH;
	} else {
		r_K = r_acc / 2.0;
	}

	return std::make_tuple(M_dot, r_K);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto compute_accretion_kernel(const double r_sqr, const double r_K) -> double
{
	return std::exp(-r_sqr / (r_K * r_K));
}

// Function to compute accretion rate for particles in a box, including the ParallelFor call
template <typename ContainerType, typename problem_t>
void ComputeAccretionRateInBox(const typename ContainerType::ParIterType &pti, const amrex::Array4<const amrex::Real> &local_state,
			       const amrex::Array4<amrex::Real> &local_accretion_rate,
			       const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> &plo,
			       const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> &dx,
			       std::array<amrex::Array4<const amrex::Real>, AMREX_SPACEDIM> fab_fc,
			       amrex::Real /*time*/, amrex::Real dt, int /*mass_index*/)
{
	const BL_PROFILE("SinkAccretionUtils::ComputeAccretionRateInBox()");

	auto &particles = pti.GetArrayOfStructs();
	auto *pData = particles().data();
	const amrex::Long np = pti.numParticles();

	const double dx_max = std::max({dx[0], dx[1], dx[2]});
	const double vol = AMREX_D_TERM(dx[0], *dx[1], *dx[2]);

	const bool use_uniform_kernel = sink_particle_use_uniform_kernel;

	amrex::ParallelFor(np, [=] AMREX_GPU_DEVICE(int64_t idx) {
		auto &p = pData[idx];

		// Find the cell containing the particle
		int ix = static_cast<int>((p.pos(0) - plo[0]) / dx[0]);
		int iy = static_cast<int>((p.pos(1) - plo[1]) / dx[1]);
		int iz = static_cast<int>((p.pos(2) - plo[2]) / dx[2]);

		auto const *fab_fc_ptr = (fab_fc[0]) ? &fab_fc : nullptr;

		const auto [M_dot, r_K] = compute_Mdot_and_r_K<problem_t>(local_state, ix, iy, iz, p.rdata(0), p.pos(0), p.pos(1),
									  p.pos(2), p.rdata(1), p.rdata(2), p.rdata(3), plo, dx, fab_fc_ptr);

		AMREX_ASSERT(M_dot >= 0.0);

		// compute the sum of the accretion kernel weight function
		double w_sum = 0.0;

		for (int ii = ix - stencil_size; ii <= ix + stencil_size; ++ii) {
			for (int jj = iy - stencil_size; jj <= iy + stencil_size; ++jj) {
				for (int kk = iz - stencil_size; kk <= iz + stencil_size; ++kk) {
					const double x = plo[0] + (ii + static_cast<amrex::Real>(0.5)) * dx[0] - p.pos(0);
					const double y = plo[1] + (jj + static_cast<amrex::Real>(0.5)) * dx[1] - p.pos(1);
					const double z = plo[2] + (kk + static_cast<amrex::Real>(0.5)) * dx[2] - p.pos(2);
					const double r_sqr = x * x + y * y + z * z;

					double r_acc_sqr = stencil_size * stencil_size * dx_max * dx_max;
					if (use_uniform_kernel) {
						r_acc_sqr = std::numeric_limits<double>::infinity();
					}

					if (r_sqr > r_acc_sqr * r_acc_tolerance) {
						continue;
					}

					double w = compute_accretion_kernel(r_sqr, r_K);
					if (use_uniform_kernel) {
						w = 1.0;
					}

					w_sum += w;
				}
			}
		}

		// compute the accretion rate at each cell; use atomic operations
		for (int ii = ix - stencil_size; ii <= ix + stencil_size; ++ii) {
			for (int jj = iy - stencil_size; jj <= iy + stencil_size; ++jj) {
				for (int kk = iz - stencil_size; kk <= iz + stencil_size; ++kk) {
					const double x = plo[0] + (ii + static_cast<amrex::Real>(0.5)) * dx[0] - p.pos(0);
					const double y = plo[1] + (jj + static_cast<amrex::Real>(0.5)) * dx[1] - p.pos(1);
					const double z = plo[2] + (kk + static_cast<amrex::Real>(0.5)) * dx[2] - p.pos(2);
					const double r_sqr = x * x + y * y + z * z;

					double r_acc_sqr = stencil_size * stencil_size * dx_max * dx_max;
					if (use_uniform_kernel) {
						r_acc_sqr = std::numeric_limits<double>::infinity();
					}

					if (r_sqr > r_acc_sqr * r_acc_tolerance) {
						continue;
					}

					double w = compute_accretion_kernel(r_sqr, r_K);
					if (use_uniform_kernel) {
						w = 1.0;
					}

					const double M_dot_cell = -M_dot * w / w_sum;

					// Compute the relative accretion rate and add it to local_accretion_rate.
					// M_dot_cell is negative, therefore rel_accretion_rate is negative.
					const double rho = local_state(ii, jj, kk, HydroSystem<problem_t>::density_index);
					AMREX_ASSERT(rho > 0.0);

					const double rel_accretion_rate = M_dot_cell * dt / (vol * rho);
					AMREX_ASSERT(rel_accretion_rate <= 0.0);

					amrex::Gpu::Atomic::AddNoRet(&local_accretion_rate(ii, jj, kk, 0), rel_accretion_rate);

					// Deposit count into the last component for roundoff algorithm.
					const int count_comp = Physics_NumVars::numHydroVars;
					amrex::Gpu::Atomic::AddNoRet(&local_accretion_rate(ii, jj, kk, count_comp), 1.0);
				}
			}
		}
	});
}

// Compute the scale-down factor for the accretion rate.
//
// This function is the single source of truth for the final accepted accretion.
//
// It modifies state_accretion_rate from the raw requested accretion rate to the
// final accepted accretion rate. It also writes scale_down so that the particle
// mass update uses the same accepted fraction.
//
// Definitions:
//   accretion_rate_original < 0:
//       raw requested fractional gas density change
//
//   accretion_rate_new <= 0:
//       final accepted fractional gas density change
//
//   scale_down:
//       accretion_rate_new / accretion_rate_original
//
// This guarantees:
//   gas_removed == particle_added
//
// The fixed density stop is enforced here:
//   rho_new = rho_cell * (1 + accretion_rate_new) >= rho_stop
template <typename problem_t>
void ComputeScaleDown(amrex::MultiFab &state, amrex::MultiFab &accretion_rate, amrex::MultiFab &scale_down, const amrex::Geometry &geom,
		      std::array<amrex::MultiFab, AMREX_SPACEDIM> const *state_fc)
{
	const BL_PROFILE("SinkAccretionUtils::ComputeScaleDown()");

	const auto &local_state_arr = state.arrays();
	const auto &local_accretion_rate_arr = accretion_rate.arrays();
	const auto &local_scale_down_arr = scale_down.arrays();

	const auto &dx = geom.CellSizeArray();
	const double dx_max = std::max({dx[0], dx[1], dx[2]});

	std::remove_reference_t<decltype((*state_fc)[0].const_arrays())> state_fc_x0{};
	std::remove_reference_t<decltype((*state_fc)[1].const_arrays())> state_fc_x1{};
	std::remove_reference_t<decltype((*state_fc)[2].const_arrays())> state_fc_x2{};

	if (state_fc != nullptr) {
		state_fc_x0 = (*state_fc)[0].const_arrays();
		state_fc_x1 = (*state_fc)[1].const_arrays();
		state_fc_x2 = (*state_fc)[2].const_arrays();
	}

	amrex::ParallelFor(accretion_rate, [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept {
		const double accretion_rate_original = local_accretion_rate_arr[bx](i, j, k);

		// Cells with zero or positive accretion request are not being accreted.
		// Set both gas accretion and particle scale-down to zero for these cells.
		if (accretion_rate_original >= 0.0) {
			local_accretion_rate_arr[bx](i, j, k) = 0.0;
			local_scale_down_arr[bx](i, j, k) = 0.0;
			return;
		}

		AMREX_ASSERT(accretion_rate_original < 0.0);

		std::array<amrex::Array4<const amrex::Real>, AMREX_SPACEDIM> fab_fc{};
		std::array<amrex::Array4<const amrex::Real>, AMREX_SPACEDIM> const *fab_fc_ptr = nullptr;

		if (state_fc != nullptr) {
			fab_fc[0] = state_fc_x0[bx];
			fab_fc[1] = state_fc_x1[bx];
			fab_fc[2] = state_fc_x2[bx];
			fab_fc_ptr = &fab_fc;
		}

		const double rho_cell = local_state_arr[bx](i, j, k, HydroSystem<problem_t>::density_index);
		AMREX_ASSERT(rho_cell > 0.0);

		// Compute local Jeans / Truelove density.
		// For MHD runs, ComputePlasmaBeta + computeJeansDensity include the beta
		// correction used by the Quokka sink implementation.
		const double cs_cell = HydroSystem<problem_t>::ComputeIsothermalSoundSpeed(local_state_arr[bx], i, j, k, fab_fc_ptr);
		const double plasma_beta = HydroSystem<problem_t>::ComputePlasmaBeta(local_state_arr[bx], i, j, k, fab_fc_ptr);
		const double rho_J = ParticleUtils::computeJeansDensity(cs_cell, dx_max, plasma_beta);

		// Fixed stop density.
		// If you want to use 0.1 * rho_J instead of a fixed density, replace this line with:
		//     const double rho_stop = 0.1 * rho_J;
		const double rho_stop = sink_accretion_rho_stop_density;

		// If this cell is already below the stop density, do not accrete from it.
		if (rho_cell <= rho_stop) {
			local_accretion_rate_arr[bx](i, j, k) = 0.0;
			local_scale_down_arr[bx](i, j, k) = 0.0;
			return;
		}

		// Start from the original accumulated Bondi-Hoyle accretion request.
		double accretion_rate_new = accretion_rate_original;

		// Original fractional limiter:
		// ordinary accretion cannot remove more than 25% in one update unless
		// the Jeans correction below requires stronger removal.
		const double accretion_rate_floor = -0.25;
		if (accretion_rate_new < accretion_rate_floor) {
			accretion_rate_new = accretion_rate_floor;
		}

		// Original Jeans-density correction:
		// If the post-accretion density would still be above rho_J, increase
		// the accretion so that rho_new becomes rho_J.
		//
		// This is the Truelove-enforcement channel. It may make scale_down > 1,
		// which is intentional: the gas side removes more mass than the raw
		// Bondi-Hoyle request, so the particle side must receive the same mass.
		if ((1.0 + accretion_rate_new) * rho_cell > rho_J) {
			const double accretion_rate_jeans = rho_J / rho_cell - 1.0;
			accretion_rate_new = accretion_rate_jeans;
		}

		// New fixed lower-density limiter:
		//
		// rho_new = rho_cell * (1 + accretion_rate_new)
		// require rho_new >= rho_stop.
		//
		// Therefore:
		// accretion_rate_new >= rho_stop / rho_cell - 1.
		const double accretion_rate_min_from_rho_stop = rho_stop / rho_cell - 1.0;
		if (accretion_rate_new < accretion_rate_min_from_rho_stop) {
			accretion_rate_new = accretion_rate_min_from_rho_stop;
		}

		// Numerical safety.
		if (accretion_rate_new > 0.0) {
			accretion_rate_new = 0.0;
		}

		AMREX_ASSERT(accretion_rate_new <= 0.0);
		AMREX_ASSERT(accretion_rate_new > -1.0);

		local_accretion_rate_arr[bx](i, j, k) = accretion_rate_new;

		// Essential consistency condition:
		//
		// UpdateParticleMassAndMomentumInBox recomputes M_dot_cell and then
		// multiplies by local_scale_down. Therefore scale_down must be exactly
		// the ratio between final accepted accretion and original requested
		// accretion.
		//
		// Do NOT clamp this to <= 1.0, because the Jeans correction can
		// intentionally make the accepted rate larger than the raw Bondi-Hoyle
		// request.
		local_scale_down_arr[bx](i, j, k) = accretion_rate_new / accretion_rate_original;
	});

	// Synchronize scale_down into ghost cells.
	// This is necessary because UpdateParticleMassAndMomentumInBox reads
	// local_scale_down over the particle stencil, which may include ghost cells.
	scale_down.FillBoundary(geom.periodicity());
}

template <typename problem_t>
auto ComputeGasRemovedMass(amrex::MultiFab &state, amrex::MultiFab &state_accretion_rate, const amrex::Geometry &geom) -> amrex::Real
{
	const auto &state_arr = state.const_arrays();
	const auto &accretion_rate_arr = state_accretion_rate.const_arrays();

	const auto dx = geom.CellSizeArray();
	const amrex::Real vol = AMREX_D_TERM(dx[0], *dx[1], *dx[2]);

	amrex::ReduceOps<amrex::ReduceOpSum> reduce_op;
	amrex::ReduceData<amrex::Real> reduce_data(reduce_op);

	reduce_op.eval(state, amrex::IntVect(0), reduce_data,
		       [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) -> amrex::Real {
			       const amrex::Real rho = state_arr[bx](i, j, k, HydroSystem<problem_t>::density_index);
			       const amrex::Real accretion_rate_cell = accretion_rate_arr[bx](i, j, k);
			       return -accretion_rate_cell * rho * vol;
		       });

	auto hv = reduce_data.value(reduce_op);
	amrex::Real gas_removed = amrex::get<0>(hv);

	amrex::ParallelDescriptor::ReduceRealSum(gas_removed);

	return gas_removed;
}

template <typename ContainerType>
auto ComputeTotalParticleMass(ContainerType *container, int lev, int mass_index) -> amrex::Real
{
	amrex::Real total_mass = 0.0;

	for (typename ContainerType::ParIterType pti(*container, lev); pti.isValid(); ++pti) {
		auto &particles = pti.GetArrayOfStructs();
		auto *pData = particles().data();
		const amrex::Long np = pti.numParticles();

		amrex::ReduceOps<amrex::ReduceOpSum> reduce_op;
		amrex::ReduceData<amrex::Real> reduce_data(reduce_op);

		reduce_op.eval(np, reduce_data, [=] AMREX_GPU_DEVICE(amrex::Long idx) -> amrex::Real {
			const auto &p = pData[idx];
			return p.rdata(mass_index);
		});

		auto hv = reduce_data.value(reduce_op);
		total_mass += amrex::get<0>(hv);
	}

	amrex::ParallelDescriptor::ReduceRealSum(total_mass);

	return total_mass;
}

// Function to update particle mass and momentum for particles in a box, including the ParallelFor call
template <typename ContainerType, typename problem_t>
void UpdateParticleMassAndMomentumInBox(const typename ContainerType::ParIterType &pti, const amrex::Array4<const amrex::Real> &local_state,
					const amrex::Array4<const amrex::Real> &local_scale_down,
					const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> &plo,
					const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> &dx,
					std::array<amrex::Array4<const amrex::Real>, AMREX_SPACEDIM> fab_fc,
					int mass_index, amrex::Real /*time*/, amrex::Real dt, amrex::Real /*vol*/,
					int mdot_index = -1, int ang_mom_index = -1)
{
	const BL_PROFILE("SinkAccretionUtils::UpdateParticleMassAndMomentumInBox()");

	auto &particles = pti.GetArrayOfStructs();
	auto *pData = particles().data();

	const double dx_max = std::max({dx[0], dx[1], dx[2]});
	const amrex::Long np = pti.numParticles();

	const bool use_uniform_kernel = sink_particle_use_uniform_kernel;

	amrex::ParallelFor(np, [=] AMREX_GPU_DEVICE(int64_t idx) {
		auto &p = pData[idx];

		// Find the cell containing the particle
		int ix = static_cast<int>((p.pos(0) - plo[0]) / dx[0]);
		int iy = static_cast<int>((p.pos(1) - plo[1]) / dx[1]);
		int iz = static_cast<int>((p.pos(2) - plo[2]) / dx[2]);

		auto const *fab_fc_ptr = (fab_fc[0]) ? &fab_fc : nullptr;

		const auto [M_dot, r_K] = compute_Mdot_and_r_K<problem_t>(local_state, ix, iy, iz, p.rdata(0), p.pos(0), p.pos(1),
									  p.pos(2), p.rdata(1), p.rdata(2), p.rdata(3), plo, dx, fab_fc_ptr);

		// compute the sum of the accretion kernel weight function
		double w_sum = 0.0;

		for (int ii = ix - stencil_size; ii <= ix + stencil_size; ++ii) {
			for (int jj = iy - stencil_size; jj <= iy + stencil_size; ++jj) {
				for (int kk = iz - stencil_size; kk <= iz + stencil_size; ++kk) {
					const double x = plo[0] + (ii + static_cast<amrex::Real>(0.5)) * dx[0] - p.pos(0);
					const double y = plo[1] + (jj + static_cast<amrex::Real>(0.5)) * dx[1] - p.pos(1);
					const double z = plo[2] + (kk + static_cast<amrex::Real>(0.5)) * dx[2] - p.pos(2);
					const double r_sqr = x * x + y * y + z * z;

					double r_acc_sqr = stencil_size * stencil_size * dx_max * dx_max;
					if (use_uniform_kernel) {
						r_acc_sqr = std::numeric_limits<double>::infinity();
					}

					if (r_sqr > r_acc_sqr * r_acc_tolerance) {
						continue;
					}

					double w = compute_accretion_kernel(r_sqr, r_K);
					if (use_uniform_kernel) {
						w = 1.0;
					}

					w_sum += w;
				}
			}
		}

		// get particle velocity
		const double pvx = p.rdata(mass_index + 1);
		const double pvy = p.rdata(mass_index + 2);
		const double pvz = p.rdata(mass_index + 3);

		// compute the accreted mass and momentum
		double accreted_mass = 0.0;
		double accreted_momentum_x = 0.0;
		double accreted_momentum_y = 0.0;
		double accreted_momentum_z = 0.0;

		double accreted_ang_mom_x = 0.0;
		double accreted_ang_mom_y = 0.0;
		double accreted_ang_mom_z = 0.0;

		for (int ii = ix - stencil_size; ii <= ix + stencil_size; ++ii) {
			for (int jj = iy - stencil_size; jj <= iy + stencil_size; ++jj) {
				for (int kk = iz - stencil_size; kk <= iz + stencil_size; ++kk) {
					const double x = plo[0] + (ii + static_cast<amrex::Real>(0.5)) * dx[0] - p.pos(0);
					const double y = plo[1] + (jj + static_cast<amrex::Real>(0.5)) * dx[1] - p.pos(1);
					const double z = plo[2] + (kk + static_cast<amrex::Real>(0.5)) * dx[2] - p.pos(2);
					const double r_sqr = x * x + y * y + z * z;

					double r_acc_sqr = stencil_size * stencil_size * dx_max * dx_max;
					if (use_uniform_kernel) {
						r_acc_sqr = std::numeric_limits<double>::infinity();
					}

					if (r_sqr > r_acc_sqr * r_acc_tolerance) {
						continue;
					}

					double w = compute_accretion_kernel(r_sqr, r_K);
					if (use_uniform_kernel) {
						w = 1.0;
					}

					const double M_dot_cell = -M_dot * w / w_sum;

					// This is the synchronization point with ComputeScaleDown().
					// scale_down_factor has already included:
					//   * ordinary 25% limiter
					//   * Jeans-density correction
					//   * fixed rho_stop lower-density limiter
					const double scale_down_factor = local_scale_down(ii, jj, kk);

					// M_dot_cell is negative, so multiply by -1 to get accreted mass.
					const double accreted_mass_cell = -M_dot_cell * dt * scale_down_factor;

					const double rho = local_state(ii, jj, kk, HydroSystem<problem_t>::density_index);
					AMREX_ASSERT(rho > 0.0);

					const double vx_lab = local_state(ii, jj, kk, HydroSystem<problem_t>::x1Momentum_index) / rho;
					const double vy_lab = local_state(ii, jj, kk, HydroSystem<problem_t>::x2Momentum_index) / rho;
					const double vz_lab = local_state(ii, jj, kk, HydroSystem<problem_t>::x3Momentum_index) / rho;

					const double vx_rel = vx_lab - pvx;
					const double vy_rel = vy_lab - pvy;
					const double vz_rel = vz_lab - pvz;

					accreted_mass += accreted_mass_cell;
					accreted_momentum_x += accreted_mass_cell * vx_lab;
					accreted_momentum_y += accreted_mass_cell * vy_lab;
					accreted_momentum_z += accreted_mass_cell * vz_lab;

					// Angular momentum:
					// L += dm * (r_cell x v_rel)
					if (ang_mom_index >= 0) {
						accreted_ang_mom_x += accreted_mass_cell * (y * vz_rel - z * vy_rel);
						accreted_ang_mom_y += accreted_mass_cell * (z * vx_rel - x * vz_rel);
						accreted_ang_mom_z += accreted_mass_cell * (x * vy_rel - y * vx_rel);
					}
				}
			}
		}

		const double par_m = p.rdata(mass_index);
		const double par_m_new = par_m + accreted_mass;

		p.rdata(mass_index) = par_m_new;
		p.rdata(mass_index + 1) = (par_m * p.rdata(mass_index + 1) + accreted_momentum_x) / par_m_new;
		p.rdata(mass_index + 2) = (par_m * p.rdata(mass_index + 2) + accreted_momentum_y) / par_m_new;
		p.rdata(mass_index + 3) = (par_m * p.rdata(mass_index + 3) + accreted_momentum_z) / par_m_new;

		if (mdot_index >= 0) {
			p.rdata(mdot_index) = accreted_mass / dt;
		}

		if (ang_mom_index >= 0) {
			p.rdata(ang_mom_index) += accreted_ang_mom_x;
			p.rdata(ang_mom_index + 1) += accreted_ang_mom_y;
			p.rdata(ang_mom_index + 2) += accreted_ang_mom_z;
		}
	});
}

template <typename ContainerType, typename problem_t>
void UpdateParticleMassAndMomentum(ContainerType *container, amrex::MultiFab &state, amrex::MultiFab &scale_down,
				   std::array<amrex::MultiFab, AMREX_SPACEDIM> const *state_fc,
				   int lev, int mass_index, amrex::Real time, amrex::Real dt,
				   int mdot_index = -1, int ang_mom_index = -1)
{
	const BL_PROFILE("SinkAccretionUtils::UpdateParticleMassAndMomentum()");

	for (typename ContainerType::ParIterType pti(*container, lev); pti.isValid(); ++pti) {
		const auto &local_state = state.array(pti);
		const auto &local_scale_down = scale_down.array(pti);

		std::array<amrex::Array4<const amrex::Real>, AMREX_SPACEDIM> local_fab_fc{};

		if (state_fc != nullptr) {
			local_fab_fc[0] = (*state_fc)[0].array(pti);
			local_fab_fc[1] = (*state_fc)[1].array(pti);
			local_fab_fc[2] = (*state_fc)[2].array(pti);
		}

		const auto &geom = container->Geom(lev);
		const auto plo = geom.ProbLoArray();
		const auto dx = geom.CellSizeArray();

		const amrex::Real vol = AMREX_D_TERM(dx[0], *dx[1], *dx[2]);

		UpdateParticleMassAndMomentumInBox<ContainerType, problem_t>(pti, local_state, local_scale_down, plo, dx, local_fab_fc,
									     mass_index, time, dt, vol, mdot_index, ang_mom_index);
	}
}

template <typename problem_t>
void UpdateHydroState(amrex::MultiFab &state, amrex::MultiFab &accretion_rate)
{
	const BL_PROFILE("SinkAccretionUtils::UpdateHydroState()");

	const auto &local_accretion_rate_arr = accretion_rate.arrays();
	const auto &state_arr = state.arrays();

	amrex::ParallelFor(state, [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept {
		const double accretion_rate_cell = local_accretion_rate_arr[bx](i, j, k);

		AMREX_ASSERT(accretion_rate_cell <= 0.0);
		AMREX_ASSERT(accretion_rate_cell > -1.0);

		const double accretion_down_factor = 1.0 + accretion_rate_cell;

		AMREX_ASSERT(accretion_down_factor > 0.0);

		state_arr[bx](i, j, k, HydroSystem<problem_t>::density_index) *= accretion_down_factor;
		state_arr[bx](i, j, k, HydroSystem<problem_t>::x1Momentum_index) *= accretion_down_factor;
		state_arr[bx](i, j, k, HydroSystem<problem_t>::x2Momentum_index) *= accretion_down_factor;
		state_arr[bx](i, j, k, HydroSystem<problem_t>::x3Momentum_index) *= accretion_down_factor;
		state_arr[bx](i, j, k, HydroSystem<problem_t>::internalEnergy_index) *= accretion_down_factor;
		state_arr[bx](i, j, k, HydroSystem<problem_t>::energy_index) *= accretion_down_factor;

		for (int n = 0; n < Physics_Traits<problem_t>::numPassiveScalars; ++n) {
			state_arr[bx](i, j, k, HydroSystem<problem_t>::scalar0_index + n) *= accretion_down_factor;
		}
	});
}

// Functor for computing the accretion rate and storing it in buffer state `accretion_rate`.
template <typename ContainerType, typename problem_t>
void computeAccretion(ContainerType *container, amrex::MultiFab &state, amrex::MultiFab &accretion_rate,
		      std::array<amrex::MultiFab, AMREX_SPACEDIM> const *state_fc,
		      int lev, amrex::Real time, amrex::Real dt, int mass_index)
{
	const BL_PROFILE("SinkAccretionUtils::computeAccretion()");

	for (typename ContainerType::ParIterType pti(*container, lev); pti.isValid(); ++pti) {
		const auto &local_state = state.array(pti);
		const auto &local_accretion_rate = accretion_rate.array(pti);

		std::array<amrex::Array4<const amrex::Real>, AMREX_SPACEDIM> local_fab_fc{};

		if (state_fc != nullptr) {
			local_fab_fc[0] = (*state_fc)[0].array(pti);
			local_fab_fc[1] = (*state_fc)[1].array(pti);
			local_fab_fc[2] = (*state_fc)[2].array(pti);
		}

		const auto &geom = container->Geom(lev);
		const auto plo = geom.ProbLoArray();
		const auto dx = geom.CellSizeArray();

		ComputeAccretionRateInBox<ContainerType, problem_t>(pti, local_state, local_accretion_rate, plo, dx, local_fab_fc, time, dt, mass_index);
	}

	// Sum boundary cell values to real cells.
	accretion_rate.SumBoundary(container->Geom(lev).periodicity());
}

// Functor for applying accretion.
template <typename ContainerType, typename problem_t>
void applyAccretion(ContainerType *container, amrex::MultiFab &state, amrex::MultiFab &state_accretion_rate,
		    std::array<amrex::MultiFab, AMREX_SPACEDIM> const *state_fc,
		    const amrex::Geometry &geom, int lev, amrex::Real time, amrex::Real dt,
		    int mass_index, int mdot_index = -1, int ang_mom_index = -1)
{
	const BL_PROFILE("SinkAccretionUtils::applyAccretion()");

	// Step 2: Compute the scale_down factor.
	//
	// Keep the default value as 1.0. This is the neutral multiplier.
	// ComputeScaleDown() will explicitly set:
	//   * 0.0 for cells with no accepted accretion
	//   * accretion_rate_new / accretion_rate_original for accreting cells
	//
	// Do NOT initialize this to 0.0 globally, otherwise cells not explicitly
	// touched before FillBoundary may accidentally suppress particle accretion.
	amrex::MultiFab scale_down(state.boxArray(), state.DistributionMap(), 1, state.nGrow());
	scale_down.setVal(1.0);

	// Update accretion_rate and compute scale_down.
	// This is where the fixed density stop is applied.
	ComputeScaleDown<problem_t>(state, state_accretion_rate, scale_down, geom, state_fc);

	amrex::Real gas_removed = 0.0;
	amrex::Real particle_mass_before = 0.0;

	if constexpr (sink_accretion_mass_check) {
		gas_removed = ComputeGasRemovedMass<problem_t>(state, state_accretion_rate, geom);
		particle_mass_before = ComputeTotalParticleMass<ContainerType>(container, lev, mass_index);
	}

	// Step 3: Update particle mass, momentum, accretion rate, and angular momentum.
	//
	// The particle update recomputes the local M_dot_cell, but multiplies it by
	// scale_down. Since scale_down was computed from the final accepted gas
	// accretion rate, particle_added remains synchronized with gas_removed.
	UpdateParticleMassAndMomentum<ContainerType, problem_t>(container, state, scale_down, state_fc, lev, mass_index, time, dt,
							       mdot_index, ang_mom_index);

	if constexpr (sink_accretion_mass_check) {
		const amrex::Real particle_mass_after = ComputeTotalParticleMass<ContainerType>(container, lev, mass_index);
		const amrex::Real particle_added = particle_mass_after - particle_mass_before;
		const amrex::Real error = particle_added - gas_removed;
		const amrex::Real rel_error = (gas_removed > 0.0) ? error / gas_removed : 0.0;

		amrex::Print() << "[SinkAccretionMassCheck]"
			       << " lev=" << lev
			       << " time=" << time
			       << " dt=" << dt
			       << " gas_removed=" << gas_removed
			       << " particle_added=" << particle_added
			       << " error=" << error
			       << " rel_error=" << rel_error << "\n";
	}

	// Step 4: Update the hydro state.
	//
	// This is done last because the original state is needed when updating
	// particle mass and momentum.
	UpdateHydroState<problem_t>(state, state_accretion_rate);
}

} // namespace SinkAccretionUtils

} // namespace quokka

#endif // PARTICLE_ACCRETION_HPP_