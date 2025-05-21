#ifndef PARTICLE_ACCRETION_HPP_
#define PARTICLE_ACCRETION_HPP_

#include "AMReX_Array4.H"
#include "AMReX_MultiFab.H"
#include "AMReX_REAL.H"
#include "gcem.hpp"
#include "hydro/hydro_system.hpp"
#include "particles/particle_types.hpp"
#include "particles/particle_utils.hpp"

namespace quokka
{

enum class AccretionScheme { Threshold = 0, BondiHoyle = 1 };

// manually set the accretion scheme
constexpr AccretionScheme accretion_scheme = AccretionScheme::BondiHoyle;

#if AMREX_SPACEDIM == 3

namespace SinkAccretionUtils
{

constexpr int stencil_size = quokka::ParticleUtils::stencil_size;

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto get_delta_rho(double rho, double rho_sink) -> double { return -0.5 * (rho - rho_sink) / rho; }

template <typename problem_t>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto compute_Mdot_and_r_K(const amrex::Array4<const amrex::Real> &local_state, int ix, int iy, int iz, double par_mass,
								   double par_x, double par_y, double par_z,
								   const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> &plo,
								   const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> &dx) -> std::tuple<double, double>
{
	const double dx_max = std::max({dx[0], dx[1], dx[2]});

	// compute the average density, momentum, and sound speed in the accretion zone
	// tot_mass = sum(rho) * V
	// tot_px = sum(px) * V
	// tot_py = sum(py) * V
	// tot_pz = sum(pz) * V
	// tot_cs = sum(cs * rho) * V
	// mean_density = tot_mass / (N * V) = sum(rho) / N
	// mean_vx = tot_px / tot_mass = sum(px) / sum(rho)
	// mean_vy = tot_py / tot_mass = sum(py) / sum(rho)
	// mean_vz = tot_pz / tot_mass = sum(pz) / sum(rho)
	// mean_cs = tot_cs / tot_mass = sum(cs * rho) / sum(rho)
	int n_cells = 0;
	double sum_rho = 0.0;
	double sum_px = 0.0;
	double sum_py = 0.0;
	double sum_pz = 0.0;
	double sum_cs = 0.0;
	for (int ii = ix - stencil_size; ii <= ix + stencil_size; ++ii) {
		for (int jj = iy - stencil_size; jj <= iy + stencil_size; ++jj) {
			for (int kk = iz - stencil_size; kk <= iz + stencil_size; ++kk) {
				const double x = par_x - plo[0] - ii * dx[0];
				const double y = par_y - plo[1] - jj * dx[1];
				const double z = par_z - plo[2] - kk * dx[2];
				const double r_sqr = x * x + y * y + z * z;
				const double r_acc_sqr = stencil_size * stencil_size * dx_max * dx_max;
				if (r_sqr > r_acc_sqr) {
					continue;
				}
				const double rho = local_state(ii, jj, kk, HydroSystem<problem_t>::density_index);
				const double px = local_state(ii, jj, kk, HydroSystem<problem_t>::x1Momentum_index);
				const double py = local_state(ii, jj, kk, HydroSystem<problem_t>::x2Momentum_index);
				const double pz = local_state(ii, jj, kk, HydroSystem<problem_t>::x3Momentum_index);
				double cs = HydroSystem<problem_t>::ComputeSoundSpeed(local_state, ii, jj, kk);
				if constexpr (quokka::EOS_Traits<problem_t>::gamma == 1.0) {
					cs = quokka::EOS_Traits<problem_t>::cs_isothermal;
				}
				sum_rho += rho;
				sum_px += px;
				sum_py += py;
				sum_pz += pz;
				sum_cs += cs * rho;
				n_cells += 1;
			}
		}
	}
	const double rho_infty = sum_rho / n_cells;
	const double vx_infty = sum_px / sum_rho;
	const double vy_infty = sum_py / sum_rho;
	const double vz_infty = sum_pz / sum_rho;
	const double cs_infty = sum_cs / sum_rho;
	AMREX_ASSERT(!std::isnan(rho_infty));
	AMREX_ASSERT(rho_infty > 0.0);
	AMREX_ASSERT(!std::isnan(cs_infty));
	AMREX_ASSERT(cs_infty > 0.0);

	// compute Bondi-Hoyle accretion radius, r_BH = G M / (v^2 + c^2)
	const double v_infty_sqr = vx_infty * vx_infty + vy_infty * vy_infty + vz_infty * vz_infty;
	const double r_BH = C::Gconst * par_mass / (v_infty_sqr + cs_infty * cs_infty);

	// Compute the accretion rate in the accretion zone,
	// M_dot = 4 pi rho_infty r_BH^2 * sqrt(v_infty^2 + lambda^2 c_s^2), where lambda = exp(3/2) / 4
	constexpr double lambda = gcem::exp(1.5) / 4.0;
	AMREX_ASSERT(rho_infty > 0.0);
	const double M_dot = 4.0 * M_PI * rho_infty * r_BH * r_BH * std::sqrt(v_infty_sqr + lambda * lambda * cs_infty * cs_infty);
	AMREX_ASSERT(M_dot >= 0.0);

	// Compute accretion kernel radius,
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
			       const amrex::Array4<amrex::Real> &local_accretion_rate, const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> &plo,
			       const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> &dx, amrex::Real /*time*/, amrex::Real dt, int /*mass_index*/)
{
	// Get the particle array of structs
	auto &particles = pti.GetArrayOfStructs();
	auto *pData = particles().data();
	const amrex::Long np = pti.numParticles();
	const double dx_max = std::max({dx[0], dx[1], dx[2]});
	const double vol = AMREX_D_TERM(dx[0], *dx[1], *dx[2]);

	const bool use_uniform_kernel = sink_particle_use_uniform_kernel;

	amrex::ParallelFor(np, [=] AMREX_GPU_DEVICE(int64_t idx) {
		auto &p = pData[idx]; // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)

		// Find the cell containing the particle
		int ix = static_cast<int>((p.pos(0) - plo[0]) / dx[0]);
		int iy = static_cast<int>((p.pos(1) - plo[1]) / dx[1]);
		int iz = static_cast<int>((p.pos(2) - plo[2]) / dx[2]);

		const auto [M_dot, r_K] = compute_Mdot_and_r_K<problem_t>(local_state, ix, iy, iz, p.rdata(0), p.pos(0), p.pos(1), p.pos(2), plo, dx);
		AMREX_ASSERT(M_dot >= 0.0);

		// compute the sum of the accretion kernel weight function, w = exp(- r^2 / r_K^2)
		double w_sum = 0.0;
		for (int ii = ix - stencil_size; ii <= ix + stencil_size; ++ii) {
			for (int jj = iy - stencil_size; jj <= iy + stencil_size; ++jj) {
				for (int kk = iz - stencil_size; kk <= iz + stencil_size; ++kk) {
					const double x = p.pos(0) - plo[0] - ii * dx[0];
					const double y = p.pos(1) - plo[1] - jj * dx[1];
					const double z = p.pos(2) - plo[2] - kk * dx[2];
					const double r_sqr = x * x + y * y + z * z;
					double r_acc_sqr = stencil_size * stencil_size * dx_max * dx_max;
					if (use_uniform_kernel) {
						// use a large accretion radius; this has the effect of using a cubic, uniform kernel
						r_acc_sqr = std::numeric_limits<double>::infinity();
					}
					if (r_sqr > r_acc_sqr) {
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
					const double x = p.pos(0) - plo[0] - ii * dx[0];
					const double y = p.pos(1) - plo[1] - jj * dx[1];
					const double z = p.pos(2) - plo[2] - kk * dx[2];
					const double r_sqr = x * x + y * y + z * z;
					double r_acc_sqr = stencil_size * stencil_size * dx_max * dx_max;
					if (use_uniform_kernel) {
						// use a large accretion radius; this has the effect of using a cubic, uniform kernel
						r_acc_sqr = std::numeric_limits<double>::infinity();
					}
					if (r_sqr > r_acc_sqr) {
						continue;
					}
					double w = compute_accretion_kernel(r_sqr, r_K);
					if (use_uniform_kernel) {
						w = 1.0;
					}
					const double M_dot_cell = -M_dot * w / w_sum;

					//------------ This is the part that is different from UpdateParticleMassAndMomentumInBox -----------
					// Compute the relative accretion rate and add it to local_accretion_rate
					const double rho = local_state(ii, jj, kk, HydroSystem<problem_t>::density_index);
					AMREX_ASSERT(rho > 0.0);
					const double rel_accretion_rate = M_dot_cell * dt / (vol * rho);
					AMREX_ASSERT(rel_accretion_rate <= 0.0);
					amrex::Gpu::Atomic::AddNoRet(&local_accretion_rate(ii, jj, kk), rel_accretion_rate);
					//----------------------------------------------------------------------------------------------------
				}
			}
		}
	});
}

// Compute the scale down factor for the accretion rate. We first prevent the gas density from dropping below 75% of its initial value.
// Then, if the density in the end state is above the Jeans density, we increase the accretion rate so that the density in the end state is
// equal to the Jeans density.
template <typename problem_t>
void ComputeScaleDown(amrex::MultiFab &state, amrex::MultiFab &accretion_rate, amrex::MultiFab &scale_down, const amrex::Geometry &geom)
{
	const auto &local_state_arr = state.arrays();
	const auto &local_accretion_rate_arr = accretion_rate.arrays();
	const auto &local_scale_down_arr = scale_down.arrays();
	const auto &dx = geom.CellSizeArray();
	const double dx_max = std::max({dx[0], dx[1], dx[2]});

	amrex::ParallelFor(accretion_rate, [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept {
		const double accretion_rate_cell = local_accretion_rate_arr[bx](i, j, k);
		const double accretion_rate_floor = -0.25;
		if (accretion_rate_cell < accretion_rate_floor) {
			// scale down the accretion rate to the minimum allowed value
			local_accretion_rate_arr[bx](i, j, k) = accretion_rate_floor;
			local_scale_down_arr[bx](i, j, k) = accretion_rate_floor / accretion_rate_cell;
		}
		AMREX_ASSERT(local_accretion_rate_arr[bx](i, j, k) <= 0.0);
		AMREX_ASSERT(local_accretion_rate_arr[bx](i, j, k) > -1.0);

		// In the accretion zone, if (1 + accretion_rate_cell) * rho > rho_J, set accretion_rate_cell = rho_J / rho - 1
		if (accretion_rate_cell > 0.0) {
			// Compute Jeans density rho_J = J^2 * pi * cs^2 / (G * dx^2)
			constexpr double J = 0.25;
			double cs_cell = HydroSystem<problem_t>::ComputeSoundSpeed(local_state_arr[bx], i, j, k);
			if constexpr (quokka::EOS_Traits<problem_t>::gamma == 1.0) {
				cs_cell = quokka::EOS_Traits<problem_t>::cs_isothermal;
			}
			const double rho_J = J * J * M_PI * cs_cell * cs_cell / (C::Gconst * (dx_max * dx_max));
			const double rho_cell = local_state_arr[bx](i, j, k, HydroSystem<problem_t>::density_index);
			if ((1.0 + accretion_rate_cell) * rho_cell > rho_J) {
				const double accretion_rate_cell_new = rho_J / rho_cell - 1.0;
				local_accretion_rate_arr[bx](i, j, k) = accretion_rate_cell_new;
				local_scale_down_arr[bx](i, j, k) = accretion_rate_cell_new / accretion_rate_cell;
			}
			AMREX_ASSERT(local_accretion_rate_arr[bx](i, j, k) <= 0.0);
			AMREX_ASSERT(local_accretion_rate_arr[bx](i, j, k) > -1.0);
		}
	});

	// synchronize scale_down
	scale_down.FillBoundary(geom.periodicity());
}

// Function to update particle mass and momentum for particles in a box, including the ParallelFor call
template <typename ContainerType, typename problem_t>
void UpdateParticleMassAndMomentumInBox(const typename ContainerType::ParIterType &pti, const amrex::Array4<const amrex::Real> &local_state,
					const amrex::Array4<const amrex::Real> &local_scale_down, const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> &plo,
					const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> &dx, int mass_index, amrex::Real /*time*/, amrex::Real dt,
					amrex::Real /*vol*/)
{
	// Get the particle array of structs
	auto &particles = pti.GetArrayOfStructs();
	auto *pData = particles().data();
	const double dx_max = std::max({dx[0], dx[1], dx[2]});
	const amrex::Long np = pti.numParticles();

	const bool use_uniform_kernel = sink_particle_use_uniform_kernel;

	amrex::ParallelFor(np, [=] AMREX_GPU_DEVICE(int64_t idx) {
		auto &p = pData[idx]; // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)

		// Find the cell containing the particle
		int ix = static_cast<int>((p.pos(0) - plo[0]) / dx[0]);
		int iy = static_cast<int>((p.pos(1) - plo[1]) / dx[1]);
		int iz = static_cast<int>((p.pos(2) - plo[2]) / dx[2]);

		const auto [M_dot, r_K] = compute_Mdot_and_r_K<problem_t>(local_state, ix, iy, iz, p.rdata(0), p.pos(0), p.pos(1), p.pos(2), plo, dx);

		// compute the sum of the accretion kernel weight function, w = exp(- r^2 / r_K^2)
		double w_sum = 0.0;
		for (int ii = ix - stencil_size; ii <= ix + stencil_size; ++ii) {
			for (int jj = iy - stencil_size; jj <= iy + stencil_size; ++jj) {
				for (int kk = iz - stencil_size; kk <= iz + stencil_size; ++kk) {
					const double x = p.pos(0) - plo[0] - ii * dx[0];
					const double y = p.pos(1) - plo[1] - jj * dx[1];
					const double z = p.pos(2) - plo[2] - kk * dx[2];
					const double r_sqr = x * x + y * y + z * z;
					double r_acc_sqr = stencil_size * stencil_size * dx_max * dx_max;
					if (use_uniform_kernel) {
						// use a large accretion radius; this has the effect of using a cubic, uniform kernel
						r_acc_sqr = std::numeric_limits<double>::infinity();
					}
					if (r_sqr > r_acc_sqr) {
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
		double accreted_mass = 0.0;
		double accreted_momentum_x = 0.0;
		double accreted_momentum_y = 0.0;
		double accreted_momentum_z = 0.0;
		for (int ii = ix - stencil_size; ii <= ix + stencil_size; ++ii) {
			for (int jj = iy - stencil_size; jj <= iy + stencil_size; ++jj) {
				for (int kk = iz - stencil_size; kk <= iz + stencil_size; ++kk) {
					const double x = p.pos(0) - plo[0] - ii * dx[0];
					const double y = p.pos(1) - plo[1] - jj * dx[1];
					const double z = p.pos(2) - plo[2] - kk * dx[2];
					const double r_sqr = x * x + y * y + z * z;
					double r_acc_sqr = stencil_size * stencil_size * dx_max * dx_max;
					if (use_uniform_kernel) {
						// use a large accretion radius; this has the effect of using a cubic, uniform kernel
						r_acc_sqr = std::numeric_limits<double>::infinity();
					}
					if (r_sqr > r_acc_sqr) {
						continue;
					}
					double w = compute_accretion_kernel(r_sqr, r_K);
					if (use_uniform_kernel) {
						w = 1.0;
					}
					const double M_dot_cell = -M_dot * w / w_sum;

					//----------------- This is the part that is different from ComputeAccretionRateInBox ---------------
					// Compute the accreted mass and momentum onto the particle
					const double scale_down_factor = local_scale_down(ii, jj, kk);
					// M_dot_cell is negative, so we multiply it by -1 to get the accreted mass
					const double accreted_mass_cell = -M_dot_cell * dt * scale_down_factor;
					const double rho = local_state(ii, jj, kk, HydroSystem<problem_t>::density_index);
					const double vx = local_state(ii, jj, kk, HydroSystem<problem_t>::x1Momentum_index) / rho;
					const double vy = local_state(ii, jj, kk, HydroSystem<problem_t>::x2Momentum_index) / rho;
					const double vz = local_state(ii, jj, kk, HydroSystem<problem_t>::x3Momentum_index) / rho;
					accreted_mass += accreted_mass_cell;
					accreted_momentum_x += accreted_mass_cell * vx;
					accreted_momentum_y += accreted_mass_cell * vy;
					accreted_momentum_z += accreted_mass_cell * vz;
					//-----------------------------------------------------------------------------------------------------
				}
			}
		}

		const double par_m = p.rdata(mass_index);
		const double par_m_new = par_m + accreted_mass;
		p.rdata(mass_index) = par_m_new;
		p.rdata(mass_index + 1) = (par_m * p.rdata(mass_index + 1) + accreted_momentum_x) / par_m_new;
		p.rdata(mass_index + 2) = (par_m * p.rdata(mass_index + 2) + accreted_momentum_y) / par_m_new;
		p.rdata(mass_index + 3) = (par_m * p.rdata(mass_index + 3) + accreted_momentum_z) / par_m_new;
	});
}

template <typename ContainerType, typename problem_t>
void UpdateParticleMassAndMomentum(ContainerType *container, amrex::MultiFab &state, amrex::MultiFab &scale_down, int lev, int mass_index, amrex::Real time,
				   amrex::Real dt)
{
	for (typename ContainerType::ParIterType pti(*container, lev); pti.isValid(); ++pti) {
		// Get the local deposit array for this box
		const auto &local_state = state.array(pti);
		const auto &local_scale_down = scale_down.array(pti);

		// Get geometry information for this level
		const auto &geom = container->Geom(lev);
		const auto plo = geom.ProbLoArray();
		const auto dx = geom.CellSizeArray();

		// Calculate cell volume
		const amrex::Real vol = AMREX_D_TERM(dx[0], *dx[1], *dx[2]);

		// Process particles in this box
		UpdateParticleMassAndMomentumInBox<ContainerType, problem_t>(pti, local_state, local_scale_down, plo, dx, mass_index, time, dt, vol);
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
		state_arr[bx](i, j, k, HydroSystem<problem_t>::internalEnergy_index) *= accretion_down_factor;
		state_arr[bx](i, j, k, HydroSystem<problem_t>::energy_index) *= accretion_down_factor;

		// update passive scalars
		for (int n = 0; n < Physics_Traits<problem_t>::numPassiveScalars; ++n) {
			state_arr[bx](i, j, k, HydroSystem<problem_t>::scalar0_index + n) *= accretion_down_factor;
		}
	});
}

// Functor for computing the accretion rate and store it in a buffer state `accretion_rate`.
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
		const auto dx = geom.CellSizeArray();

		// Process particles in this box
		ComputeAccretionRateInBox<ContainerType, problem_t>(pti, local_state, local_accretion_rate, plo, dx, time, dt, mass_index);
	}

	// Sum boundary cell values to real cells
	accretion_rate.SumBoundary(container->Geom(lev).periodicity());
}

// Functor for applying accretion.
template <typename ContainerType, typename problem_t>
void applyAccretion(ContainerType *container, amrex::MultiFab &state, amrex::MultiFab &state_accretion_rate, const amrex::Geometry &geom, int lev,
		    amrex::Real time, amrex::Real dt, int mass_index)
{
	// Step 2: Compute the scale_down factor. We scale down the accretion rate to prevent accretion rates from exceeding 100%
	// of the available mass.
	amrex::MultiFab scale_down(state.boxArray(), state.DistributionMap(), 1, state.nGrow());
	scale_down.setVal(1.0);
	// Update accretion_rate and compute scale_down
	ComputeScaleDown<problem_t>(state, state_accretion_rate, scale_down, geom);

	// Step 3: Update particle mass and momentum
	UpdateParticleMassAndMomentum<ContainerType, problem_t>(container, state, scale_down, lev, mass_index, time, dt);

	// Step 4: Update the hydro state. We do this at last because the original state is needed for updating particles in step 3.
	UpdateHydroState<problem_t>(state, state_accretion_rate);
}

} // namespace SinkAccretionUtils

#endif // AMREX_SPACEDIM == 3

} // namespace quokka

#endif // PARTICLE_ACCRETION_HPP_
