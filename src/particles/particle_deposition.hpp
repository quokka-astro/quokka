#ifndef PARTICLE_DEPOSITION_HPP_
#define PARTICLE_DEPOSITION_HPP_
//
// Particle deposition functions for transferring particle quantities to the mesh.
//
// To enable deterministic (reproducible) mass deposition on GPUs, compile with:
//   -DQUOKKA_DETERMINISTIC_DEPOSITION
// This sorts particles by cell before deposition to ensure consistent floating-point
// summation order, eliminating GPU race conditions at the cost of some performance.
//

#include <algorithm>

#include "AMReX_Array.H"
#include "AMReX_Array4.H"
#include "AMReX_BLProfiler.H"
#include "AMReX_Extension.H"
#include "AMReX_FArrayBox.H"
#include "AMReX_MakeParticle.H"
#include "AMReX_MultiFab.H"
#include "AMReX_ParticleInterpolators.H"
#include "AMReX_REAL.H"
#include "hydro/hydro_system.hpp"
#include "particles/particle_types.hpp"

namespace quokka
{

constexpr int SN_stencil_size = 3;
constexpr int SN_stencil_array_size = SN_stencil_size + 1;

constexpr double cloudy_H_mass_fraction = 1.0 / (1.0 + 0.1 * 3.971);
constexpr double m_u = C::m_u;

//-------------------- Radiation depositions --------------------

// Functor for depositing radiation energy from particles onto the grid
struct RadDeposition {
	double current_time{}; // Current simulation time
	int start_part_comp{}; // Starting component in particle data
	int start_mesh_comp{}; // Starting component in mesh data
	int num_comp{};	       // Number of components to deposit
	int birthTimeIndex{};  // Index for particle birth time

	// Operator to perform radiation deposition using linear interpolation
	template <typename ContainerType>
	AMREX_GPU_DEVICE AMREX_FORCE_INLINE void operator()(const ContainerType &p, amrex::Array4<amrex::Real> const &radEnergySource,
							    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &plo,
							    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dxi) const noexcept
	{
		amrex::ParticleInterpolator::Linear interp(p, plo, dxi);
		// Deposit radiation energy only if particle is active
		interp.ParticleToMesh(p, radEnergySource, start_part_comp, start_mesh_comp, num_comp,
				      [=] AMREX_GPU_DEVICE(const ContainerType &part, int comp) {
					      if (current_time < part.rdata(birthTimeIndex) || current_time >= part.rdata(birthTimeIndex + 1)) {
						      return 0.0;
					      }
					      return part.rdata(comp) * (AMREX_D_TERM(dxi[0], *dxi[1], *dxi[2]));
				      });
	}
};

#if AMREX_SPACEDIM == 3

//-------------------- Mass depositions --------------------

// Functor for depositing particle mass onto the grid (original non-deterministic version)
struct MassDeposition {
	amrex::Real Gconst{};  // Gravitational constant
	int start_part_comp{}; // Starting component in particle data
	int start_mesh_comp{}; // Starting component in mesh data
	int num_comp{};	       // Number of components to deposit

	// Operator to perform mass deposition using linear interpolation
	template <typename ContainerType>
	AMREX_GPU_DEVICE AMREX_FORCE_INLINE void operator()(const ContainerType &p, amrex::Array4<amrex::Real> const &rho,
						    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &plo,
						    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dxi) const noexcept
	{
		amrex::ParticleInterpolator::Linear interp(p, plo, dxi);
		// Deposit mass weighted by 4 pi G
		interp.ParticleToMesh(p, rho, start_part_comp, start_mesh_comp, num_comp, [=] AMREX_GPU_DEVICE(const ContainerType &part, int comp) {
			return 4.0 * M_PI * Gconst * part.rdata(comp) * (AMREX_D_TERM(dxi[0], *dxi[1], *dxi[2]));
		});
	}
};

// Function interface for deterministic mass deposition
template<typename PTD>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE void depositMassDeterministic(
	const PTD& ptd,
	amrex::Long np,
	amrex::Array4<amrex::Real> const& rho,
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& plo,
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& dxi,
	amrex::Real Gconst,
	int mass_comp) noexcept
{
	// Process particles sequentially to ensure deterministic order
	for (amrex::Long i = 0; i < np; ++i) {
		auto p = amrex::make_particle<typename PTD::ParticleType::ConstType>{}(ptd, i);
		
		// Calculate mass contribution using same formula as original MassDeposition
		const amrex::Real mass_contrib = 4.0 * M_PI * Gconst * p.rdata(mass_comp) * (AMREX_D_TERM(dxi[0], *dxi[1], *dxi[2]));
		
		// Perform linear interpolation to deposit to 8 neighboring cells
		amrex::ParticleInterpolator::Linear interp(p, plo, dxi);
		
		// Manually perform the interpolation without atomic operations
		// This replicates the logic from ParticleInterpolator::Linear::ParticleToMesh
		static constexpr int stencil_width = 2;
		for (int kk = 0; kk <= 1; ++kk) {
			for (int jj = 0; jj <= 1; ++jj) {
				for (int ii = 0; ii <= 1; ++ii) {
					const amrex::Real weight = interp.w[0*stencil_width+ii] *
					                          interp.w[1*stencil_width+jj] *
					                          interp.w[2*stencil_width+kk];
					const amrex::Real val = weight * mass_contrib;
					
					// Direct assignment without atomics (safe because single-threaded per tile)
					rho(interp.index[0]+ii, interp.index[1]+jj, interp.index[2]+kk, 0) += val;
				}
			}
		}
	}
}

//-------------------- Supernova depositions --------------------

namespace SNFeedbackUtils
{

// Function to deposit thermal supernova remnant quantities
template <typename problem_t>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE void
depositThermalSNR(amrex::Array4<amrex::Real> const &local_buffer, const int ix, const int iy, const int iz, const amrex::Real m_ej, const amrex::Real E_blast,
		  const amrex::Real SN_kin_energy, const amrex::Real p_vx, const amrex::Real p_vy, const amrex::Real p_vz, const amrex::Real vol_inverse,
		  const amrex::GpuArray<amrex::GpuArray<amrex::GpuArray<amrex::Real, SN_stencil_array_size>, SN_stencil_array_size>, SN_stencil_array_size>
		      &stencil_weights_gpu) noexcept
{
	for (int ii = -SN_stencil_size; ii <= SN_stencil_size; ++ii) {
		for (int jj = -SN_stencil_size; jj <= SN_stencil_size; ++jj) {
			for (int kk = -SN_stencil_size; kk <= SN_stencil_size; ++kk) {
				const int iii = std::abs(ii);
				const int jjj = std::abs(jj);
				const int kkk = std::abs(kk);
				const double kernel_times_vol_inverse = stencil_weights_gpu[iii][jjj][kkk] * vol_inverse; // NOLINT
				const amrex::Real SNR_rho_per_cell = m_ej * kernel_times_vol_inverse;
				const amrex::Real SNR_energy_per_cell = (E_blast + SN_kin_energy) * kernel_times_vol_inverse;
				const amrex::Real SNR_px_per_cell = m_ej * p_vx * kernel_times_vol_inverse;
				const amrex::Real SNR_py_per_cell = m_ej * p_vy * kernel_times_vol_inverse;
				const amrex::Real SNR_pz_per_cell = m_ej * p_vz * kernel_times_vol_inverse;
				amrex::Gpu::Atomic::AddNoRet(&local_buffer(ix + ii, iy + jj, iz + kk, HydroSystem<problem_t>::density_index), SNR_rho_per_cell);
				amrex::Gpu::Atomic::AddNoRet(&local_buffer(ix + ii, iy + jj, iz + kk, HydroSystem<problem_t>::x1Momentum_index),
							     SNR_px_per_cell);
				amrex::Gpu::Atomic::AddNoRet(&local_buffer(ix + ii, iy + jj, iz + kk, HydroSystem<problem_t>::x2Momentum_index),
							     SNR_py_per_cell);
				amrex::Gpu::Atomic::AddNoRet(&local_buffer(ix + ii, iy + jj, iz + kk, HydroSystem<problem_t>::x3Momentum_index),
							     SNR_pz_per_cell);
				amrex::Gpu::Atomic::AddNoRet(&local_buffer(ix + ii, iy + jj, iz + kk, HydroSystem<problem_t>::energy_index),
							     SNR_energy_per_cell);
			}
		}
	}
}

template <typename problem_t>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE void depositThermalKineticMomentumSNR(
    amrex::Array4<amrex::Real> const &local_state, amrex::Array4<amrex::Real> const &local_buffer, const int ix, const int iy, const int iz,
    const amrex::Real stencil_volume, const amrex::Real pos_x, const amrex::Real pos_y, const amrex::Real pos_z, const amrex::Real m_ej,
    const amrex::Real E_blast, const amrex::Real SN_kin_energy, const amrex::Real p_snr_0, const amrex::Real vol_inverse,
    const amrex::GpuArray<amrex::GpuArray<amrex::GpuArray<amrex::Real, SN_stencil_array_size>, SN_stencil_array_size>, SN_stencil_array_size>
	&stencil_weights_gpu,
    const amrex::Real avg_density, const amrex::Real vol, const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> &dx,
    const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> &plo, const SNScheme SN_scheme_d)
{
	const double n_H_amb = avg_density * cloudy_H_mass_fraction / m_u;
	const amrex::Real M_snr = (avg_density * stencil_volume * vol) + m_ej;	 // SNR mass
	const amrex::Real M_sf = 1679.0 * C::M_solar * std::pow(n_H_amb, -0.26); // Shell-formation mass
	const amrex::Real RM = M_snr / M_sf;					 // R_M factor = M_snr / M_sf
	const amrex::Real p_snr = p_snr_0 * std::pow(n_H_amb, -0.17);		 // = 1.89e5 when n = 10

	// fraction of terminal SN momentum to go to gas momentum
	amrex::Real f_factor = 1.0;
	if (SN_scheme_d == SNScheme::SN_thermal_or_thermal_momentum) {
		if (RM < 1.0) {
			if (RM > 0.027) {
				f_factor = 0.529 * std::sqrt(RM); // f^2 = 0.28. 28% kinetic (Kim & Ostriker 2017)
			} else {
				f_factor = 0.0; // pure thermal in well-resolved limit
			}
		}
	} else if (SN_scheme_d == SNScheme::SN_thermal_kinetic_or_thermal_momentum) {
		if (RM < 1.0) {
			if (RM > 0.027) {
				f_factor = 0.529 * std::sqrt(RM); // f^2 = 0.28. 28% kinetic (Kim & Ostriker 2017)
			} else {
				f_factor = 1.414 * std::sqrt(RM); // pure kinetic in well-resolved limit: (f^2 / RM) * (p_snr^2
								  // / 2 M_sf) = 2 * (p_snr^2 / 2 M_sf) ~= 1e51 erg
			}
		}
	}
	// SNScheme::SN_pure_kinetic_or_thermal_momentum: keep f_factor = 1.0

	// // log RM and f_factor, for debugging on CPU.
	// if (particle_verbose) {
	// 	printf("SNR logging -- RM: %.2e, f_factor: %.2e\n", RM, f_factor);
	// }

	for (int ii = ix - SN_stencil_size; ii <= ix + SN_stencil_size; ++ii) {
		for (int jj = iy - SN_stencil_size; jj <= iy + SN_stencil_size; ++jj) {
			for (int kk = iz - SN_stencil_size; kk <= iz + SN_stencil_size; ++kk) {
				const int iii = std::abs(ii - ix);
				const int jjj = std::abs(jj - iy);
				const int kkk = std::abs(kk - iz);
				const double kernel_times_vol_inverse = stencil_weights_gpu[iii][jjj][kkk] * vol_inverse;

				const double delta_x = (ii + 0.5) * dx[0] + plo[0] - pos_x; // NOLINT
				const double delta_y = (jj + 0.5) * dx[1] + plo[1] - pos_y; // NOLINT
				const double delta_z = (kk + 0.5) * dx[2] + plo[2] - pos_z; // NOLINT
				const double r_sq = (delta_x * delta_x) + (delta_y * delta_y) + (delta_z * delta_z);

				const amrex::Real delta_rho_i = m_ej * kernel_times_vol_inverse;
				const amrex::Real e_snr_per_cell = (E_blast + SN_kin_energy) * kernel_times_vol_inverse;
				const amrex::Real momentum_per_cell = f_factor * p_snr * kernel_times_vol_inverse;

				// Compute unit vector from particle to cell center
				const amrex::Real r = std::sqrt(r_sq);
				// Avoid division by zero
				const amrex::Real inv_r = (r > 0.0) ? 1.0 / r : 0.0;

				// unit vector from particle to cell center
				const amrex::Real r_hat_x = delta_x * inv_r;
				const amrex::Real r_hat_y = delta_y * inv_r;
				const amrex::Real r_hat_z = delta_z * inv_r;

				const double rho = local_state(ii, jj, kk, HydroSystem<problem_t>::density_index);
				const double px = local_state(ii, jj, kk, HydroSystem<problem_t>::x1Momentum_index);
				const double py = local_state(ii, jj, kk, HydroSystem<problem_t>::x2Momentum_index);
				const double pz = local_state(ii, jj, kk, HydroSystem<problem_t>::x3Momentum_index);

				// Compute momentum directed along unit vector
				const double dpx = (delta_rho_i * px / rho) + (momentum_per_cell * r_hat_x);
				const double dpy = (delta_rho_i * py / rho) + (momentum_per_cell * r_hat_y);
				const double dpz = (delta_rho_i * pz / rho) + (momentum_per_cell * r_hat_z);

				amrex::Gpu::Atomic::AddNoRet(&local_buffer(ii, jj, kk, HydroSystem<problem_t>::density_index), delta_rho_i);
				amrex::Gpu::Atomic::AddNoRet(&local_buffer(ii, jj, kk, HydroSystem<problem_t>::x1Momentum_index), dpx);
				amrex::Gpu::Atomic::AddNoRet(&local_buffer(ii, jj, kk, HydroSystem<problem_t>::x2Momentum_index), dpy);
				amrex::Gpu::Atomic::AddNoRet(&local_buffer(ii, jj, kk, HydroSystem<problem_t>::x3Momentum_index), dpz);
				amrex::Gpu::Atomic::AddNoRet(&local_buffer(ii, jj, kk, HydroSystem<problem_t>::energy_index), e_snr_per_cell);
			}
		}
	}
}

template <typename ContainerType, typename problem_t>
void depositToBuffer(ContainerType *container, amrex::MultiFab &state, amrex::MultiFab &state_buffer, int lev, amrex::Real time, amrex::Real dt, int mass_index,
		     int evolutionStageIndex, int birthTimeIndex, const SNScheme SN_scheme_d)
{
	const BL_PROFILE("SNFeedbackUtils::depositToBuffer()");
	constexpr amrex::Real stencil_volume = 4.0 / 3.0 * M_PI * SN_stencil_size * SN_stencil_size * SN_stencil_size;
	constexpr amrex::GpuArray<amrex::GpuArray<amrex::GpuArray<amrex::Real, SN_stencil_array_size>, SN_stencil_array_size>, SN_stencil_array_size>
	    stencil_weights_gpu = {{{{{0.00884198143074, 0.00884198143074, 0.00884198143074, 0.00416240696843},
				      {0.00884198143074, 0.00884198143074, 0.00884198143074, 0.00262865918549},
				      {0.00884198143074, 0.00884198143074, 0.00596795726055, 0.00005052308190},
				      {0.00416240696843, 0.00262865918549, 0.00005052308190, 0.00000000000000}}},
				    {{{0.00884198143074, 0.00884198143074, 0.00884198143074, 0.00262865918549},
				      {0.00884198143074, 0.00884198143074, 0.00861063982859, 0.00119306623841},
				      {0.00884198143074, 0.00861063982859, 0.00400459528385, 0.00000136166514},
				      {0.00262865918549, 0.00119306623841, 0.00000136166514, 0.00000000000000}}},
				    {{{0.00884198143074, 0.00884198143074, 0.00596795726055, 0.00005052308190},
				      {0.00884198143074, 0.00861063982859, 0.00400459528385, 0.00000136166514},
				      {0.00596795726055, 0.00400459528385, 0.00045652034325, 0.00000000000000},
				      {0.00005052308190, 0.00000136166514, 0.00000000000000, 0.00000000000000}}},
				    {{{0.00416240696843, 0.00262865918549, 0.00005052308190, 0.00000000000000},
				      {0.00262865918549, 0.00119306623841, 0.00000136166514, 0.00000000000000},
				      {0.00005052308190, 0.00000136166514, 0.00000000000000, 0.00000000000000},
				      {0.00000000000000, 0.00000000000000, 0.00000000000000, 0.00000000000000}}}}};

	const amrex::Real step_end_time = time + dt;

	constexpr double E_blast = 1.0e51;		       // ergs
	constexpr double m_ej = 10.0 * C::M_solar;	       // ejecta mass in cgs
	constexpr double m_dead_min = 1.4 * C::M_solar;	       // minimum mass of a dead star
	constexpr double p_snr_0 = 2.8e5 * C::M_solar * 1.0e5; // SN terminal momentum in cgs

	// Step 1: Local deposition within each box
	for (typename ContainerType::ParIterType pti(*container, lev); pti.isValid(); ++pti) {
		// Get the particle array of structs
		auto &particles = pti.GetArrayOfStructs();
		auto *pData = particles().data();
		const amrex::Long np = pti.numParticles();

		// Get the local deposit array for this box
		const auto &local_buffer = state_buffer.array(pti);
		const auto &local_state = state.array(pti);

		// Get geometry information for this level
		const auto &geom = container->Geom(lev);
		const auto plo = geom.ProbLoArray();
		const auto dxi = geom.InvCellSizeArray();
		const auto dx = geom.CellSizeArray();

		// Calculate inverse cell volume
		const amrex::Real vol_inverse = AMREX_D_TERM(dxi[0], *dxi[1], *dxi[2]);
		const amrex::Real vol = AMREX_D_TERM(dx[0], *dx[1], *dx[2]);

		// Deposit particle data into the local buffer
		amrex::ParallelFor(np, [=] AMREX_GPU_DEVICE(int64_t idx) {
			auto &p = pData[idx]; // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)

			// Check if this is a supernova progenitor
			const bool is_sn_progenitor = (p.idata(evolutionStageIndex) == static_cast<int>(StellarEvolutionStage::SNProgenitor));

			if (is_sn_progenitor && step_end_time > p.rdata(birthTimeIndex + 1)) {
				// Update the particle's evolution stage to SNRemnant
				p.idata(evolutionStageIndex) = static_cast<int>(StellarEvolutionStage::SNRemnant);

				// update the particle's mass: subtract ejecta mass
				const amrex::Real mass_dead_star = p.rdata(mass_index) - m_ej;
				// AMREX_ASSERT_WITH_MESSAGE(mass_dead_star > 0.0, "SN progenitor mass should be greater than ejecta mass (10 M_sun)");
				p.rdata(mass_index) = std::max(m_dead_min, mass_dead_star);

				// get particle velocity and kinetic energy
				const amrex::Real p_vx = p.rdata(mass_index + 1);
				const amrex::Real p_vy = p.rdata(mass_index + 2);
				const amrex::Real p_vz = p.rdata(mass_index + 3);
				const amrex::Real SN_kin_energy = 0.5 * m_ej * (p_vx * p_vx + p_vy * p_vy + p_vz * p_vz);

				const amrex::Real pos_x = p.pos(0);
				const amrex::Real pos_y = p.pos(1);
				const amrex::Real pos_z = p.pos(2);

				// Find the cell containing the particle
				int ix = static_cast<int>(amrex::Math::floor((pos_x - plo[0]) * dxi[0]));
				int iy = static_cast<int>(amrex::Math::floor((pos_y - plo[1]) * dxi[1]));
				int iz = static_cast<int>(amrex::Math::floor((pos_z - plo[2]) * dxi[2]));

				amrex::Real avg_density = 0.0;
				for (int ii = -SN_stencil_size; ii <= SN_stencil_size; ++ii) {
					for (int jj = -SN_stencil_size; jj <= SN_stencil_size; ++jj) {
						for (int kk = -SN_stencil_size; kk <= SN_stencil_size; ++kk) {
							const int iii = std::abs(ii);
							const int jjj = std::abs(jj);
							const int kkk = std::abs(kk);
							const double kernel = stencil_weights_gpu[iii][jjj][kkk];
							avg_density += kernel * local_state(ix + ii, iy + jj, iz + kk, HydroSystem<problem_t>::density_index);
						}
					}
				}

				if (SN_scheme_d == SNScheme::SN_thermal_only) {
					// Deposit mass and energy into (2 * stencil_width + 1)³ cells centered on the particle's cell
					depositThermalSNR<problem_t>(local_buffer, ix, iy, iz, m_ej, E_blast, SN_kin_energy, p_vx, p_vy, p_vz, vol_inverse,
								     stencil_weights_gpu);
				} else {
					// Deposit momentum and energy into (2 * stencil_width + 1)³ cells centered on the particle's cell
					depositThermalKineticMomentumSNR<problem_t>(local_state, local_buffer, ix, iy, iz, stencil_volume, pos_x, pos_y, pos_z,
										    m_ej, E_blast, SN_kin_energy, p_snr_0, vol_inverse, stencil_weights_gpu,
										    avg_density, vol, dx, plo, SN_scheme_d);
				}
			}
		});
	}
}

template <typename problem_t>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE void addCompositeBufferToState(amrex::Array4<amrex::Real> const &local_state,
								   amrex::Array4<amrex::Real> const &local_buffer, int i, int j, int k,
								   amrex::Real *p_max_velocity)
{
	// For SN_thermal_or_thermal_momentum, SN_thermal_kinetic_or_thermal_momentum, and SN_pure_kinetic_or_thermal_momentum,
	// the buffer contains mass, momentum, and energy. We need to add the buffer to the state in a way that guarantees
	// that the internal energy is positive. In fact, we demand that the cell temperature should not decrease.
	const double rho = local_state(i, j, k, HydroSystem<problem_t>::density_index);
	const double px = local_state(i, j, k, HydroSystem<problem_t>::x1Momentum_index);
	const double py = local_state(i, j, k, HydroSystem<problem_t>::x2Momentum_index);
	const double pz = local_state(i, j, k, HydroSystem<problem_t>::x3Momentum_index);
	const double e_int = local_state(i, j, k, HydroSystem<problem_t>::internalEnergy_index);
	const double e_tot = local_state(i, j, k, HydroSystem<problem_t>::energy_index);

	const double d_rho = local_buffer(i, j, k, HydroSystem<problem_t>::density_index);

	// Skip if there is no SN feedback
	if (d_rho == 0.0) {
		return;
	}

	const double d_px = local_buffer(i, j, k, HydroSystem<problem_t>::x1Momentum_index);
	const double d_py = local_buffer(i, j, k, HydroSystem<problem_t>::x2Momentum_index);
	const double d_pz = local_buffer(i, j, k, HydroSystem<problem_t>::x3Momentum_index);
	const double d_e = local_buffer(i, j, k, HydroSystem<problem_t>::energy_index);

	const double rho_new = rho + d_rho;
	double px_new = px + d_px;
	double py_new = py + d_py;
	double pz_new = pz + d_pz;
	double e_tot_new = e_tot + d_e;

	const double d_e_int_d_rho = e_int / rho;
	const double e_int_new_tmp = d_e_int_d_rho * rho_new;
	const double e_int_plus_kinetic = e_int_new_tmp + (0.5 * ((px_new * px_new) + (py_new * py_new) + (pz_new * pz_new)) / rho_new);

	const Real uncertainty_tol = static_cast<Real>(5.) * std::numeric_limits<Real>::epsilon();

	if (e_int_plus_kinetic <= e_tot_new * (1.0 + uncertainty_tol)) {
		e_tot_new = std::max(e_int_plus_kinetic, e_tot_new);
	} else {
		// find the lambda such that e_int_plus_kinetic == e_tot_new
		const double e_kinetic_max = e_tot_new - e_int_new_tmp;
		AMREX_ASSERT(e_kinetic_max >= 0.0);

		// If e_kinetic_max < (0.5 * (px * px + py * py + pz * pz) / rho_new), it means the SN energy (10^51 erg) is not enough to accelerate
		// the SNR to match the velocity of the gas. This may ONLY happen when the SN star is nearly static at first and the background gas is
		// moving at a speed >3171 km/s. This should NOT happen in practice. In this case, we keep temperature constant and match remnant
		// velocity with the background gas.  A lazy workaround: keep temperature constant and match remnant velocity with the background gas.
		if (e_kinetic_max < (0.5 * (px * px + py * py + pz * pz) / rho_new) * (1.0 + uncertainty_tol)) {
			// keep temperature constant and match remnant velocity with the background gas.
			px_new = rho_new * (px / rho);
			py_new = rho_new * (py / rho);
			pz_new = rho_new * (pz / rho);
			e_tot_new = e_int_new_tmp + (0.5 * ((px_new * px_new) + (py_new * py_new) + (pz_new * pz_new)) / rho_new);
		} else {

			// Find analytical solution of the following equation:
			// 0.5 * (p + lambda d_p)^2 / rho_new = e_kinetic_max
			// which simplifies to:
			// d_p^2 lambda^2 + 2 d_p p lambda + p^2 - 2 rho_new e_kinetic_max = 0
			// This quadratic equation, denoted as F(x) = 0, has some known properties which make the
			// solution well constrained:
			// 1. F(0) < 0 (see assertion above)
			// 2. F(1) > 0 (otherwise we won't be in this else clause)
			// Therefore, there must be one and only one solution in the range [0, 1], and it's the bigger
			// one of the two solutions (so we take the plus sign in the quadratic formula).
			// A special case is when a = 0, in which case the physical solution is the no kinetic energy is added to the state, therefore lambda =
			// 0.
			double lambda = 0.0;
			const double a = (d_px * d_px) + (d_py * d_py) + (d_pz * d_pz);				      // a = d_p^2
			if (a > std::numeric_limits<double>::min()) {						      // a > 0
				const double b = 2.0 * (px * d_px + py * d_py + pz * d_pz);			      // b = 2 d_p p
				const double c = (px * px) + (py * py) + (pz * pz) - (2.0 * rho_new * e_kinetic_max); // c = p^2 - 2 rho_new e_kinetic_max
				const double discriminant = (b * b) - (4.0 * a * c);
				AMREX_ASSERT(discriminant >= 0.0);
				lambda = (-b + std::sqrt(discriminant)) / (2.0 * a);
				// lambda = std::max(lambda, 0.0);
				AMREX_ASSERT(lambda >= 0.0);
				AMREX_ASSERT(lambda <= 1.0);
			}

			// assert that lambda is a valid solution
			AMREX_ASSERT_WITH_MESSAGE(std::abs((0.5 *
							    ((px + lambda * d_px) * (px + lambda * d_px) + (py + lambda * d_py) * (py + lambda * d_py) +
							     (pz + lambda * d_pz) * (pz + lambda * d_pz)) /
							    rho_new) -
							   e_kinetic_max) <= 1.0e-10 * e_kinetic_max,
						  "lambda is not a valid solution. This should NOT happen. @chongchonghe should be responsible for this.");

			px_new = px + lambda * d_px;
			py_new = py + lambda * d_py;
			pz_new = pz + lambda * d_pz;
		}
	}

	const double e_int_new = e_tot_new - (0.5 * ((px_new * px_new) + (py_new * py_new) + (pz_new * pz_new)) / rho_new);
	AMREX_ASSERT(e_int_new > 0.0);
	local_state(i, j, k, HydroSystem<problem_t>::density_index) = rho_new;
	local_state(i, j, k, HydroSystem<problem_t>::x1Momentum_index) = px_new;
	local_state(i, j, k, HydroSystem<problem_t>::x2Momentum_index) = py_new;
	local_state(i, j, k, HydroSystem<problem_t>::x3Momentum_index) = pz_new;
	local_state(i, j, k, HydroSystem<problem_t>::internalEnergy_index) = e_int_new;
	local_state(i, j, k, HydroSystem<problem_t>::energy_index) = e_tot_new;

	// Compute sound speed
	Real cs = NAN;
	if constexpr (HydroSystem<problem_t>::is_eos_isothermal()) {
		cs = quokka::EOS_Traits<problem_t>::cs_isothermal;
	} else {
		cs = HydroSystem<problem_t>::ComputeSoundSpeed(local_state, i, j, k);
	}

	// Compute velocity magnitude and track maximum
	const Real vx = px_new / rho_new;
	const Real vy = py_new / rho_new;
	const Real vz = pz_new / rho_new;
	const Real velocity_magnitude = std::sqrt(vx * vx + vy * vy + vz * vz);

	amrex::Gpu::Atomic::Max(&p_max_velocity[0], velocity_magnitude + cs);

	// // log the state, for debugging on CPU.
	// if (d_rho / rho > 1.0e-12) {
	// 	// print original rho, px, py, pz, e_int, e_tot; new rho_new, px_new, py_new, pz_new, e_int_new,
	// 	// e_tot_new
	// 	printf("original: rho = %e, px = %e, py = %e, pz = %e, e_int = %e, e_tot = %e\n", rho, px, py, pz, e_int, e_tot);
	// 	printf("new: rho_new = %e, px_new = %e, py_new = %e, pz_new = %e, e_int_new = %e, e_tot_new = %e\n", rho_new, px_new,
	// 	       py_new, pz_new, e_int_new, e_tot_new);
	// 	// print d e_int / d e_tot
	// 	printf("d e_int / d e_tot = %e\n", (e_int_new - e_int) / (e_tot_new - e_tot));
	// 	printf("e_int / rho = %e, e_int_new / rho_new = %e\n", e_int / rho, e_int_new / rho_new);
	// }
}

template <typename problem_t>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE void addThermalOnlyBufferToState(amrex::Array4<amrex::Real> const &local_state,
								     amrex::Array4<amrex::Real> const &local_buffer, int i, int j, int k,
								     amrex::Real *p_max_velocity)
{
	const Real d_rho = local_buffer(i, j, k, HydroSystem<problem_t>::density_index);

	// Skip if there is no SN feedback
	if (d_rho == 0.0) {
		return;
	}

	// For SN_thermal_only, the buffer contains only mass and energy (and a small amount of momentum), so it's safe to add the
	// buffer directly to the state.
	const double rho_new = local_state(i, j, k, HydroSystem<problem_t>::density_index) + d_rho;
	const double px_new = local_state(i, j, k, HydroSystem<problem_t>::x1Momentum_index) + local_buffer(i, j, k, HydroSystem<problem_t>::x1Momentum_index);
	const double py_new = local_state(i, j, k, HydroSystem<problem_t>::x2Momentum_index) + local_buffer(i, j, k, HydroSystem<problem_t>::x2Momentum_index);
	const double pz_new = local_state(i, j, k, HydroSystem<problem_t>::x3Momentum_index) + local_buffer(i, j, k, HydroSystem<problem_t>::x3Momentum_index);
	const double e_new = local_state(i, j, k, HydroSystem<problem_t>::energy_index) + local_buffer(i, j, k, HydroSystem<problem_t>::energy_index);
	const double e_int_new = e_new - (0.5 * ((px_new * px_new) + (py_new * py_new) + (pz_new * pz_new)) / rho_new);

	local_state(i, j, k, HydroSystem<problem_t>::density_index) = rho_new;
	local_state(i, j, k, HydroSystem<problem_t>::x1Momentum_index) = px_new;
	local_state(i, j, k, HydroSystem<problem_t>::x2Momentum_index) = py_new;
	local_state(i, j, k, HydroSystem<problem_t>::x3Momentum_index) = pz_new;
	local_state(i, j, k, HydroSystem<problem_t>::internalEnergy_index) = e_int_new;
	local_state(i, j, k, HydroSystem<problem_t>::energy_index) = e_new;

	// Compute sound speed. For thermal-only feedback, the gas velocity stays unchanged, so we only report sound speed.
	Real cs = NAN;
	if constexpr (HydroSystem<problem_t>::is_eos_isothermal()) {
		cs = quokka::EOS_Traits<problem_t>::cs_isothermal;
	} else {
		cs = HydroSystem<problem_t>::ComputeSoundSpeed(local_state, i, j, k);
	}

	amrex::Gpu::Atomic::Max(&p_max_velocity[0], cs);
}

template <typename problem_t>
void addBufferToState(amrex::MultiFab &state, amrex::MultiFab &state_buffer, const SNScheme SN_scheme_d, amrex::Real *p_max_velocity)
{
	const BL_PROFILE("SNFeedbackUtils::addBufferToState()");
	for (amrex::MFIter mfi(state); mfi.isValid(); ++mfi) {
		const amrex::Box &box = mfi.validbox();
		auto const &local_state = state.array(mfi);
		auto const &local_buffer = state_buffer.array(mfi);

		// add buffer to state
		amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
			auto p_max_velocity_local = p_max_velocity; // NOLINT
			if (SN_scheme_d == SNScheme::SN_thermal_only) {
				addThermalOnlyBufferToState<problem_t>(local_state, local_buffer, i, j, k, p_max_velocity_local);
			} else {
				addCompositeBufferToState<problem_t>(local_state, local_buffer, i, j, k, p_max_velocity_local);
			}
		});
	}
}

// Function to update particle evolution stages from SNProgenitor to SNRemnant
template <typename ContainerType>
void updateEvolutionStage(ContainerType *container, int lev_min, amrex::Real step_end_time, int birthTimeIndex, int evolutionStageIndex)
{
	const BL_PROFILE("SNFeedbackUtils::updateEvolutionStage()");
	if (container == nullptr || evolutionStageIndex < 0 || birthTimeIndex < 0) {
		return;
	}

	for (int lev = lev_min; lev <= container->finestLevel(); ++lev) {
		for (typename ContainerType::ParIterType pti(*container, lev); pti.isValid(); ++pti) {
			auto &particles = pti.GetArrayOfStructs();
			auto *pData = particles().data();
			const amrex::Long np = pti.numParticles();

			amrex::ParallelFor(np, [=] AMREX_GPU_DEVICE(int64_t idx) {
				auto &p = pData[idx]; // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)

				// Check if this is a supernova progenitor
				const bool is_sn_progenitor = (p.idata(evolutionStageIndex) == static_cast<int>(StellarEvolutionStage::SNProgenitor));

				// Update the particle's evolution stage to SNRemnant if it's time
				if (is_sn_progenitor && step_end_time > p.rdata(birthTimeIndex + 1)) {
					p.idata(evolutionStageIndex) = static_cast<int>(StellarEvolutionStage::SNRemnant);
				}
			});
		}
	}
}

} // namespace SNFeedbackUtils

template <typename ContainerType, typename problem_t>
auto SNDeposition(ContainerType *container, amrex::MultiFab &state, amrex::MultiFab &state_buffer, int lev, amrex::Real time, amrex::Real dt, int mass_index,
		  int evolutionStageIndex, int birthTimeIndex) -> Real
{
	const BL_PROFILE("[particle_deposition] SNDeposition()");
	static_assert(SN_stencil_size <= 3,
		      "SN_stencil_size must be <= 3"); // SN_stencil_size must be <= n_ghost - 1 = 3. SN particle may drift 1 cell before being deposited.

	// Zero the buffer for each particle type
	state_buffer.setVal(0.0);

	// copy host variables to device
	const SNScheme SN_scheme_d = SN_scheme;

	// Initialize maximum velocity tracking
	amrex::Gpu::Buffer<amrex::Real> max_velocity_buffer({0.0});
	amrex::Real *p_max_velocity = max_velocity_buffer.data();

	// Step 1: Local deposition within each box
	SNFeedbackUtils::depositToBuffer<ContainerType, problem_t>(container, state, state_buffer, lev, time, dt, mass_index, evolutionStageIndex,
								   birthTimeIndex, SN_scheme_d);

	// Step 2: Sum boundary values
	state_buffer.SumBoundary(container->Geom(lev).periodicity());

	// Step 3: Add the buffer to the state
	SNFeedbackUtils::addBufferToState<problem_t>(state, state_buffer, SN_scheme_d, p_max_velocity);

	// Step 4: Check maximum velocity and print warning if needed
	auto *h_max_velocity = max_velocity_buffer.copyToHost();
	Real max_velocity = h_max_velocity[0];
	amrex::ParallelDescriptor::ReduceRealMax(max_velocity);

	return max_velocity;
}

#endif // AMREX_SPACEDIM == 3

} // namespace quokka

#endif // PARTICLE_DEPOSITION_HPP_
