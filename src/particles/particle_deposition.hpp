#ifndef PARTICLE_DEPOSITION_HPP_
#define PARTICLE_DEPOSITION_HPP_

#include <algorithm>
#include <array>
#include <numbers>

#include "AMReX_Array.H"
#include "AMReX_Array4.H"
#include "AMReX_BLProfiler.H"
#include "AMReX_Extension.H"
#include "AMReX_MultiFab.H"
#include "AMReX_ParticleInterpolators.H"
#include "AMReX_REAL.H"
#include "hydro/hydro_system.hpp"
#include "particles/particle_types.hpp"
#include "particles/particle_utils.hpp"

namespace amrex::ParticleInterpolator
{
/** \brief A class that implements nearest-eight-cell interpolation.
 */
struct NearestEight : public Base<NearestEight, amrex::Real> {
	static constexpr int stencil_width = 2;

	static constexpr int nx = stencil_width - 1; // NOLINT
	static constexpr int ny = stencil_width - 1; // NOLINT
	static constexpr int nz = stencil_width - 1; // NOLINT

	amrex::Real weights[3 * stencil_width]; // NOLINT

	template <typename P>
	AMREX_GPU_DEVICE AMREX_FORCE_INLINE NearestEight(const P &p, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &plo, // NOLINT
							 amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dxi)
	{
		w = &weights[0]; // NOLINT
		for (int i = 0; i < AMREX_SPACEDIM; ++i) {
			amrex::Real l = (p.pos(i) - plo[i]) * dxi[i] + 0.5;
			index[i] = static_cast<int>(amrex::Math::floor(l)) - 1;
			w[stencil_width * i + 0] = 1.;
			w[stencil_width * i + 1] = 1.;
		}
		for (int i = AMREX_SPACEDIM; i < 3; ++i) {
			index[i] = 0;
			w[stencil_width * i + 0] = 1.;
			w[stencil_width * i + 1] = 0.;
		}
	}
};

/** \brief A class that implements Gaussian kernel particle/mesh interpolation
 *  with a spherical radial cutoff.
 *
 *  Template parameter N controls stencil extent: the kernel is non-zero only
 *  within a sphere of radius N*dx centred on the particle.  The stencil box
 *  is (2N+1)^3 cells; cells outside the sphere receive zero weight.
 *  N is limited by the number of ghost cells (max 6).
 *
 *  The kernel weight at distance r (in units of dx) is
 *      W(r) = exp(-r^2 / (2 sigma^2))   for r <= N
 *           = 0                           for r >  N
 *  and the weights are normalised so that their sum over the sphere equals 1.
 *
 *  This struct does NOT inherit from amrex::ParticleInterpolator::Base because
 *  the spherical cutoff breaks the separability assumed by Base::ParticleToMesh.
 */
template <int N = 3> struct Gaussian {
	static_assert(N >= 1 && N <= 6, "N must be between 1 and 6 (limited by ghost cells)");
	static constexpr int stencil_width = 2 * N + 1;
	static constexpr amrex::Real sigma = 1.5;			   // Gaussian width in units of cell size (dx)
	static constexpr amrex::Real cutoff_r2 = static_cast<amrex::Real>(N * N); // spherical cutoff radius squared

	int index[3]{};	     // NOLINT lower-left corner of stencil box
	amrex::Real frac[3]{}; // NOLINT fractional cell position per dimension
	amrex::Real inv_norm{}; // NOLINT 1 / (sum of weights within sphere)

	template <typename P>
	AMREX_GPU_DEVICE AMREX_FORCE_INLINE Gaussian(const P &p, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &plo, // NOLINT
						     amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dxi)
	{
		// Compute stencil origin and fractional position for each active dimension
		for (int i = 0; i < AMREX_SPACEDIM; ++i) {
			const amrex::Real l = (p.pos(i) - plo[i]) * dxi[i] + 0.5;
			index[i] = static_cast<int>(amrex::Math::floor(l)) - N;
			frac[i] = l - amrex::Math::floor(l);
		}
		for (int i = AMREX_SPACEDIM; i < 3; ++i) {
			index[i] = 0;
			frac[i] = 0.0;
		}

		// Compute normalization: sum of Gaussian weights within the sphere
		const amrex::Real inv_2sigma2 = 0.5 / (sigma * sigma);
		amrex::Real norm_sum = 0.0;
		const int nz_loop = (AMREX_SPACEDIM >= 3) ? stencil_width : 1;
		const int ny_loop = (AMREX_SPACEDIM >= 2) ? stencil_width : 1;
		for (int kk = 0; kk < nz_loop; ++kk) {
			const amrex::Real dz = (AMREX_SPACEDIM >= 3) ? static_cast<amrex::Real>(N - kk) + frac[2] - 1.0 : 0.0;
			for (int jj = 0; jj < ny_loop; ++jj) {
				const amrex::Real dy = (AMREX_SPACEDIM >= 2) ? static_cast<amrex::Real>(N - jj) + frac[1] - 1.0 : 0.0;
				for (int ii = 0; ii < stencil_width; ++ii) {
					const amrex::Real dx = static_cast<amrex::Real>(N - ii) + frac[0] - 1.0;
					const amrex::Real r2 = AMREX_D_TERM(dx * dx, +dy * dy, +dz * dz);
					if (r2 <= cutoff_r2) {
						norm_sum += std::exp(-r2 * inv_2sigma2);
					}
				}
			}
		}
		inv_norm = 1.0 / norm_sum;
	}

	/// Deposit particle data onto the mesh using the spherical Gaussian kernel.
	/// Same interface as amrex::ParticleInterpolator::Base::ParticleToMesh.
	template <typename P, typename V, typename F>
	AMREX_GPU_DEVICE AMREX_FORCE_INLINE void ParticleToMesh(const P &p, amrex::Array4<V> const &arr, int src_comp, int dst_comp,
								int num_comps, F const &f)
	{
		const amrex::Real inv_2sigma2 = 0.5 / (sigma * sigma);
		const int nz_loop = (AMREX_SPACEDIM >= 3) ? stencil_width : 1;
		const int ny_loop = (AMREX_SPACEDIM >= 2) ? stencil_width : 1;

		for (int ic = 0; ic < num_comps; ++ic) {
			const auto pval = f(p, src_comp + ic);
			for (int kk = 0; kk < nz_loop; ++kk) {
				const amrex::Real dz = (AMREX_SPACEDIM >= 3) ? static_cast<amrex::Real>(N - kk) + frac[2] - 1.0 : 0.0;
				for (int jj = 0; jj < ny_loop; ++jj) {
					const amrex::Real dy =
					    (AMREX_SPACEDIM >= 2) ? static_cast<amrex::Real>(N - jj) + frac[1] - 1.0 : 0.0;
					for (int ii = 0; ii < stencil_width; ++ii) {
						const amrex::Real dx = static_cast<amrex::Real>(N - ii) + frac[0] - 1.0;
						const amrex::Real r2 = AMREX_D_TERM(dx * dx, +dy * dy, +dz * dz);
						if (r2 <= cutoff_r2) {
							const amrex::Real wt = std::exp(-r2 * inv_2sigma2) * inv_norm;
							amrex::Gpu::Atomic::AddNoRet(
							    &arr(index[0] + ii, index[1] + jj, index[2] + kk, ic + dst_comp),
							    static_cast<V>(wt * pval));
						}
					}
				}
			}
		}
	}
};
} // namespace amrex::ParticleInterpolator

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

	// Operator to perform radiation deposition using Gaussian kernel interpolation
	template <typename ContainerType>
	AMREX_GPU_DEVICE AMREX_FORCE_INLINE void operator()(const ContainerType &p, amrex::Array4<amrex::Real> const &radEnergySource,
							    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &plo,
							    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dxi) const noexcept
	{
		amrex::ParticleInterpolator::Gaussian<> interp(p, plo, dxi);
		const auto currentTime = current_time;
		const auto birthIndex = birthTimeIndex;
		// Deposit radiation energy only if particle is active
		interp.ParticleToMesh(p, radEnergySource, start_part_comp, start_mesh_comp, num_comp,
					      [=] AMREX_GPU_DEVICE(const ContainerType &part, int comp) {
						      if (currentTime < part.rdata(birthIndex) || currentTime >= part.rdata(birthIndex + 1)) {
							      return 0.0;
						      }
						      return part.rdata(comp) * (AMREX_D_TERM(dxi[0], *dxi[1], *dxi[2]));
					      });
	}
};

//-------------------- Mass depositions --------------------

// Functor for depositing particle mass onto the grid
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
		const amrex::Real gConstLocal = Gconst;
		const amrex::Real cellVolumeFactor = (AMREX_D_TERM(dxi[0], *dxi[1], *dxi[2]));
		// Deposit mass weighted by 4 pi G
		interp.ParticleToMesh(p, rho, start_part_comp, start_mesh_comp, num_comp, [=] AMREX_GPU_DEVICE(const ContainerType &part, int comp) {
			return 4.0 * M_PI * gConstLocal * part.rdata(comp) * cellVolumeFactor;
		});
	}
};

struct DepositionCount {
	int start_part_comp{}; // Starting component in particle data
	int start_mesh_comp{}; // Starting component in mesh data
	int num_comp{};	       // Number of components to deposit

	// Operator to perform mass deposition using linear interpolation
	template <typename ContainerType>
	AMREX_GPU_DEVICE AMREX_FORCE_INLINE void operator()(const ContainerType &p, amrex::Array4<amrex::Real> const &rho_count,
							    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &plo,
							    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dxi) const noexcept
	{
		amrex::ParticleInterpolator::NearestEight interp(p, plo, dxi);
		// Deposit to 1.0 to all eight cells that the particle interacts with
		interp.ParticleToMesh(p, rho_count, start_part_comp, start_mesh_comp, num_comp,
				      [=] AMREX_GPU_DEVICE(const ContainerType & /*part*/, int /*comp*/) { return 1.0; });
	}
};

//-------------------- Supernova depositions --------------------

namespace SNFeedbackUtils
{

// Function to deposit thermal supernova remnant quantities
template <typename problem_t>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE void
depositThermalSNR(amrex::Array4<amrex::Real> const &local_buffer, const int ix, const int iy, const int iz, const amrex::Real m_ej, const amrex::Real E_blast,
		  const amrex::Real SN_kin_energy, const amrex::Real p_vx, const amrex::Real p_vy, const amrex::Real p_vz, const amrex::Real vol_inverse,
		  const amrex::GpuArray<amrex::GpuArray<amrex::GpuArray<amrex::Real, SN_stencil_array_size>, SN_stencil_array_size>, SN_stencil_array_size>
		      &stencil_weights_gpu,
		  const amrex::Real scalar_yield_per_SN_d) noexcept
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

				// Deposit passive scalar if enabled
				// TODO(chongchonghe): Add support for multiple passive scalars (currently only deposits to scalar0)
				if constexpr (Physics_Traits<problem_t>::numPassiveScalars > 0) {
					const amrex::Real scalar_per_cell = scalar_yield_per_SN_d * kernel_times_vol_inverse;
					amrex::Gpu::Atomic::AddNoRet(&local_buffer(ix + ii, iy + jj, iz + kk, HydroSystem<problem_t>::scalar0_index),
								     scalar_per_cell);
				}

				// Deposit count into the last component for roundoff algorithm
				const int count_comp = HydroSystem<problem_t>::nHydroScalars_; // Last component is the count
				amrex::Gpu::Atomic::AddNoRet(&local_buffer(ix + ii, iy + jj, iz + kk, count_comp), 1.0);
			}
		}
	}
}

template <typename problem_t>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE void
depositThermalKineticMomentumSNR(amrex::Array4<amrex::Real> const &local_state, amrex::Array4<amrex::Real> const &local_buffer, const int ix, const int iy,
				 const int iz, const amrex::Real stencil_volume, const amrex::Real pos_x, const amrex::Real pos_y, const amrex::Real pos_z,
				 const amrex::Real m_ej, const amrex::Real E_blast, const amrex::Real p_snr_0, const amrex::Real vol_inverse,
				 const amrex::GpuArray<amrex::GpuArray<amrex::GpuArray<amrex::Real, SN_stencil_array_size>, SN_stencil_array_size>,
						       SN_stencil_array_size> &stencil_weights_gpu,
				 const amrex::Real avg_density, const amrex::Real vol, const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> &dx,
				 const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> &plo, const SNScheme SN_scheme_d, const Real pvx, const Real pvy,
				 const Real pvz, const bool SN_smooth_gas_velocity, const amrex::Real scalar_yield_per_SN_d)
{
	const double n_H_amb = avg_density * cloudy_H_mass_fraction / m_u;
	const amrex::Real M_gas = avg_density * stencil_volume * vol; // Gas mass in stencil
	const amrex::Real M_snr = M_gas + m_ej;			      // SNR mass
	constexpr amrex::Real M_sf_canonical = 1679.0 * C::M_solar;   // canonical pre-factor [g], n_H^{-0.26} applied below
	constexpr amrex::Real p_snr_0_canonical = quokka::SN_p_term_Msunkmps_canonical * C::M_solar * 1.0e5; // canonical SN terminal momentum [g cm/s]
	// Scale M_sf so that the kinetic energy p_snr^2 / (2 M_sf) is invariant under changes of p_snr_0:
	//   M_sf_scaled = M_sf_canonical * (p_snr_0 / p_snr_0_canonical)^2
	const amrex::Real p_ratio = p_snr_0 / p_snr_0_canonical;
	const amrex::Real M_sf = M_sf_canonical * std::pow(n_H_amb, -0.26) * p_ratio * p_ratio; // Shell-formation mass (scaled)
	const amrex::Real RM = M_snr / M_sf;							// R_M factor = M_snr / M_sf
	const amrex::Real p_snr = p_snr_0 * std::pow(n_H_amb, -0.17);				// = 1.89e5 when n = 10

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
				f_factor = std::numbers::sqrt2 * std::sqrt(RM); // pure kinetic in well-resolved limit: (f^2 / RM) * (p_snr^2
										// / 2 M_sf) = 2 * (p_snr^2 / 2 M_sf) ~= 1e51 erg
			}
		}
	}
	// SNScheme::SN_pure_kinetic_or_thermal_momentum: keep f_factor = 1.0

	// Step 1: Compute kernel-weighted average momentum of gas in stencil (for COM velocity)
	amrex::Real Px_gas_avg = 0.0;
	amrex::Real Py_gas_avg = 0.0;
	amrex::Real Pz_gas_avg = 0.0;

	for (int ii = ix - SN_stencil_size; ii <= ix + SN_stencil_size; ++ii) {
		for (int jj = iy - SN_stencil_size; jj <= iy + SN_stencil_size; ++jj) {
			for (int kk = iz - SN_stencil_size; kk <= iz + SN_stencil_size; ++kk) {
				const int iii = std::abs(ii - ix);
				const int jjj = std::abs(jj - iy);
				const int kkk = std::abs(kk - iz);
				const double kernel = stencil_weights_gpu[iii][jjj][kkk];
				Px_gas_avg += kernel * local_state(ii, jj, kk, HydroSystem<problem_t>::x1Momentum_index);
				Py_gas_avg += kernel * local_state(ii, jj, kk, HydroSystem<problem_t>::x2Momentum_index);
				Pz_gas_avg += kernel * local_state(ii, jj, kk, HydroSystem<problem_t>::x3Momentum_index);
			}
		}
	}

	// Total momentum of gas (kernel weights sum to 1, multiply by stencil volume)
	const amrex::Real Px_gas_total = Px_gas_avg * stencil_volume * vol;
	const amrex::Real Py_gas_total = Py_gas_avg * stencil_volume * vol;
	const amrex::Real Pz_gas_total = Pz_gas_avg * stencil_volume * vol;

	// Step 2: Compute COM velocity of SNR (gas + ejecta)
	// v_COM = (P_gas + m_ej * v_ej) / (M_gas + m_ej)
	const amrex::Real v_COM_x = (Px_gas_total + m_ej * pvx) / M_snr;
	const amrex::Real v_COM_y = (Py_gas_total + m_ej * pvy) / M_snr;
	const amrex::Real v_COM_z = (Pz_gas_total + m_ej * pvz) / M_snr;

	// Step 3: Deposit to cells
	// After SN, each cell should have velocity v_COM + v_radial (isotropic expansion in COM frame)
	// Momentum change: delta_p = (rho_new * v_COM - p_old) + p_radial
	// Energy change: delta_E = (E_blast + SN_kin_energy) * kernel + v_COM . p_radial (cross term for Galilean invariance)
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
				const amrex::Real momentum_per_cell = f_factor * p_snr * kernel_times_vol_inverse;

				// Compute unit vector from particle to cell center
				const amrex::Real r = std::sqrt(r_sq);
				// Avoid division by zero
				const amrex::Real inv_r = (r > 0.0) ? 1.0 / r : 0.0;

				// unit vector from particle to cell center
				const amrex::Real r_hat_x = delta_x * inv_r;
				const amrex::Real r_hat_y = delta_y * inv_r;
				const amrex::Real r_hat_z = delta_z * inv_r;

				// Radial momentum components
				const amrex::Real p_radial_x = momentum_per_cell * r_hat_x;
				const amrex::Real p_radial_y = momentum_per_cell * r_hat_y;
				const amrex::Real p_radial_z = momentum_per_cell * r_hat_z;

				const double rho = local_state(ii, jj, kk, HydroSystem<problem_t>::density_index);
				const double px = local_state(ii, jj, kk, HydroSystem<problem_t>::x1Momentum_index);
				const double py = local_state(ii, jj, kk, HydroSystem<problem_t>::x2Momentum_index);
				const double pz = local_state(ii, jj, kk, HydroSystem<problem_t>::x3Momentum_index);

				// Energy deposition: thermal + ejecta kinetic + cross term (v_COM . p_radial)
				// The cross term ensures Galilean invariance: it accounts for the kinetic energy change
				// from the velocity "reset" to v_COM. This term sums to zero over all cells (momentum conserving).
				const amrex::Real v_COM_dot_p_radial = SN_smooth_gas_velocity
									   ? (v_COM_x * p_radial_x) + (v_COM_y * p_radial_y) + (v_COM_z * p_radial_z)
									   : ((px * p_radial_x) + (py * p_radial_y) + (pz * p_radial_z)) / rho;
				const amrex::Real SN_kin_energy = 0.5 * m_ej * (pvx * pvx + pvy * pvy + pvz * pvz);
				const amrex::Real e_snr_per_cell = (E_blast + SN_kin_energy) * kernel_times_vol_inverse + v_COM_dot_p_radial;

				// Compute momentum change for Galilean invariance:
				// After SN, cell velocity = v_COM + v_radial
				// p_new = rho_new * v_COM + p_radial
				// delta_p = p_new - p_old = (rho + delta_rho) * v_COM + p_radial - p_old
				const double rho_new = rho + delta_rho_i;
				const double dpx =
				    SN_smooth_gas_velocity ? (rho_new * v_COM_x - px) + p_radial_x : p_radial_x + m_ej * pvx * kernel_times_vol_inverse;
				const double dpy =
				    SN_smooth_gas_velocity ? (rho_new * v_COM_y - py) + p_radial_y : p_radial_y + m_ej * pvy * kernel_times_vol_inverse;
				const double dpz =
				    SN_smooth_gas_velocity ? (rho_new * v_COM_z - pz) + p_radial_z : p_radial_z + m_ej * pvz * kernel_times_vol_inverse;

				amrex::Gpu::Atomic::AddNoRet(&local_buffer(ii, jj, kk, HydroSystem<problem_t>::density_index), delta_rho_i);
				amrex::Gpu::Atomic::AddNoRet(&local_buffer(ii, jj, kk, HydroSystem<problem_t>::x1Momentum_index), dpx);
				amrex::Gpu::Atomic::AddNoRet(&local_buffer(ii, jj, kk, HydroSystem<problem_t>::x2Momentum_index), dpy);
				amrex::Gpu::Atomic::AddNoRet(&local_buffer(ii, jj, kk, HydroSystem<problem_t>::x3Momentum_index), dpz);
				amrex::Gpu::Atomic::AddNoRet(&local_buffer(ii, jj, kk, HydroSystem<problem_t>::energy_index), e_snr_per_cell);

				// Deposit passive scalar if enabled
				// TODO(chongchonghe): Add support for multiple passive scalars (currently only deposits to scalar0)
				if constexpr (Physics_Traits<problem_t>::numPassiveScalars > 0) {
					const amrex::Real scalar_per_cell = scalar_yield_per_SN_d * kernel_times_vol_inverse;
					amrex::Gpu::Atomic::AddNoRet(&local_buffer(ii, jj, kk, HydroSystem<problem_t>::scalar0_index), scalar_per_cell);
				}

				// Deposit count into the last component for roundoff algorithm
				const int count_comp = HydroSystem<problem_t>::nHydroScalars_; // Last component is the count
				amrex::Gpu::Atomic::AddNoRet(&local_buffer(ii, jj, kk, count_comp), 1.0);
			}
		}
	}
}

template <ParticleType particleType, typename ContainerType, typename problem_t>
void depositToBuffer(ContainerType *container, amrex::MultiFab &state, amrex::MultiFab &state_buffer, int lev, amrex::Real time, amrex::Real dt, int mass_index,
		     int evolutionStageIndex, int birthTimeIndex, const SNScheme SN_scheme_d, int *p_sn_count = nullptr)
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

	constexpr double E_blast = 1.0e51;		// ergs
	constexpr double m_ej = 10.0 * C::M_solar;	// ejecta mass in cgs
	constexpr double m_dead_min = 1.4 * C::M_solar; // minimum mass of a dead star
	const double p_snr_0 =
	    quokka::SN_p_term_Msunkmps * C::M_solar * 1.0e5; // SN terminal momentum in cgs (runtime parameter: particles.SN_p_term_Msunkmps [M_sun km/s])

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

		const bool SN_smooth_gas_velocity_d = SN_smooth_gas_velocity;
		const amrex::Real scalar_yield_per_SN_d = scalar_yield_per_SN;

		// Deposit particle data into the local buffer
		amrex::ParallelFor(np, [=] AMREX_GPU_DEVICE(int64_t idx) {
			auto &p = pData[idx]; // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
			auto const local_state_capture = local_state;
			auto const plo_capture = plo;
			auto const dxi_capture = dxi;
			amrex::ignore_unused(local_state_capture, plo_capture, dxi_capture);

			// Check if this is a supernova progenitor
			const bool is_sn_progenitor = (p.idata(evolutionStageIndex) == static_cast<int>(StellarEvolutionStage::SNProgenitor));

			if (is_sn_progenitor && step_end_time > p.rdata(birthTimeIndex + 1)) {
				// Count this SN explosion
				if (p_sn_count != nullptr) {
					amrex::Gpu::Atomic::AddNoRet(p_sn_count, 1);
				}

				// Update the particle's evolution stage to SNRemnant
				p.idata(evolutionStageIndex) = static_cast<int>(StellarEvolutionStage::SNRemnant);

				// update the particle's mass: subtract ejecta mass
				const amrex::Real mass_dead_star = p.rdata(mass_index) - m_ej;
				// AMREX_ASSERT_WITH_MESSAGE(mass_dead_star > 0.0, "SN progenitor mass should be greater than ejecta mass (10 M_sun)");
				p.rdata(mass_index) = std::max(m_dead_min, mass_dead_star);

				// get particle velocity
				const amrex::Real p_vx = p.rdata(mass_index + 1);
				const amrex::Real p_vy = p.rdata(mass_index + 2);
				const amrex::Real p_vz = p.rdata(mass_index + 3);

				const amrex::Real pos_x = p.pos(0);
				const amrex::Real pos_y = p.pos(1);
				const amrex::Real pos_z = p.pos(2);

				if constexpr (particleType == ParticleType::StochasticStellarPop) {
					p.rdata(StochasticStellarPopParticleDeathPosXIdx) = pos_x;
					p.rdata(StochasticStellarPopParticleDeathPosYIdx) = pos_y;
					p.rdata(StochasticStellarPopParticleDeathPosZIdx) = pos_z;
				}

				// Find the cell containing the particle
				int ix = static_cast<int>(amrex::Math::floor((pos_x - plo_capture[0]) * dxi_capture[0]));
				int iy = static_cast<int>(amrex::Math::floor((pos_y - plo_capture[1]) * dxi_capture[1]));
				int iz = static_cast<int>(amrex::Math::floor((pos_z - plo_capture[2]) * dxi_capture[2]));

				if constexpr (particleType == ParticleType::StochasticStellarPop) {
					p.rdata(StochasticStellarPopParticleDeathDensityIdx) =
					    local_state_capture(ix, iy, iz, HydroSystem<problem_t>::density_index);
				}

				amrex::Real avg_density = 0.0;
				for (int ii = -SN_stencil_size; ii <= SN_stencil_size; ++ii) {
					for (int jj = -SN_stencil_size; jj <= SN_stencil_size; ++jj) {
						for (int kk = -SN_stencil_size; kk <= SN_stencil_size; ++kk) {
							const int iii = std::abs(ii);
							const int jjj = std::abs(jj);
							const int kkk = std::abs(kk);
							const double kernel = stencil_weights_gpu[iii][jjj][kkk];
							avg_density +=
							    kernel * local_state_capture(ix + ii, iy + jj, iz + kk, HydroSystem<problem_t>::density_index);
						}
					}
				}

				if (SN_scheme_d == SNScheme::SN_thermal_only) {
					// For thermal-only scheme, compute lab-frame kinetic energy
					const amrex::Real SN_kin_energy = 0.5 * m_ej * (p_vx * p_vx + p_vy * p_vy + p_vz * p_vz);
					// Deposit mass and energy into (2 * stencil_width + 1)³ cells centered on the particle's cell
					depositThermalSNR<problem_t>(local_buffer, ix, iy, iz, m_ej, E_blast, SN_kin_energy, p_vx, p_vy, p_vz, vol_inverse,
								     stencil_weights_gpu, scalar_yield_per_SN_d);
				} else {
					// Deposit momentum and energy into (2 * stencil_width + 1)³ cells centered on the particle's cell
					// (SN kinetic energy computed inside function using COM frame for Galilean invariance)
					depositThermalKineticMomentumSNR<problem_t>(local_state, local_buffer, ix, iy, iz, stencil_volume, pos_x, pos_y, pos_z,
										    m_ej, E_blast, p_snr_0, vol_inverse, stencil_weights_gpu, avg_density, vol,
										    dx, plo, SN_scheme_d, p_vx, p_vy, p_vz, SN_smooth_gas_velocity_d,
										    scalar_yield_per_SN_d);
				}
			}
		});
	}
}

template <typename problem_t>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE void
addCompositeBufferToState(amrex::Array4<amrex::Real> const &local_state, amrex::Array4<amrex::Real> const &local_buffer,
			  std::array<amrex::Array4<const amrex::Real>, AMREX_SPACEDIM> const *fab_fc, int i, int j, int k, amrex::Real *p_max_velocity)
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

	// Add passive scalars from buffer to state (scalars are conserved densities)
	if constexpr (Physics_Traits<problem_t>::numPassiveScalars > 0) {
		for (int n = 0; n < Physics_Traits<problem_t>::numPassiveScalars; ++n) {
			local_state(i, j, k, HydroSystem<problem_t>::scalar0_index + n) += local_buffer(i, j, k, HydroSystem<problem_t>::scalar0_index + n);
		}
	}

	// Compute sound speed
	Real cs = NAN;
	if constexpr (HydroSystem<problem_t>::is_eos_isothermal()) {
		cs = quokka::EOS_Traits<problem_t>::cs_isothermal;
	} else {
		cs = HydroSystem<problem_t>::ComputeSoundSpeed(local_state, i, j, k, fab_fc);
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
AMREX_GPU_DEVICE AMREX_FORCE_INLINE void
addThermalOnlyBufferToState(amrex::Array4<amrex::Real> const &local_state, amrex::Array4<amrex::Real> const &local_buffer,
			    std::array<amrex::Array4<const amrex::Real>, AMREX_SPACEDIM> const *fab_fc, int i, int j, int k, amrex::Real *p_max_velocity)
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

	// Add passive scalars from buffer to state (scalars are conserved densities)
	if constexpr (Physics_Traits<problem_t>::numPassiveScalars > 0) {
		for (int n = 0; n < Physics_Traits<problem_t>::numPassiveScalars; ++n) {
			local_state(i, j, k, HydroSystem<problem_t>::scalar0_index + n) += local_buffer(i, j, k, HydroSystem<problem_t>::scalar0_index + n);
		}
	}

	// Compute sound speed. For thermal-only feedback, the gas velocity stays unchanged, so we only report sound speed.
	Real cs = NAN;
	if constexpr (HydroSystem<problem_t>::is_eos_isothermal()) {
		cs = quokka::EOS_Traits<problem_t>::cs_isothermal;
	} else {
		cs = HydroSystem<problem_t>::ComputeSoundSpeed(local_state, i, j, k, fab_fc);
	}

	amrex::Gpu::Atomic::Max(&p_max_velocity[0], cs);
}

template <typename problem_t>
void addBufferToState(amrex::MultiFab &state, std::array<amrex::MultiFab, AMREX_SPACEDIM> const *state_fc, amrex::MultiFab &state_buffer,
		      const SNScheme SN_scheme_d, amrex::Real *p_max_velocity)
{
	const BL_PROFILE("SNFeedbackUtils::addBufferToState()");
	for (amrex::MFIter mfi(state); mfi.isValid(); ++mfi) {
		const amrex::Box &box = mfi.validbox();
		auto const &local_state = state.array(mfi);
		auto const &local_buffer = state_buffer.array(mfi);

		bool has_fab_fc = false;
		std::array<amrex::Array4<const amrex::Real>, AMREX_SPACEDIM> fab_fc{};
		if (state_fc != nullptr) {
			for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
				fab_fc[dir] = (*state_fc)[dir].const_array(mfi);
			}
			has_fab_fc = true;
		}

		// add buffer to state
		amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
			auto p_max_velocity_local = p_max_velocity; // NOLINT
			const auto *const fab_fc_ptr = has_fab_fc ? &fab_fc : nullptr;
			if (SN_scheme_d == SNScheme::SN_thermal_only) {
				addThermalOnlyBufferToState<problem_t>(local_state, local_buffer, fab_fc_ptr, i, j, k, p_max_velocity_local);
			} else {
				addCompositeBufferToState<problem_t>(local_state, local_buffer, fab_fc_ptr, i, j, k, p_max_velocity_local);
			}
		});
	}
}

// Function to update particle evolution stages from SNProgenitor to SNRemnant
template <ParticleType particleType, typename ContainerType, typename problem_t>
void updateEvolutionStageAndDeathDensity(ContainerType *container, amrex::MultiFab &state, int lev, amrex::Real step_end_time, int birthTimeIndex,
					 int evolutionStageIndex)
{
	const BL_PROFILE("SNFeedbackUtils::updateEvolutionStageAndDeathDensity()");
	if (container == nullptr || evolutionStageIndex < 0 || birthTimeIndex < 0) {
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

		amrex::ParallelFor(np, [=] AMREX_GPU_DEVICE(int64_t idx) {
			auto &p = pData[idx]; // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
			auto const local_state_capture = local_state;
			auto const plo_capture = plo;
			auto const dxi_capture = dxi;
			amrex::ignore_unused(local_state_capture, plo_capture, dxi_capture);

			// Check if this is a supernova progenitor
			const bool is_sn_progenitor = (p.idata(evolutionStageIndex) == static_cast<int>(StellarEvolutionStage::SNProgenitor));

			// Update the particle's evolution stage to SNRemnant if it's time
			if (is_sn_progenitor && step_end_time > p.rdata(birthTimeIndex + 1)) {
				p.idata(evolutionStageIndex) = static_cast<int>(StellarEvolutionStage::SNRemnant);
				if constexpr (particleType == ParticleType::StochasticStellarPop) {
					const amrex::Real pos_x = p.pos(0);
					const amrex::Real pos_y = p.pos(1);
					const amrex::Real pos_z = p.pos(2);
					p.rdata(StochasticStellarPopParticleDeathPosXIdx) = pos_x;
					p.rdata(StochasticStellarPopParticleDeathPosYIdx) = pos_y;
					p.rdata(StochasticStellarPopParticleDeathPosZIdx) = pos_z;

					const int ix = static_cast<int>(amrex::Math::floor((pos_x - plo_capture[0]) * dxi_capture[0]));
					const int iy = static_cast<int>(amrex::Math::floor((pos_y - plo_capture[1]) * dxi_capture[1]));
					const int iz = static_cast<int>(amrex::Math::floor((pos_z - plo_capture[2]) * dxi_capture[2]));
					p.rdata(StochasticStellarPopParticleDeathDensityIdx) =
					    local_state_capture(ix, iy, iz, HydroSystem<problem_t>::density_index);
				}
			}
		});
	}
}

template <ParticleType particleType, typename ContainerType>
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
					if constexpr (particleType == ParticleType::StochasticStellarPop) {
						p.rdata(StochasticStellarPopParticleDeathPosXIdx) = p.pos(0);
						p.rdata(StochasticStellarPopParticleDeathPosYIdx) = p.pos(1);
						p.rdata(StochasticStellarPopParticleDeathPosZIdx) = p.pos(2);
					}
				}
			});
		}
	}
}

} // namespace SNFeedbackUtils

template <ParticleType particleType, typename ContainerType, typename problem_t>
auto SNDeposition(ContainerType *container, amrex::MultiFab &state, std::array<amrex::MultiFab, AMREX_SPACEDIM> const *state_fc, int lev, amrex::Real time,
		  amrex::Real dt, int mass_index, int evolutionStageIndex, int birthTimeIndex) -> std::pair<int, Real>
{
	const BL_PROFILE("[particle_deposition] SNDeposition()");
	static_assert(SN_stencil_size <= 3,
		      "SN_stencil_size must be <= 3"); // SN_stencil_size must be <= n_ghost - 1 = 3. SN particle may drift 1 cell before being deposited.

	// Create buffer for this particle type with extra component for count
	amrex::MultiFab state_buffer(state.boxArray(), state.DistributionMap(), state.nComp() + 1, state.nGrow());
	state_buffer.setVal(0.0);

	// copy host variables to device
	const SNScheme SN_scheme_d = SN_scheme;

	// Initialize maximum velocity tracking
	amrex::Gpu::Buffer<amrex::Real> max_velocity_buffer({0.0});
	amrex::Real *p_max_velocity = max_velocity_buffer.data();

	// Initialize SN explosion count tracking
	amrex::Gpu::Buffer<int> sn_count_buffer({0});
	int *p_sn_count = sn_count_buffer.data();

	// Step 1: Local deposition within each box
	SNFeedbackUtils::depositToBuffer<particleType, ContainerType, problem_t>(container, state, state_buffer, lev, time, dt, mass_index, evolutionStageIndex,
										 birthTimeIndex, SN_scheme_d, p_sn_count);

	// Step 2: Sum boundary values
	state_buffer.SumBoundary(container->Geom(lev).periodicity());

	// Apply roundoff to state_buffer
	ParticleUtils::roundoffMultiFab(state_buffer);

	// Step 3: Add the buffer to the state
	SNFeedbackUtils::addBufferToState<problem_t>(state, state_fc, state_buffer, SN_scheme_d, p_max_velocity);

	// Step 4: Check maximum velocity and SN count, then reduce across ranks
	auto *h_max_velocity = max_velocity_buffer.copyToHost();
	Real max_velocity = h_max_velocity[0];
	amrex::ParallelDescriptor::ReduceRealMax(max_velocity);

	auto *h_sn_count = sn_count_buffer.copyToHost();
	int sn_count = h_sn_count[0];
	amrex::ParallelDescriptor::ReduceIntSum(sn_count);

	return {sn_count, max_velocity};
}

} // namespace quokka

#endif // PARTICLE_DEPOSITION_HPP_
