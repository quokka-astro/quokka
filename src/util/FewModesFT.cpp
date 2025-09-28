//==============================================================================
// ABOUTME: Gaussian random vector field generator using few-modes inverse FFT
// ABOUTME: Based on AthenaPK implementation for turbulent driving fields
//==============================================================================
/// \file FewModesFT.cpp
/// \brief Implementation of Gaussian random vector field generator

#include "FewModesFT.hpp"
#include "AMReX_GpuDevice.H"
#include "AMReX_GpuLaunch.H"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <utility>

namespace quokka::util
{

namespace detail
{
// Standalone functions to avoid CUDA extended device lambda restrictions

namespace
{
void initialize_coefficients(amrex::Real *var_hat_real_ptr, amrex::Real *var_hat_imag_ptr, int size)
{
	amrex::ParallelFor(size, [=] AMREX_GPU_DEVICE(int idx) {
		var_hat_real_ptr[idx] = 0.0;
		var_hat_imag_ptr[idx] = 0.0;
	});
}

void compute_phase_i(int size, int lo_i, int ni, int num_modes, int gnx1, const amrex::Real *k_vec_ptr, amrex::Array4<amrex::Real> const &phases_i_arr)
{
	amrex::ParallelFor(size, [=] AMREX_GPU_DEVICE(int idx) {
		const int i = lo_i + (idx % ni);
		const int m = idx / ni;

		const auto gi = static_cast<amrex::Real>(i % gnx1);
		const amrex::Real w_kx = k_vec_ptr[0 * num_modes + m] * 2.0 * M_PI / static_cast<amrex::Real>(gnx1);

		const amrex::Real cos_phase = std::cos(w_kx * gi);
		const amrex::Real sin_phase = std::sin(w_kx * gi);

		// Adjust phase factor for Complex->Real IFT: u_hat*(k) = u_hat(-k)
		if (k_vec_ptr[0 * num_modes + m] == 0.0) {
			phases_i_arr(i, 0, 0, m * 2) = 0.5 * cos_phase;
			phases_i_arr(i, 0, 0, m * 2 + 1) = 0.5 * sin_phase;
		} else {
			phases_i_arr(i, 0, 0, m * 2) = cos_phase;
			phases_i_arr(i, 0, 0, m * 2 + 1) = sin_phase;
		}
	});
}

void compute_phase_j(int size, int lo_j, int nj, int num_modes, int gnx2, const amrex::Real *k_vec_ptr, amrex::Array4<amrex::Real> const &phases_j_arr)
{
	amrex::ParallelFor(size, [=] AMREX_GPU_DEVICE(int idx) {
		const int j = lo_j + (idx % nj);
		const int m = idx / nj;

		const auto gj = static_cast<amrex::Real>(j % gnx2);
		const amrex::Real w_ky = k_vec_ptr[1 * num_modes + m] * 2.0 * M_PI / static_cast<amrex::Real>(gnx2);

		const amrex::Real cos_phase = std::cos(w_ky * gj);
		const amrex::Real sin_phase = std::sin(w_ky * gj);

		phases_j_arr(0, j, 0, m * 2) = cos_phase;
		phases_j_arr(0, j, 0, m * 2 + 1) = sin_phase;
	});
}

void compute_phase_k(int size, int lo_k, int nk, int num_modes, int gnx3, const amrex::Real *k_vec_ptr, amrex::Array4<amrex::Real> const &phases_k_arr)
{
	amrex::ParallelFor(size, [=] AMREX_GPU_DEVICE(int idx) {
		const int k = lo_k + (idx % nk);
		const int m = idx / nk;

		const auto gk = static_cast<amrex::Real>(k % gnx3);
		const amrex::Real w_kz = k_vec_ptr[2 * num_modes + m] * 2.0 * M_PI / static_cast<amrex::Real>(gnx3);

		const amrex::Real cos_phase = std::cos(w_kz * gk);
		const amrex::Real sin_phase = std::sin(w_kz * gk);

		phases_k_arr(0, 0, k, m * 2) = cos_phase;
		phases_k_arr(0, 0, k, m * 2 + 1) = sin_phase;
	});
}

void generate_power_spectrum(int size, int num_modes, amrex::Real k_peak, const amrex::Real *k_vec_ptr, const amrex::Real *random_num_ptr,
			     amrex::Real *var_hat_new_real_ptr, amrex::Real *var_hat_new_imag_ptr)
{
	amrex::ParallelFor(size, [=] AMREX_GPU_DEVICE(int idx) {
		const int n = idx / num_modes;
		const int m = idx % num_modes;

		const amrex::Real kx = k_vec_ptr[0 * num_modes + m];
		const amrex::Real ky = k_vec_ptr[1 * num_modes + m];
		const amrex::Real kz = k_vec_ptr[2 * num_modes + m];

		const amrex::Real kmag = std::sqrt(kx * kx + ky * ky + kz * kz);

		amrex::Real tmp = std::pow(kmag / k_peak, 2.0) * (2.0 - std::pow(kmag / k_peak, 2.0));
		tmp = std::max(tmp, 0.0);

		const int idx_real = (n * num_modes + m) * 2;
		const int idx_imag = (n * num_modes + m) * 2 + 1;
		const amrex::Real v_sqr_local = random_num_ptr[idx_real] * random_num_ptr[idx_real] + random_num_ptr[idx_imag] * random_num_ptr[idx_imag];
		const amrex::Real norm = std::sqrt(-2.0 * std::log(v_sqr_local) / v_sqr_local);

		var_hat_new_real_ptr[idx] = tmp * norm * random_num_ptr[idx_real];
		var_hat_new_imag_ptr[idx] = tmp * norm * random_num_ptr[idx_imag];
	});
}

void enforce_symmetry(int size, int num_modes, const amrex::Real *k_vec_ptr, amrex::Real *var_hat_new_real_ptr, amrex::Real *var_hat_new_imag_ptr)
{
	amrex::ParallelFor(size, [=] AMREX_GPU_DEVICE(int idx) {
		const int n = idx / num_modes;
		const int m = idx % num_modes;

		if (k_vec_ptr[0 * num_modes + m] == 0.0) {
			for (int m2 = 0; m2 < m; ++m2) {
				if (k_vec_ptr[1 * num_modes + m] == -k_vec_ptr[1 * num_modes + m2] &&
				    k_vec_ptr[2 * num_modes + m] == -k_vec_ptr[2 * num_modes + m2]) {
					const int idx2 = n * num_modes + m2;
					var_hat_new_real_ptr[idx] = var_hat_new_real_ptr[idx2];
					var_hat_new_imag_ptr[idx] = -var_hat_new_imag_ptr[idx2];
				}
			}
		}
	});
}

void apply_projection(int num_modes, amrex::Real sol_weight, const amrex::Real *k_vec_ptr, amrex::Real *var_hat_new_real_ptr, amrex::Real *var_hat_new_imag_ptr)
{
	amrex::ParallelFor(num_modes, [=] AMREX_GPU_DEVICE(int m) {
		const amrex::Real kx = k_vec_ptr[0 * num_modes + m];
		const amrex::Real ky = k_vec_ptr[1 * num_modes + m];
		const amrex::Real kz = k_vec_ptr[2 * num_modes + m];

		amrex::Real kmag = std::sqrt(kx * kx + ky * ky + kz * kz);

		// Avoid division by zero
		if (kmag == 0.0) {
			kmag = 1.0;
		}

		// Make unit vector
		const amrex::Real kx_unit = kx / kmag;
		const amrex::Real ky_unit = ky / kmag;
		const amrex::Real kz_unit = kz / kmag;

		// Calculate dot product for each mode
		const amrex::Real dot_real = var_hat_new_real_ptr[0 * num_modes + m] * kx_unit + var_hat_new_real_ptr[1 * num_modes + m] * ky_unit +
					     var_hat_new_real_ptr[2 * num_modes + m] * kz_unit;
		const amrex::Real dot_imag = var_hat_new_imag_ptr[0 * num_modes + m] * kx_unit + var_hat_new_imag_ptr[1 * num_modes + m] * ky_unit +
					     var_hat_new_imag_ptr[2 * num_modes + m] * kz_unit;

		// Apply projection to all components
		const amrex::Real factor = 1.0 - 2.0 * sol_weight;

		var_hat_new_real_ptr[0 * num_modes + m] = var_hat_new_real_ptr[0 * num_modes + m] * sol_weight + factor * dot_real * kx_unit;
		var_hat_new_imag_ptr[0 * num_modes + m] = var_hat_new_imag_ptr[0 * num_modes + m] * sol_weight + factor * dot_imag * kx_unit;

		var_hat_new_real_ptr[1 * num_modes + m] = var_hat_new_real_ptr[1 * num_modes + m] * sol_weight + factor * dot_real * ky_unit;
		var_hat_new_imag_ptr[1 * num_modes + m] = var_hat_new_imag_ptr[1 * num_modes + m] * sol_weight + factor * dot_imag * ky_unit;

		var_hat_new_real_ptr[2 * num_modes + m] = var_hat_new_real_ptr[2 * num_modes + m] * sol_weight + factor * dot_real * kz_unit;
		var_hat_new_imag_ptr[2 * num_modes + m] = var_hat_new_imag_ptr[2 * num_modes + m] * sol_weight + factor * dot_imag * kz_unit;
	});
}

void evolve_coefficients(int size, amrex::Real c_drift, amrex::Real c_diff, amrex::Real *var_hat_real_ptr, amrex::Real *var_hat_imag_ptr,
			 const amrex::Real *var_hat_new_real_ptr, const amrex::Real *var_hat_new_imag_ptr)
{
	amrex::ParallelFor(size, [=] AMREX_GPU_DEVICE(int idx) {
		// Update persistent GPU storage with evolved coefficients
		var_hat_real_ptr[idx] = var_hat_real_ptr[idx] * c_drift + var_hat_new_real_ptr[idx] * c_diff;
		var_hat_imag_ptr[idx] = var_hat_imag_ptr[idx] * c_drift + var_hat_new_imag_ptr[idx] * c_diff;
	});
}

void compute_inverse_fourier_transform(const amrex::Box &bx, int num_modes, const amrex::Real *var_hat_real_ptr, const amrex::Real *var_hat_imag_ptr,
				       amrex::Array4<amrex::Real> const &mf_arr, amrex::Array4<amrex::Real const> const &phases_i_arr,
				       amrex::Array4<amrex::Real const> const &phases_j_arr, amrex::Array4<amrex::Real const> const &phases_k_arr)
{
	amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		for (int n = 0; n < 3; ++n) {
			mf_arr(i, j, k, n) = 0.0;

			for (int m = 0; m < num_modes; ++m) {
				const amrex::Real phase_i_real = phases_i_arr(i, 0, 0, m * 2);
				const amrex::Real phase_i_imag = phases_i_arr(i, 0, 0, m * 2 + 1);
				const amrex::Real phase_j_real = phases_j_arr(0, j, 0, m * 2);
				const amrex::Real phase_j_imag = phases_j_arr(0, j, 0, m * 2 + 1);
				const amrex::Real phase_k_real = phases_k_arr(0, 0, k, m * 2);
				const amrex::Real phase_k_imag = phases_k_arr(0, 0, k, m * 2 + 1);

				// Complex multiplication: phase = phase_i * phase_j * phase_k
				const amrex::Real temp_real = phase_i_real * phase_j_real - phase_i_imag * phase_j_imag;
				const amrex::Real temp_imag = phase_i_real * phase_j_imag + phase_i_imag * phase_j_real;

				const amrex::Real phase_real = temp_real * phase_k_real - temp_imag * phase_k_imag;
				const amrex::Real phase_imag = temp_real * phase_k_imag + temp_imag * phase_k_real;

				const int idx = n * num_modes + m;
				mf_arr(i, j, k, n) += 2.0 * (var_hat_real_ptr[idx] * phase_real - var_hat_imag_ptr[idx] * phase_imag);
			}
		}
	});
}

} // anonymous namespace

} // namespace detail

FewModesFT::FewModesFT(std::string prefix, int num_modes, const std::vector<std::vector<amrex::Real>> &k_vec, amrex::Real k_peak, amrex::Real sol_weight,
		       amrex::Real t_corr, uint32_t rseed, const amrex::BoxArray &ba, const amrex::DistributionMapping &dm, bool fill_ghosts)
    : num_modes_(num_modes), prefix_(std::move(prefix)), var_hat_real_d_(static_cast<std::size_t>(3) * static_cast<std::size_t>(num_modes)),
      var_hat_imag_d_(static_cast<std::size_t>(3) * static_cast<std::size_t>(num_modes)),
      k_vec_d_(static_cast<std::size_t>(3) * static_cast<std::size_t>(num_modes)), k_vec_(k_vec), k_peak_(k_peak), sol_weight_(sol_weight), t_corr_(t_corr),
      fill_ghosts_(fill_ghosts)
{

	if (num_modes > 100) {
		amrex::Print() << "### WARNING: Using more than 100 explicit modes will significantly increase runtime.\n";
		amrex::Print() << "If many modes are required, consider using a full FFT-based driving mechanism.\n";
	}

	AMREX_ALWAYS_ASSERT((sol_weight == -1.0) || (sol_weight >= 0.0 && sol_weight <= 1.0));

	// Initialize CPU random number storage
	random_num_.resize(3);
	for (int n = 0; n < 3; ++n) {
		random_num_[n].resize(num_modes);
		for (int m = 0; m < num_modes; ++m) {
			random_num_[n][m].resize(2); // real and imaginary parts
		}
	}

	// Initialize GPU-resident Fourier coefficients to zero
	amrex::Real *var_hat_real_ptr = var_hat_real_d_.data();
	amrex::Real *var_hat_imag_ptr = var_hat_imag_d_.data();

	detail::initialize_coefficients(var_hat_real_ptr, var_hat_imag_ptr, 3 * num_modes);

	// Initialize GPU-resident wave vectors
	amrex::Real *k_vec_ptr = k_vec_d_.data();

	// Copy k_vec data to device (one-time initialization)
	for (int dim = 0; dim < 3; ++dim) {
		for (int m = 0; m < num_modes; ++m) {
			k_vec_ptr[dim * num_modes + m] = k_vec[dim][m];
		}
	}

	amrex::Gpu::streamSynchronize();

	// Initialize phase arrays
	const int nghost = fill_ghosts ? 1 : 0;
	phases_i_.define(ba, dm, num_modes * 2, nghost); // 2 components for real/imag
	phases_j_.define(ba, dm, num_modes * 2, nghost);
	phases_k_.define(ba, dm, num_modes * 2, nghost);

	// Initialize random number generator
	rng_.seed(rseed);
	dist_ = std::uniform_real_distribution<>(-1.0, 1.0);
}

void FewModesFT::SetPhases(const amrex::Geometry &geom)
{
	const auto *prob_lo = geom.ProbLo();
	const auto *prob_hi = geom.ProbHi();

	const amrex::Real Lx = prob_hi[0] - prob_lo[0];
	const amrex::Real Ly = prob_hi[1] - prob_lo[1];
	const amrex::Real Lz = prob_hi[2] - prob_lo[2];

	const auto domain = geom.Domain();
	const int gnx1 = domain.length(0);
	const int gnx2 = domain.length(1);
	const int gnx3 = domain.length(2);

	// Check that the domain is cubic and has uniform spacing
	AMREX_ALWAYS_ASSERT(gnx1 == gnx2 && gnx2 == gnx3);
	AMREX_ALWAYS_ASSERT(std::abs(Lx - Ly) < 1e-12 && std::abs(Ly - Lz) < 1e-12);

	const Complex I(0.0, 1.0);

	// Set phases for each direction
	for (amrex::MFIter mfi(phases_i_); mfi.isValid(); ++mfi) {
		const amrex::Box &gbx = mfi.growntilebox();

		auto &phases_i_fab = phases_i_[mfi];
		auto &phases_j_fab = phases_j_[mfi];
		auto &phases_k_fab = phases_k_[mfi];

		// Calculate phases for i-direction
		const auto lo_i = gbx.loVect()[0];
		const auto hi_i = gbx.hiVect()[0];
		const auto num_modes = num_modes_;
		const auto *k_vec_ptr = k_vec_d_.data();
		auto phases_i_arr = phases_i_fab.array();

		const int ni = hi_i - lo_i + 1;
		detail::compute_phase_i(ni * num_modes, lo_i, ni, num_modes, gnx1, k_vec_ptr, phases_i_arr);

		// Calculate phases for j-direction
		const auto lo_j = gbx.loVect()[1];
		const auto hi_j = gbx.hiVect()[1];
		auto phases_j_arr = phases_j_fab.array();

		const int nj = hi_j - lo_j + 1;
		detail::compute_phase_j(nj * num_modes, lo_j, nj, num_modes, gnx2, k_vec_ptr, phases_j_arr);

		// Calculate phases for k-direction
		const auto lo_k = gbx.loVect()[2];
		const auto hi_k = gbx.hiVect()[2];
		auto phases_k_arr = phases_k_fab.array();

		const int nk = hi_k - lo_k + 1;
		detail::compute_phase_k(nk * num_modes, lo_k, nk, num_modes, gnx3, k_vec_ptr, phases_k_arr);
	}
}

void FewModesFT::Generate(amrex::MultiFab &mf, amrex::Real dt)
{
	const Complex I(0.0, 1.0);

	// Generate random numbers on host to ensure deterministic behavior
	amrex::Real v1 = 0.0;
	amrex::Real v2 = 0.0;
	amrex::Real v_sqr = 0.0;
	for (int n = 0; n < 3; ++n) {
		for (int m = 0; m < num_modes_; ++m) {
			v_sqr = 1.0; // Initialize to enter the loop
			while (v_sqr >= 1.0 || v_sqr == 0.0) {
				v1 = dist_(rng_);
				v2 = dist_(rng_);
				v_sqr = v1 * v1 + v2 * v2;
			}

			random_num_[n][m][0] = v1;
			random_num_[n][m][1] = v2;
		}
	}

	// Copy random numbers to device-accessible memory
	amrex::Gpu::DeviceVector<amrex::Real> random_num_d(static_cast<std::size_t>(3) * static_cast<std::size_t>(num_modes_) * static_cast<std::size_t>(2));
	amrex::Real *random_num_ptr = random_num_d.data();

	for (int n = 0; n < 3; ++n) {
		for (int m = 0; m < num_modes_; ++m) {
			const int idx_real = (n * num_modes_ + m) * 2;
			const int idx_imag = (n * num_modes_ + m) * 2 + 1;
			random_num_ptr[idx_real] = random_num_[n][m][0];
			random_num_ptr[idx_imag] = random_num_[n][m][1];
		}
	}

	// Use GPU-resident k_vec data (initialized once in constructor)
	amrex::Real *k_vec_ptr = k_vec_d_.data();

	// Copy var_hat_new_ to device-accessible memory
	amrex::Gpu::DeviceVector<amrex::Real> var_hat_new_real_d(static_cast<std::size_t>(3) * static_cast<std::size_t>(num_modes_));
	amrex::Gpu::DeviceVector<amrex::Real> var_hat_new_imag_d(static_cast<std::size_t>(3) * static_cast<std::size_t>(num_modes_));
	amrex::Real *var_hat_new_real_ptr = var_hat_new_real_d.data();
	amrex::Real *var_hat_new_imag_ptr = var_hat_new_imag_d.data();

	amrex::Gpu::streamSynchronize();

	// Generate new power spectrum (injection) on GPU
	const auto k_peak = k_peak_;
	const auto num_modes = num_modes_;

	detail::generate_power_spectrum(3 * num_modes_, num_modes, k_peak, k_vec_ptr, random_num_ptr, var_hat_new_real_ptr, var_hat_new_imag_ptr);

	// Enforce symmetry for complex to real transform on GPU
	detail::enforce_symmetry(3 * num_modes_, num_modes, k_vec_ptr, var_hat_new_real_ptr, var_hat_new_imag_ptr);

	// Apply projection if requested on GPU
	if (sol_weight_ >= 0.0) {
		const auto sol_weight = sol_weight_;
		detail::apply_projection(num_modes_, sol_weight, k_vec_ptr, var_hat_new_real_ptr, var_hat_new_imag_ptr);
	}

	// Get pointers to GPU-resident var_hat_ arrays
	amrex::Real *var_hat_real_ptr = var_hat_real_d_.data();
	amrex::Real *var_hat_imag_ptr = var_hat_imag_d_.data();

	// Evolve (Ornstein-Uhlenbeck process) on GPU using persistent storage
	const amrex::Real c_drift = std::exp(-dt / t_corr_);
	const amrex::Real c_diff = std::sqrt(1.0 - c_drift * c_drift);

	detail::evolve_coefficients(3 * num_modes_, c_drift, c_diff, var_hat_real_ptr, var_hat_imag_ptr, var_hat_new_real_ptr, var_hat_new_imag_ptr);

	amrex::Gpu::streamSynchronize();

	// Perform inverse Fourier transform
	for (amrex::MFIter mfi(mf); mfi.isValid(); ++mfi) {
		const amrex::Box &bx = mfi.validbox();
		auto &mf_fab = mf[mfi];
		const auto &phases_i_fab = phases_i_[mfi];
		const auto &phases_j_fab = phases_j_[mfi];
		const auto &phases_k_fab = phases_k_[mfi];

		auto mf_arr = mf_fab.array();
		auto phases_i_arr = phases_i_fab.const_array();
		auto phases_j_arr = phases_j_fab.const_array();
		auto phases_k_arr = phases_k_fab.const_array();

		const auto num_modes = num_modes_;

		detail::compute_inverse_fourier_transform(bx, num_modes, var_hat_real_ptr, var_hat_imag_ptr, mf_arr, phases_i_arr, phases_j_arr, phases_k_arr);
	}
}

auto MakeRandomModes(int num_modes, amrex::Real k_peak, uint32_t rseed) -> std::vector<std::vector<amrex::Real>>
{
	std::vector<std::vector<amrex::Real>> k_vec(3);
	for (int i = 0; i < 3; ++i) {
		k_vec[i].resize(num_modes);
	}

	const int k_low = std::floor(k_peak / 2.0);
	const int k_high = std::ceil(2.0 * k_peak);

	// Use random_device for truly random seed if rseed is 0, otherwise use provided seed
	uint32_t actual_seed = rseed;
	if (rseed == 0) {
		std::random_device rd;
		actual_seed = rd();
	}
	std::mt19937 rng(actual_seed);
	std::uniform_int_distribution<> dist(-k_high, k_high);

	int n_mode = 0;
	int n_attempt = 0;
	constexpr int max_attempts = 1000000;
	amrex::Real kx1 = 0.0;
	amrex::Real kx2 = 0.0;
	amrex::Real kx3 = 0.0;
	amrex::Real k_mag = 0.0;
	amrex::Real ampl = 0.0;
	bool mode_exists = false;

	while (n_mode < num_modes && n_attempt < max_attempts) {
		n_attempt++;

		kx1 = dist(rng);
		kx2 = dist(rng);
		kx3 = dist(rng);
		k_mag = std::sqrt(kx1 * kx1 + kx2 * kx2 + kx3 * kx3);

		// Expected amplitude of the spectral function
		ampl = (k_mag / k_peak) * (k_mag / k_peak) * (2.0 - (k_mag / k_peak) * (k_mag / k_peak));

		// Check if mode was already picked
		mode_exists = false;
		for (int n_mode_exist = 0; n_mode_exist < n_mode; ++n_mode_exist) {
			if (k_vec[0][n_mode_exist] == kx1 && k_vec[1][n_mode_exist] == kx2 && k_vec[2][n_mode_exist] == kx3) {
				mode_exists = true;
				break;
			}
		}

		// kx1 < 0.0 because we use an explicit symmetric Complex to Real transform
		if (ampl < 0.0 || k_mag < k_low || k_mag > k_high || mode_exists || kx1 < 0.0) {
			continue;
		}

		k_vec[0][n_mode] = kx1;
		k_vec[1][n_mode] = kx2;
		k_vec[2][n_mode] = kx3;
		n_mode++;
	}

	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(n_attempt < max_attempts, "MakeRandomModes did not succeed in calculating perturbation modes.");

	return k_vec;
}

} // namespace quokka::util