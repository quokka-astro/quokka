#ifndef FEWMODESFT_HPP_ // NOLINT
#define FEWMODESFT_HPP_
//==============================================================================
// ABOUTME: Gaussian random vector field generator using few-modes inverse FFT
// ABOUTME: Based on AthenaPK implementation for turbulent driving fields
//==============================================================================
/// \file FewModesFT.hpp
/// \brief Helper functions for an inverse (explicit complex to real) FFT
///        generating Gaussian random vector fields with specified power spectrum

#include <complex>
#include <random>
#include <string>
#include <vector>

#include "AMReX.H"
#include "AMReX_Array.H"
#include "AMReX_BaseFab.H"
#include "AMReX_FArrayBox.H"
#include "AMReX_Geometry.H"
#include "AMReX_GpuDevice.H"
#include "AMReX_MultiFab.H"
#include "AMReX_REAL.H"

namespace quokka::util
{

using Complex = std::complex<amrex::Real>;

class FewModesFT
{
      private:
	int num_modes_;
	std::string prefix_;
	// GPU-resident Fourier coefficient storage (real and imaginary parts separate for GPU efficiency)
	amrex::Gpu::DeviceVector<amrex::Real> var_hat_real_d_; // [component * num_modes + mode] - current coefficients
	amrex::Gpu::DeviceVector<amrex::Real> var_hat_imag_d_; // [component * num_modes + mode] - current coefficients

	// GPU-resident wave vector storage
	amrex::Gpu::DeviceVector<amrex::Real> k_vec_d_;			// [dimension * num_modes + mode] - wave vectors
	std::vector<std::vector<amrex::Real>> k_vec_;			// [dimension][mode] - kept for host access if needed
	amrex::Real k_peak_;						// peak of the power spectrum
	std::vector<std::vector<std::vector<amrex::Real>>> random_num_; // [component][mode][real/imag]
	std::mt19937 rng_;
	std::uniform_real_distribution<> dist_;
	amrex::Real sol_weight_; // power in solenoidal modes for projection. Set to negative to disable projection
	amrex::Real t_corr_;	 // correlation time for evolution of Ornstein-Uhlenbeck process
	bool fill_ghosts_;	 // if the inverse transform should also fill ghost zones

	int gnx1_{};
	int gnx2_{};
	int gnx3_{};
	bool phases_initialized_{false};

	amrex::Gpu::DeviceVector<amrex::Real> phase_i_real_d_;
	amrex::Gpu::DeviceVector<amrex::Real> phase_i_imag_d_;
	amrex::Gpu::DeviceVector<amrex::Real> phase_j_real_d_;
	amrex::Gpu::DeviceVector<amrex::Real> phase_j_imag_d_;
	amrex::Gpu::DeviceVector<amrex::Real> phase_k_real_d_;
	amrex::Gpu::DeviceVector<amrex::Real> phase_k_imag_d_;

      public:
	FewModesFT(std::string prefix, int num_modes, const std::vector<std::vector<amrex::Real>> &k_vec, amrex::Real k_peak, amrex::Real sol_weight,
		   amrex::Real t_corr, uint32_t rseed, const amrex::BoxArray &ba, const amrex::DistributionMapping &dm, bool fill_ghosts = false);

	void SetPhases(const amrex::Geometry &geom);
	void Generate(amrex::MultiFab &mf, amrex::Real dt);

	[[nodiscard]] auto GetNumModes() const -> int { return num_modes_; }
	[[nodiscard]] auto GetKVec() const -> const std::vector<std::vector<amrex::Real>> & { return k_vec_; }
	[[nodiscard]] auto GetKPeak() const -> amrex::Real { return k_peak_; }
};

// Creates a random set of wave vectors with k_mag within k_peak/2 and 2*k_peak
auto MakeRandomModes(int num_modes, amrex::Real k_peak, uint32_t rseed = 0) -> std::vector<std::vector<amrex::Real>>;

} // namespace quokka::util

#endif // FEWMODESFT_HPP_
