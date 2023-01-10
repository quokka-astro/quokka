//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file generate_modes.cpp
/// \brief Sample a Gaussian random field.
///

#include <random>

#include "generate_modes.hpp"

using amrex::Real;

auto generateRandomModes(const int kmin, const int kmax, const int alpha_PL, const int seed) -> amrex::TableData<Real, 4>
{
	// generate random amplitudes and phases

	amrex::Array<int, 4> tlo{kmin, kmin, kmin, 0};
	amrex::Array<int, 4> thi{kmax, kmax, kmax, 1};
	amrex::TableData<Real, 4> h_table_data(tlo, thi, amrex::The_Pinned_Arena());
	auto const &h_table = h_table_data.table();

	// use 64-bit Mersenne Twister (do not use 32-bit version for sampling doubles!)
	std::mt19937_64 rng(seed); // NOLINT
	std::uniform_real_distribution<double> sample_phase(0., 2.0 * M_PI);
	std::uniform_real_distribution<double> sample_unit(0., 1.0);

	Real rms = 0;
	for (int i = tlo[0]; i <= thi[0]; ++i) {
		for (int j = tlo[1]; j <= thi[1]; ++j) {
			for (int k = tlo[2]; k <= thi[2]; ++k) {
				// compute wavenumber |k|
				const Real kx = static_cast<Real>(i);
				const Real ky = static_cast<Real>(j);
				const Real kz = static_cast<Real>(k);
				const Real k_abs = std::sqrt(kx * kx + ky * ky + kz * kz);

				// sample amplitude from Rayleigh distribution
				Real amp = std::sqrt(-2.0 * std::log(sample_unit(rng) + 1.0e-20));

				// apply power spectrum
				amp /= pow(k_abs, alpha_PL);

				if (i != 0 || j != 0 || k != 0) {
					rms += amp * amp;
					h_table(i, j, k, 0) = amp;
					h_table(i, j, k, 1) = sample_phase(rng);
				} else { // k == 0, set it to zero
					h_table(i, j, k, 0) = 0;
					h_table(i, j, k, 1) = 0;
				}
			}
		}
	}
	return h_table_data;
}

void projectModes(const int kmin, const int kmax, amrex::TableData<Real, 4> &dvx, amrex::TableData<Real, 4> &dvy, amrex::TableData<Real, 4> &dvz)
{
	amrex::Array<int, 4> tlo{kmin, kmin, kmin, 0};
	amrex::Array<int, 4> thi{kmax, kmax, kmax, 1};
	auto const &dvx_table = dvx.table();
	auto const &dvy_table = dvy.table();
	auto const &dvz_table = dvz.table();

	// delete compressive modes
	for (int i = tlo[0]; i <= thi[0]; ++i) {
		for (int j = tlo[1]; j <= thi[1]; ++j) {
			for (int k = tlo[2]; k <= thi[2]; ++k) {
				if (i != 0 || j != 0 || k != 0) {
					// compute k_hat = (kx, ky, kz)
					Real kx = std::sin(2.0 * M_PI * i / kmax);
					Real ky = std::sin(2.0 * M_PI * j / kmax);
					Real kz = std::sin(2.0 * M_PI * k / kmax);
					Real kabs = std::sqrt(kx * kx + ky * ky + kz * kz);
					kx /= kabs;
					ky /= kabs;
					kz /= kabs;

					Real vx = dvx_table(i, j, k, 0);
					Real vy = dvy_table(i, j, k, 0);
					Real vz = dvz_table(i, j, k, 0);
					Real v_dot_khat = vx * kx + vy * ky + vz * kz;

					// return v - (v dot k_hat) k_hat
					dvx_table(i, j, k, 0) -= v_dot_khat * kx;
					dvy_table(i, j, k, 0) -= v_dot_khat * ky;
					dvz_table(i, j, k, 0) -= v_dot_khat * kz;
				}
			}
		}
	}
}

auto computeRms(const int kmin, const int kmax, amrex::TableData<Real, 4> &dvx, amrex::TableData<Real, 4> &dvy, amrex::TableData<Real, 4> &dvz) -> Real
{
	amrex::Array<int, 4> tlo{kmin, kmin, kmin, 0};
	amrex::Array<int, 4> thi{kmax, kmax, kmax, 1};
	auto const &dvx_table = dvx.const_table();
	auto const &dvy_table = dvy.const_table();
	auto const &dvz_table = dvz.const_table();

	// compute rms power
	Real rms_sq = 0;
	for (int i = tlo[0]; i <= thi[0]; ++i) {
		for (int j = tlo[1]; j <= thi[1]; ++j) {
			for (int k = tlo[2]; k <= thi[2]; ++k) {
				Real vx = dvx_table(i, j, k, 0);
				Real vy = dvy_table(i, j, k, 0);
				Real vz = dvz_table(i, j, k, 0);
				rms_sq += vx * vx + vy * vy + vz * vz;
			}
		}
	}
	return std::sqrt(rms_sq);
}
