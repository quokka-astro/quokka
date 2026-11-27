//==============================================================================
// Copyright 2026 Neco Kriel.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file testSlowWave.cpp
/// \brief Combined slow-wave test: fixed-resolution answer test and
///        Richardson convergence test in a single binary.
///
/// The answer test runs at the TOML-specified resolution and checks that the
/// L1 error norm is below a threshold.  The convergence test sweeps
/// resolutions with the Richardson harness and verifies second-order
/// convergence.  Both tests must pass for the binary to exit with status 0.
///

#include <algorithm>
#include <bitset>
#include <cassert>
#include <cmath>
#include <gcem.hpp>
#include <iostream>

#include "AMReX_Array.H"
#include "AMReX_Array4.H"
#include "AMReX_ParallelDescriptor.H"
#include "AMReX_REAL.H"

#include "QuokkaSimulation.hpp"
#include "grid.hpp"
#include "physics_info.hpp"
#include "util/BC.hpp"
#include "util/richardson.hpp"

struct SlowWave {
};

template <> struct quokka::EOS_Traits<SlowWave> {
	static constexpr double gamma = 5. / 3.;
	static constexpr double mean_molecular_weight = C::m_u;
	static constexpr double boltzmann_constant = C::k_B;
};

template <> struct Physics_Traits<SlowWave> {
	static constexpr bool is_hydro_enabled = true;
	static constexpr int numMassScalars = 0;
	static constexpr int numPassiveScalars = numMassScalars + 0;
	static constexpr bool is_self_gravity_enabled = false;
	static constexpr bool is_radiation_enabled = false;
	static constexpr bool is_mhd_enabled = true;
	static constexpr int nGroups = 1;
	static constexpr bool is_dust_enabled = false;
	static constexpr int nDustGroups = 1;
	static constexpr UnitSystem unit_system = UnitSystem::CGS;
};

constexpr double sound_speed = 1.0;
constexpr double gamma_gas = quokka::EOS_Traits<SlowWave>::gamma;
constexpr double bg_density = 1.0;
constexpr double bg_pressure = sound_speed * sound_speed * bg_density / gamma_gas;
constexpr double b0_magn = 1.0;
constexpr double delta_b_magn = 1e-6;
constexpr double alfven_speed = b0_magn / gcem::sqrt(bg_density);

// Set in setupWaveGeometry() before any IC kernel launches.
AMREX_GPU_MANAGED double angle_between_k_b0_rad = 0.0;				// NOLINT
AMREX_GPU_MANAGED double k_rotation_in_xy_rad = 0.0;				// NOLINT
AMREX_GPU_MANAGED double k_elevation_from_xy_rad = 0.0;				// NOLINT
AMREX_GPU_MANAGED std::array<amrex::Real, 3> k_dir_prf{1.0, 0.0, 0.0};		// NOLINT
AMREX_GPU_MANAGED std::array<amrex::Real, 3> inplane_dir_prf{0.0, 1.0, 0.0};	// NOLINT
AMREX_GPU_MANAGED std::array<amrex::Real, 3> outofplane_dir_prf{0.0, 0.0, 1.0}; // NOLINT
AMREX_GPU_MANAGED double k_magn = 2.0 * M_PI;					// NOLINT

AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE auto computeMagnitude(const std::array<amrex::Real, 3> &vfield) -> double
{
	return std::sqrt(vfield[0] * vfield[0] + vfield[1] * vfield[1] + vfield[2] * vfield[2]);
}

AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE auto computeDotProduct(const std::array<amrex::Real, 3> &vfield1, const std::array<amrex::Real, 3> &vfield2) -> double
{
	return vfield1[0] * vfield2[0] + vfield1[1] * vfield2[1] + vfield1[2] * vfield2[2];
}

AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE auto computeCrossProduct(const std::array<amrex::Real, 3> &vfield1, const std::array<amrex::Real, 3> &vfield2)
    -> std::array<amrex::Real, 3>
{
	return {vfield1[1] * vfield2[2] - vfield1[2] * vfield2[1], vfield1[2] * vfield2[0] - vfield1[0] * vfield2[2],
		vfield1[0] * vfield2[1] - vfield1[1] * vfield2[0]};
}

AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE void normalizeVector(std::array<amrex::Real, 3> &vfield)
{
	const double vfield_magn = computeMagnitude(vfield);
	if (vfield_magn > 1e-14) {
		vfield[0] /= vfield_magn;
		vfield[1] /= vfield_magn;
		vfield[2] /= vfield_magn;
	}
}

// PRF<->MRF rotation. See the SlowWaveConvergence reference-frame derivation;
// the slow-mode eigenvector is most naturally written in the MRF (k along
// x1_mrf, B0 in the (x1_mrf, x3_mrf) plane), then rotated back to the PRF
// before being written into AMReX state arrays.
AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE auto rotatePRF2MRF(const std::array<amrex::Real, 3> &vec_prf) -> std::array<amrex::Real, 3>
{
	return {vec_prf[0] * k_dir_prf[0] + vec_prf[1] * k_dir_prf[1] + vec_prf[2] * k_dir_prf[2],
		vec_prf[0] * inplane_dir_prf[0] + vec_prf[1] * inplane_dir_prf[1] + vec_prf[2] * inplane_dir_prf[2],
		vec_prf[0] * outofplane_dir_prf[0] + vec_prf[1] * outofplane_dir_prf[1] + vec_prf[2] * outofplane_dir_prf[2]};
}

AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE auto rotateMRF2PRF(const std::array<amrex::Real, 3> &vec_mrf) -> std::array<amrex::Real, 3>
{
	return {vec_mrf[0] * k_dir_prf[0] + vec_mrf[1] * inplane_dir_prf[0] + vec_mrf[2] * outofplane_dir_prf[0],
		vec_mrf[0] * k_dir_prf[1] + vec_mrf[1] * inplane_dir_prf[1] + vec_mrf[2] * outofplane_dir_prf[1],
		vec_mrf[0] * k_dir_prf[2] + vec_mrf[1] * inplane_dir_prf[2] + vec_mrf[2] * outofplane_dir_prf[2]};
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE auto computeVectorPotentialComponent_prf(const double x1_prf, const double x2_prf, const double x3_prf, const double time,
									     const int icomp) -> double
{
	const std::array<amrex::Real, 3> x_vec_mrf = rotatePRF2MRF({x1_prf, x2_prf, x3_prf});
	const double tiny = 1e-16;

	const double cos_angle = std::cos(angle_between_k_b0_rad);
	const double sin_angle = std::sin(angle_between_k_b0_rad);
	const double B0_1 = b0_magn * cos_angle;
	const double B0_2 = b0_magn * sin_angle;

	const double bg_A1 = 0.0;
	const double bg_A2 = 0.0;
	const double bg_A3 = -B0_2 * x_vec_mrf[0] + B0_1 * x_vec_mrf[1];

	const double cs =
	    std::sqrt(0.5 * (sound_speed * sound_speed + alfven_speed * alfven_speed -
			     std::sqrt((sound_speed * sound_speed + alfven_speed * alfven_speed) * (sound_speed * sound_speed + alfven_speed * alfven_speed) -
				       4.0 * sound_speed * sound_speed * alfven_speed * alfven_speed * cos_angle * cos_angle)));

	const double omega = cs * k_magn;
	const double phase = omega * time - k_magn * x_vec_mrf[0];
	const double delta_A1 = 0.0;
	const double delta_A2 = 0.0;
	double delta_A3 = 0.0;

	if (std::abs(sin_angle) < tiny || std::abs(cos_angle) < tiny) {
		delta_A3 = 0.0;
	} else {
		delta_A3 = (delta_b_magn / k_magn) * std::sin(phase);
	}
	const double A1_mrf = bg_A1 + delta_A1;
	const double A2_mrf = bg_A2 + delta_A2;
	const double A3_mrf = bg_A3 + delta_A3;
	const std::array<amrex::Real, 3> A_prf = rotateMRF2PRF({A1_mrf, A2_mrf, A3_mrf});
	return A_prf[icomp];
}

AMREX_GPU_DEVICE inline auto Ax_prf(const double x1_prf, const double x2_prf, const double x3_prf, const double time) -> double
{
	return computeVectorPotentialComponent_prf(x1_prf, x2_prf, x3_prf, time, 0);
}

AMREX_GPU_DEVICE inline auto Ay_prf(const double x1_prf, const double x2_prf, const double x3_prf, const double time) -> double
{
	return computeVectorPotentialComponent_prf(x1_prf, x2_prf, x3_prf, time, 1);
}

AMREX_GPU_DEVICE inline auto Az_prf(const double x1_prf, const double x2_prf, const double x3_prf, const double time) -> double
{
	return computeVectorPotentialComponent_prf(x1_prf, x2_prf, x3_prf, time, 2);
}

AMREX_GPU_DEVICE
void computeWaveSolution(int i, int j, int k, amrex::Array4<amrex::Real> const &state, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx,
			 amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_lo, quokka::centering cen, quokka::direction dir, amrex::Real time)
{
	const amrex::Real x1_prf_L = prob_lo[0] + i * dx[0];
	const amrex::Real x2_prf_L = prob_lo[1] + j * dx[1];
	const amrex::Real x3_prf_L = prob_lo[2] + k * dx[2];

	if (cen == quokka::centering::cc) {
		const double tiny = 1e-16;
		const amrex::Real x1_prf_C = x1_prf_L + static_cast<amrex::Real>(0.5) * dx[0];
		const amrex::Real x2_prf_C = x2_prf_L + static_cast<amrex::Real>(0.5) * dx[1];
		const amrex::Real x3_prf_C = x3_prf_L + static_cast<amrex::Real>(0.5) * dx[2];
		const std::array<amrex::Real, 3> x_vec_mrf_C = rotatePRF2MRF({x1_prf_C, x2_prf_C, x3_prf_C});

		const double cos_angle = std::cos(angle_between_k_b0_rad);
		const double sin_angle = std::sin(angle_between_k_b0_rad);

		const double cs = std::sqrt(
		    0.5 * (sound_speed * sound_speed + alfven_speed * alfven_speed -
			   std::sqrt((sound_speed * sound_speed + alfven_speed * alfven_speed) * (sound_speed * sound_speed + alfven_speed * alfven_speed) -
				     4.0 * sound_speed * sound_speed * alfven_speed * alfven_speed * cos_angle * cos_angle)));

		const double omega = cs * k_magn;
		const double phase = omega * time - k_magn * x_vec_mrf_C[0];
		const double cos_phase = std::cos(phase);
		double epsilon = (std::abs(sin_angle) < tiny)
				     ? 0.0
				     : (delta_b_magn / b0_magn * (cs * cs - alfven_speed * alfven_speed * cos_angle * cos_angle) / (cs * cs * sin_angle));
		const double B0_1 = b0_magn * cos_angle;
		const double B0_2 = b0_magn * sin_angle;

		double v1_mrf = 0.0;
		double v2_mrf = 0.0;
		double delta_B2 = 0.0;

		if (std::abs(sin_angle) < tiny) {
			// theta = 0 deg: slow mode reduces to a pure sound wave.
			v1_mrf = -delta_b_magn / b0_magn * cs * cos_phase;
			v2_mrf = 0.0;
			delta_B2 = 0.0;
		} else if (std::abs(cos_angle) < tiny) {
			// theta = 90 deg: c_s = 0; mode becomes a static pressure-balanced
			// structure with no perturbation.
			v1_mrf = 0.0;
			v2_mrf = 0.0;
			delta_B2 = 0.0;
			epsilon = 0.0;
		} else {
			delta_B2 = delta_b_magn * cos_phase;
			v1_mrf = -epsilon * cs * cos_phase;
			v2_mrf = delta_b_magn / b0_magn * alfven_speed * alfven_speed * cos_angle / cs * cos_phase;
		}

		double const v3_mrf = 0.0;

		const double density = bg_density * (1.0 + epsilon * cos_phase);
		const double pressure = bg_pressure * (1.0 + gamma_gas * epsilon * cos_phase);

		const auto v_prf = rotateMRF2PRF({v1_mrf, v2_mrf, v3_mrf});
		const auto dB_prf = rotateMRF2PRF({0.0, delta_B2, 0.0});
		const auto B0_prf = rotateMRF2PRF({B0_1, B0_2, 0.0});
		const double b_x1_prf = B0_prf[0] + dB_prf[0];
		const double b_x2_prf = B0_prf[1] + dB_prf[1];
		const double b_x3_prf = B0_prf[2] + dB_prf[2];

		const double v_magn_sq = v_prf[0] * v_prf[0] + v_prf[1] * v_prf[1] + v_prf[2] * v_prf[2];
		const double b_magn_sq = b_x1_prf * b_x1_prf + b_x2_prf * b_x2_prf + b_x3_prf * b_x3_prf;
		const double Ekin = 0.5 * density * v_magn_sq;
		const double Emag = 0.5 * b_magn_sq;
		const double Eint = pressure / (gamma_gas - 1);
		const double Etot = Ekin + Emag + Eint;

		state(i, j, k, HydroSystem<SlowWave>::density_index) = density;
		state(i, j, k, HydroSystem<SlowWave>::x1Momentum_index) = v_prf[0] * density;
		state(i, j, k, HydroSystem<SlowWave>::x2Momentum_index) = v_prf[1] * density;
		state(i, j, k, HydroSystem<SlowWave>::x3Momentum_index) = v_prf[2] * density;
		state(i, j, k, HydroSystem<SlowWave>::energy_index) = Etot;
		state(i, j, k, HydroSystem<SlowWave>::internalEnergy_index) = Eint;

	} else if (cen == quokka::centering::fc) {
		// Compute B from the vector potential to preserve div(B) = 0 exactly.
		if (dir == quokka::direction::x) {
			const double b_x1 =
			    (Az_prf(x1_prf_L, x2_prf_L + dx[1], x3_prf_L + dx[2] / 2.0, time) - Az_prf(x1_prf_L, x2_prf_L, x3_prf_L + dx[2] / 2.0, time)) /
				dx[1] -
			    (Ay_prf(x1_prf_L, x2_prf_L + dx[1] / 2.0, x3_prf_L + dx[2], time) - Ay_prf(x1_prf_L, x2_prf_L + dx[1] / 2.0, x3_prf_L, time)) /
				dx[2];
			state(i, j, k, MHDSystem<SlowWave>::bfield_index) = b_x1;
		} else if (dir == quokka::direction::y) {
			const double b_x2 =
			    (Ax_prf(x1_prf_L + dx[0] / 2.0, x2_prf_L, x3_prf_L + dx[2], time) - Ax_prf(x1_prf_L + dx[0] / 2.0, x2_prf_L, x3_prf_L, time)) /
				dx[2] -
			    (Az_prf(x1_prf_L + dx[0], x2_prf_L, x3_prf_L + dx[2] / 2.0, time) - Az_prf(x1_prf_L, x2_prf_L, x3_prf_L + dx[2] / 2.0, time)) /
				dx[0];
			state(i, j, k, MHDSystem<SlowWave>::bfield_index) = b_x2;
		} else if (dir == quokka::direction::z) {
			const double b_x3 =
			    (Ay_prf(x1_prf_L + dx[0], x2_prf_L + dx[1] / 2.0, x3_prf_L, time) - Ay_prf(x1_prf_L, x2_prf_L + dx[1] / 2.0, x3_prf_L, time)) /
				dx[0] -
			    (Ax_prf(x1_prf_L + dx[0] / 2.0, x2_prf_L + dx[1], x3_prf_L, time) - Ax_prf(x1_prf_L + dx[0] / 2.0, x2_prf_L, x3_prf_L, time)) /
				dx[1];
			state(i, j, k, MHDSystem<SlowWave>::bfield_index) = b_x3;
		}
	}
}

template <> void QuokkaSimulation<SlowWave>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = grid_elem.dx_;
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = grid_elem.prob_lo_;
	const amrex::Array4<double> &state_cc = grid_elem.array_;
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const quokka::centering cen = grid_elem.cen_;
	const quokka::direction dir = grid_elem.dir_;

	const int ncomp_cc = Physics_Indices<SlowWave>::nvarTotal_cc;
	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
		for (int n = 0; n < ncomp_cc; ++n) {
			state_cc(i, j, k, n) = 0;
		}
		computeWaveSolution(i, j, k, state_cc, dx, prob_lo, cen, dir, 0);
	});
}

template <> void QuokkaSimulation<SlowWave>::setInitialConditionsOnGridFaceVars(quokka::grid const &grid_elem)
{
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = grid_elem.dx_;
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = grid_elem.prob_lo_;
	const amrex::Array4<double> &state_fc = grid_elem.array_;
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const quokka::centering cen = grid_elem.cen_;
	const quokka::direction dir = grid_elem.dir_;

	const int ncomp_fc = Physics_Indices<SlowWave>::nvarPerDim_fc;
	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
		for (int n = 0; n < ncomp_fc; ++n) {
			state_fc(i, j, k, n) = 0;
		}
		computeWaveSolution(i, j, k, state_fc, dx, prob_lo, cen, dir, 0);
	});
}

template <>
void QuokkaSimulation<SlowWave>::computeReferenceSolution(amrex::MultiFab &ref, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx,
							  amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_lo)
{
	for (amrex::MFIter iter(ref); iter.isValid(); ++iter) {
		const amrex::Box &indexRange = iter.validbox();
		auto const &stateExact = ref.array(iter);
		auto const ncomp = ref.nComp();
		const amrex::Real time = tNew_[0];

		amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
			for (int n = 0; n < ncomp; ++n) {
				stateExact(i, j, k, n) = 0.0;
			}
			computeWaveSolution(i, j, k, stateExact, dx, prob_lo, quokka::centering::cc, quokka::direction::na, time);
		});
	}
}

template <>
void QuokkaSimulation<SlowWave>::computeReferenceSolution_fc(amrex::MultiFab &ref, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx,
							     amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_lo, quokka::direction const dir)
{
	for (amrex::MFIter iter(ref); iter.isValid(); ++iter) {
		const amrex::Box &indexRange = iter.validbox();
		auto const &stateExact = ref.array(iter);
		auto const ncomp = ref.nComp();
		const amrex::Real time = tNew_[0];

		amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
			for (int n = 0; n < ncomp; ++n) {
				stateExact(i, j, k, n) = 0.0;
			}
			computeWaveSolution(i, j, k, stateExact, dx, prob_lo, quokka::centering::fc, dir, time);
		});
	}
}

namespace
{
// Read wave-geometry parameters from the "setup" ParmParse table and populate
// all GPU-managed globals (angle, k direction, orthonormal basis).
void setupWaveGeometry()
{
	amrex::ParmParse const setup_pp("setup");
	double angle_between_k_b0_deg = 45.0;
	setup_pp.query("angle_between_k_b0", angle_between_k_b0_deg);
	constexpr double deg2rad = M_PI / 180.0;
	angle_between_k_b0_rad = deg2rad * angle_between_k_b0_deg;

	int num_modes_x = 1;
	int num_modes_y = 0;
	int num_modes_z = 0;
	setup_pp.query("num_modes_x", num_modes_x);
	setup_pp.query("num_modes_y", num_modes_y);
	setup_pp.query("num_modes_z", num_modes_z);
	if ((num_modes_x == 0) && (num_modes_y == 0) && (num_modes_z == 0)) {
		amrex::Abort("Invalid k modes: the triplet (0,0,0) is not allowed.");
	}

	const std::array<amrex::Real, 3> k_vec_prf = {2.0 * M_PI * static_cast<amrex::Real>(num_modes_x), 2.0 * M_PI * static_cast<amrex::Real>(num_modes_y),
						      2.0 * M_PI * static_cast<amrex::Real>(num_modes_z)};
	k_magn = computeMagnitude(k_vec_prf);
	constexpr double tiny = 1e-16;
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(std::abs(b0_magn) > tiny, "b0_magn must be nonzero.");
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(std::abs(k_magn) > tiny, "k_magn must be nonzero.");
	k_dir_prf = {k_vec_prf[0] / k_magn, k_vec_prf[1] / k_magn, k_vec_prf[2] / k_magn};

	k_rotation_in_xy_rad = std::atan2(k_dir_prf[1], k_dir_prf[0]);
	k_elevation_from_xy_rad = std::atan2(k_dir_prf[2], std::hypot(k_dir_prf[0], k_dir_prf[1]));

	std::array<amrex::Real, 3> ref_prf{0.0, 0.0, 1.0};
	if (std::abs(computeDotProduct(ref_prf, k_dir_prf)) > 0.9999) {
		ref_prf = {0.0, 1.0, 0.0};
	}
	inplane_dir_prf = computeCrossProduct(ref_prf, k_dir_prf);
	normalizeVector(inplane_dir_prf);
	outofplane_dir_prf = computeCrossProduct(k_dir_prf, inplane_dir_prf);
	normalizeVector(outofplane_dir_prf);
}
} // namespace

// Run the fixed-resolution answer test using parameters from the TOML file.
// Returns 0 on pass, 1 on failure.
auto runAnswerTest() -> int
{
	setupWaveGeometry();

	const double cos_angle = std::cos(angle_between_k_b0_rad);
	const double cs =
	    std::sqrt(0.5 * (sound_speed * sound_speed + alfven_speed * alfven_speed -
			     std::sqrt((sound_speed * sound_speed + alfven_speed * alfven_speed) * (sound_speed * sound_speed + alfven_speed * alfven_speed) -
				       4.0 * sound_speed * sound_speed * alfven_speed * alfven_speed * cos_angle * cos_angle)));
	const double wavelength = 2.0 * M_PI / k_magn;
	const double wave_period = wavelength / cs;

	amrex::Print() << "SlowWave answer test: angle=" << angle_between_k_b0_rad * 180.0 / M_PI << " deg, |k|=" << k_magn << ", c_s=" << cs
		       << ", wave_period=" << wave_period << "\n";

	QuokkaSimulation<SlowWave> sim;
	sim.setInitialConditions();
	sim.evolve();

	const double error_tol = 0.002;
	auto comp_errors = sim.computeComponentErrors();

	const auto n_cells = static_cast<amrex::Real>(sim.state_new_cc_[0].boxArray().numPts());
	amrex::Real sum_sq_err = 0.0;
	amrex::Real sum_sq_ref = 0.0;
	for (const auto &[name, abs_err, rel_err] : comp_errors) {
		amrex::Real const L1_err = abs_err * n_cells;
		sum_sq_err += L1_err * L1_err;
		if (!std::isnan(rel_err) && rel_err != 0.0 && abs_err > 10E-15) {
			amrex::Real const L1_ref = (abs_err / rel_err) * n_cells;
			sum_sq_ref += L1_ref * L1_ref;
		}
	}

	const amrex::Real err_norm = std::sqrt(sum_sq_err);
	const amrex::Real sol_norm = std::sqrt(sum_sq_ref);
	amrex::Real error_norm = 0.0;
	if (sol_norm > 0.0) {
		error_norm = err_norm / sol_norm;
		amrex::Print() << "Relative RMS L1 error norm = " << error_norm << "\n";
	} else {
		error_norm = err_norm;
		amrex::Print() << "Absolute L1 error norm = " << error_norm << "\n";
	}

	return (error_norm > error_tol) ? 1 : 0;
}

// Run a single convergence-test simulation at resolution nx and return the
// error norm.  Called repeatedly by the Richardson harness.
// Note: uses unconditional pp.add() so that ParmParse's last-added-value
// semantics override any per-direction entries already set by the TOML.
auto runWaveTest(int nx) -> double
{
	setupWaveGeometry();

	const double cos_angle = std::cos(angle_between_k_b0_rad);
	const double cs =
	    std::sqrt(0.5 * (sound_speed * sound_speed + alfven_speed * alfven_speed -
			     std::sqrt((sound_speed * sound_speed + alfven_speed * alfven_speed) * (sound_speed * sound_speed + alfven_speed * alfven_speed) -
				       4.0 * sound_speed * sound_speed * alfven_speed * alfven_speed * cos_angle * cos_angle)));
	const int max_timesteps = std::max(20000, nx * 100);
	const double wavelength = 2.0 * M_PI / k_magn;
	const double max_time = wavelength / cs;

	{
		amrex::ParmParse pp("amr");
		const amrex::Vector<int> ncells = {nx, 8, 8};
		const int blocking_x = std::max(16, nx);
		// Unconditional adds: ParmParse searches from the most-recently-added entry
		// first, so these override any same-named entries already in the TOML.
		pp.add("blocking_factor_x", blocking_x);
		pp.add("blocking_factor_y", 8);
		pp.add("blocking_factor_z", 8);
		pp.add("max_grid_size", nx);
		pp.add("max_grid_size_x", nx);
		pp.add("max_level", 0);
		pp.addarr("n_cell", ncells);
	}

	{
		amrex::ParmParse pp_geom("geometry");
		pp_geom.addarr("prob_lo", amrex::Vector<double>{0.0, 0.0, 0.0});
		pp_geom.addarr("prob_hi", amrex::Vector<double>{1.0, 1.0, 1.0});
		pp_geom.addarr("is_periodic", amrex::Vector<int>{1, 1, 1});
	}

	auto BCs_cc = quokka::BC<SlowWave>(quokka::BCType::int_dir);
	const int nvars_fc = Physics_Indices<SlowWave>::nvarTotal_fc;
	amrex::Vector<amrex::BCRec> BCs_fc(nvars_fc);
	for (int icomp = 0; icomp < nvars_fc; ++icomp) {
		for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
			BCs_fc[icomp].setLo(idim, amrex::BCType::int_dir);
			BCs_fc[icomp].setHi(idim, amrex::BCType::int_dir);
		}
	}

	QuokkaSimulation<SlowWave> sim(BCs_cc, BCs_fc);
	sim.cflNumber_ = 0.2;
	sim.stopTime_ = max_time;
	sim.maxTimesteps_ = max_timesteps;
	sim.setInitialConditions();
	sim.evolve();

	return sim.computeErrorNorm();
}

// Run the Richardson convergence test. Returns 0 on pass, 1 on failure.
auto runConvergenceTest() -> int
{
	quokka::richardson::applyQuietDefaults();

	quokka::richardson::Parameters params{};
	params.machine_precision_target = 2.0e-9;
	params.nx_initial = 16;
	params.nx_max = 128;
	params.expected_rate = 2.0;
	params.tolerance = 0.3;
	params.test_name = "Slow Wave";
	params.csv_filename = "slow_wave_convergence.csv";

	return quokka::richardson::run(params, [](int nx) { return runWaveTest(nx); });
}

auto problem_main() -> int
{
	amrex::Print() << std::string(70, '=') << "\n";
	amrex::Print() << "Running SlowWave answer test\n";
	amrex::Print() << std::string(70, '=') << "\n";
	const int answer_status = runAnswerTest();
	amrex::Print() << "SlowWave answer test: " << (answer_status == 0 ? "PASSED" : "FAILED") << "\n\n";

	amrex::Print() << std::string(70, '=') << "\n";
	amrex::Print() << "Running SlowWave convergence test\n";
	amrex::Print() << std::string(70, '=') << "\n";
	const int convergence_status = runConvergenceTest();
	amrex::Print() << "SlowWave convergence test: " << (convergence_status == 0 ? "PASSED" : "FAILED") << "\n\n";

	amrex::Print() << std::string(70, '=') << "\n";
	if (answer_status == 0 && convergence_status == 0) {
		amrex::Print() << "All SlowWave tests PASSED\n";
	} else {
		amrex::Print() << "Some SlowWave tests FAILED\n";
	}
	amrex::Print() << std::string(70, '=') << "\n";

	return (answer_status != 0 || convergence_status != 0) ? 1 : 0;
}
