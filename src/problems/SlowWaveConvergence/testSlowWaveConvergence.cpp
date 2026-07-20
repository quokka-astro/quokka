//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file testSlowWaveConvergence.cpp
/// \brief Defines a Richardson convergence test for the slow MHD wave.
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

struct SlowWaveConvergence {
};

template <> struct quokka::EOS_Traits<SlowWaveConvergence> {
	static constexpr double gamma = 5. / 3.;
	static constexpr double mean_molecular_weight = C::m_u;
};

template <> struct Physics_Traits<SlowWaveConvergence> : DefaultPhysicsTraits {
	static constexpr bool is_hydro_enabled = true;
	static constexpr bool is_mhd_enabled = true;
};

constexpr double sound_speed = 1.0;
constexpr double gamma_gas = quokka::EOS_Traits<SlowWaveConvergence>::gamma;
constexpr double bg_density = 1.0;
constexpr double bg_pressure = sound_speed * sound_speed * bg_density / gamma_gas;
constexpr double b0_magn = 1.0;
constexpr double delta_b_magn = 1e-6;
constexpr double alfven_speed = b0_magn / gcem::sqrt(bg_density);

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

// angles (radians) in the math reference frame (MRF)
AMREX_GPU_MANAGED double angle_between_k_b0_rad = 0.0; // NOLINT

// rotation from the problem reference frame (PRF) to the MRF
AMREX_GPU_MANAGED double k_rotation_in_xy_rad = 0.0;	// NOLINT
AMREX_GPU_MANAGED double k_elevation_from_xy_rad = 0.0; // NOLINT

//------------------------------------------------------------------------------
// Reference frames and rotation matrix
//
// We work with two right-handed, orthonormal frames:
//
// PRF (Problem Reference Frame): the simulation's grid-aligned axes.
// MRF (Math Reference Frame):    a wave-aligned basis:
//                                e1 = k_dir_prf          (propagation direction)
//                                e2 = inplane_dir_prf    (lies in the k-b0 plane)
//                                e3 = outofplane_dir_prf (perpendicular to that plane)
//
// The three unit vectors (k_dir_prf, inplane_dir_prf, outofplane_dir_prf) are
// stored as 3-vectors *expressed in PRF coordinates*. Arranged as the *rows*
// of a 3x3 matrix
//
//             [ k_dir_prf^T         ]    [ r00 r01 r02   (row 0)
// R  =  rows  [ inplane_dir_prf^T   ]  =   r10 r11 r12   (row 1)
//             [ outofplane_dir_prf^T]      r20 r21 r22 ] (row 2)
//
// R maps PRF component vectors into MRF component vectors via a standard
// rotation (passive change of basis):
//
// v_mrf = R * v_prf
//
// Because the rows are orthonormal, R is a pure rotation, so the inverse is
// its transpose:
//
// v_prf = R^T * v_mrf
//
// The two helpers below implement exactly these operations using dot products
// with the basis vectors (to stay GPU-friendly and avoid building dynamic
// matrices):
//
// - rotatePRF2MRF(v_prf) -> R * v_prf
// - rotateMRF2PRF(v_mrf) -> R^T * v_mrf
//
// Preconditions:
//  (k_dir_prf, inplane_dir_prf, outofplane_dir_prf) form a right-handed,
//  orthonormal basis (constructed in problem_main()).
//------------------------------------------------------------------------------

// Unit basis vectors of the MRF, expressed in PRF coordinates (rows of R):
// row 0: e1 = k_dir_prf (propagation)
// row 1: e2 = inplane_dir_prf (k-b0 plane)
// row 2: e3 = outofplane_dir_prf (perpendicular to that plane)
AMREX_GPU_MANAGED std::array<amrex::Real, 3> k_dir_prf{1.0, 0.0, 0.0};		// NOLINT
AMREX_GPU_MANAGED std::array<amrex::Real, 3> inplane_dir_prf{0.0, 1.0, 0.0};	// NOLINT
AMREX_GPU_MANAGED std::array<amrex::Real, 3> outofplane_dir_prf{0.0, 0.0, 1.0}; // NOLINT

// wavefront
AMREX_GPU_MANAGED double k_magn = 2.0 * M_PI; // NOLINT

/// \brief Rotate a vector from PRF to MRF by multiplying with the rotation matrix R.
/// \details Implements v_mrf = R * v_prf, where the rows of R are the
///          MRF basis vectors expressed in PRF coordinates:
///          R = [k_dir_prf^T; inplane_dir_prf^T; outofplane_dir_prf^T].
/// \param vec_prf Components of the vector in the PRF.
/// \return Components of the same geometric vector in the MRF.
AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE auto rotatePRF2MRF(const std::array<amrex::Real, 3> &vec_prf) -> std::array<amrex::Real, 3>
{
	// v_mrf[i] = e_i^T * v_prf  (i = 0:k, 1:in-plane, 2:out-of-plane)
	return {vec_prf[0] * k_dir_prf[0] + vec_prf[1] * k_dir_prf[1] + vec_prf[2] * k_dir_prf[2],
		vec_prf[0] * inplane_dir_prf[0] + vec_prf[1] * inplane_dir_prf[1] + vec_prf[2] * inplane_dir_prf[2],
		vec_prf[0] * outofplane_dir_prf[0] + vec_prf[1] * outofplane_dir_prf[1] + vec_prf[2] * outofplane_dir_prf[2]};
}

/// \brief Rotate a vector from MRF back to PRF by multiplying with R^T.
/// \details Implements v_prf = R^T * v_mrf. Because R is orthonormal, R^{-1}=R^T.
/// \param vec_mrf Components of the vector in the MRF.
/// \return Components of the same geometric vector in the PRF.
AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE auto rotateMRF2PRF(const std::array<amrex::Real, 3> &vec_mrf) -> std::array<amrex::Real, 3>
{
	// v_prf = k_dir_prf * v_mrf[0] + inplane_dir_prf * v_mrf[1] + outofplane_dir_prf * v_mrf[2]
	return {vec_mrf[0] * k_dir_prf[0] + vec_mrf[1] * inplane_dir_prf[0] + vec_mrf[2] * outofplane_dir_prf[0],
		vec_mrf[0] * k_dir_prf[1] + vec_mrf[1] * inplane_dir_prf[1] + vec_mrf[2] * outofplane_dir_prf[1],
		vec_mrf[0] * k_dir_prf[2] + vec_mrf[1] * inplane_dir_prf[2] + vec_mrf[2] * outofplane_dir_prf[2]};
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE auto computeVectorPotentialComponent_prf(const double x1_prf, const double x2_prf, const double x3_prf, const double time,
									     const int icomp) -> double
{
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(icomp == 0 || icomp == 1 || icomp == 2, "computeVectorPotentialComponent_prf(): icomp must be 0,1,2");

	// rotate PRF -> MRF
	const std::array<amrex::Real, 3> x_vec_mrf = rotatePRF2MRF({x1_prf, x2_prf, x3_prf});
	const double tiny = 1e-16;
	// Assumes: k along x1_mrf, B0 in (x1_mrf, x3_mrf) plane, and A along x2_mrf.

	// background B0 in MRF
	const double θ = angle_between_k_b0_rad;
	const double B0_1 = b0_magn * std::cos(θ);
	const double B0_2 = b0_magn * std::sin(θ);

	// background vector potential that yields B0 in MRF:
	// bg_A = (0, B0_1 * x3 - B0_3 * x1, 0)
	const double bg_A1 = 0.0;
	const double bg_A2 = 0.0;
	const double bg_A3 = -B0_2 * x_vec_mrf[0] + B0_1 * x_vec_mrf[1];

	// slow speed and phase
	const double a = sound_speed;
	const double vA = alfven_speed;
	const double cosθ = std::cos(θ);
	const double sinθ = std::sin(θ);

	const double cs = std::sqrt(0.5 * (a * a + vA * vA - std::sqrt((a * a + vA * vA) * (a * a + vA * vA) - 4.0 * a * a * vA * vA * cosθ * cosθ)));

	const double omega = cs * k_magn;
	const double phase = omega * time - k_magn * x_vec_mrf[0];
	const double delta_A1 = 0.0;
	const double delta_A2 = 0.0;
	double delta_A3 = 0.0;

	if (std::abs(sinθ) < tiny || std::abs(cosθ) < tiny) {
		// theta = 0 or 180 deg: slow mode is pure sound wave → no B perturbation
		// theta = 90 deg: no perturbations in B1 or B2
		delta_A3 = 0.0; // δB = 0
	} else {
		const double dB2_mrf = delta_b_magn; // δB3
		delta_A3 = (dB2_mrf / k_magn) * std::sin(phase);
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
		AMREX_ALWAYS_ASSERT_WITH_MESSAGE(std::abs(b0_magn) > tiny, "computeWaveSolution: background magnetic field magnitude b0_magn must be nonzero.");
		AMREX_ALWAYS_ASSERT_WITH_MESSAGE(std::abs(k_magn) > tiny, "computeWaveSolution: wavevector magnitude k_magn must be nonzero.");
		const amrex::Real x1_prf_C = x1_prf_L + static_cast<amrex::Real>(0.5) * dx[0];
		const amrex::Real x2_prf_C = x2_prf_L + static_cast<amrex::Real>(0.5) * dx[1];
		const amrex::Real x3_prf_C = x3_prf_L + static_cast<amrex::Real>(0.5) * dx[2];
		const std::array<amrex::Real, 3> x_vec_mrf_C = rotatePRF2MRF({x1_prf_C, x2_prf_C, x3_prf_C});

		// speeds & geometry
		const double a = sound_speed;
		const double vA = alfven_speed;
		const double θ = angle_between_k_b0_rad;
		const double cosθ = std::cos(θ);
		const double sinθ = std::sin(θ);

		const double cs = std::sqrt(0.5 * (a * a + vA * vA - std::sqrt((a * a + vA * vA) * (a * a + vA * vA) - 4.0 * a * a * vA * vA * cosθ * cosθ)));

		const double omega = cs * k_magn;
		const double phase = omega * time - k_magn * x_vec_mrf_C[0];
		const double cos_phase = std::cos(phase);
		double epsilon =
		    (std::abs(sinθ) < tiny) ? 0.0 : (delta_b_magn / b0_magn * (cs * cs - vA * vA * cosθ * cosθ) / (cs * cs * sinθ)); // normalized amplitude
		const double B0_1 = b0_magn * cosθ;
		const double B0_2 = b0_magn * sinθ;

		// Velocity perturbations in MRF (from slow mode eigenvector)
		double v1_mrf = 0.0;
		double v2_mrf = 0.0;

		// Magnetic field perturbation in MRF: δB = (0, 0, δB3)
		double delta_B2 = 0.0;

		// compute velocity perturbations in MRF
		if (std::abs(sinθ) < tiny) {
			// Pure sound wave: set amplitude via epsilon (velocity/density perturbation)
			if (i == 0 && j == 0 && k == 0 && time == 0.0) {
				amrex::Warning(
				    "Warning: angle between k and B0 is 0 or 180 deg. Slow wave reduces to pure sound wave with no magnetic perturbation.");
			}
			v1_mrf = -delta_b_magn / b0_magn * cs * cos_phase; // velocity along k̂ (parallel component)
			v2_mrf = 0.0;					   // perpendicular velocity suppressed
			delta_B2 = 0.0;					   // no transverse magnetic perturbation

		} else if (std::abs(cosθ) < tiny) {
			if (i == 0 && j == 0 && k == 0 && time == 0.0) {
				amrex::Warning(
				    "Slow wave at 90 degrees: c_s = 0, mode becomes static pressure-balanced structure. Setting all perturbations to zero.");
			}
			v1_mrf = 0.0;	// no parallel velocity
			v2_mrf = 0.0;	// no perpendicular velocity
			delta_B2 = 0.0; // no magnetic perturbation
			epsilon = 0.0;	// density/pressure perturbation set to zero
		} else {
			// --- Oblique slow magnetosonic wave ---
			delta_B2 = delta_b_magn * cos_phase;
			v1_mrf = -epsilon * cs * cos_phase; // velocity along k̂ (parallel component)
			v2_mrf = delta_b_magn / b0_magn * vA * vA * cosθ / cs * cos_phase;
		}

		double const v3_mrf = 0.0;

		// density & pressure perturbations (linear compressive slow mode)
		const double density = bg_density * (1.0 + epsilon * cos_phase);
		const double pressure = bg_pressure * (1.0 + gamma_gas * epsilon * cos_phase);

		const auto v_prf = rotateMRF2PRF({v1_mrf, v2_mrf, v3_mrf});
		const auto dB_prf = rotateMRF2PRF({0.0, delta_B2, 0.0});
		const auto B0_prf = rotateMRF2PRF({B0_1, B0_2, 0.0});
		const double b_x1_prf = B0_prf[0] + dB_prf[0];
		const double b_x2_prf = B0_prf[1] + dB_prf[1];
		const double b_x3_prf = B0_prf[2] + dB_prf[2];

		// energy bookkeeping
		const double v_magn_sq = v_prf[0] * v_prf[0] + v_prf[1] * v_prf[1] + v_prf[2] * v_prf[2];
		const double b_magn_sq = b_x1_prf * b_x1_prf + b_x2_prf * b_x2_prf + b_x3_prf * b_x3_prf;
		const double Ekin = 0.5 * density * v_magn_sq;
		const double Emag = 0.5 * b_magn_sq;
		const double Eint = pressure / (gamma_gas - 1);
		const double Etot = Ekin + Emag + Eint;

		// write state
		state(i, j, k, HydroSystem<SlowWaveConvergence>::density_index) = density;
		state(i, j, k, HydroSystem<SlowWaveConvergence>::x1Momentum_index) = v_prf[0] * density;
		state(i, j, k, HydroSystem<SlowWaveConvergence>::x2Momentum_index) = v_prf[1] * density;
		state(i, j, k, HydroSystem<SlowWaveConvergence>::x3Momentum_index) = v_prf[2] * density;
		state(i, j, k, HydroSystem<SlowWaveConvergence>::energy_index) = Etot;
		state(i, j, k, HydroSystem<SlowWaveConvergence>::internalEnergy_index) = Eint;

	} else if (cen == quokka::centering::fc) {
		// compute b-field using the magnetic vector potential to preserve div(b) = 0 topology
		if (dir == quokka::direction::x) {
			const double b_x1 =
			    (Az_prf(x1_prf_L, x2_prf_L + dx[1], x3_prf_L + dx[2] / 2.0, time) - Az_prf(x1_prf_L, x2_prf_L, x3_prf_L + dx[2] / 2.0, time)) /
				dx[1] -
			    (Ay_prf(x1_prf_L, x2_prf_L + dx[1] / 2.0, x3_prf_L + dx[2], time) - Ay_prf(x1_prf_L, x2_prf_L + dx[1] / 2.0, x3_prf_L, time)) /
				dx[2];
			state(i, j, k, MHDSystem<SlowWaveConvergence>::bfield_index) = b_x1;
		} else if (dir == quokka::direction::y) {
			const double b_x2 =
			    (Ax_prf(x1_prf_L + dx[0] / 2.0, x2_prf_L, x3_prf_L + dx[2], time) - Ax_prf(x1_prf_L + dx[0] / 2.0, x2_prf_L, x3_prf_L, time)) /
				dx[2] -
			    (Az_prf(x1_prf_L + dx[0], x2_prf_L, x3_prf_L + dx[2] / 2.0, time) - Az_prf(x1_prf_L, x2_prf_L, x3_prf_L + dx[2] / 2.0, time)) /
				dx[0];
			state(i, j, k, MHDSystem<SlowWaveConvergence>::bfield_index) = b_x2;
		} else if (dir == quokka::direction::z) {
			const double b_x3 =
			    (Ay_prf(x1_prf_L + dx[0], x2_prf_L + dx[1] / 2.0, x3_prf_L, time) - Ay_prf(x1_prf_L, x2_prf_L + dx[1] / 2.0, x3_prf_L, time)) /
				dx[0] -
			    (Ax_prf(x1_prf_L + dx[0] / 2.0, x2_prf_L + dx[1], x3_prf_L, time) - Ax_prf(x1_prf_L + dx[0] / 2.0, x2_prf_L, x3_prf_L, time)) /
				dx[1];
			state(i, j, k, MHDSystem<SlowWaveConvergence>::bfield_index) = b_x3;
		}
	}
}

template <> void QuokkaSimulation<SlowWaveConvergence>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = grid_elem.dx_;
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = grid_elem.prob_lo_;
	const amrex::Array4<double> &state_cc = grid_elem.array_;
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const quokka::centering cen = grid_elem.cen_;
	const quokka::direction dir = grid_elem.dir_;

	const int ncomp_cc = Physics_Indices<SlowWaveConvergence>::nvarTotal_cc;
	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
		for (int n = 0; n < ncomp_cc; ++n) {
			state_cc(i, j, k, n) = 0;
		}
		computeWaveSolution(i, j, k, state_cc, dx, prob_lo, cen, dir, 0);
	});
}

template <> void QuokkaSimulation<SlowWaveConvergence>::setInitialConditionsOnGridFaceVars(quokka::grid const &grid_elem)
{
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = grid_elem.dx_;
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = grid_elem.prob_lo_;
	const amrex::Array4<double> &state_fc = grid_elem.array_;
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const quokka::centering cen = grid_elem.cen_;
	const quokka::direction dir = grid_elem.dir_;

	const int ncomp_fc = Physics_Indices<SlowWaveConvergence>::nvarPerDim_fc;
	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
		for (int n = 0; n < ncomp_fc; ++n) {
			state_fc(i, j, k, n) = 0;
		}
		computeWaveSolution(i, j, k, state_fc, dx, prob_lo, cen, dir, 0);
	});
}

template <>
void QuokkaSimulation<SlowWaveConvergence>::computeReferenceSolution(amrex::MultiFab &ref, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx,
								     amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_lo)
{
	for (amrex::MFIter iter(ref); iter.isValid(); ++iter) {
		const amrex::Box &indexRange = iter.validbox();
		auto const &stateExact = ref.array(iter);
		auto const ncomp = ref.nComp();

		amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
			for (int n = 0; n < ncomp; ++n) {
				stateExact(i, j, k, n) = 0.0;
			}
			computeWaveSolution(i, j, k, stateExact, dx, prob_lo, quokka::centering::cc, quokka::direction::na, 0);
		});
	}
}

template <>
void QuokkaSimulation<SlowWaveConvergence>::computeReferenceSolution_fc(amrex::MultiFab &ref, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx,
									amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_lo,
									quokka::direction const dir)
{
	for (amrex::MFIter iter(ref); iter.isValid(); ++iter) {
		const amrex::Box &indexRange = iter.validbox();
		auto const &stateExact = ref.array(iter);
		auto const ncomp = ref.nComp();

		amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
			for (int n = 0; n < ncomp; ++n) {
				stateExact(i, j, k, n) = 0.0;
			}
			computeWaveSolution(i, j, k, stateExact, dx, prob_lo, quokka::centering::fc, dir, 0);
		});
	}
}

auto runWaveTest(int nx, int ny, int nz) -> double
{
	// Read problem parameters
	amrex::ParmParse const hpp("setup");
	double angle_between_k_b0_deg = 0.0;
	hpp.query("angle_between_k_b0", angle_between_k_b0_deg);
	constexpr double deg2rad = M_PI / 180.0;
	angle_between_k_b0_rad = deg2rad * angle_between_k_b0_deg;

	const double a = sound_speed;
	const double vA = alfven_speed;
	const double cosθ = std::cos(angle_between_k_b0_rad);
	const double cs = std::sqrt(0.5 * (a * a + vA * vA - std::sqrt((a * a + vA * vA) * (a * a + vA * vA) - 4.0 * a * a * vA * vA * cosθ * cosθ)));
	const int max_timesteps = std::max(20000, nx * 100);

	int num_modes_x = 0;
	int num_modes_y = 0;
	int num_modes_z = 0;
	hpp.query("num_modes_x", num_modes_x);
	hpp.query("num_modes_y", num_modes_y);
	hpp.query("num_modes_z", num_modes_z);

	if ((num_modes_x == 0) && (num_modes_y == 0) && (num_modes_z == 0)) {
		amrex::Abort("Invalid k modes: the triplet (0,0,0) is not allowed.");
	}

	if (num_modes_y != 0 && ny == 8) {
		amrex::Abort("num_modes_y != 0 requires refine_n_dims >= 2 to converge.");
	}
	if (num_modes_z != 0 && nz == 8) {
		amrex::Abort("num_modes_z != 0 requires refine_n_dims >= 3 to converge.");
	}

	// we assume box length = 1.0
	const std::array<amrex::Real, 3> k_vec_prf = {2.0 * M_PI * static_cast<amrex::Real>(num_modes_x), 2.0 * M_PI * static_cast<amrex::Real>(num_modes_y),
						      2.0 * M_PI * static_cast<amrex::Real>(num_modes_z)};
	k_magn = computeMagnitude(k_vec_prf);
	const double wavelength = 2.0 * M_PI / k_magn;
	const double max_time = wavelength / cs;
	k_dir_prf = {k_vec_prf[0] / k_magn, k_vec_prf[1] / k_magn, k_vec_prf[2] / k_magn};

	k_rotation_in_xy_rad = std::atan2(k_dir_prf[1], k_dir_prf[0]);
	k_elevation_from_xy_rad = std::atan2(k_dir_prf[2], std::hypot(k_dir_prf[0], k_dir_prf[1]));

	// to build our orthonormal basis in the problem reference frame (PRF)
	// first choose a vector that is not aligned/parallel with the wave propagation direction
	std::array<amrex::Real, 3> ref_prf{0.0, 0.0, 1.0}; // guess a direction
	if (std::abs(computeDotProduct(ref_prf, k_dir_prf)) > 0.9999) {
		ref_prf = {0.0, 1.0, 0.0};
	}

	inplane_dir_prf = computeCrossProduct(ref_prf, k_dir_prf);
	normalizeVector(inplane_dir_prf);

	outofplane_dir_prf = computeCrossProduct(k_dir_prf, inplane_dir_prf);
	normalizeVector(outofplane_dir_prf);

	// Set grid dimensions using AMReX parameter system
	amrex::ParmParse pp("amr");
	amrex::Vector<int> const ncells = {nx, ny, nz};

	if (!pp.contains("blocking_factor")) {
		pp.add("blocking_factor", 8);
	}

	if (!pp.contains("max_grid_size")) {
		pp.add("max_grid_size", 128);
	}

	pp.add("max_level", 0);
	pp.addarr("n_cell", ncells);

	// Set domain bounds using AMReX parameter system
	amrex::ParmParse pp_geom("geometry");
	amrex::Vector<double> const prob_lo = {0.0, 0.0, 0.0};
	amrex::Vector<double> const prob_hi = {1.0, 1.0, 1.0};
	amrex::Vector<int> const is_periodic = {1, 1, 1};
	pp_geom.addarr("prob_lo", prob_lo);
	pp_geom.addarr("prob_hi", prob_hi);
	pp_geom.addarr("is_periodic", is_periodic);

	// Setup boundary conditions
	auto BCs_cc = quokka::BC<SlowWaveConvergence>(quokka::BCType::int_dir);

	const int nvars_fc = Physics_Indices<SlowWaveConvergence>::nvarTotal_fc;
	amrex::Vector<amrex::BCRec> BCs_fc(nvars_fc);
	for (int icomp = 0; icomp < nvars_fc; ++icomp) {
		for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
			BCs_fc[icomp].setLo(idim, amrex::BCType::int_dir);
			BCs_fc[icomp].setHi(idim, amrex::BCType::int_dir);
		}
	}

	// Run simulation
	QuokkaSimulation<SlowWaveConvergence> sim(BCs_cc, BCs_fc);

	sim.stopTime_ = max_time;
	sim.maxTimesteps_ = max_timesteps;
	sim.setInitialConditions();

	// Main time loop
	sim.evolve();

	return sim.computeErrorNorm();
}

auto problem_main() -> int
{
	bool run_convergence = true;
	bool run_sim = false;
	double error_tol = 0.002;
	{
		amrex::ParmParse const pp("setup");
		pp.query("run_convergence", run_convergence);
		pp.query("run_sim", run_sim);
		pp.query("error_tol", error_tol);
	}

	// SlowWaveConvergence does not model resistivity; abort early rather than silently
	// producing a wrong reference solution if mhd.resistivity is set (applies to both modes).
	{
		double eta = 0.0;
		amrex::ParmParse const mhd_pp("mhd");
		mhd_pp.query("resistivity", eta);
		if (eta != 0.0) {
			amrex::Abort("SlowWaveConvergence does not support mhd.resistivity != 0; use AlfvenWaveLinearConvergence "
				     "for resistivity validation.");
		}
	}

	int status = 0;

	if (run_sim) {
		{
			amrex::ParmParse const pp("setup");
			double angle_between_k_b0_deg = 0.0;
			pp.query("angle_between_k_b0", angle_between_k_b0_deg);
			constexpr double deg2rad = M_PI / 180.0;
			angle_between_k_b0_rad = deg2rad * angle_between_k_b0_deg;

			int num_modes_x = 0;
			int num_modes_y = 0;
			int num_modes_z = 0;
			pp.query("num_modes_x", num_modes_x);
			pp.query("num_modes_y", num_modes_y);
			pp.query("num_modes_z", num_modes_z);
			if ((num_modes_x == 0) && (num_modes_y == 0) && (num_modes_z == 0)) {
				amrex::Abort("Invalid k modes: the triplet (0,0,0) is not allowed.");
			}

			const std::array<amrex::Real, 3> k_vec_prf = {2.0 * M_PI * static_cast<amrex::Real>(num_modes_x),
								      2.0 * M_PI * static_cast<amrex::Real>(num_modes_y),
								      2.0 * M_PI * static_cast<amrex::Real>(num_modes_z)};
			k_magn = computeMagnitude(k_vec_prf);
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

		auto BCs_cc = quokka::BC<SlowWaveConvergence>(quokka::BCType::int_dir);
		const int nvars_fc = Physics_Indices<SlowWaveConvergence>::nvarTotal_fc;
		amrex::Vector<amrex::BCRec> BCs_fc(nvars_fc);
		for (int icomp = 0; icomp < nvars_fc; ++icomp) {
			for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
				BCs_fc[icomp].setLo(idim, amrex::BCType::int_dir);
				BCs_fc[icomp].setHi(idim, amrex::BCType::int_dir);
			}
		}

		QuokkaSimulation<SlowWaveConvergence> sim(BCs_cc, BCs_fc);
		sim.setInitialConditions();
		sim.evolve();

		const double error_norm = sim.computeErrorNorm();
		amrex::Print() << std::format("\nrun_sim error norm = {:.6e}  (tol = {:.6e})\n", error_norm, error_tol);
		if (error_norm > error_tol) {
			status = 1;
		}
	}

	if (run_convergence) {
		quokka::richardson::applyQuietDefaults();

		quokka::richardson::Parameters params{};
		params.machine_precision_target = 2.0e-9;
		params.nx_initial = 16;
		params.nx_max = 128;
		{
			amrex::ParmParse const pp("setup");
			pp.query("nx_start", params.nx_initial);
			pp.query("nx_max", params.nx_max);
			pp.query("machine_precision_target", params.machine_precision_target);
			pp.query("refine_n_dims", params.refine_n_dims);
		}
		params.expected_rate = 2.0;
		params.tolerance = 0.3;
		params.test_name = "Slow Wave";
		params.csv_filename = "slow_wave_convergence.csv";

		if (quokka::richardson::run(params, [](int nx, int ny, int nz) { return runWaveTest(nx, ny, nz); }) != 0) {
			status = 1;
		}
	}

	return status;
}
