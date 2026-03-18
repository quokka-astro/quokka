//==============================================================================
// Copyright 2022 Neco Kriel.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file testAlfvenWaveLinear.cpp
/// \brief Setup a linear Alfven wave test with optional resistivity.
///

#include <array>
#include <cassert>
#include <cmath>
#include <gcem.hpp>

#include "AMReX_Array.H"
#include "AMReX_Array4.H"
#include "AMReX_Gpu.H"
#include "AMReX_ParmParse.H"
#include "AMReX_REAL.H"

#include "QuokkaSimulation.hpp"
#include "grid.hpp"
#include "hydro/EOS.hpp"
#include "physics_info.hpp"
#include "util/BC.hpp"
#include "util/fextract.hpp"
#ifdef HAVE_PYTHON
#include "util/matplotlibcpp.h"
#endif

struct AlfvenWaveLinear {
};

template <> struct quokka::EOS_Traits<AlfvenWaveLinear> {
	static constexpr double gamma = 5. / 3.;
	static constexpr double mean_molecular_weight = C::m_u;
	static constexpr double boltzmann_constant = C::k_B;
};

template <> struct Physics_Traits<AlfvenWaveLinear> {
	static constexpr bool is_hydro_enabled = true;
	static constexpr int numMassScalars = 0;
	static constexpr int numPassiveScalars = numMassScalars + 0;
	static constexpr bool is_self_gravity_enabled = false;
	static constexpr bool is_radiation_enabled = false;
	static constexpr bool is_dust_enabled = false;
	static constexpr int nDustGroups = 1;
	static constexpr bool is_mhd_enabled = true;
	static constexpr int nGroups = 1;
	static constexpr UnitSystem unit_system = UnitSystem::CGS;
};

constexpr double sound_speed = 1.0;
constexpr double gamma_gas = quokka::EOS_Traits<AlfvenWaveLinear>::gamma;
constexpr double bg_density = 1.0;
constexpr double bg_pressure = sound_speed * sound_speed * bg_density / gamma_gas;
constexpr double bg_b_magn = 1.0;
constexpr double delta_b_magn = 1e-6;
constexpr double alfven_speed = bg_b_magn / gcem::sqrt(bg_density);

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
AMREX_GPU_MANAGED double angle_between_k_bg_b_rad = 0.0; // NOLINT

// rotation from the problem reference frame (PRF) to the MRF
AMREX_GPU_MANAGED double k_rotation_in_plane_rad = 0.0;	// NOLINT
AMREX_GPU_MANAGED double k_elevation_from_plane_rad = 0.0; // NOLINT

//------------------------------------------------------------------------------
// Reference frames and rotation matrix
//
// We work with two right-handed, orthonormal frames:
//
// PRF (Problem Reference Frame): the simulation's grid-aligned axes.
// MRF (Math Reference Frame):    a wave-aligned basis:
//                                e0 = k_dir_prf          (propagation direction)
//                                e1 = inplane_dir_prf    (lies in the k-bg_b plane)
//                                e2 = outofplane_dir_prf (perpendicular to that plane)
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
// row 0: e0 = k_dir_prf (propagation)
// row 1: e1 = inplane_dir_prf (k-bg_b plane)
// row 2: e2 = outofplane_dir_prf (perpendicular to that plane)
AMREX_GPU_MANAGED std::array<amrex::Real, 3> k_dir_prf{1.0, 0.0, 0.0};		// NOLINT
AMREX_GPU_MANAGED std::array<amrex::Real, 3> inplane_dir_prf{0.0, 1.0, 0.0};	// NOLINT
AMREX_GPU_MANAGED std::array<amrex::Real, 3> outofplane_dir_prf{0.0, 0.0, 1.0}; // NOLINT

// wavefront
AMREX_GPU_MANAGED double k_magn = 2.0 * M_PI; // NOLINT

//------------------------------------------------------------------------------
// Resistive dispersion relation parameters (all computed in problem_main).
//
// For a resistive Alfven wave with constant resistivity eta, the dispersion
// relation is:
//
//   omega^2 - i eta k^2 omega - v_A^2 k^2 cos^2(theta) = 0
//
// which gives omega = omega_r + i gamma, where:
//
//   gamma   = eta k^2 / 2                            (decay rate)
//   omega_r = sqrt(omega_ideal^2 - gamma^2)           (reduced oscillation frequency)
//   omega_ideal = v_A k cos(theta)                    (ideal Alfven frequency)
//
// The analytical solution is then:
//
//   b_x2(x,t) = bg_b_magn delta_b exp(-gamma t) cos(omega_r t - k x0_mrf)
//   u_x2(x,t) = -(|omega|/omega_ideal) v_A delta_b exp(-gamma t)
//                   cos(omega_r t - k x0_mrf - phi)
//
// where |omega| = sqrt(omega_r^2 + gamma^2) = omega_ideal (exactly, by construction),
// and phi = atan2(gamma, omega_r) is the phase lag of u_x2 behind b_x2.
//
// In the ideal limit eta -> 0: gamma -> 0, omega_r -> omega_ideal, phi -> 0,
// and the solution reduces to the standard ideal Alfven wave.
//
// In the overdamped regime omega_r = 0 (Rm_k < 1): the wave does not propagate.
// The two roots are purely imaginary and the solution is a superposition of
// two purely decaying modes. This code handles it gracefully (omega_r = 0,
// phi = pi/2) but the sinusoidal spatial structure is no longer an eigenmode,
// so the overdamped case is not a clean benchmark and should be avoided in
// practice (choose Rm_k >> 1).
//------------------------------------------------------------------------------
AMREX_GPU_MANAGED double resistivity = 0.0; // NOLINT: resistivity read from input
AMREX_GPU_MANAGED double omega_ideal = 0.0;	 // NOLINT: v_A k cos(theta)
AMREX_GPU_MANAGED double gamma_decay = 0.0;	 // NOLINT: eta k^2 / 2
AMREX_GPU_MANAGED double omega_r = 0.0;		 // NOLINT: sqrt(omega_ideal^2 - gamma^2)
AMREX_GPU_MANAGED double phase_shift = 0.0;	 // NOLINT: atan2(gamma, omega_r)

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

AMREX_GPU_DEVICE AMREX_FORCE_INLINE auto computeVectorPotentialComponent_prf(const double x0_prf, const double x1_prf, const double x2_prf, const double time,
									     const int icomp) -> double
{
	// Computes A in PRF by:
	// 1. rotating x_vec from PRF->MRF,
	// 2. building A in MRF,
	// 3. rotating A back MRF->PRF and selecting the relevant component.
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(icomp == 0 || icomp == 1 || icomp == 2,
					 "computeVectorPotentialComponent_prf(): icomp must be an integer in {0, 1, 2}");
	const std::array<amrex::Real, 3> x_vec_mrf = rotatePRF2MRF({x0_prf, x1_prf, x2_prf});
	const double bg_bx0_mrf = bg_b_magn * std::cos(angle_between_k_bg_b_rad);
	const double bg_bx1_mrf = bg_b_magn * std::sin(angle_between_k_bg_b_rad);
	// bg_A = (0, 0, bg_bx0 * x1 - bg_bx1 * x0) -> curl(bg_A) = (bg_bx0, bg_bx1, 0)
	const double bg_A0_mrf = 0.0;
	const double bg_A1_mrf = 0.0;
	const double bg_A2_mrf = bg_bx0_mrf * x_vec_mrf[1] - bg_bx1_mrf * x_vec_mrf[0];
	// delta_b_x2_mrf = bg_b_magn delta_b exp(-gamma t) cos(omega_r t - k x0_mrf)
	//   => delta_A1_mrf = -(bg_b_magn delta_b / k) exp(-gamma t) sin(omega_r t - k x0_mrf)
	const double delta_A0_mrf = 0.0;
	const double delta_A1_mrf = -(bg_b_magn * delta_b_magn / k_magn) * std::exp(-gamma_decay * time) * std::sin(omega_r * time - k_magn * x_vec_mrf[0]);
	const double delta_A2_mrf = 0.0;
	const double A0_mrf = bg_A0_mrf + delta_A0_mrf;
	const double A1_mrf = bg_A1_mrf + delta_A1_mrf;
	const double A2_mrf = bg_A2_mrf + delta_A2_mrf;
	const std::array<amrex::Real, 3> A_vec_prf = rotateMRF2PRF({A0_mrf, A1_mrf, A2_mrf});
	return A_vec_prf[icomp];
}

AMREX_GPU_DEVICE inline auto Ax0_prf(const double x0_prf, const double x1_prf, const double x2_prf, const double time) -> double
{
	return computeVectorPotentialComponent_prf(x0_prf, x1_prf, x2_prf, time, 0);
}

AMREX_GPU_DEVICE inline auto Ax1_prf(const double x0_prf, const double x1_prf, const double x2_prf, const double time) -> double
{
	return computeVectorPotentialComponent_prf(x0_prf, x1_prf, x2_prf, time, 1);
}

AMREX_GPU_DEVICE inline auto Ax2_prf(const double x0_prf, const double x1_prf, const double x2_prf, const double time) -> double
{
	return computeVectorPotentialComponent_prf(x0_prf, x1_prf, x2_prf, time, 2);
}

AMREX_GPU_DEVICE
void computeWaveSolution(int i, int j, int k, amrex::Array4<amrex::Real> const &state, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx,
			 amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_lo, quokka::centering cen, quokka::direction dir, amrex::Real time)
{
	const amrex::Real x0_prf_L = prob_lo[0] + i * dx[0];
	const amrex::Real x1_prf_L = prob_lo[1] + j * dx[1];
	const amrex::Real x2_prf_L = prob_lo[2] + k * dx[2];

	if (cen == quokka::centering::cc) {
		const amrex::Real x0_prf_C = x0_prf_L + static_cast<amrex::Real>(0.5) * dx[0];
		const amrex::Real x1_prf_C = x1_prf_L + static_cast<amrex::Real>(0.5) * dx[1];
		const amrex::Real x2_prf_C = x2_prf_L + static_cast<amrex::Real>(0.5) * dx[2];
		const std::array<amrex::Real, 3> x_vec_mrf_C = rotatePRF2MRF({x0_prf_C, x1_prf_C, x2_prf_C});

		// phase argument: omega_r t - k x0_mrf (k = k e0 in MRF by construction)
		const double phase_b = omega_r * time - k_magn * x_vec_mrf_C[0];
		const double phase_u = phase_b - phase_shift; // u_x2 lags b_x2 by phi

		const double envelope = std::exp(-gamma_decay * time);

		// delta_b_x2_mrf = bg_b_magn delta_b exp(-gamma t) cos(phase_b)
		const double delta_b_cos = delta_b_magn * envelope * std::cos(phase_b);

		// u_x2_mrf = -(|omega|/omega_ideal) v_A delta_b exp(-gamma t) cos(phase_u)
		// |omega|/omega_ideal = 1 exactly; ratio kept explicit for safety when omega_ideal -> 0
		constexpr double elsasser_sgn = -1.0;
		const double omega_magn = std::sqrt(omega_r * omega_r + gamma_decay * gamma_decay); // = omega_ideal
		const double v_amp = (omega_ideal > 0.0) ? elsasser_sgn * (omega_magn / omega_ideal) * alfven_speed * delta_b_magn : 0.0;
		const double delta_v_magn = v_amp * envelope * std::cos(phase_u);

		// Project velocity perturbation from MRF e2 back into PRF
		const double v_x0_prf = delta_v_magn * outofplane_dir_prf[0];
		const double v_x1_prf = delta_v_magn * outofplane_dir_prf[1];
		const double v_x2_prf = delta_v_magn * outofplane_dir_prf[2];

		// Background b in PRF: bg_b_magn (cos(theta) e0_prf + sin(theta) e1_prf)
		const double bg_bx0_prf = bg_b_magn * (std::cos(angle_between_k_bg_b_rad) * k_dir_prf[0] + std::sin(angle_between_k_bg_b_rad) * inplane_dir_prf[0]);
		const double bg_bx1_prf = bg_b_magn * (std::cos(angle_between_k_bg_b_rad) * k_dir_prf[1] + std::sin(angle_between_k_bg_b_rad) * inplane_dir_prf[1]);
		const double bg_bx2_prf = bg_b_magn * (std::cos(angle_between_k_bg_b_rad) * k_dir_prf[2] + std::sin(angle_between_k_bg_b_rad) * inplane_dir_prf[2]);

		// Perturbed b in PRF: bg_b_magn delta_b_cos (e2_prf direction)
		const double delta_b_x0_prf = bg_b_magn * delta_b_cos * outofplane_dir_prf[0];
		const double delta_b_x1_prf = bg_b_magn * delta_b_cos * outofplane_dir_prf[1];
		const double delta_b_x2_prf = bg_b_magn * delta_b_cos * outofplane_dir_prf[2];

		// Total b in PRF
		const double b_x0_prf = bg_bx0_prf + delta_b_x0_prf;
		const double b_x1_prf = bg_bx1_prf + delta_b_x1_prf;
		const double b_x2_prf = bg_bx2_prf + delta_b_x2_prf;

		const double density = bg_density;
		const double pressure = bg_pressure;

		const double v_magn_sq = v_x0_prf * v_x0_prf + v_x1_prf * v_x1_prf + v_x2_prf * v_x2_prf;
		const double b_magn_sq = b_x0_prf * b_x0_prf + b_x1_prf * b_x1_prf + b_x2_prf * b_x2_prf;
		const double Ekin = 0.5 * density * v_magn_sq;
		const double Emag = 0.5 * b_magn_sq;
		const double Eint = pressure / (gamma_gas - 1);
		const double Etot = Ekin + Emag + Eint;

		state(i, j, k, HydroSystem<AlfvenWaveLinear>::density_index) = density;
		state(i, j, k, HydroSystem<AlfvenWaveLinear>::x1Momentum_index) = v_x0_prf * density;
		state(i, j, k, HydroSystem<AlfvenWaveLinear>::x2Momentum_index) = v_x1_prf * density;
		state(i, j, k, HydroSystem<AlfvenWaveLinear>::x3Momentum_index) = v_x2_prf * density;
		state(i, j, k, HydroSystem<AlfvenWaveLinear>::energy_index) = Etot;
		state(i, j, k, HydroSystem<AlfvenWaveLinear>::internalEnergy_index) = Eint;
	} else if (cen == quokka::centering::fc) {
		// compute b-field using the magnetic vector potential to preserve div(b) = 0 topology
		const double b_x0 =
		    (Ax2_prf(x0_prf_L, x1_prf_L + dx[1], x2_prf_L + dx[2] / 2.0, time) - Ax2_prf(x0_prf_L, x1_prf_L, x2_prf_L + dx[2] / 2.0, time)) / dx[1] -
		    (Ax1_prf(x0_prf_L, x1_prf_L + dx[1] / 2.0, x2_prf_L + dx[2], time) - Ax1_prf(x0_prf_L, x1_prf_L + dx[1] / 2.0, x2_prf_L, time)) / dx[2];

		const double b_x1 =
		    (Ax0_prf(x0_prf_L + dx[0] / 2.0, x1_prf_L, x2_prf_L + dx[2], time) - Ax0_prf(x0_prf_L + dx[0] / 2.0, x1_prf_L, x2_prf_L, time)) / dx[2] -
		    (Ax2_prf(x0_prf_L + dx[0], x1_prf_L, x2_prf_L + dx[2] / 2.0, time) - Ax2_prf(x0_prf_L, x1_prf_L, x2_prf_L + dx[2] / 2.0, time)) / dx[0];

		const double b_x2 =
		    (Ax1_prf(x0_prf_L + dx[0], x1_prf_L + dx[1] / 2.0, x2_prf_L, time) - Ax1_prf(x0_prf_L, x1_prf_L + dx[1] / 2.0, x2_prf_L, time)) / dx[0] -
		    (Ax0_prf(x0_prf_L + dx[0] / 2.0, x1_prf_L + dx[1], x2_prf_L, time) - Ax0_prf(x0_prf_L + dx[0] / 2.0, x1_prf_L, x2_prf_L, time)) / dx[1];

		if (dir == quokka::direction::x) {
			state(i, j, k, MHDSystem<AlfvenWaveLinear>::bfield_index) = b_x0;
		} else if (dir == quokka::direction::y) {
			state(i, j, k, MHDSystem<AlfvenWaveLinear>::bfield_index) = b_x1;
		} else if (dir == quokka::direction::z) {
			state(i, j, k, MHDSystem<AlfvenWaveLinear>::bfield_index) = b_x2;
		}
	}
}

template <> void QuokkaSimulation<AlfvenWaveLinear>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = grid_elem.dx_;
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = grid_elem.prob_lo_;
	const amrex::Array4<amrex::Real> &state_cc = grid_elem.array_;
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const quokka::centering cen = grid_elem.cen_;
	const quokka::direction dir = grid_elem.dir_;

	const int ncomp_cc = Physics_Indices<AlfvenWaveLinear>::nvarTotal_cc;
	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		for (int n = 0; n < ncomp_cc; ++n) {
			state_cc(i, j, k, n) = 0;
		}
		computeWaveSolution(i, j, k, state_cc, dx, prob_lo, cen, dir, 0);
	});
}

template <> void QuokkaSimulation<AlfvenWaveLinear>::setInitialConditionsOnGridFaceVars(quokka::grid const &grid_elem)
{
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = grid_elem.dx_;
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = grid_elem.prob_lo_;
	const amrex::Array4<amrex::Real> &state_fc = grid_elem.array_;
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const quokka::centering cen = grid_elem.cen_;
	const quokka::direction dir = grid_elem.dir_;

	const int ncomp_fc = Physics_Indices<AlfvenWaveLinear>::nvarPerDim_fc;
	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		for (int n = 0; n < ncomp_fc; ++n) {
			state_fc(i, j, k, n) = 0;
		}
		computeWaveSolution(i, j, k, state_fc, dx, prob_lo, cen, dir, 0);
	});
}

template <>
void QuokkaSimulation<AlfvenWaveLinear>::computeReferenceSolution(amrex::MultiFab &ref, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx,
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
void QuokkaSimulation<AlfvenWaveLinear>::computeReferenceSolution_fc(amrex::MultiFab &ref, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx,
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

auto problem_main() -> int
{
	amrex::ParmParse const hpp("setup");

	double angle_between_k_bg_b_deg = 0.0;
	hpp.query("angle_between_k_bg_b", angle_between_k_bg_b_deg);

	constexpr double deg2rad = M_PI / 180.0;
	angle_between_k_bg_b_rad = deg2rad * angle_between_k_bg_b_deg;

	int num_modes_0 = 0;
	int num_modes_1 = 0;
	int num_modes_2 = 0;
	hpp.query("num_modes_0", num_modes_0);
	hpp.query("num_modes_1", num_modes_1);
	hpp.query("num_modes_2", num_modes_2);

	if ((num_modes_0 == 0) && (num_modes_1 == 0) && (num_modes_2 == 0)) {
		amrex::Abort("Invalid k modes: the triplet (0,0,0) is not allowed.");
	}

	// we assume box length = 1.0
	const std::array<amrex::Real, 3> k_vec_prf = {2.0 * M_PI * static_cast<amrex::Real>(num_modes_0), 2.0 * M_PI * static_cast<amrex::Real>(num_modes_1),
						      2.0 * M_PI * static_cast<amrex::Real>(num_modes_2)};
	k_magn = computeMagnitude(k_vec_prf);
	k_dir_prf = {k_vec_prf[0] / k_magn, k_vec_prf[1] / k_magn, k_vec_prf[2] / k_magn};

	k_rotation_in_plane_rad = std::atan2(k_dir_prf[1], k_dir_prf[0]);
	k_elevation_from_plane_rad = std::atan2(k_dir_prf[2], std::hypot(k_dir_prf[0], k_dir_prf[1]));

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

	// Read resistivity from mhd namespace in the input file.
	// This mirrors how Quokka reads it internally, so the analytical solution
	// automatically matches whatever value the solver uses.
	amrex::ParmParse const mhd_pp("mhd");
	mhd_pp.query("resistivity", resistivity);

	// Compute resistive dispersion relation parameters.
	// All quantities are scalars in the MRF so no rotation is needed here.
	omega_ideal = alfven_speed * k_magn * std::cos(angle_between_k_bg_b_rad);
	gamma_decay = 0.5 * resistivity * k_magn * k_magn;
	const double discriminant = omega_ideal * omega_ideal - gamma_decay * gamma_decay;

	if (discriminant < 0.0) {
		// Overdamped regime: wave does not propagate. The sinusoidal initial
		// condition is not an eigenmode here; warn the user.
		amrex::Print() << "WARNING (testAlfvenWaveLinear): overdamped regime detected "
			       << "(Rm_k = " << omega_ideal / gamma_decay << " < 1). "
			       << "The resistive Alfven wave does not propagate; "
			       << "this initial condition is not a clean eigenmode test.\n";
		omega_r = 0.0;
		phase_shift = M_PI / 2.0;
	} else {
		omega_r = std::sqrt(discriminant);
		phase_shift = std::atan2(gamma_decay, omega_r);
	}

	QuokkaSimulation<AlfvenWaveLinear> sim;

	sim.setInitialConditions();
	sim.evolve();

	int status = 1;
	const double error_tol = 0.005;
	amrex::Real const error_norm = sim.computeErrorNorm();
	if (error_norm < error_tol) {
		status = 0;
		amrex::Print() << "Error norm = " << error_norm << "\n";
		amrex::Print() << "test passed\n";
	} else {
		amrex::Print() << "Error norm = " << error_norm << "\n";
		amrex::Print() << "test failed\n";
	}

	return status;
}