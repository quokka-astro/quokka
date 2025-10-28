//==============================================================================
// Copyright 2022 Neco Kriel.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file test_fc_quantities.cpp
/// \brief Defines a test problem to make sure face-centered quantities are created correctly.
///

#include <array>
#include <cassert>
#include <cmath>

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
	static constexpr amrex::Real gamma = 5. / 3.;
	static constexpr amrex::Real mean_molecular_weight = C::m_u;
	static constexpr amrex::Real boltzmann_constant = C::k_B;
};

template <> struct Physics_Traits<AlfvenWaveLinear> {
	static constexpr bool is_hydro_enabled = true;
	static constexpr int numMassScalars = 0;
	static constexpr int numPassiveScalars = numMassScalars + 0;
	static constexpr bool is_self_gravity_enabled = false;
	static constexpr bool is_radiation_enabled = false;
	static constexpr bool is_mhd_enabled = true;
	static constexpr int nGroups = 1;
	static constexpr UnitSystem unit_system = UnitSystem::CGS;
};

constexpr amrex::Real sound_speed = 1.0;
constexpr amrex::Real gamma_gas = quokka::EOS_Traits<AlfvenWaveLinear>::gamma;
constexpr amrex::Real bg_density = 1.0;
constexpr amrex::Real bg_pressure = sound_speed * sound_speed * bg_density / gamma_gas;
constexpr amrex::Real b0_magn = 1.0;
constexpr amrex::Real delta_b_magn = 1e-6;

// AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE
// amrex::Real ensure_safe_zero(amrex::Real value) {
// 	if (amrex::Math::abs(value) < 1e-9) {
// 		return +0.0;
// 	} else {
// 		return value;
// 	}
// }

AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE auto computeMagnitude(const std::array<amrex::Real, 3> &vfield) -> amrex::Real
{
	return std::sqrt(vfield[0] * vfield[0] + vfield[1] * vfield[1] + vfield[2] * vfield[2]);
}

AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE auto computeDotProduct(const std::array<amrex::Real, 3> &vfield1, const std::array<amrex::Real, 3> &vfield2)
    -> amrex::Real
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
	const amrex::Real vfield_magn = computeMagnitude(vfield);
	if (vfield_magn > static_cast<amrex::Real>(1e-14)) {
		vfield[0] /= vfield_magn;
		vfield[1] /= vfield_magn;
		vfield[2] /= vfield_magn;
	}
}

// angles (radians) in the math reference frame (MRF)
AMREX_GPU_MANAGED amrex::Real angle_between_k_b0_rad = 0.0; // NOLINT

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
AMREX_GPU_MANAGED std::array<amrex::Real, 3> k_dir_prf{amrex::Real(1.0), amrex::Real(0.0), amrex::Real(0.0)};	       // NOLINT
AMREX_GPU_MANAGED std::array<amrex::Real, 3> inplane_dir_prf{amrex::Real(0.0), amrex::Real(1.0), amrex::Real(0.0)};    // NOLINT
AMREX_GPU_MANAGED std::array<amrex::Real, 3> outofplane_dir_prf{amrex::Real(0.0), amrex::Real(0.0), amrex::Real(1.0)}; // NOLINT

// wavefront
AMREX_GPU_MANAGED amrex::Real k_magn = static_cast<amrex::Real>(2.0 * M_PI); // NOLINT

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

AMREX_GPU_DEVICE AMREX_FORCE_INLINE auto computeVectorPotentialComponent_prf(const amrex::Real x1_prf, const amrex::Real x2_prf, const amrex::Real x3_prf,
									     const amrex::Real time, const int icomp) -> amrex::Real
{
	// Computes A in PRF by:
	// 1. rotating x_vec from PRF->MRF,
	// 2. building A in MRF,
	// 3. rotating A back MRF->PRF and selecting the relevant component.
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(icomp == 0 || icomp == 1 || icomp == 2,
					 "computeVectorPotentialComponent_prf(): icomp must be an integer in {0, 1, 2}");
	const std::array<amrex::Real, 3> x_vec_mrf = rotatePRF2MRF({x1_prf, x2_prf, x3_prf});
	const amrex::Real b0_x1_mrf = b0_magn * std::cos(angle_between_k_b0_rad);
	const amrex::Real b0_x2_mrf = b0_magn * std::sin(angle_between_k_b0_rad);
	// bg_A = (0, 0, b0_x1 * x2 - b0_x2 * x1) -> curl(bg_A) = (b0_x1, b0_x2, 0)
	const auto bg_A1_mrf = static_cast<amrex::Real>(0.0);
	const auto bg_A2_mrf = static_cast<amrex::Real>(0.0);
	const amrex::Real bg_A3_mrf = b0_x1_mrf * x_vec_mrf[1] - b0_x2_mrf * x_vec_mrf[0];
	// d/dx A_x2 = bg_b * delta_b * cos(omega t - k x1); A_x1 = A_x3 = 0 -> delta_b_x1 = delta_b_x3 = 0
	const amrex::Real alfven_speed = b0_magn / std::sqrt(bg_density);
	const amrex::Real omega = alfven_speed * k_magn * std::cos(angle_between_k_b0_rad);
	const auto delta_A1_mrf = static_cast<amrex::Real>(0.0);
	const amrex::Real delta_A2_mrf = -((b0_magn * delta_b_magn) / k_magn) * std::sin(omega * time - k_magn * x_vec_mrf[0]);
	const auto delta_A3_mrf = static_cast<amrex::Real>(0.0);
	const amrex::Real A1_mrf = bg_A1_mrf + delta_A1_mrf;
	const amrex::Real A2_mrf = bg_A2_mrf + delta_A2_mrf;
	const amrex::Real A3_mrf = bg_A3_mrf + delta_A3_mrf;
	const std::array<amrex::Real, 3> A_vec_prf = rotateMRF2PRF({A1_mrf, A2_mrf, A3_mrf});
	return A_vec_prf[icomp];
}

AMREX_GPU_DEVICE inline auto Ax_prf(const amrex::Real x1_prf, const amrex::Real x2_prf, const amrex::Real x3_prf, const amrex::Real time) -> amrex::Real
{
	return computeVectorPotentialComponent_prf(x1_prf, x2_prf, x3_prf, time, 0);
}

AMREX_GPU_DEVICE inline auto Ay_prf(const amrex::Real x1_prf, const amrex::Real x2_prf, const amrex::Real x3_prf, const amrex::Real time) -> amrex::Real
{
	return computeVectorPotentialComponent_prf(x1_prf, x2_prf, x3_prf, time, 1);
}

AMREX_GPU_DEVICE inline auto Az_prf(const amrex::Real x1_prf, const amrex::Real x2_prf, const amrex::Real x3_prf, const amrex::Real time) -> amrex::Real
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
		const amrex::Real x1_prf_C = x1_prf_L + static_cast<amrex::Real>(0.5) * dx[0];
		const amrex::Real x2_prf_C = x2_prf_L + static_cast<amrex::Real>(0.5) * dx[1];
		const amrex::Real x3_prf_C = x3_prf_L + static_cast<amrex::Real>(0.5) * dx[2];
		const std::array<amrex::Real, 3> x_vec_mrf_C = rotatePRF2MRF({x1_prf_C, x2_prf_C, x3_prf_C});

		const amrex::Real alfven_speed = b0_magn / std::sqrt(bg_density);
		// this is agnostic to the choice of reference frame: vec(k) dot vec(x) is invariant under rotation
		const amrex::Real omega = alfven_speed * k_magn * std::cos(angle_between_k_b0_rad);
		const amrex::Real cos_phase = std::cos(omega * time - k_magn * x_vec_mrf_C[0]);

		constexpr auto elsasser_sgn = static_cast<amrex::Real>(-1.0);
		// equivalent to, but numerically safer than -omega / (k_magn * cos_theta)
		const amrex::Real delta_v_magn = elsasser_sgn * alfven_speed * delta_b_magn * cos_phase;

		const amrex::Real v_x1_prf = delta_v_magn * outofplane_dir_prf[0];
		const amrex::Real v_x2_prf = delta_v_magn * outofplane_dir_prf[1];
		const amrex::Real v_x3_prf = delta_v_magn * outofplane_dir_prf[2];

		// background b
		const amrex::Real b0_x1_prf =
		    b0_magn * (std::cos(angle_between_k_b0_rad) * k_dir_prf[0] + std::sin(angle_between_k_b0_rad) * inplane_dir_prf[0]);
		const amrex::Real b0_x2_prf =
		    b0_magn * (std::cos(angle_between_k_b0_rad) * k_dir_prf[1] + std::sin(angle_between_k_b0_rad) * inplane_dir_prf[1]);
		const amrex::Real b0_x3_prf =
		    b0_magn * (std::cos(angle_between_k_b0_rad) * k_dir_prf[2] + std::sin(angle_between_k_b0_rad) * inplane_dir_prf[2]);
		// perturbed b
		const amrex::Real delta_b_x1_prf = b0_magn * delta_b_magn * cos_phase * outofplane_dir_prf[0];
		const amrex::Real delta_b_x2_prf = b0_magn * delta_b_magn * cos_phase * outofplane_dir_prf[1];
		const amrex::Real delta_b_x3_prf = b0_magn * delta_b_magn * cos_phase * outofplane_dir_prf[2];
		// total b
		const amrex::Real b_x1_prf = b0_x1_prf + delta_b_x1_prf;
		const amrex::Real b_x2_prf = b0_x2_prf + delta_b_x2_prf;
		const amrex::Real b_x3_prf = b0_x3_prf + delta_b_x3_prf;

		const amrex::Real density = bg_density;
		const amrex::Real pressure = bg_pressure;

		const amrex::Real v_magn_sq = v_x1_prf * v_x1_prf + v_x2_prf * v_x2_prf + v_x3_prf * v_x3_prf;
		const amrex::Real b_magn_sq = b_x1_prf * b_x1_prf + b_x2_prf * b_x2_prf + b_x3_prf * b_x3_prf;
		const amrex::Real Ekin = static_cast<amrex::Real>(0.5) * density * v_magn_sq;
		const amrex::Real Emag = static_cast<amrex::Real>(0.5) * b_magn_sq;
		const amrex::Real Eint = pressure / (gamma_gas - 1);
		const amrex::Real Etot = Ekin + Emag + Eint;

		state(i, j, k, HydroSystem<AlfvenWaveLinear>::density_index) = density;
		state(i, j, k, HydroSystem<AlfvenWaveLinear>::x1Momentum_index) = v_x1_prf * density;
		state(i, j, k, HydroSystem<AlfvenWaveLinear>::x2Momentum_index) = v_x2_prf * density;
		state(i, j, k, HydroSystem<AlfvenWaveLinear>::x3Momentum_index) = v_x3_prf * density;
		state(i, j, k, HydroSystem<AlfvenWaveLinear>::energy_index) = Etot;
		state(i, j, k, HydroSystem<AlfvenWaveLinear>::internalEnergy_index) = Eint;
	} else if (cen == quokka::centering::fc) {
		const amrex::Real delta_x1 = dx[0];
		const amrex::Real delta_x2 = dx[1];
		const amrex::Real delta_x3 = dx[2];
		const amrex::Real x1_prf_C = x1_prf_L + 0.5 * delta_x1;
		const amrex::Real x2_prf_C = x2_prf_L + 0.5 * delta_x2;
		const amrex::Real x3_prf_C = x3_prf_L + 0.5 * delta_x3;
		const amrex::Real x1_prf_R = x1_prf_L + delta_x1;
		const amrex::Real x2_prf_R = x2_prf_L + delta_x2;
		const amrex::Real x3_prf_R = x3_prf_L + delta_x3;
		// b-field computed using the magnetic vector potential to preserve div(b) = 0 topology
		// dAz/dy - dAy/dz
		const amrex::Real b_x1_L = (Az_prf(x1_prf_L, x2_prf_R, x3_prf_C, time) - Az_prf(x1_prf_L, x2_prf_L, x3_prf_C, time)) / delta_x2 -
					   (Ay_prf(x1_prf_L, x2_prf_C, x3_prf_R, time) - Ay_prf(x1_prf_L, x2_prf_C, x3_prf_L, time)) / delta_x3;
		// dAx/dz - dAz/dx
		const amrex::Real b_x2_L = (Ax_prf(x1_prf_C, x2_prf_L, x3_prf_R, time) - Ax_prf(x1_prf_C, x2_prf_L, x3_prf_L, time)) / delta_x3 -
					   (Az_prf(x1_prf_R, x2_prf_L, x3_prf_C, time) - Az_prf(x1_prf_L, x2_prf_L, x3_prf_C, time)) / delta_x1;
		// dAy/dx - dAx/dy
		const amrex::Real b_x3_L = (Ay_prf(x1_prf_R, x2_prf_C, x3_prf_L, time) - Ay_prf(x1_prf_L, x2_prf_C, x3_prf_L, time)) / delta_x1 -
					   (Ax_prf(x1_prf_C, x2_prf_R, x3_prf_L, time) - Ax_prf(x1_prf_C, x2_prf_L, x3_prf_L, time)) / delta_x2;
		if (dir == quokka::direction::x) {
			state(i, j, k, MHDSystem<AlfvenWaveLinear>::bfield_index) = b_x1_L;
		} else if (dir == quokka::direction::y) {
			state(i, j, k, MHDSystem<AlfvenWaveLinear>::bfield_index) = b_x2_L;
		} else if (dir == quokka::direction::z) {
			state(i, j, k, MHDSystem<AlfvenWaveLinear>::bfield_index) = b_x3_L;
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
			state_cc(i, j, k, n) = 0.0; // fill unused quantities with zeros
		}
		computeWaveSolution(i, j, k, state_cc, dx, prob_lo, cen, dir, 0);
	});
}

template <> void QuokkaSimulation<AlfvenWaveLinear>::setInitialConditionsOnGridFaceVars(quokka::grid const &grid_elem)
{
	// extract grid information
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = grid_elem.dx_;
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = grid_elem.prob_lo_;
	const amrex::Array4<amrex::Real> &state_fc = grid_elem.array_;
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const quokka::centering cen = grid_elem.cen_;
	const quokka::direction dir = grid_elem.dir_;

	const int ncomp_fc = Physics_Indices<AlfvenWaveLinear>::nvarPerDim_fc;
	// loop over the grid and set the initial condition
	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		for (int n = 0; n < ncomp_fc; ++n) {
			state_fc(i, j, k, n) = 0.0; // fill unused quantities with zeros
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

		const amrex::Real time = tNew_[0];
		const int ncomp_cc = Physics_Indices<AlfvenWaveLinear>::nvarTotal_cc;
		amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
			for (int n = 0; n < ncomp_cc; ++n) {
				stateExact(i, j, k, n) = 0.0; // fill unused quantities with zeros
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

		const amrex::Real time = tNew_[0];
		const int ncomp_fc = Physics_Indices<AlfvenWaveLinear>::nvarPerDim_fc;
		amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
			for (int n = 0; n < ncomp_fc; ++n) {
				stateExact(i, j, k, n) = 0.0; // fill unused quantities with zeros
			}
			computeWaveSolution(i, j, k, stateExact, dx, prob_lo, quokka::centering::fc, dir, time);
		});
	}
}

auto problem_main() -> int
{
	amrex::ParmParse const hpp("setup");

	amrex::Real angle_between_k_b0_deg = 0.0;
	hpp.query("angle_between_k_b0", angle_between_k_b0_deg);

	constexpr amrex::Real deg2rad = M_PI / 180.0;
	angle_between_k_b0_rad = deg2rad * angle_between_k_b0_deg;

	int num_modes_x = 0;
	int num_modes_y = 0;
	int num_modes_z = 0;
	hpp.query("num_modes_x", num_modes_x);
	hpp.query("num_modes_y", num_modes_y);
	hpp.query("num_modes_z", num_modes_z);

	if ((num_modes_x == 0) && (num_modes_y == 0) && (num_modes_z == 0)) {
		amrex::Abort("Invalid k modes: the triplet (0,0,0) is not allowed.");
	}

	// we assume box length = 1.0
	const std::array<amrex::Real, 3> k_vec_prf = {(2.0 * M_PI) * static_cast<amrex::Real>(num_modes_x),
						      (2.0 * M_PI) * static_cast<amrex::Real>(num_modes_y),
						      (2.0 * M_PI) * static_cast<amrex::Real>(num_modes_z)};
	k_magn = computeMagnitude(k_vec_prf);
	k_dir_prf = {k_vec_prf[0] / k_magn, k_vec_prf[1] / k_magn, k_vec_prf[2] / k_magn};

	// to build our orthonormal basis in the problem reference frame (PRF)
	// first choose a vector that is not aligned/parallel with the wave propagation direction
	std::array<amrex::Real, 3> ref_prf{static_cast<amrex::Real>(0.0), static_cast<amrex::Real>(0.0), static_cast<amrex::Real>(1.0)};
	if (std::abs(computeDotProduct(ref_prf, k_dir_prf)) > static_cast<amrex::Real>(0.9999)) {
		ref_prf = {static_cast<amrex::Real>(0.0), static_cast<amrex::Real>(1.0), static_cast<amrex::Real>(0.0)};
	}

	// define the plane in which b0 will sit
	inplane_dir_prf = computeCrossProduct(ref_prf, k_dir_prf);
	normalizeVector(inplane_dir_prf);

	// define the direction the perturbation will be induced
	outofplane_dir_prf = computeCrossProduct(k_dir_prf, inplane_dir_prf);
	normalizeVector(outofplane_dir_prf);
	amrex::Gpu::synchronize();

	auto BCs_cc = quokka::BC<AlfvenWaveLinear>(quokka::BCType::int_dir);

	const int nvars_fc = Physics_Indices<AlfvenWaveLinear>::nvarTotal_fc;
	amrex::Vector<amrex::BCRec> BCs_fc(nvars_fc);
	for (int icomp = 0; icomp < nvars_fc; ++icomp) {
		for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
			BCs_fc[icomp].setLo(idim, amrex::BCType::int_dir);
			BCs_fc[icomp].setHi(idim, amrex::BCType::int_dir);
		}
	}

	QuokkaSimulation<AlfvenWaveLinear> sim(BCs_cc, BCs_fc);
	sim.computeReferenceSolution_ = true;
	sim.setInitialConditions();
	sim.evolve();

	int status = 1;
	const amrex::Real error_tol = 0.005;
	if (sim.errorNorm_ < error_tol) {
		status = 0;
		amrex::Print() << "Error norm = " << sim.errorNorm_ << "\n";
		amrex::Print() << "test passed\n";
	} else {
		amrex::Print() << "Error norm = " << sim.errorNorm_ << "\n";
		amrex::Print() << "test failed\n";
	}

	return status;
}
