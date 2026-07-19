//==============================================================================
// Copyright 2022 Neco Kriel.
// Modified for Entropy wave (MHD) test.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file testEntropyWaveConvergence.cpp
/// \brief Defines a test problem for an MHD entropy wave (density perturbation
///        advected with background flow) and ensures face-centered quantities
///        are created correctly.
///

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <gcem.hpp>
#include <iostream>

#include "AMReX_Array.H"
#include "AMReX_Array4.H"
#include "AMReX_Gpu.H"
#include "AMReX_ParallelDescriptor.H"
#include "AMReX_ParmParse.H"
#include "AMReX_REAL.H"

#include "QuokkaSimulation.hpp"
#include "grid.hpp"
#include "hydro/EOS.hpp"
#include "physics_info.hpp"
#include "util/BC.hpp"
#include "util/fextract.hpp"
#include "util/richardson.hpp"
#ifdef HAVE_PYTHON
#include "util/matplotlibcpp.h"
#endif

struct EntropyWaveLinear {
};

template <> struct quokka::EOS_Traits<EntropyWaveLinear> {
	static constexpr double gamma = 5. / 3.;
	static constexpr double mean_molecular_weight = C::m_u;
};

template <> struct Physics_Traits<EntropyWaveLinear> : DefaultPhysicsTraits {
	static constexpr bool is_hydro_enabled = true;
	static constexpr bool is_mhd_enabled = true;
};

// Background and perturbation parameters
constexpr double adv_speed = 1.0; // advection speed of the entropy wave (in MRF along k)
constexpr double gamma_gas = quokka::EOS_Traits<EntropyWaveLinear>::gamma;
constexpr double bg_density = 1.0;
constexpr double delta_rho_magn = 1e-6; // small density perturbation amplitude
constexpr double sound_speed = 1.0;
constexpr double bg_pressure = sound_speed * sound_speed * bg_density / gamma_gas;
constexpr double b0_magn = 1.0;

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
// Reference frames and rotation matrix (same explanation as original file)
//------------------------------------------------------------------------------
AMREX_GPU_MANAGED std::array<amrex::Real, 3> k_dir_prf{1.0, 0.0, 0.0};		// NOLINT
AMREX_GPU_MANAGED std::array<amrex::Real, 3> inplane_dir_prf{0.0, 1.0, 0.0};	// NOLINT
AMREX_GPU_MANAGED std::array<amrex::Real, 3> outofplane_dir_prf{0.0, 0.0, 1.0}; // NOLINT

// wavefront
AMREX_GPU_MANAGED double k_magn = 2.0 * M_PI; // NOLINT

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

// Vector potential: construct only the background field (no perturbation)
AMREX_GPU_DEVICE AMREX_FORCE_INLINE auto computeVectorPotentialComponent_prf(const double x1_prf, const double x2_prf, const double x3_prf,
									     const double /*time*/, const int icomp) -> double
{
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(icomp == 0 || icomp == 1 || icomp == 2,
					 "computeVectorPotentialComponent_prf(): icomp must be an integer in {0, 1, 2}");
	const std::array<amrex::Real, 3> x_vec_mrf = rotatePRF2MRF({x1_prf, x2_prf, x3_prf});
	// background B components in MRF: b0_x1 = b0 * cos(theta), b0_x2 = b0 * sin(theta)
	const double b0_x1_mrf = b0_magn * std::cos(angle_between_k_b0_rad);
	const double b0_x2_mrf = b0_magn * std::sin(angle_between_k_b0_rad);
	// bg_A = (0, 0, b0_x1 * x2 - b0_x2 * x1) -> curl(bg_A) = (b0_x1, b0_x2, 0)
	const double bg_A1_mrf = 0.0;
	const double bg_A2_mrf = 0.0;
	const double bg_A3_mrf = b0_x1_mrf * x_vec_mrf[1] - b0_x2_mrf * x_vec_mrf[0];
	const std::array<amrex::Real, 3> A_vec_prf = rotateMRF2PRF({bg_A1_mrf, bg_A2_mrf, bg_A3_mrf});
	return A_vec_prf[icomp];
}

AMREX_GPU_DEVICE inline auto Ax_prf(const double x1_prf, const double x2_prf, const double x3_prf, const double time) -> double
{
	(void)time;
	return computeVectorPotentialComponent_prf(x1_prf, x2_prf, x3_prf, time, 0);
}

AMREX_GPU_DEVICE inline auto Ay_prf(const double x1_prf, const double x2_prf, const double x3_prf, const double time) -> double
{
	(void)time;
	return computeVectorPotentialComponent_prf(x1_prf, x2_prf, x3_prf, time, 1);
}

AMREX_GPU_DEVICE inline auto Az_prf(const double x1_prf, const double x2_prf, const double x3_prf, const double time) -> double
{
	(void)time;
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

		// Phase for advected entropy wave: omega = adv_speed * |k| (propagates along k)
		const double omega = adv_speed * k_magn;
		const double cos_phase = std::cos(omega * time - k_magn * x_vec_mrf_C[0]);

		// density perturbation (entropy mode) with constant pressure
		const double density = bg_density * (1.0 + delta_rho_magn * cos_phase);
		const double pressure = bg_pressure; // constant (entropy perturbation)

		// background velocity: advect the wave at adv_speed along k (MRF)
		const std::array<amrex::Real, 3> v_mrf_bg = {static_cast<amrex::Real>(adv_speed), 0.0, 0.0};
		const std::array<amrex::Real, 3> v_prf_bg = rotateMRF2PRF(v_mrf_bg);
		const double v_x1_prf = v_prf_bg[0];
		const double v_x2_prf = v_prf_bg[1];
		const double v_x3_prf = v_prf_bg[2];

		// background magnetic field (no perturbation)
		const double b0_x1_prf = b0_magn * (std::cos(angle_between_k_b0_rad) * k_dir_prf[0] + std::sin(angle_between_k_b0_rad) * inplane_dir_prf[0]);
		const double b0_x2_prf = b0_magn * (std::cos(angle_between_k_b0_rad) * k_dir_prf[1] + std::sin(angle_between_k_b0_rad) * inplane_dir_prf[1]);
		const double b0_x3_prf = b0_magn * (std::cos(angle_between_k_b0_rad) * k_dir_prf[2] + std::sin(angle_between_k_b0_rad) * inplane_dir_prf[2]);

		const double v_magn_sq = v_x1_prf * v_x1_prf + v_x2_prf * v_x2_prf + v_x3_prf * v_x3_prf;
		const double b_magn_sq = b0_x1_prf * b0_x1_prf + b0_x2_prf * b0_x2_prf + b0_x3_prf * b0_x3_prf;
		const double Ekin = 0.5 * density * v_magn_sq;
		const double Emag = 0.5 * b_magn_sq;
		// internal energy per volume: pressure / (gamma - 1)
		const double Eint = pressure / (gamma_gas - 1.0);
		const double Etot = Ekin + Emag + Eint;

		state(i, j, k, HydroSystem<EntropyWaveLinear>::density_index) = density;
		state(i, j, k, HydroSystem<EntropyWaveLinear>::x1Momentum_index) = v_x1_prf * density;
		state(i, j, k, HydroSystem<EntropyWaveLinear>::x2Momentum_index) = v_x2_prf * density;
		state(i, j, k, HydroSystem<EntropyWaveLinear>::x3Momentum_index) = v_x3_prf * density;
		state(i, j, k, HydroSystem<EntropyWaveLinear>::energy_index) = Etot;
		state(i, j, k, HydroSystem<EntropyWaveLinear>::internalEnergy_index) = Eint;
	} else if (cen == quokka::centering::fc) {
		// compute b-field using the magnetic vector potential (background only) to preserve div(b)=0
		const double b_x1 =
		    (Az_prf(x1_prf_L, x2_prf_L + dx[1], x3_prf_L + dx[2] / 2.0, time) - Az_prf(x1_prf_L, x2_prf_L, x3_prf_L + dx[2] / 2.0, time)) / dx[1] -
		    (Ay_prf(x1_prf_L, x2_prf_L + dx[1] / 2.0, x3_prf_L + dx[2], time) - Ay_prf(x1_prf_L, x2_prf_L + dx[1] / 2.0, x3_prf_L, time)) / dx[2];

		const double b_x2 =
		    (Ax_prf(x1_prf_L + dx[0] / 2.0, x2_prf_L, x3_prf_L + dx[2], time) - Ax_prf(x1_prf_L + dx[0] / 2.0, x2_prf_L, x3_prf_L, time)) / dx[2] -
		    (Az_prf(x1_prf_L + dx[0], x2_prf_L, x3_prf_L + dx[2] / 2.0, time) - Az_prf(x1_prf_L, x2_prf_L, x3_prf_L + dx[2] / 2.0, time)) / dx[0];

		const double b_x3 =
		    (Ay_prf(x1_prf_L + dx[0], x2_prf_L + dx[1] / 2.0, x3_prf_L, time) - Ay_prf(x1_prf_L, x2_prf_L + dx[1] / 2.0, x3_prf_L, time)) / dx[0] -
		    (Ax_prf(x1_prf_L + dx[0] / 2.0, x2_prf_L + dx[1], x3_prf_L, time) - Ax_prf(x1_prf_L + dx[0] / 2.0, x2_prf_L, x3_prf_L, time)) / dx[1];

		if (dir == quokka::direction::x) {
			state(i, j, k, MHDSystem<EntropyWaveLinear>::bfield_index) = b_x1;
		} else if (dir == quokka::direction::y) {
			state(i, j, k, MHDSystem<EntropyWaveLinear>::bfield_index) = b_x2;
		} else if (dir == quokka::direction::z) {
			state(i, j, k, MHDSystem<EntropyWaveLinear>::bfield_index) = b_x3;
		}
	}
}

template <> void QuokkaSimulation<EntropyWaveLinear>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = grid_elem.dx_;
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = grid_elem.prob_lo_;
	const amrex::Array4<amrex::Real> &state_cc = grid_elem.array_;
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const quokka::centering cen = grid_elem.cen_;
	const quokka::direction dir = grid_elem.dir_;

	const int ncomp_cc = Physics_Indices<EntropyWaveLinear>::nvarTotal_cc;
	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
		for (int n = 0; n < ncomp_cc; ++n) {
			state_cc(i, j, k, n) = 0; // fill unused quantities with zeros
		}
		computeWaveSolution(i, j, k, state_cc, dx, prob_lo, cen, dir, 0);
	});
}

template <> void QuokkaSimulation<EntropyWaveLinear>::setInitialConditionsOnGridFaceVars(quokka::grid const &grid_elem)
{
	// extract grid information
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = grid_elem.dx_;
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = grid_elem.prob_lo_;
	const amrex::Array4<amrex::Real> &state_fc = grid_elem.array_;
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const quokka::centering cen = grid_elem.cen_;
	const quokka::direction dir = grid_elem.dir_;

	const int ncomp_fc = Physics_Indices<EntropyWaveLinear>::nvarPerDim_fc;
	// loop over the grid and set the initial condition
	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
		for (int n = 0; n < ncomp_fc; ++n) {
			state_fc(i, j, k, n) = 0; // fill unused quantities with zeros
		}
		computeWaveSolution(i, j, k, state_fc, dx, prob_lo, cen, dir, 0);
	});
}

template <>
void QuokkaSimulation<EntropyWaveLinear>::computeReferenceSolution(amrex::MultiFab &ref, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx,
								   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_lo)
{
	for (amrex::MFIter iter(ref); iter.isValid(); ++iter) {
		const amrex::Box &indexRange = iter.validbox();
		auto const &stateExact = ref.array(iter);
		auto const ncomp = ref.nComp();

		amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
			for (int n = 0; n < ncomp; ++n) {
				stateExact(i, j, k, n) = 0.0; // fill unused quantities with zeros
			}
			computeWaveSolution(i, j, k, stateExact, dx, prob_lo, quokka::centering::cc, quokka::direction::na, 0);
		});
	}
}

template <>
void QuokkaSimulation<EntropyWaveLinear>::computeReferenceSolution_fc(amrex::MultiFab &ref, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx,
								      amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_lo, quokka::direction const dir)
{
	for (amrex::MFIter iter(ref); iter.isValid(); ++iter) {
		const amrex::Box &indexRange = iter.validbox();
		auto const &stateExact = ref.array(iter);
		auto const ncomp = ref.nComp();

		amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept -> void {
			for (int n = 0; n < ncomp; ++n) {
				stateExact(i, j, k, n) = 0.0; // fill unused quantities with zeros
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
	const double max_time = wavelength / adv_speed;
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
	pp.addarr("n_cell", ncells);

	if (!pp.contains("blocking_factor")) {
		pp.add("blocking_factor", 8);
	}

	if (!pp.contains("max_grid_size")) {
		pp.add("max_grid_size", 128);
	}

	pp.add("max_level", 0);

	// Set domain bounds using AMReX parameter system
	amrex::ParmParse pp_geom("geometry");
	amrex::Vector<double> const prob_lo = {0.0, 0.0, 0.0};
	amrex::Vector<double> const prob_hi = {1.0, 1.0, 1.0};
	amrex::Vector<int> const is_periodic = {1, 1, 1};
	pp_geom.addarr("prob_lo", prob_lo);
	pp_geom.addarr("prob_hi", prob_hi);
	pp_geom.addarr("is_periodic", is_periodic);

	// Setup boundary conditions
	auto BCs_cc = quokka::BC<EntropyWaveLinear>(quokka::BCType::int_dir);

	const int nvars_fc = Physics_Indices<EntropyWaveLinear>::nvarTotal_fc;
	amrex::Vector<amrex::BCRec> BCs_fc(nvars_fc);
	for (int icomp = 0; icomp < nvars_fc; ++icomp) {
		for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
			BCs_fc[icomp].setLo(idim, amrex::BCType::int_dir);
			BCs_fc[icomp].setHi(idim, amrex::BCType::int_dir);
		}
	}

	// Run simulation
	QuokkaSimulation<EntropyWaveLinear> sim(BCs_cc, BCs_fc);

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

		auto BCs_cc = quokka::BC<EntropyWaveLinear>(quokka::BCType::int_dir);
		const int nvars_fc = Physics_Indices<EntropyWaveLinear>::nvarTotal_fc;
		amrex::Vector<amrex::BCRec> BCs_fc(nvars_fc);
		for (int icomp = 0; icomp < nvars_fc; ++icomp) {
			for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
				BCs_fc[icomp].setLo(idim, amrex::BCType::int_dir);
				BCs_fc[icomp].setHi(idim, amrex::BCType::int_dir);
			}
		}

		QuokkaSimulation<EntropyWaveLinear> sim(BCs_cc, BCs_fc);
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
		params.test_name = "Entropy Wave";
		params.csv_filename = "entropy_wave_convergence.csv";

		if (quokka::richardson::run(params, [](int nx, int ny, int nz) { return runWaveTest(nx, ny, nz); }) != 0) {
			status = 1;
		}
	}

	return status;
}
