//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file testFastWaveConvergence.cpp
/// \brief Defines a Richardson convergence test for the fast MHD wave.
///

#include <bitset>
#include <cassert>
#include <cmath>
#include <gcem.hpp>
#include <iostream>

#include "AMReX_Array.H"
#include "AMReX_Array4.H"
#include "AMReX_REAL.H"

#include "QuokkaSimulation.hpp"
#include "grid.hpp"
#include "physics_info.hpp"
#include "util/BC.hpp"
#include "util/fextract.hpp"

struct FastWaveConvergence {
};

template <> struct quokka::EOS_Traits<FastWaveConvergence> {
	static constexpr double gamma = 5. / 3.;
	static constexpr double mean_molecular_weight = C::m_u;
	static constexpr double boltzmann_constant = C::k_B;
};

template <> struct Physics_Traits<FastWaveConvergence> {
	static constexpr bool is_hydro_enabled = true;
	static constexpr int numMassScalars = 0;		     // number of mass scalars
	static constexpr int numPassiveScalars = numMassScalars + 0; // number of passive scalars
	static constexpr bool is_self_gravity_enabled = false;
	static constexpr bool is_radiation_enabled = false;
	static constexpr bool is_mhd_enabled = true;
	static constexpr int nGroups = 1; // number of radiation groups
	static constexpr UnitSystem unit_system = UnitSystem::CGS;
};

// constants
constexpr double sound_speed = 1.0;
constexpr double gamma_gas = quokka::EOS_Traits<FastWaveConvergence>::gamma;

// background states
constexpr double bg_density = 1.0;
constexpr double bg_pressure = sound_speed * sound_speed * bg_density / gamma_gas;
constexpr double bg_mag_amplitude = 1.;

// theta is the angle between k and background magnetic field bg_mag
constexpr double theta_degrees = 90.0; // degrees
constexpr double cos_theta = gcem::cos(theta_degrees * M_PI / 180.0);

// k = 2 pi / wave length
// box length = 1, so |k| in [1, inf)
constexpr double num_modes = 1;
AMREX_GPU_MANAGED double k_amplitude = 2.0 * M_PI; // Will be updated
// constexpr double k_amplitude = 2 * M_PI * num_modes;

// input perturbation: choose to do this via the relative density field in [0, 1]. remember, the linear regime is valid when this perturbation is small
constexpr double delta_b = 1e-4;

constexpr double alfven_speed = bg_mag_amplitude / gcem::sqrt(bg_density);
constexpr double magnetosonic_speed = gcem::sqrt(alfven_speed * alfven_speed + sound_speed * sound_speed);
constexpr double bg_mag_x3 = bg_mag_amplitude;

// constexpr double omega =
//     gcem::sqrt(gcem::pow(k_amplitude, 2) / 2.0 *
// 	       (gcem::pow(magnetosonic_speed, 2) + gcem::sqrt(gcem::pow(magnetosonic_speed, 4) - 4.0 * gcem::pow(alfven_speed, 2) * gcem::pow(sound_speed, 2) *
// 												     gcem::pow(cos_theta, 2)))); // NOLINT(cert-err58-cpp)

// angles (radians) in the math reference frame (MRF)
AMREX_GPU_MANAGED double angle_between_k_b0_rad = 0.0; // NOLINT

// Unit basis vectors of the MRF, expressed in PRF coordinates
AMREX_GPU_MANAGED std::array<amrex::Real, 3> k_dir_prf{1.0, 0.0, 0.0};		// NOLINT
AMREX_GPU_MANAGED std::array<amrex::Real, 3> inplane_dir_prf{0.0, 1.0, 0.0};	// NOLINT
AMREX_GPU_MANAGED std::array<amrex::Real, 3> outofplane_dir_prf{0.0, 0.0, 1.0}; // NOLINT

// Helper functions (add these before computeWaveSolution)
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

// Rotation helpers
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

// Modified omega calculation (replaces your constexpr omega)
AMREX_GPU_MANAGED double omega = 0.0; // Will be computed in problem_main

// AMREX_GPU_DEVICE auto computeMagneticVectorPotential_x(double x1, double x2, double /*x3*/, double time)
// {
// 	return -x2 / 2.0 * (bg_mag_amplitude + delta_b * std::cos(omega * time - k_amplitude * x1));
// }
// AMREX_GPU_DEVICE auto computeMagneticVectorPotential_y(double x1, double /*x2*/, double /*x3*/, double time) -> double
// {
// 	return bg_mag_amplitude * x1 / 2.0 + ((delta_b)*std::sin(omega * time - k_amplitude * x1) / (-2.0 * k_amplitude));
// }
// AMREX_GPU_DEVICE auto computeMagneticVectorPotential_z(double /*x1*/, double /*x2*/, double /*x3*/, double /*time*/) -> double { return 0.0; }

// Vector potential in PRF - computes component icomp of A at position (x1,x2,x3) in PRF
AMREX_GPU_DEVICE AMREX_FORCE_INLINE auto computeVectorPotentialComponent_prf(const double x1_prf, const double x2_prf, const double x3_prf, const double time,
									     const int icomp) -> double
{
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(icomp == 0 || icomp == 1 || icomp == 2, "computeVectorPotentialComponent_prf(): icomp must be in {0, 1, 2}");

	// Rotate position to MRF
	const std::array<amrex::Real, 3> x_vec_mrf = rotatePRF2MRF({x1_prf, x2_prf, x3_prf});

	// Background B field in MRF: B0 = (b0_x1, b0_x2, 0) lying in the k-inplane plane
	const double b0_x1_mrf = bg_mag_amplitude * std::cos(angle_between_k_b0_rad);
	const double b0_x2_mrf = bg_mag_amplitude * std::sin(angle_between_k_b0_rad);

	// Background vector potential: curl(A_bg) = B0
	// Choose A_bg = (0, 0, b0_x1 * x2 - b0_x2 * x1) so curl gives (b0_x1, b0_x2, 0)
	const double bg_A1_mrf = 0.0;
	const double bg_A2_mrf = 0.0;
	const double bg_A3_mrf = b0_x1_mrf * x_vec_mrf[1] - b0_x2_mrf * x_vec_mrf[0];

	// Perturbation: delta_B is in x3 direction (out of k-b0 plane)
	// For delta_B_x3 = delta_b * cos(omega*t - k*x1), we need:
	// dA2/dx1 = delta_b * cos(omega*t - k*x1)
	// So A2 = -(delta_b/k) * sin(omega*t - k*x1)
	const double delta_A1_mrf = 0.0;
	const double delta_A2_mrf = -(bg_mag_amplitude * delta_b / k_amplitude) * std::sin(omega * time - k_amplitude * x_vec_mrf[0]);
	const double delta_A3_mrf = 0.0;

	// Total vector potential in MRF
	const double A1_mrf = bg_A1_mrf + delta_A1_mrf;
	const double A2_mrf = bg_A2_mrf + delta_A2_mrf;
	const double A3_mrf = bg_A3_mrf + delta_A3_mrf;

	// Rotate back to PRF
	const std::array<amrex::Real, 3> A_vec_prf = rotateMRF2PRF({A1_mrf, A2_mrf, A3_mrf});

	return A_vec_prf[icomp];
}

AMREX_GPU_DEVICE inline auto computeMagneticVectorPotential_x(double x1, double x2, double x3, double time) -> double
{
	return computeVectorPotentialComponent_prf(x1, x2, x3, time, 0);
}

AMREX_GPU_DEVICE inline auto computeMagneticVectorPotential_y(double x1, double x2, double x3, double time) -> double
{
	return computeVectorPotentialComponent_prf(x1, x2, x3, time, 1);
}

AMREX_GPU_DEVICE inline auto computeMagneticVectorPotential_z(double x1, double x2, double x3, double time) -> double
{
	return computeVectorPotentialComponent_prf(x1, x2, x3, time, 2);
}

AMREX_GPU_DEVICE void computeWaveSolution(int i, int j, int k, amrex::Array4<amrex::Real> const &state, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx,
					  amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_lo, quokka::centering cen, quokka::direction dir,
					  double time)
{

	const amrex::Real x1_L = prob_lo[0] + i * dx[0];
	const amrex::Real x2_L = prob_lo[1] + j * dx[1];
	const amrex::Real x3_L = prob_lo[2] + k * dx[2];

	if (cen == quokka::centering::cc) {
		const amrex::Real x1_C = x1_L + static_cast<amrex::Real>(0.5) * dx[0];
		const amrex::Real x2_C = x2_L + static_cast<amrex::Real>(0.5) * dx[1];
		const amrex::Real x3_C = x3_L + static_cast<amrex::Real>(0.5) * dx[2];

		// Rotate position to MRF
		const std::array<amrex::Real, 3> x_vec_mrf = rotatePRF2MRF({x1_C, x2_C, x3_C});

		// Wave phase (invariant under rotation: k·x is the same in both frames)
		const double cos_wave = std::cos(omega * time - k_amplitude * x_vec_mrf[0]);

		// Background magnetic field in MRF
		const double b0_x1_mrf = bg_mag_amplitude * std::cos(angle_between_k_b0_rad);
		const double b0_x2_mrf = bg_mag_amplitude * std::sin(angle_between_k_b0_rad);
		const double b0_x3_mrf = 0.0;

		// Perturbed magnetic field in MRF (perpendicular to k-b0 plane)
		const double delta_b_x1_mrf = 0.0;
		const double delta_b_x2_mrf = 0.0;
		const double delta_b_x3_mrf = delta_b * cos_wave;

		// Total B field in MRF
		const double b_x1_mrf = b0_x1_mrf + delta_b_x1_mrf;
		const double b_x2_mrf = b0_x2_mrf + delta_b_x2_mrf;
		const double b_x3_mrf = b0_x3_mrf + delta_b_x3_mrf;

		// Rotate B field back to PRF
		const std::array<amrex::Real, 3> b_vec_prf = rotateMRF2PRF({b_x1_mrf, b_x2_mrf, b_x3_mrf});

		// Density and pressure perturbations
		const double density = bg_density + bg_density * delta_b / bg_mag_amplitude * cos_wave;
		const double pressure = bg_pressure + bg_pressure * gamma_gas * delta_b / bg_mag_amplitude * cos_wave;

		// Velocity perturbation in MRF (aligned with perturbed B)
		const double v_x1_mrf = 0.0;
		const double v_x2_mrf = 0.0;
		const double v_x3_mrf = magnetosonic_speed * delta_b / bg_mag_amplitude * cos_wave;

		// Rotate velocity back to PRF
		const std::array<amrex::Real, 3> v_vec_prf = rotateMRF2PRF({v_x1_mrf, v_x2_mrf, v_x3_mrf});
		if (i == 4 && j == 4 && k == 4) {
			std::cout << "Time: " << time << ", B_prf: (" << b_vec_prf[0] << ", " << b_vec_prf[1] << ", " << b_vec_prf[2] << ")\n";
			std::cout << "  delta_b_mrf: (" << delta_b_x1_mrf << ", " << delta_b_x2_mrf << ", " << delta_b_x3_mrf << ")\n";
			std::cout << "  cos_wave: " << cos_wave << "\n";
			std::cout << "  density: " << density << " (bg: " << bg_density << ")\n";
			std::cout << "  pressure: " << pressure << " (bg: " << bg_pressure << ")\n";
			std::cout << "  delta_rho/rho: " << (density - bg_density) / bg_density << "\n";
			std::cout << "  delta_P/P: " << (pressure - bg_pressure) / bg_pressure << "\n";
		}
		// Compute energies
		const double v_sq = v_vec_prf[0] * v_vec_prf[0] + v_vec_prf[1] * v_vec_prf[1] + v_vec_prf[2] * v_vec_prf[2];
		const double b_sq = b_vec_prf[0] * b_vec_prf[0] + b_vec_prf[1] * b_vec_prf[1] + b_vec_prf[2] * b_vec_prf[2];
		const double Ekin = 0.5 * density * v_sq;
		const double Emag = 0.5 * b_sq;
		const double Eint = pressure / (gamma_gas - 1.0);
		const double Etot = Ekin + Emag + Eint;

		// Store in PRF coordinates
		state(i, j, k, HydroSystem<FastWaveConvergence>::density_index) = density;
		state(i, j, k, HydroSystem<FastWaveConvergence>::x1Momentum_index) = v_vec_prf[0] * density;
		state(i, j, k, HydroSystem<FastWaveConvergence>::x2Momentum_index) = v_vec_prf[1] * density;
		state(i, j, k, HydroSystem<FastWaveConvergence>::x3Momentum_index) = v_vec_prf[2] * density;
		state(i, j, k, HydroSystem<FastWaveConvergence>::energy_index) = Etot;
		state(i, j, k, HydroSystem<FastWaveConvergence>::internalEnergy_index) = Eint;

	} else if (cen == quokka::centering::fc) {
		// Face-centered B: compute analytical B at face location
		amrex::Real x1_F = x1_L, x2_F = x2_L, x3_F = x3_L;
		if (dir == quokka::direction::x) {
			x1_F += 0.0;
			x2_F += 0.5 * dx[1];
			x3_F += 0.5 * dx[2];
		} else if (dir == quokka::direction::y) {
			x1_F += 0.5 * dx[0];
			x2_F += 0.0;
			x3_F += 0.5 * dx[2];
		} else if (dir == quokka::direction::z) {
			x1_F += 0.5 * dx[0];
			x2_F += 0.5 * dx[1];
			x3_F += 0.0;
		}
		const std::array<amrex::Real, 3> x_vec_mrf = rotatePRF2MRF({x1_F, x2_F, x3_F});
		const double cos_wave = std::cos(omega * time - k_amplitude * x_vec_mrf[0]);

		const double b0_x1_mrf = bg_mag_amplitude * std::cos(angle_between_k_b0_rad);
		const double b0_x2_mrf = bg_mag_amplitude * std::sin(angle_between_k_b0_rad);
		const double b0_x3_mrf = 0.0;

		const double delta_b_x1_mrf = 0.0;
		const double delta_b_x2_mrf = 0.0;
		const double delta_b_x3_mrf = delta_b * cos_wave;

		const double b_x1_mrf = b0_x1_mrf + delta_b_x1_mrf;
		const double b_x2_mrf = b0_x2_mrf + delta_b_x2_mrf;
		const double b_x3_mrf = b0_x3_mrf + delta_b_x3_mrf;

		const std::array<amrex::Real, 3> b_vec_prf = rotateMRF2PRF({b_x1_mrf, b_x2_mrf, b_x3_mrf});

		if (dir == quokka::direction::x)
			state(i, j, k, MHDSystem<FastWaveConvergence>::bfield_index) = b_vec_prf[0];
		else if (dir == quokka::direction::y)
			state(i, j, k, MHDSystem<FastWaveConvergence>::bfield_index) = b_vec_prf[1];
		else if (dir == quokka::direction::z)
			state(i, j, k, MHDSystem<FastWaveConvergence>::bfield_index) = b_vec_prf[2];
	}
}

template <> void QuokkaSimulation<FastWaveConvergence>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	// extract grid information
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = grid_elem.dx_;
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = grid_elem.prob_lo_;
	const amrex::Array4<double> &state_cc = grid_elem.array_;
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const quokka::centering cen = grid_elem.cen_;
	const quokka::direction dir = grid_elem.dir_;

	const int ncomp_cc = Physics_Indices<FastWaveConvergence>::nvarTotal_cc;
	// loop over the grid and set the initial condition
	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		for (int n = 0; n < ncomp_cc; ++n) {
			state_cc(i, j, k, n) = 0; // fill unused quantities with zeros
		}
		computeWaveSolution(i, j, k, state_cc, dx, prob_lo, cen, dir, 0);
	});
}

template <> void QuokkaSimulation<FastWaveConvergence>::setInitialConditionsOnGridFaceVars(quokka::grid const &grid_elem)
{
	// extract grid information
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = grid_elem.dx_;
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = grid_elem.prob_lo_;
	const amrex::Array4<double> &state_fc = grid_elem.array_;
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const quokka::centering cen = grid_elem.cen_;
	const quokka::direction dir = grid_elem.dir_;

	const int ncomp_fc = Physics_Indices<FastWaveConvergence>::nvarPerDim_fc;
	// loop over the grid and set the initial condition
	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		for (int n = 0; n < ncomp_fc; ++n) {
			state_fc(i, j, k, n) = 0; // fill unused quantities with zeros
		}
		computeWaveSolution(i, j, k, state_fc, dx, prob_lo, cen, dir, 0);
	});
}

template <>
void QuokkaSimulation<FastWaveConvergence>::computeReferenceSolution(amrex::MultiFab &ref, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx,
								     amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_lo)
{
	for (amrex::MFIter iter(ref); iter.isValid(); ++iter) {
		const amrex::Box &indexRange = iter.validbox();
		auto const &stateExact = ref.array(iter);
		auto const ncomp = ref.nComp();
		const amrex::Real time = tNew_[0];

		amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
			for (int n = 0; n < ncomp; ++n) {
				stateExact(i, j, k, n) = 0.0; // fill unused quantities with zeros
			}
			computeWaveSolution(i, j, k, stateExact, dx, prob_lo, quokka::centering::cc, quokka::direction::na, time);
		});
	}
}

template <>
void QuokkaSimulation<FastWaveConvergence>::computeReferenceSolution_fc(amrex::MultiFab &ref, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx,
									amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_lo,
									quokka::direction const dir)
{
	for (amrex::MFIter iter(ref); iter.isValid(); ++iter) {
		const amrex::Box &indexRange = iter.validbox();
		auto const &stateExact = ref.array(iter);
		auto const ncomp = ref.nComp();
		const amrex::Real time = tNew_[0];

		amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
			for (int n = 0; n < ncomp; ++n) {
				stateExact(i, j, k, n) = 0.0; // fill unused quantities with zeros
			}
			computeWaveSolution(i, j, k, stateExact, dx, prob_lo, quokka::centering::fc, dir, time);
		});
	}
}

auto runWaveTest(int nx) -> double
{
	amrex::Print() << "runWaveTest: omega = " << omega << ", k_amplitude = " << k_amplitude << ", max_time = " << k_amplitude / omega << "\n";

	// Problem parameters
	const double CFL_number = 0.3;
	const double max_time = 2.0 * M_PI / omega;

	const int max_timesteps = std::max(20000, nx * 100);

	// Problem initialization
	const int ncomp_cc = Physics_Indices<FastWaveConvergence>::nvarTotal_cc;
	amrex::Vector<amrex::BCRec> BCs_cc(ncomp_cc);
	for (int n = 0; n < ncomp_cc; ++n) {
		for (int i = 0; i < AMREX_SPACEDIM; ++i) {
			BCs_cc[n].setLo(i, amrex::BCType::int_dir); // periodic
			BCs_cc[n].setHi(i, amrex::BCType::int_dir);
		}
	}

	const int nvars_fc = Physics_Indices<FastWaveConvergence>::nvarTotal_fc;
	amrex::Vector<amrex::BCRec> BCs_fc(nvars_fc);
	for (int icomp = 0; icomp < nvars_fc; ++icomp) {
		for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
			BCs_fc[icomp].setLo(idim, amrex::BCType::int_dir); // periodic
			BCs_fc[icomp].setHi(idim, amrex::BCType::int_dir);
		}
	}

	// Set grid dimensions using AMReX parameter system
	amrex::ParmParse pp("amr");
	amrex::Vector<int> const ncells = {nx, 8, 8};
	pp.add("max_level", 0);
	pp.add("blocking_factor_x", nx);
	pp.add("blocking_factor_y", 8);
	pp.add("blocking_factor_z", 8);
	pp.add("max_grid_size", nx);
	pp.addarr("n_cell", ncells);

	// Set domain bounds using AMReX parameter system
	amrex::ParmParse pp_geom("geometry");
	amrex::Vector<double> const prob_lo = {0.0, 0.0, 0.0};
	amrex::Vector<double> const prob_hi = {1.0, 1.0, 1.0};
	amrex::Vector<int> const is_periodic = {1, 1, 1};
	pp_geom.addarr("prob_lo", prob_lo);
	pp_geom.addarr("prob_hi", prob_hi);
	pp_geom.addarr("is_periodic", is_periodic);

	QuokkaSimulation<FastWaveConvergence> sim(BCs_cc, BCs_fc);

	sim.computeReferenceSolution_ = true;
	sim.cflNumber_ = CFL_number;
	sim.stopTime_ = max_time;
	sim.maxTimesteps_ = max_timesteps;

	// set initial conditions
	sim.setInitialConditions();
	auto [pos_exact, val_exact] = fextract(sim.state_new_cc_[0], sim.geom[0], 0, 0.5);
	// Main time loop
	sim.evolve();
	auto [position, values] = fextract(sim.state_new_cc_[0], sim.geom[0], 0, 0.5);
	int const nx_final = static_cast<int>(position.size());

	// // compute error norm
	amrex::Real err_sq = 0.;
	for (int n = 0; n < QuokkaSimulation<FastWaveConvergence>::ncompHydro_; ++n) {
		if (n == HydroSystem<FastWaveConvergence>::internalEnergy_index) {
			continue;
		}
		amrex::Real dU_k = 0.;
		for (int i = 0; i < nx_final; ++i) {
			// Δ Uk = ∑i |Uk,in - Uk,i0| / Nx
			const amrex::Real U_k0 = val_exact.at(n)[i];
			const amrex::Real U_k1 = values.at(n)[i];
			dU_k += std::abs(U_k1 - U_k0) / static_cast<double>(nx_final);
		}
		// ε = || Δ U || = [&sum_k (Δ Uk)2]^{1/2}
		err_sq += dU_k * dU_k;
	}
	const amrex::Real epsilon = std::sqrt(err_sq);

	return epsilon;
	// const double error = sim.errorNorm_;
	// return error;
}

auto problem_main() -> int
{
	// Richardson convergence test: run at increasing resolution until machine precision is reached
	const double machine_precision_target = 2.0e-11;
	const int nx_initial = 32;
	const int nx_max = 128;
	bool reached_target = false;

	// Silence TinyProfiler so convergence logs stay readable
	{
		amrex::ParmParse pp_tp("tiny_profiler");
		if (!pp_tp.contains("output_file")) {
			pp_tp.add("output_file", std::string("/dev/null"));
		}
	}

	// Suppress per-step logging from the coarse timestep loop
	{
		amrex::ParmParse pp_general;
		if (!pp_general.contains("suppress_output")) {
			pp_general.add("suppress_output", 1);
		}
	}

	amrex::Vector<int> resolutions;
	amrex::Vector<double> errors;
	amrex::Vector<double> dx_values;

	amrex::Print() << "Running Richardson convergence test for FastWave:\n";
	amrex::Print() << "Resolution\tError Norm\n";
	amrex::Print() << "----------\t----------\n";

	for (int nx = nx_initial; nx <= nx_max; nx *= 2) {

		amrex::ParmParse const hpp("setup");

		double angle_between_k_b0_deg = 90.0; // default to 90 degrees
		hpp.query("angle_between_k_b0", angle_between_k_b0_deg);

		constexpr double deg2rad = M_PI / 180.0;
		angle_between_k_b0_rad = deg2rad * angle_between_k_b0_deg;

		// Read k-vector modes
		int num_modes_x = 1;
		int num_modes_y = 0;
		int num_modes_z = 0;
		hpp.query("num_modes_x", num_modes_x);
		hpp.query("num_modes_y", num_modes_y);
		hpp.query("num_modes_z", num_modes_z);

		// Compute k vector and basis
		const std::array<amrex::Real, 3> k_vec_prf = {2.0 * M_PI * static_cast<amrex::Real>(num_modes_x),
							      2.0 * M_PI * static_cast<amrex::Real>(num_modes_y),
							      2.0 * M_PI * static_cast<amrex::Real>(num_modes_z)};

		const double k_magn = computeMagnitude(k_vec_prf);
		k_amplitude = k_magn;
		k_dir_prf = {k_vec_prf[0] / k_magn, k_vec_prf[1] / k_magn, k_vec_prf[2] / k_magn};
		amrex::Print() << "k_dir_prf: (" << k_dir_prf[0] << ", " << k_dir_prf[1] << ", " << k_dir_prf[2] << ")\n";
		amrex::Print() << "inplane_dir_prf: (" << inplane_dir_prf[0] << ", " << inplane_dir_prf[1] << ", " << inplane_dir_prf[2] << ")\n";
		amrex::Print() << "outofplane_dir_prf: (" << outofplane_dir_prf[0] << ", " << outofplane_dir_prf[1] << ", " << outofplane_dir_prf[2] << ")\n";

		// Build orthonormal basis
		std::array<amrex::Real, 3> ref_prf{0.0, 0.0, 1.0};
		if (std::abs(computeDotProduct(ref_prf, k_dir_prf)) > 0.9999) {
			ref_prf = {0.0, 1.0, 0.0};
		}

		inplane_dir_prf = computeCrossProduct(ref_prf, k_dir_prf);
		normalizeVector(inplane_dir_prf);

		outofplane_dir_prf = computeCrossProduct(k_dir_prf, inplane_dir_prf);
		normalizeVector(outofplane_dir_prf);

		// Compute omega with the angle
		const double cos_theta = std::cos(angle_between_k_b0_rad);
		omega = std::sqrt(
		    std::pow(k_magn, 2) / 2.0 *
		    (std::pow(magnetosonic_speed, 2) +
		     std::sqrt(std::pow(magnetosonic_speed, 4) - 4.0 * std::pow(alfven_speed, 2) * std::pow(sound_speed, 2) * std::pow(cos_theta, 2))));

		amrex::Print() << "Fast wave configuration:\n";
		amrex::Print() << "  Angle k-B0: " << angle_between_k_b0_deg << " degrees\n";
		amrex::Print() << "  k magnitude: " << k_magn << "\n";
		amrex::Print() << "  omega: " << omega << "\n";
		amrex::Print() << "  Phase speed: " << omega / k_magn << "\n\n";

		double const error = runWaveTest(nx);

		resolutions.push_back(nx);
		errors.push_back(error);
		dx_values.push_back(1.0 / static_cast<double>(nx)); // dx = L / nx for unit domain

		amrex::Print() << fmt::format("{:10d}\t{:.6e}\n", nx, error);

		if (error <= machine_precision_target) {
			reached_target = true;
			break;
		}

		if (nx == nx_max) {
			amrex::Print() << fmt::format("\nReached maximum resolution (nx = {}) without achieving the target error {:.3e}\n", nx_max,
						      machine_precision_target);
			break;
		}
	}

	// Calculate convergence rates using Richardson extrapolation
	amrex::Print() << "\nConvergence Rate Analysis:\n";
	amrex::Print() << "Resolution Pair\tObserved Rate\tExpected Rate\n";
	amrex::Print() << "---------------\t-------------\t-------------\n";

	bool convergence_passed = true;
	const double expected_rate = 2.0; // PPM should give ~2nd order for smooth problems
	const double tolerance = 0.3;	  // Allow 30% deviation from expected rate

	for (int i = 1; i < resolutions.size(); ++i) {
		// Calculate convergence rate: p = log(E(2h)/E(h)) / log(2)
		double const log_error_ratio = std::log(errors[i - 1] / errors[i]);
		double const log_dx_ratio = std::log(dx_values[i - 1] / dx_values[i]);
		double const observed_rate = log_error_ratio / log_dx_ratio;

		amrex::Print() << fmt::format("{:4d} -> {:4d}\t{:13.2f}\t{:13.1f}\n", resolutions[i - 1], resolutions[i], observed_rate, expected_rate);

		// Check if convergence rate is within acceptable range
		if (observed_rate + tolerance < expected_rate) {
			convergence_passed = false;
		}
	}

	// Calculate overall convergence rate from first to last resolution
	if (resolutions.size() >= 2) {
		double const overall_log_error_ratio = std::log(errors[0] / errors.back());
		double const overall_log_dx_ratio = std::log(dx_values[0] / dx_values.back());
		double const overall_rate = overall_log_error_ratio / overall_log_dx_ratio;

		amrex::Print() << fmt::format("\nOverall convergence rate: {:.2f}\n", overall_rate);
		amrex::Print() << fmt::format("Expected rate: {:.1f}\n", expected_rate);

		if (overall_rate + tolerance < expected_rate) {
			convergence_passed = false;
		}
	}

	// Output results for analysis
	if (amrex::ParallelDescriptor::IOProcessor()) {
		std::ofstream file("fast_wave_convergence.csv");
		file << "nx,dx,error\n";
		for (int i = 0; i < resolutions.size(); ++i) {
			file << fmt::format("{},{:.6e},{:.6e}\n", resolutions[i], dx_values[i], errors[i]);
		}
		file.close();
		amrex::Print() << "\nConvergence data written to fast_wave_convergence.csv\n";
	}

	// Test status
	if (convergence_passed) {
		if (reached_target) {
			amrex::Print() << fmt::format("\n✓ Richardson convergence test PASSED (target error {:.3e} reached)\n", machine_precision_target);
		} else {
			amrex::Print() << "\n✓ Richardson convergence test PASSED\n";
		}
		return 0;
	}

	amrex::Print() << "\n✗ Richardson convergence test FAILED\n";
	amrex::Print() << "Observed convergence rate deviates from expected rate by more than " << tolerance << "\n";
	return 1;
}
