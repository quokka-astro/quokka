//==============================================================================
// Copyright 2022 Neco Kriel.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file test_fc_quantities.cpp
/// \brief Defines a test problem to make sure face-centred quantities are created correctly.
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
	static constexpr bool is_mhd_enabled = true;
	static constexpr int nGroups = 1;
	static constexpr UnitSystem unit_system = UnitSystem::CGS;
};

constexpr double sound_speed = 1.0;
constexpr double gamma_gas = quokka::EOS_Traits<AlfvenWaveLinear>::gamma;
constexpr double bg_density = 1.0;
constexpr double bg_pressure = sound_speed * sound_speed * bg_density / gamma_gas;
constexpr double b0_magn = 1.0;
constexpr double delta_b_magn = 1e-6;
constexpr double alfven_speed = b0_magn / gcem::sqrt(bg_density);

AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE double computeMagnitude(const std::array<double, 3> &vfield)
{
	return gcem::sqrt(vfield[0] * vfield[0] + vfield[1] * vfield[1] + vfield[2] * vfield[2]);
}

AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE double computeDotProduct(const std::array<double, 3> &vfield1, const std::array<double, 3> &vfield2)
{
	return vfield1[0] * vfield2[0] + vfield1[1] * vfield2[1] + vfield1[2] * vfield2[2];
}

AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE std::array<double, 3> computeCrossProduct(const std::array<double, 3> &vfield1, const std::array<double, 3> &vfield2)
{
	return {vfield1[1] * vfield2[2] - vfield1[2] * vfield2[1], vfield1[2] * vfield2[0] - vfield1[0] * vfield2[2],
		vfield1[0] * vfield2[1] - vfield1[1] * vfield2[0]};
}

AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE void normaliseVector(std::array<double, 3> &vfield)
{
	const double vfield_magn = computeMagnitude(vfield);
	if (vfield_magn > 1e-14) {
		vfield[0] /= vfield_magn;
		vfield[1] /= vfield_magn;
		vfield[2] /= vfield_magn;
	}
}

struct ProblemSetup {
	// angles (radians) in the math reference frame (MRF)
	double angle_between_k_b0_rad = 0.0;
	double cos_angle_between_k_b0 = std::cos(angle_between_k_b0_rad);
	double sin_angle_between_k_b0 = std::sin(angle_between_k_b0_rad);

	// rotation from the problem reference frame (PRF) to the mrf
	double k_rotation_in_xy_rad = 0.0;
	double k_elevation_from_xy_rad = 0.0;

	// MRF expressed in the PRF
	std::array<double, 3> k_dir_prf{1.0, 0.0, 0.0};
	std::array<double, 3> inplane_dir_prf{0.0, 1.0, 0.0};
	std::array<double, 3> outofplane_dir_prf{0.0, 0.0, 1.0};

	// wavefront
	double k_magn = 2.0 * M_PI;
	double omega = alfven_speed * k_magn;
};

AMREX_GPU_MANAGED ProblemSetup ps; // NOLINT

std::array<double,3> rotatePRF2MRF(const std::array<double, 3>& vec_prf)
{
	return {
		vec_prf[0] * ps.k_dir_prf[0] + vec_prf[1] * ps.k_dir_prf[1] + vec_prf[2] * ps.k_dir_prf[2],
		vec_prf[0] * ps.inplane_dir_prf[0] + vec_prf[1] * ps.inplane_dir_prf[1] + vec_prf[2] * ps.inplane_dir_prf[2],
		vec_prf[0] * ps.outofplane_dir_prf[0] + vec_prf[1] * ps.outofplane_dir_prf[1] + vec_prf[2] * ps.outofplane_dir_prf[2]
	};
}

AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE
std::array<double,3> rotateMRF2PRF(const std::array<double, 3>& vec_mrf)
{
	return {
		vec_mrf[0] * ps.k_dir_prf[0] + vec_mrf[1] * ps.inplane_dir_prf[0] + vec_mrf[2] * ps.outofplane_dir_prf[0],
		vec_mrf[0] * ps.k_dir_prf[1] + vec_mrf[1] * ps.inplane_dir_prf[1] + vec_mrf[2] * ps.outofplane_dir_prf[1],
		vec_mrf[0] * ps.k_dir_prf[2] + vec_mrf[1] * ps.inplane_dir_prf[2] + vec_mrf[2] * ps.outofplane_dir_prf[2]
	};
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE auto computeVectorPotentialComponent_prf(const double x1_prf, const double x2_prf, const double x3_prf, const double time, const int icomp)
    -> double
{
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(icomp == 0 || icomp == 1 || icomp == 2, "computeVectorPotentialComponent_prf(): icomp must be an integer in {0, 1, 2}");
	const std::array<double, 3> x_vec_mrf = rotatePRF2MRF({x1_prf, x2_prf, x3_prf});
	const double b0_x1_mrf = b0_magn * ps.cos_angle_between_k_b0;
	const double b0_x2_mrf = b0_magn * ps.sin_angle_between_k_b0;
	// bg_A = (0, 0, b0x * y - b0y * x) -> curl(bg_A) = (b0x, b0y, 0)
	const double bg_A1_mrf = 0.0;
	const double bg_A2_mrf = 0.0;
	const double bg_A3_mrf = b0_x1_mrf * x_vec_mrf[1] - b0_x2_mrf * x_vec_mrf[0];
	const double delta_A1_mrf = 0.0;
	const double delta_A2_mrf = -(b0_magn * delta_b_magn / ps.k_magn) * std::sin(ps.omega * time - ps.k_magn * x_vec_mrf[0]);
	const double delta_A3_mrf = 0.0;
	const double A1_mrf = bg_A1_mrf + delta_A1_mrf;
	const double A2_mrf = bg_A2_mrf + delta_A2_mrf;
	const double A3_mrf = bg_A3_mrf + delta_A3_mrf;
	const std::array<double, 3> A_vec_prf = rotateMRF2PRF({A1_mrf, A2_mrf, A3_mrf});
	return A_vec_prf[icomp];
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
			 amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_lo, quokka::centering cen, quokka::direction dir, double time)
{
	const amrex::Real x1_prf_L = prob_lo[0] + i * dx[0];
	const amrex::Real x2_prf_L = prob_lo[1] + j * dx[1];
	const amrex::Real x3_prf_L = prob_lo[2] + k * dx[2];

	if (cen == quokka::centering::cc) {
		const amrex::Real x1_prf_C = x1_prf_L + static_cast<amrex::Real>(0.5) * dx[0];
		const amrex::Real x2_prf_C = x2_prf_L + static_cast<amrex::Real>(0.5) * dx[1];
		const amrex::Real x3_prf_C = x3_prf_L + static_cast<amrex::Real>(0.5) * dx[2];
		const std::array<double, 3> x_vec_mrf_C = rotatePRF2MRF({x1_prf_C, x2_prf_C, x3_prf_C});

		// this is agnostic to the choice of reference frame: vec(k) dot vec(x) is invariant under rotation
		const double cos_phase = std::cos(ps.omega * time - ps.k_magn * x_vec_mrf_C[0]);

		constexpr double elsasser_sgn = -1.0;
		// equivelant to, but numerically safer than -omega / (k_magn * cos_theta)
		double delta_v_magn = elsasser_sgn * alfven_speed * delta_b_magn * cos_phase;
		
		const double v_x1_prf = delta_v_magn * ps.outofplane_dir_prf[0];
		const double v_x2_prf = delta_v_magn * ps.outofplane_dir_prf[1];
		const double v_x3_prf = delta_v_magn * ps.outofplane_dir_prf[2];

		// background b
		const double b0_x1_prf = b0_magn * (ps.cos_angle_between_k_b0 * ps.k_dir_prf[0] + ps.sin_angle_between_k_b0 * ps.inplane_dir_prf[0]);
		const double b0_x2_prf = b0_magn * (ps.cos_angle_between_k_b0 * ps.k_dir_prf[1] + ps.sin_angle_between_k_b0 * ps.inplane_dir_prf[1]);
		const double b0_x3_prf = b0_magn * (ps.cos_angle_between_k_b0 * ps.k_dir_prf[2] + ps.sin_angle_between_k_b0 * ps.inplane_dir_prf[2]);
		// perturbed b
		const double delta_b_x1_prf = b0_magn * delta_b_magn * cos_phase * ps.outofplane_dir_prf[0];
		const double delta_b_x2_prf = b0_magn * delta_b_magn * cos_phase * ps.outofplane_dir_prf[1];
		const double delta_b_x3_prf = b0_magn * delta_b_magn * cos_phase * ps.outofplane_dir_prf[2];
		// total b
		const double b_x1_prf = b0_x1_prf + delta_b_x1_prf;
		const double b_x2_prf = b0_x2_prf + delta_b_x2_prf;
		const double b_x3_prf = b0_x3_prf + delta_b_x3_prf;

		const double density = bg_density;
		const double pressure = bg_pressure;

		const double v_magn_sq = v_x1_prf * v_x1_prf + v_x2_prf * v_x2_prf + v_x3_prf * v_x3_prf;
		const double b_magn_sq = b_x1_prf * b_x1_prf + b_x2_prf * b_x2_prf + b_x3_prf * b_x3_prf;
		const double Ekin = 0.5 * density * v_magn_sq;
		const double Emag = 0.5 * b_magn_sq;
		const double Eint = pressure / (gamma_gas - 1);
		const double Etot = Ekin + Emag + Eint;

		state(i, j, k, HydroSystem<AlfvenWaveLinear>::density_index) = density;
		state(i, j, k, HydroSystem<AlfvenWaveLinear>::x1Momentum_index) = v_x1_prf * density;
		state(i, j, k, HydroSystem<AlfvenWaveLinear>::x2Momentum_index) = v_x2_prf * density;
		state(i, j, k, HydroSystem<AlfvenWaveLinear>::x3Momentum_index) = v_x3_prf * density;
		state(i, j, k, HydroSystem<AlfvenWaveLinear>::energy_index) = Etot;
		state(i, j, k, HydroSystem<AlfvenWaveLinear>::internalEnergy_index) = Eint;
	} else if (cen == quokka::centering::fc) {
		const double b_x1 = (
				Az_prf(x1_prf_L, x2_prf_L + dx[1], x3_prf_L + dx[2] / 2.0, time) - Az_prf(x1_prf_L, x2_prf_L, x3_prf_L + dx[2] / 2.0, time)
			) / dx[1] - (
				Ay_prf(x1_prf_L, x2_prf_L + dx[1] / 2.0, x3_prf_L + dx[2], time) - Ay_prf(x1_prf_L, x2_prf_L + dx[1] / 2.0, x3_prf_L, time)
			) / dx[2];

		const double b_x2 = (
				Ax_prf(x1_prf_L + dx[0] / 2.0, x2_prf_L, x3_prf_L + dx[2], time) - Ax_prf(x1_prf_L + dx[0] / 2.0, x2_prf_L, x3_prf_L, time)
			) / dx[2] - (
				Az_prf(x1_prf_L + dx[0], x2_prf_L, x3_prf_L + dx[2] / 2.0, time) - Az_prf(x1_prf_L, x2_prf_L, x3_prf_L + dx[2] / 2.0, time)
			) / dx[0];

		const double b_x3 = (
				Ay_prf(x1_prf_L + dx[0], x2_prf_L + dx[1] / 2.0, x3_prf_L, time) - Ay_prf(x1_prf_L, x2_prf_L + dx[1] / 2.0, x3_prf_L, time)
			) / dx[0] - (
				Ax_prf(x1_prf_L + dx[0] / 2.0, x2_prf_L + dx[1], x3_prf_L, time) - Ax_prf(x1_prf_L + dx[0] / 2.0, x2_prf_L, x3_prf_L, time)
			) / dx[1];

		if (dir == quokka::direction::x) {
			state(i, j, k, MHDSystem<AlfvenWaveLinear>::bfield_index) = b_x1;
		} else if (dir == quokka::direction::y) {
			state(i, j, k, MHDSystem<AlfvenWaveLinear>::bfield_index) = b_x2;
		} else if (dir == quokka::direction::z) {
			state(i, j, k, MHDSystem<AlfvenWaveLinear>::bfield_index) = b_x3;
		}
	}
}

template <> void QuokkaSimulation<AlfvenWaveLinear>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = grid_elem.dx_;
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = grid_elem.prob_lo_;
	const amrex::Array4<double> &state_cc = grid_elem.array_;
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const quokka::centering cen = grid_elem.cen_;
	const quokka::direction dir = grid_elem.dir_;

	const int ncomp_cc = Physics_Indices<AlfvenWaveLinear>::nvarTotal_cc;
	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		for (int n = 0; n < ncomp_cc; ++n) {
			state_cc(i, j, k, n) = 0; // fill unused quantities with zeros
		}
		computeWaveSolution(i, j, k, state_cc, dx, prob_lo, cen, dir, 0);
	});
}

template <> void QuokkaSimulation<AlfvenWaveLinear>::setInitialConditionsOnGridFaceVars(quokka::grid const &grid_elem)
{
	// extract grid information
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = grid_elem.dx_;
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = grid_elem.prob_lo_;
	const amrex::Array4<double> &state_fc = grid_elem.array_;
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const quokka::centering cen = grid_elem.cen_;
	const quokka::direction dir = grid_elem.dir_;

	const int ncomp_fc = Physics_Indices<AlfvenWaveLinear>::nvarPerDim_fc;
	// loop over the grid and set the initial condition
	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		for (int n = 0; n < ncomp_fc; ++n) {
			state_fc(i, j, k, n) = 0; // fill unused quantities with zeros
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

		amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
			for (int n = 0; n < ncomp; ++n) {
				stateExact(i, j, k, n) = 0.0; // fill unused quantities with zeros
			}
			computeWaveSolution(i, j, k, stateExact, dx, prob_lo, quokka::centering::cc, quokka::direction::na, 0);
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

		amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
			for (int n = 0; n < ncomp; ++n) {
				stateExact(i, j, k, n) = 0.0; // fill unused quantities with zeros
			}
			computeWaveSolution(i, j, k, stateExact, dx, prob_lo, quokka::centering::fc, dir, 0);
		});
	}
}

auto problem_main() -> int
{
	amrex::ParmParse const hpp("setup");

	double angle_between_k_b0_deg = 0.0;
	hpp.query("angle_between_k_b0", angle_between_k_b0_deg);

	constexpr double deg2rad = M_PI / 180.0;
	const double angle_between_k_b0_rad = deg2rad * angle_between_k_b0_deg;
	const double cos_angle_between_k_b0 = std::cos(angle_between_k_b0_rad);
	const double sin_angle_between_k_b0 = std::sin(angle_between_k_b0_rad);

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
	const std::array<double, 3> k_vec_prf = {
		2.0 * M_PI * static_cast<double>(num_modes_x), 
		2.0 * M_PI * static_cast<double>(num_modes_y),
		2.0 * M_PI * static_cast<double>(num_modes_z)
	};
	const double k_magn = computeMagnitude(k_vec_prf);
	const std::array<double, 3> k_dir_prf = {k_vec_prf[0] / k_magn, k_vec_prf[1] / k_magn, k_vec_prf[2] / k_magn};

	const double k_rotation_in_xy_rad = std::atan2(k_dir_prf[1], k_dir_prf[0]);
	const double k_elevation_from_xy_rad = std::atan2(k_dir_prf[2], std::hypot(k_dir_prf[0], k_dir_prf[1]));

	// note that this is rotation invariant
	const double omega = alfven_speed * k_magn * cos_angle_between_k_b0;

	// to build our orthonormal basis in the problem reference frame (PRF)
	// first choose a vector that is not aligned/parallel with the wave propagation direction
	std::array<double, 3> ref_prf{0.0, 0.0, 1.0}; // guess a direction
	if (std::abs(computeDotProduct(ref_prf, k_dir_prf)) > 0.9999) {
		ref_prf = {0.0, 1.0, 0.0};
	}

	// define the plane in which b0 will sit
	std::array<double, 3> inplane_dir_prf = computeCrossProduct(ref_prf, k_dir_prf);
	normaliseVector(inplane_dir_prf);

	// define the direction the perturbation will be induced
	std::array<double, 3> outofplane_dir_prf = computeCrossProduct(k_dir_prf, inplane_dir_prf);
	normaliseVector(outofplane_dir_prf);

	ps.angle_between_k_b0_rad = angle_between_k_b0_rad;
	ps.cos_angle_between_k_b0 = cos_angle_between_k_b0;
	ps.sin_angle_between_k_b0 = sin_angle_between_k_b0;
	ps.k_rotation_in_xy_rad = k_rotation_in_xy_rad;
	ps.k_elevation_from_xy_rad = k_elevation_from_xy_rad;
	ps.k_dir_prf = k_dir_prf;
	ps.k_magn = k_magn;
	ps.omega = omega;
	ps.inplane_dir_prf = inplane_dir_prf;
	ps.outofplane_dir_prf = outofplane_dir_prf;

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
	const double error_tol = 0.002;
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
