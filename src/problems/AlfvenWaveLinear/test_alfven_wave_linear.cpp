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
constexpr double bg_b_magn = 1.0;
constexpr double delta_b_magn = 1e-6;
constexpr double alfven_speed = bg_b_magn / gcem::sqrt(bg_density);

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
	const double field_magn = std::sqrt(vfield[0] * vfield[0] + vfield[1] * vfield[1] + vfield[2] * vfield[2]);
	if (field_magn > 1e-14) {
		vfield[0] /= field_magn;
		vfield[1] /= field_magn;
		vfield[2] /= field_magn;
	}
}

struct ProblemSetup {
	// angles (radians) in the math reference frame (MRF)
	double angle_between_k_b0 = 0.0;
	double cos_angle_k_b0 = 1.0;
	double sin_angle_k_b0 = 0.0;

	// rotation from the mrf to the prf
	double k_rotation_xy = 0.0;
	double k_elevation_xy = 0.0;

	// MRF expressed in problem refrence frame (PRF)
	std::array<double, 3> k_dir_prf{1.0, 0.0, 0.0};
	std::array<double, 3> inplane_dir_prf{0.0, 1.0, 0.0};
	std::array<double, 3> outofplane_dir_prf{0.0, 0.0, 1.0};

	double k_magn = 2.0 * M_PI;
	double omega = alfven_speed * k_magn;

	// convenient diagnostics: b-field resolved in the PRF
	double bg_mag_x1_prf = bg_b_magn;
	double bg_mag_x2_prf = 0.0;
	double bg_mag_x3_prf = 0.0;
};

AMREX_GPU_MANAGED ProblemSetup problem_setup;

AMREX_GPU_DEVICE AMREX_FORCE_INLINE double computeVectorPotential(const double x1, const double x2, const double x3, const double time, const int component)
{
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(component == 0 || component == 1 || component == 2,
					 "computeVectorPotential(): component must be an integer in {0, 1, 2}");

	// position in the problem reference frame (PRF)
	const std::array<double, 3> x_prf{x1, x2, x3};

	// map PRF vector to MRF coordinates by projecting onto the MRF basis
	// equivalently: x_mrf = transpose(R) * x_prf, where R = [k_dir_prf, inplane_dir_prf, outofplane_dir_prf]
	// the columns of R are the MRF basis vectors expressed in the PRF
	// because R is orthonormal, its inverse is its transpose, so multiplying by R^T is exactly taking dot products
	const double x0_mrf = computeDotProduct(x_prf, problem_setup.k_dir_prf);
	const double x1_mrf = computeDotProduct(x_prf, problem_setup.inplane_dir_prf);
	const double x2_mrf = computeDotProduct(x_prf, problem_setup.outofplane_dir_prf);

	// b_mrf = (b0*cos(theta), b0*sin(theta), 0)
	const double b0_mrf = bg_b_magn * problem_setup.cos_angle_k_b0;
	const double b1_mrf = bg_b_magn * problem_setup.sin_angle_k_b0;
	const double b2_mrf = 0.0;

	// we choose a Coulomb-gauge vector potential to compute the uniform field
	// A_uniform = 0.5 * cross(x, b) -> curl(A_uniform) = B and div(A_uniform) = 0
	const double A0_uniform_mrf = 0.5 * (x2_mrf * b1_mrf - x1_mrf * b2_mrf);
	const double A1_uniform_mrf = 0.5 * (x0_mrf * b2_mrf - x2_mrf * b0_mrf);
	const double A2_uniform_mrf = 0.5 * (x1_mrf * b0_mrf - x0_mrf * b1_mrf);

	// add a linearly polarised Alfven-wave perturbation
	// note, this requires:
	// (1) the phase only depends on the coordinate along k (x0_mrf),
	// (2) delta_b_magn is perpendicular to both k and B0 (out-of-plane),
	// (3) delta v is parallel to deltaB (Alfven polarization).
	// we achieve point (2) by perturbing ONLY the in-plane component: A1 = A1(x0_mrf, t).
	// curl(A_perturb) with A_perturb = A1(x0) * e1 yields delta_b_magn = dA1 / dx0 * e2, which is purely out-of-plane
	// and so the phase = omega * t - |k| * x0_mrf
	const double phase_mrf = problem_setup.omega * time - problem_setup.k_magn * x0_mrf;

	// small-amplitude perturbation: amplitude = b0 * delta_b_magn / |k| where delta_b_magn << b0
	const double A1_perturb_mrf = -(bg_b_magn * delta_b_magn / problem_setup.k_magn) * std::sin(phase_mrf);

	// total vector potential
	const double A0_mrf = A0_uniform_mrf;
	const double A1_mrf = A1_uniform_mrf + A1_perturb_mrf;
	const double A2_mrf = A2_uniform_mrf;

	// map the vector potential back to the PRF
	const double A0_prf = A0_mrf * problem_setup.k_dir_prf[0] + A1_mrf * problem_setup.inplane_dir_prf[0] + A2_mrf * problem_setup.outofplane_dir_prf[0];
	const double A1_prf = A0_mrf * problem_setup.k_dir_prf[1] + A1_mrf * problem_setup.inplane_dir_prf[1] + A2_mrf * problem_setup.outofplane_dir_prf[1];
	const double A2_prf = A0_mrf * problem_setup.k_dir_prf[2] + A1_mrf * problem_setup.inplane_dir_prf[2] + A2_mrf * problem_setup.outofplane_dir_prf[2];

	if (component == 0) {
		return A0_prf;
	}
	if (component == 1) {
		return A1_prf;
	}
	if (component == 2) {
		return A2_prf;
	}
}

AMREX_GPU_DEVICE inline double Ax(const double x1, const double x2, const double x3, const double t) { return computeVectorPotential(x1, x2, x3, t, 0); }

AMREX_GPU_DEVICE inline double Ay(const double x1, const double x2, const double x3, const double t) { return computeVectorPotential(x1, x2, x3, t, 1); }

AMREX_GPU_DEVICE inline double Az(const double x1, const double x2, const double x3, const double t) { return computeVectorPotential(x1, x2, x3, t, 2); }

AMREX_GPU_DEVICE
void computeWaveSolution(int i, int j, int k, amrex::Array4<amrex::Real> const &state, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx,
			 amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_lo, quokka::centering cen, quokka::direction dir, double time)
{
	const amrex::Real x1_L = prob_lo[0] + i * dx[0];
	const amrex::Real x2_L = prob_lo[1] + j * dx[1];
	const amrex::Real x3_L = prob_lo[2] + k * dx[2];

	const amrex::Real x1_C = x1_L + static_cast<amrex::Real>(0.5) * dx[0];
	const amrex::Real x2_C = x2_L + static_cast<amrex::Real>(0.5) * dx[1];
	const amrex::Real x3_C = x3_L + static_cast<amrex::Real>(0.5) * dx[2];

	if (cen == quokka::centering::cc) {
		const std::array<double, 3> xC_prf{x1_C, x2_C, x3_C};
		const double x0_mrf = computeDotProduct(xC_prf, problem_setup.k_dir_prf);
		const double phase_mrf = problem_setup.omega * time - problem_setup.k_magn * x0_mrf;
		const double cos_phase_mrf = std::cos(phase_mrf);

		double delta_v = 0.0;
		if (std::abs(problem_setup.cos_angle_k_b0) > 1e-14) {
			delta_v = -problem_setup.omega * delta_b_magn / (sound_speed * problem_setup.k_magn * problem_setup.cos_angle_k_b0) * cos_phase_mrf;
		}
		const double v_x1 = delta_v * problem_setup.outofplane_dir_prf[0];
		const double v_x2 = delta_v * problem_setup.outofplane_dir_prf[1];
		const double v_x3 = delta_v * problem_setup.outofplane_dir_prf[2];

		const double b0_x1 =
		    bg_b_magn * (problem_setup.cos_angle_k_b0 * problem_setup.k_dir_prf[0] + problem_setup.sin_angle_k_b0 * problem_setup.inplane_dir_prf[0]);
		const double b0_x2 =
		    bg_b_magn * (problem_setup.cos_angle_k_b0 * problem_setup.k_dir_prf[1] + problem_setup.sin_angle_k_b0 * problem_setup.inplane_dir_prf[1]);
		const double b0_x3 =
		    bg_b_magn * (problem_setup.cos_angle_k_b0 * problem_setup.k_dir_prf[2] + problem_setup.sin_angle_k_b0 * problem_setup.inplane_dir_prf[2]);

		const double delta_b_x1 = bg_b_magn * delta_b_magn * cos_phase_mrf * problem_setup.outofplane_dir_prf[0];
		const double delta_b_x2 = bg_b_magn * delta_b_magn * cos_phase_mrf * problem_setup.outofplane_dir_prf[1];
		const double delta_b_x3 = bg_b_magn * delta_b_magn * cos_phase_mrf * problem_setup.outofplane_dir_prf[2];

		const double b_x1 = b0_x1 + delta_b_x1;
		const double b_x2 = b0_x2 + delta_b_x2;
		const double b_x3 = b0_x3 + delta_b_x3;

		const double density = bg_density;
		const double pressure = bg_pressure;

		const double v_magn_sq = v_x1 * v_x1 + v_x2 * v_x2 + v_x3 * v_x3;
		const double b_magn_sq = b_x1 * b_x1 + b_x2 * b_x2 + b_x3 * b_x3;
		const double Ekin = 0.5 * density * v_magn_sq;
		const double Emag = 0.5 * b_magn_sq;
		const double Eint = pressure / (gamma_gas - 1);
		const double Etot = Ekin + Emag + Eint;

		state(i, j, k, HydroSystem<AlfvenWaveLinear>::density_index) = density;
		state(i, j, k, HydroSystem<AlfvenWaveLinear>::x1Momentum_index) = v_x1 * density;
		state(i, j, k, HydroSystem<AlfvenWaveLinear>::x2Momentum_index) = v_x2 * density;
		state(i, j, k, HydroSystem<AlfvenWaveLinear>::x3Momentum_index) = v_x3 * density;
		state(i, j, k, HydroSystem<AlfvenWaveLinear>::energy_index) = Etot;
		state(i, j, k, HydroSystem<AlfvenWaveLinear>::internalEnergy_index) = Eint;
	} else if (cen == quokka::centering::fc) {
		const double b_x1 = (Az(x1_L, x2_L + dx[1], x3_L + dx[2] / 2.0, time) - Az(x1_L, x2_L, x3_L + dx[2] / 2.0, time)) / dx[1] -
				    (Ay(x1_L, x2_L + dx[1] / 2.0, x3_L + dx[2], time) - Ay(x1_L, x2_L + dx[1] / 2.0, x3_L, time)) / dx[2];

		const double b_x2 = (Ax(x1_L + dx[0] / 2.0, x2_L, x3_L + dx[2], time) - Ax(x1_L + dx[0] / 2.0, x2_L, x3_L, time)) / dx[2] -
				    (Az(x1_L + dx[0], x2_L, x3_L + dx[2] / 2.0, time) - Az(x1_L, x2_L, x3_L + dx[2] / 2.0, time)) / dx[0];

		const double b_x3 = (Ay(x1_L + dx[0], x2_L + dx[1] / 2.0, x3_L, time) - Ay(x1_L, x2_L + dx[1] / 2.0, x3_L, time)) / dx[0] -
				    (Ax(x1_L + dx[0] / 2.0, x2_L + dx[1], x3_L, time) - Ax(x1_L + dx[0] / 2.0, x2_L, x3_L, time)) / dx[1];

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

	double angle_between_k_b0_degrees = 0.0;
	hpp.query("angle_between_k_b0", angle_between_k_b0_degrees);

	// we assume box length of 1.0
	int num_modes_x = 0;
	int num_modes_y = 0;
	int num_modes_z = 0;
	hpp.query("num_modes_x", num_modes_x);
	hpp.query("num_modes_y", num_modes_y);
	hpp.query("num_modes_z", num_modes_z);

	if ((num_modes_x == 0) && (num_modes_y == 0) && (num_modes_z == 0)) {
		amrex::Abort("Invalid k modes: the triplet (0,0,0) is not allowed.");
	}

	const std::array<double, 3> k_vec_prf = {2.0 * M_PI * static_cast<double>(num_modes_x), 2.0 * M_PI * static_cast<double>(num_modes_y),
						 2.0 * M_PI * static_cast<double>(num_modes_z)};
	const double k_magn = std::sqrt(k_vec_prf[0] * k_vec_prf[0] + k_vec_prf[1] * k_vec_prf[1] + k_vec_prf[2] * k_vec_prf[2]);
	std::array<double, 3> k_dir_prf = {k_vec_prf[0] / k_magn, k_vec_prf[1] / k_magn, k_vec_prf[2] / k_magn};

	const double k_rotation_xy = std::atan2(k_dir_prf[1], k_dir_prf[0]);
	const double k_elevation_xy = std::atan2(k_dir_prf[2], std::hypot(k_dir_prf[0], k_dir_prf[1]));

	constexpr double deg2rad = M_PI / 180.0;
	const double angle_between_k_b0 = deg2rad * angle_between_k_b0_degrees;
	const double cos_angle_k_b0 = std::cos(angle_between_k_b0);
	const double sin_angle_k_b0 = std::sin(angle_between_k_b0);

	const double omega = alfven_speed * k_magn * cos_angle_k_b0;

	std::array<double, 3> ref_prf{0.0, 0.0, 1.0};
	if (std::abs(computeDotProduct(ref_prf, k_dir_prf)) > 0.9999)
		ref_prf = {1.0, 0.0, 0.0};

	std::array<double, 3> inplane_dir_prf = computeCrossProduct(ref_prf, k_dir_prf);
	normaliseVector(inplane_dir_prf);

	std::array<double, 3> outofplane_dir_prf = computeCrossProduct(k_dir_prf, inplane_dir_prf);
	normaliseVector(outofplane_dir_prf);

	problem_setup.angle_between_k_b0 = angle_between_k_b0;
	problem_setup.k_rotation_xy = k_rotation_xy;
	problem_setup.k_elevation_xy = k_elevation_xy;
	problem_setup.cos_angle_k_b0 = cos_angle_k_b0;
	problem_setup.sin_angle_k_b0 = sin_angle_k_b0;
	problem_setup.k_magn = k_magn;
	problem_setup.omega = omega;
	problem_setup.k_dir_prf = k_dir_prf;
	problem_setup.inplane_dir_prf = inplane_dir_prf;
	problem_setup.outofplane_dir_prf = outofplane_dir_prf;

	problem_setup.bg_mag_x1_prf = bg_b_magn * (cos_angle_k_b0 * k_dir_prf[0] + sin_angle_k_b0 * inplane_dir_prf[0]);
	problem_setup.bg_mag_x2_prf = bg_b_magn * (cos_angle_k_b0 * k_dir_prf[1] + sin_angle_k_b0 * inplane_dir_prf[1]);
	problem_setup.bg_mag_x3_prf = bg_b_magn * (cos_angle_k_b0 * k_dir_prf[2] + sin_angle_k_b0 * inplane_dir_prf[2]);

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
