//==============================================================================
// Copyright 2025 Neco Kriel.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file testMHDBalsaraVortex.cpp
/// \brief advecting MHD Balsara vortex.
///

#include <cassert>
#include <cmath>
#include <gcem.hpp>
#include <numbers>

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

struct MHDBalsaraVortex {
};

template <> struct quokka::EOS_Traits<MHDBalsaraVortex> {
	static constexpr double gamma = 5.0 / 3.0;
	static constexpr double mean_molecular_weight = C::m_u;
};

template <> struct Physics_Traits<MHDBalsaraVortex> : DefaultPhysicsTraits {
	static constexpr bool is_hydro_enabled = true;
	static constexpr bool is_mhd_enabled = true;
};

// background state
constexpr double gamma_gas = quokka::EOS_Traits<MHDBalsaraVortex>::gamma;
constexpr double bg_density = 1.0;
constexpr double bg_pressure = 1.0;
constexpr double sound_speed = gcem::sqrt(gamma_gas * bg_pressure / bg_density);
// vortex parameters
AMREX_GPU_MANAGED double vortex_radius = 1.0;  // NOLINT
AMREX_GPU_MANAGED double vortex_Mach = 0.01;   // NOLINT
AMREX_GPU_MANAGED double vortex_b_magn = 0.01; // NOLINT
// domain extends over [-5, 5] by default
constexpr double vortex_center_x1 = 0.0;
constexpr double vortex_center_x2 = 0.0;
// no drift by default
AMREX_GPU_MANAGED double vortex_drift_x1 = 0.0; // NOLINT
AMREX_GPU_MANAGED double vortex_drift_x2 = 0.0; // NOLINT

AMREX_GPU_DEVICE AMREX_FORCE_INLINE auto computeRadiusSq(const double x1, const double x2) -> double
{
	const double delta_x1_from_center = x1 - vortex_center_x1;
	const double delta_x2_from_center = x2 - vortex_center_x2;
	const double vortex_radius_inv = 1.0 / vortex_radius;
	const double delta_x1_scaled = delta_x1_from_center * vortex_radius_inv;
	const double delta_x2_scaled = delta_x2_from_center * vortex_radius_inv;
	return delta_x1_scaled * delta_x1_scaled + delta_x2_scaled * delta_x2_scaled; // (r/R)^2
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE auto computeRadialProfile(const double radius_sq) -> double { return std::exp(0.5 * (1.0 - radius_sq)); }

AMREX_GPU_DEVICE AMREX_FORCE_INLINE auto Az(const double x1, const double x2) -> double
{
	// vec(a) = (0, 0, a_z), with:
	//   a_z = b_magn * R * exp((1 - (r/R)^2)/2)
	// so:
	//   b_x =  d(a_z)/dy = -(y/R) * b_magn * exp((1 - (r/R)^2)/2)
	//   b_y = -d(a_z)/dx =  (x/R) * b_magn * exp((1 - (r/R)^2)/2)
	const double radius_sq = computeRadiusSq(x1, x2);
	const double radial_profile = computeRadialProfile(radius_sq);
	return vortex_b_magn * vortex_radius * radial_profile;
}

AMREX_GPU_DEVICE
void computeVortexSolution(int i, int j, int k, amrex::Array4<amrex::Real> const &state, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx,
			   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_lo, quokka::centering cen, quokka::direction dir)
{
	// lower-left corner
	const amrex::Real x1_L = prob_lo[0] + i * dx[0];
	const amrex::Real x2_L = prob_lo[1] + j * dx[1];

	if (cen == quokka::centering::cc) {
		const amrex::Real x1_C = x1_L + static_cast<amrex::Real>(0.5) * dx[0];
		const amrex::Real x2_C = x2_L + static_cast<amrex::Real>(0.5) * dx[1];

		const double radius_sq = computeRadiusSq(x1_C, x2_C);
		const double radial_profile = computeRadialProfile(radius_sq);
		const double radial_profile_sq = radial_profile * radial_profile;
		const double delta_x1_from_center = x1_C - vortex_center_x1;
		const double delta_x2_from_center = x2_C - vortex_center_x2;

		const double vortex_u_magn = vortex_Mach * sound_speed;

		const double pressure =
		    bg_pressure + 0.5 * (vortex_b_magn * vortex_b_magn * (1.0 - radius_sq) - bg_density * vortex_u_magn * vortex_u_magn) * radial_profile_sq;
		const double Eint = pressure / (gamma_gas - 1.0);

		const double delta_u_x1 = -(delta_x2_from_center / vortex_radius) * vortex_u_magn * radial_profile;
		const double delta_u_x2 = (delta_x1_from_center / vortex_radius) * vortex_u_magn * radial_profile;
		const double u_x1 = vortex_drift_x1 + delta_u_x1;
		const double u_x2 = vortex_drift_x2 + delta_u_x2;
		const double u_x3 = 0.0;
		const double mom_x1 = bg_density * u_x1;
		const double mom_x2 = bg_density * u_x2;
		const double mom_x3 = bg_density * u_x3;
		const double Ekin = 0.5 * bg_density * (u_x1 * u_x1 + u_x2 * u_x2 + u_x3 * u_x3);

		const double b_x1 = -(delta_x2_from_center / vortex_radius) * vortex_b_magn * radial_profile;
		const double b_x2 = (delta_x1_from_center / vortex_radius) * vortex_b_magn * radial_profile;
		const double b_x3 = 0.0;
		const double Emag = 0.5 * (b_x1 * b_x1 + b_x2 * b_x2 + b_x3 * b_x3);

		const double Etot = Eint + Ekin + Emag;

		state(i, j, k, HydroSystem<MHDBalsaraVortex>::density_index) = bg_density;
		state(i, j, k, HydroSystem<MHDBalsaraVortex>::x1Momentum_index) = mom_x1;
		state(i, j, k, HydroSystem<MHDBalsaraVortex>::x2Momentum_index) = mom_x2;
		state(i, j, k, HydroSystem<MHDBalsaraVortex>::x3Momentum_index) = mom_x3;
		state(i, j, k, HydroSystem<MHDBalsaraVortex>::energy_index) = Etot;
		state(i, j, k, HydroSystem<MHDBalsaraVortex>::internalEnergy_index) = Eint;

	} else if (cen == quokka::centering::fc) {
		const auto b_x1_L = (Az(x1_L, x2_L + dx[1]) - Az(x1_L, x2_L)) / dx[1];
		const auto b_x2_L = (Az(x1_L, x2_L) - Az(x1_L + dx[0], x2_L)) / dx[0];
		const auto b_x3_L = 0.0;

		if (dir == quokka::direction::x) {
			state(i, j, k, MHDSystem<MHDBalsaraVortex>::bfield_index) = b_x1_L;
		} else if (dir == quokka::direction::y) {
			state(i, j, k, MHDSystem<MHDBalsaraVortex>::bfield_index) = b_x2_L;
		} else if (dir == quokka::direction::z) {
			state(i, j, k, MHDSystem<MHDBalsaraVortex>::bfield_index) = b_x3_L;
		}
	}
}

template <> void QuokkaSimulation<MHDBalsaraVortex>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = grid_elem.dx_;
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = grid_elem.prob_lo_;
	const amrex::Array4<amrex::Real> &state_cc = grid_elem.array_;
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const quokka::centering cen = grid_elem.cen_;
	const quokka::direction dir = grid_elem.dir_;

	const int ncomp_cc = Physics_Indices<MHDBalsaraVortex>::nvarTotal_cc;

	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		for (int icomp = 0; icomp < ncomp_cc; ++icomp) {
			state_cc(i, j, k, icomp) = 0.0; // fill unused quantities with zeros
		}
		computeVortexSolution(i, j, k, state_cc, dx, prob_lo, cen, dir);
	});
}

template <> void QuokkaSimulation<MHDBalsaraVortex>::setInitialConditionsOnGridFaceVars(quokka::grid const &grid_elem)
{
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = grid_elem.dx_;
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = grid_elem.prob_lo_;
	const amrex::Array4<amrex::Real> &state_fc = grid_elem.array_;
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const quokka::centering cen = grid_elem.cen_;
	const quokka::direction dir = grid_elem.dir_;

	const int ncomp_fc = Physics_Indices<MHDBalsaraVortex>::nvarPerDim_fc;

	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		for (int icomp = 0; icomp < ncomp_fc; ++icomp) {
			state_fc(i, j, k, icomp) = 0.0; // fill unused quantities with zeros
		}
		computeVortexSolution(i, j, k, state_fc, dx, prob_lo, cen, dir);
	});
}

template <>
void QuokkaSimulation<MHDBalsaraVortex>::computeReferenceSolution(amrex::MultiFab &ref, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx,
								  amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_lo)
{
	for (amrex::MFIter iter(ref); iter.isValid(); ++iter) {
		const amrex::Box &indexRange = iter.validbox();
		auto const &stateExact = ref.array(iter);
		auto const ncomp = ref.nComp();

		amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
			for (int icomp = 0; icomp < ncomp; ++icomp) {
				stateExact(i, j, k, icomp) = 0.0; // fill unused quantities with zeros
			}
			computeVortexSolution(i, j, k, stateExact, dx, prob_lo, quokka::centering::cc, quokka::direction::na);
		});
	}
}

template <>
void QuokkaSimulation<MHDBalsaraVortex>::computeReferenceSolution_fc(amrex::MultiFab &ref, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx,
								     amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_lo, quokka::direction const dir)
{
	for (amrex::MFIter iter(ref); iter.isValid(); ++iter) {
		const amrex::Box &indexRange = iter.validbox();
		auto const &stateExact = ref.array(iter);
		auto const ncomp = ref.nComp();

		amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
			for (int icomp = 0; icomp < ncomp; ++icomp) {
				stateExact(i, j, k, icomp) = 0.0; // fill unused quantities with zeros
			}
			computeVortexSolution(i, j, k, stateExact, dx, prob_lo, quokka::centering::fc, dir);
		});
	}
}

auto problem_main() -> int
{
	amrex::ParmParse const hpp("setup");

	int advection_int = 0;
	int num_orbits = 1;
	hpp.query("vortex_Mach", vortex_Mach);
	hpp.query("vortex_b_magn", vortex_b_magn);
	hpp.query("advection", advection_int);
	hpp.query("num_orbits", num_orbits);
	const double vortex_u_magn = vortex_Mach * sound_speed;
	const bool is_advection_enabled = (advection_int != 0);
	if (vortex_radius <= 0.0) {
		amrex::Abort("vortex_radius must be > 0.");
	}

	auto BCs_cc = quokka::BC<MHDBalsaraVortex>(quokka::BCType::int_dir);

	const int nvars_fc = Physics_Indices<MHDBalsaraVortex>::nvarTotal_fc;
	amrex::Vector<amrex::BCRec> BCs_fc(nvars_fc);
	for (int icomp = 0; icomp < nvars_fc; ++icomp) {
		for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
			BCs_fc[icomp].setLo(idim, amrex::BCType::int_dir);
			BCs_fc[icomp].setHi(idim, amrex::BCType::int_dir);
		}
	}

	QuokkaSimulation<MHDBalsaraVortex> sim(BCs_cc, BCs_fc);

	double stop_time = 0.0;
	if (is_advection_enabled) {
		const double advection_speed = vortex_u_magn;
		vortex_drift_x2 = vortex_drift_x1 = advection_speed / std::numbers::sqrt2;
		const double length_x1 = sim.geom[0].ProbLength(0);
		const double length_x2 = sim.geom[0].ProbLength(1);
		if (std::abs(length_x1 - length_x2) > 1e-12) {
			amrex::Abort("The domain must be square for advection.");
		}
		const double advection_distance = std::sqrt(length_x1 * length_x1 + length_x2 * length_x2);
		const double advection_duration = advection_distance / vortex_u_magn;
		stop_time = static_cast<double>(num_orbits) * advection_duration;
	} else {
		vortex_drift_x1 = vortex_drift_x2 = 0.0;
		const double orbital_duration = 2.0 * std::numbers::pi / vortex_u_magn;
		stop_time = static_cast<double>(num_orbits) * orbital_duration;
	}

	sim.stopTime_ = stop_time;

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
