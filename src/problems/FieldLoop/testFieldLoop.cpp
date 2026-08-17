//==============================================================================
// Copyright 2025 Ben Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file testFieldLoop.cpp
/// \brief
///   This problem is based on the test described here:
///   https://www.astro.princeton.edu/~jstone/Athena/tests/field-loop/Field-loop.html
///

#include <algorithm>
#include <cmath>
#include <limits>

#include "AMReX_Array.H"
#include "AMReX_Array4.H"
#include "AMReX_ParmParse.H"
#include "AMReX_REAL.H"

#include "QuokkaSimulation.hpp"
#include "grid.hpp"
#include "hydro/EOS.hpp"
#include "hydro/hydro_system.hpp"
#include "physics_info.hpp"
#include "util/BC.hpp"

struct FieldLoop {
};

AMREX_ENUM(RefineOn, Region, MagneticEnergy); // NOLINT

template <> struct quokka::EOS_Traits<FieldLoop> {
	static constexpr double gamma = 5. / 3.;
	static constexpr double mean_molecular_weight = C::m_u;
};

template <> struct Physics_Traits<FieldLoop> : DefaultPhysicsTraits {
	static constexpr bool is_hydro_enabled = true;
	static constexpr bool is_mhd_enabled = true;
};

constexpr double A = 1.0e-3;

// loop geometry, set from the `setup.*` TOML group in problem_main()
AMREX_GPU_MANAGED double loop_radius = 0.3;   // NOLINT
AMREX_GPU_MANAGED double loop_center_x = 0.0; // NOLINT
AMREX_GPU_MANAGED double loop_center_y = 0.0; // NOLINT

// in-plane advection velocity (unit speed, angle measured from the y-axis)
AMREX_GPU_MANAGED double advection_vx = 0.0; // NOLINT
AMREX_GPU_MANAGED double advection_vy = 0.0; // NOLINT

// out-of-plane (z) drift; the loop's structure is invariant along z, so this
// exercises the solver's handling of a flow component the true solution lacks
AMREX_GPU_MANAGED double advection_vz = 1.0; // NOLINT

// static refinement region: a 3D box, defaulting to the full domain in z unless set
AMREX_GPU_MANAGED double region_lo_x = 0.4;					 // NOLINT
AMREX_GPU_MANAGED double region_hi_x = 0.6;					 // NOLINT
AMREX_GPU_MANAGED double region_lo_y = -0.23125;				 // NOLINT
AMREX_GPU_MANAGED double region_hi_y = 0.23125;					 // NOLINT
AMREX_GPU_MANAGED double region_lo_z = -std::numeric_limits<double>::infinity(); // NOLINT
AMREX_GPU_MANAGED double region_hi_z = std::numeric_limits<double>::infinity();	 // NOLINT

template <> void QuokkaSimulation<FieldLoop>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	// extract grid information
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = grid_elem.dx_;
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = grid_elem.prob_lo_;
	const amrex::Array4<double> &state_cc = grid_elem.array_;
	const amrex::Box &indexRange = grid_elem.indexRange_;

	constexpr double gamma_gas = quokka::EOS_Traits<FieldLoop>::gamma;
	constexpr double rho0 = 1.0;
	constexpr double P0 = 1.0;

	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		const double x = prob_lo[0] + ((i + 0.5) * dx[0]);
		const double y = prob_lo[1] + ((j + 0.5) * dx[1]);

		const double Ekin = 0.5 * rho0 * (advection_vx * advection_vx + advection_vy * advection_vy + advection_vz * advection_vz);
		const double Eint = P0 / (gamma_gas - 1.0);

		// Az = MAX([A ( loop_radius - r )],0)
		auto A_z = [=](double x, double y) -> double {
			const double dx_c = x - loop_center_x;
			const double dy_c = y - loop_center_y;
			const double R = std::sqrt(dx_c * dx_c + dy_c * dy_c);
			return std::max(A * (loop_radius - R), 0.);
		};
		auto B_x = [=](double xL, double yL) { return (A_z(xL, yL + dx[1]) - A_z(xL, yL)) / dx[1]; };
		auto B_y = [=](double xL, double yL) { return -(A_z(xL + dx[0], yL) - A_z(xL, yL)) / dx[0]; };
		const double bx = 0.5 * (B_x(x - 0.5 * dx[0], y - 0.5 * dx[1]) + B_x(x + 0.5 * dx[0], y - 0.5 * dx[1]));
		const double by = 0.5 * (B_y(x - 0.5 * dx[0], y - 0.5 * dx[1]) + B_y(x - 0.5 * dx[0], y + 0.5 * dx[1]));
		const double Emag = 0.5 * (bx * bx + by * by);

		state_cc(i, j, k, HydroSystem<FieldLoop>::density_index) = rho0;
		state_cc(i, j, k, HydroSystem<FieldLoop>::x1Momentum_index) = rho0 * advection_vx;
		state_cc(i, j, k, HydroSystem<FieldLoop>::x2Momentum_index) = rho0 * advection_vy;
		state_cc(i, j, k, HydroSystem<FieldLoop>::x3Momentum_index) = rho0 * advection_vz;
		state_cc(i, j, k, HydroSystem<FieldLoop>::internalEnergy_index) = Eint;
		state_cc(i, j, k, HydroSystem<FieldLoop>::energy_index) = Eint + Ekin + Emag;
	});
}

template <> void QuokkaSimulation<FieldLoop>::setInitialConditionsOnGridFaceVars(quokka::grid const &grid_elem)
{
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = grid_elem.dx_;
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = grid_elem.prob_lo_;
	const amrex::Array4<double> &state_fc = grid_elem.array_;
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const quokka::direction dir = grid_elem.dir_;

	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		const double xL = prob_lo[0] + (i * dx[0]);
		const double yL = prob_lo[1] + (j * dx[1]);

		// Az = MAX([A ( loop_radius - r )],0)
		auto A_z = [=](double x, double y) -> double {
			const double dx_c = x - loop_center_x;
			const double dy_c = y - loop_center_y;
			const double R = std::sqrt(dx_c * dx_c + dy_c * dy_c);
			return std::max(A * (loop_radius - R), 0.);
		};
		auto B_x = [=](double xL, double yL) { return (A_z(xL, yL + dx[1]) - A_z(xL, yL)) / dx[1]; };
		auto B_y = [=](double xL, double yL) { return -(A_z(xL + dx[0], yL) - A_z(xL, yL)) / dx[0]; };
		const double bx = B_x(xL, yL);
		const double by = B_y(xL, yL);

		if (dir == quokka::direction::x) {
			state_fc(i, j, k, Physics_Indices<FieldLoop>::mhdFirstIndex) = bx;
		} else if (dir == quokka::direction::y) {
			state_fc(i, j, k, Physics_Indices<FieldLoop>::mhdFirstIndex) = by;
		} else if (dir == quokka::direction::z) {
			state_fc(i, j, k, Physics_Indices<FieldLoop>::mhdFirstIndex) = 0;
		}
	});
}

template <> void QuokkaSimulation<FieldLoop>::refineGrid(int lev, amrex::TagBoxArray &tags, amrex::Real /*time*/, int /*ngrow*/)
{
	RefineOn refine_based_on{};
	amrex::ParmParse const pp("setup");
	pp.query("refine_based_on", refine_based_on);

	auto const &dx = geom[lev].CellSizeArray();
	auto const &plo = geom[lev].ProbLoArray();

	for (amrex::MFIter mfi(state_new_cc_[lev]); mfi.isValid(); ++mfi) {
		const amrex::Box &box = mfi.validbox();
		const auto tag = tags.array(mfi);
		const auto &Bx_fc = state_new_fc_[lev][0].const_array(mfi);
		const auto &By_fc = state_new_fc_[lev][1].const_array(mfi);
		const auto &Bz_fc = state_new_fc_[lev][2].const_array(mfi);

		if (refine_based_on == RefineOn::Region) {
			// static mesh refinement within a fixed physical region
			amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
				const double x = plo[0] + ((i + 0.5) * dx[0]);
				const double y = plo[1] + ((j + 0.5) * dx[1]);
				const double z = plo[2] + ((k + 0.5) * dx[2]);

				if (x >= region_lo_x && x <= region_hi_x && y >= region_lo_y && y <= region_hi_y && z >= region_lo_z && z <= region_hi_z) {
					tag(i, j, k) = amrex::TagBox::SET;
				}
			});
		} else if (refine_based_on == RefineOn::MagneticEnergy) {
			// refine on magnetic energy density
			constexpr int idx = Physics_Indices<FieldLoop>::mhdFirstIndex;
			amrex::Real const threshold = 0.5 * A * A;
			amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
				const amrex::Real bx = 0.5 * (Bx_fc(i, j, k, idx) + Bx_fc(i + 1, j, k, idx));
				const amrex::Real by = 0.5 * (By_fc(i, j, k, idx) + By_fc(i, j + 1, k, idx));
				const amrex::Real bz = 0.5 * (Bz_fc(i, j, k, idx) + Bz_fc(i, j, k + 1, idx));
				const amrex::Real Emag = 0.5 * (bx * bx + by * by + bz * bz);
				if (Emag >= threshold) {
					tag(i, j, k) = amrex::TagBox::SET;
				}
			});
		}
	}
}

template <>
void QuokkaSimulation<FieldLoop>::ComputeDerivedVar(int lev, std::string const &dname, amrex::MultiFab &mf, const int ncomp,
						    amrex::MultiFab const & /*state_cc*/, amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> const &state_fc) const
{
	// compute derived variables and save in 'mf'
	if (dname == "magnetic_divergence") {
		const amrex::Geometry &geom_lev = geom[lev];
		const auto dx = geom_lev.CellSizeArray();
		auto output = mf.arrays();

		// Get the face-centered magnetic field arrays
		auto const &Bx_arr = state_fc[0].const_arrays();
		auto const &By_arr = state_fc[1].const_arrays();
		auto const &Bz_arr = state_fc[2].const_arrays();

		amrex::ParallelFor(mf, {0, 0, 0}, [=] AMREX_GPU_DEVICE(int box, int i, int j, int k) noexcept {
			// Compute divergence using finite differences
			constexpr int idx = Physics_Indices<FieldLoop>::mhdFirstIndex;
			amrex::Real const Bx_p = Bx_arr[box](i + 1, j, k, idx);
			amrex::Real const Bx_m = Bx_arr[box](i, j, k, idx);
			amrex::Real const By_p = By_arr[box](i, j + 1, k, idx);
			amrex::Real const By_m = By_arr[box](i, j, k, idx);
			amrex::Real const Bz_p = Bz_arr[box](i, j, k + 1, idx);
			amrex::Real const Bz_m = Bz_arr[box](i, j, k, idx);
			amrex::Real const divB_x = (Bx_p - Bx_m) / dx[0];
			amrex::Real const divB_y = (By_p - By_m) / dx[1];
			amrex::Real const divB_z = (Bz_p - Bz_m) / dx[2];
			output[box](i, j, k, ncomp) = divB_x + divB_y + divB_z;
		});
	}
	amrex::Gpu::streamSynchronizeAll();
}

auto problem_main() -> int
{
	amrex::ParmParse const pp("setup");

	pp.query("loop_radius", loop_radius);
	pp.query("loop_center_x", loop_center_x);
	pp.query("loop_center_y", loop_center_y);
	pp.query("advection_vz", advection_vz);

	RefineOn refine_based_on{};
	pp.query("refine_based_on", refine_based_on);

	pp.query("region_lo_x", region_lo_x);
	pp.query("region_hi_x", region_hi_x);
	pp.query("region_lo_y", region_lo_y);
	pp.query("region_hi_y", region_hi_y);

	if (loop_radius <= 0.0) {
		amrex::Abort("setup.loop_radius must be > 0.");
	}
	if (region_lo_x >= region_hi_x || region_lo_y >= region_hi_y) {
		amrex::Abort("setup.region_lo_* must be < setup.region_hi_* in both dimensions.");
	}

	QuokkaSimulation<FieldLoop> sim;
	// idealized test in non-physical CGS units: disable the default 5 K temperature floor
	sim.tempFloor_ = 0.0;

	// default in-plane advection direction: the domain's own diagonal (x-y extent), so the loop
	// crosses the periodic domain and returns to its starting position; independent of loop_center
	double advection_angle_deg = std::atan2(sim.geom[0].ProbLength(0), sim.geom[0].ProbLength(1)) * 180.0 / M_PI;
	pp.query("advection_angle_deg", advection_angle_deg);
	const double advection_angle_rad = advection_angle_deg * M_PI / 180.0;
	advection_vx = std::sin(advection_angle_rad);
	advection_vy = std::cos(advection_angle_rad);

	// default the region's z-bounds to the full domain, matching the prior (z-unbounded) behavior
	if (pp.query("region_lo_z", region_lo_z) == 0) {
		region_lo_z = sim.geom[0].ProbLo(2);
	}
	if (pp.query("region_hi_z", region_hi_z) == 0) {
		region_hi_z = sim.geom[0].ProbHi(2);
	}
	if (region_lo_z >= region_hi_z) {
		amrex::Abort("setup.region_lo_z must be < setup.region_hi_z.");
	}

	if (refine_based_on == RefineOn::Region) {
		// sanity check: the static refinement region must not overlap the field loop's support in the
		// x-y plane; the loop's structure is z-invariant, so the region's z-bounds don't affect this
		const double closest_x = std::clamp(loop_center_x, region_lo_x, region_hi_x);
		const double closest_y = std::clamp(loop_center_y, region_lo_y, region_hi_y);
		const double dx_c = closest_x - loop_center_x;
		const double dy_c = closest_y - loop_center_y;
		const double dist_to_region = std::sqrt(dx_c * dx_c + dy_c * dy_c);
		if (dist_to_region < loop_radius) {
			amrex::Abort("the static refinement region overlaps the field loop; move setup.region_* or shrink setup.loop_radius.");
		}
	}

	sim.setInitialConditions();
	sim.evolve();
	return 0;
}
