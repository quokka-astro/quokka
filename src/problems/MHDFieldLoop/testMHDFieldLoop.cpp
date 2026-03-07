//==============================================================================
// Copyright 2025 Ben Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file testMHDFieldLoop.cpp
/// \brief
///   This problem is based on the test described here:
///   https://www.astro.princeton.edu/~jstone/Athena/tests/field-loop/Field-loop.html
///

#include <cmath>

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

struct MHDFieldLoop {
};

AMREX_ENUM(RefineOn, Region, MagneticEnergy); // NOLINT

template <> struct quokka::EOS_Traits<MHDFieldLoop> {
	static constexpr double gamma = 5. / 3.;
	static constexpr double mean_molecular_weight = C::m_u;
	static constexpr double boltzmann_constant = C::k_B;
};

template <> struct Physics_Traits<MHDFieldLoop> {
	static constexpr bool is_hydro_enabled = true;
	static constexpr int numMassScalars = 0;		     // number of mass scalars
	static constexpr int numPassiveScalars = numMassScalars + 0; // number of passive scalars
	static constexpr bool is_self_gravity_enabled = false;
	static constexpr bool is_radiation_enabled = false;
	static constexpr bool is_dust_enabled = false;
	static constexpr int nDustGroups = 1; // number of dust groups
	static constexpr bool is_mhd_enabled = true;
	static constexpr int nGroups = 1; // number of radiation groups
	static constexpr UnitSystem unit_system = UnitSystem::CGS;
};

constexpr double A = 1.0e-3;
constexpr double R_0 = 0.3;

template <> void QuokkaSimulation<MHDFieldLoop>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	// extract grid information
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = grid_elem.dx_;
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = grid_elem.prob_lo_;
	const amrex::Array4<double> &state_cc = grid_elem.array_;
	const amrex::Box &indexRange = grid_elem.indexRange_;

	constexpr double gamma_gas = quokka::EOS_Traits<MHDFieldLoop>::gamma;
	constexpr double rho0 = 1.0;
	constexpr double P0 = 1.0;

	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		const double x = prob_lo[0] + ((i + 0.5) * dx[0]);
		const double y = prob_lo[1] + ((j + 0.5) * dx[1]);

		//  Vx=sin(60 degrees) and Vy=cos(60 degrees)
		const double vx = std::sin(M_PI / 3.0);
		const double vy = std::cos(M_PI / 3.0);
		const double vz = 1.0; // this should not affect the solution!

		const double Ekin = 0.5 * rho0 * (vx * vx + vy * vy);
		const double Eint = P0 / (gamma_gas - 1.0);

		// Az = MAX([A ( R0 - r )],0)
		auto A_z = [=](double x, double y) {
			const double R = std::sqrt(x * x + y * y);
			return std::max(A * (R_0 - R), 0.);
		};
		auto B_x = [=](double xL, double yL) { return (A_z(xL, yL + dx[1]) - A_z(xL, yL)) / dx[1]; };
		auto B_y = [=](double xL, double yL) { return -(A_z(xL + dx[0], yL) - A_z(xL, yL)) / dx[0]; };
		const double bx = 0.5 * (B_x(x - 0.5 * dx[0], y - 0.5 * dx[1]) + B_x(x + 0.5 * dx[0], y - 0.5 * dx[1]));
		const double by = 0.5 * (B_y(x - 0.5 * dx[0], y - 0.5 * dx[1]) + B_y(x - 0.5 * dx[0], y + 0.5 * dx[1]));
		const double Emag = 0.5 * (bx * bx + by * by);

		state_cc(i, j, k, HydroSystem<MHDFieldLoop>::density_index) = rho0;
		state_cc(i, j, k, HydroSystem<MHDFieldLoop>::x1Momentum_index) = rho0 * vx;
		state_cc(i, j, k, HydroSystem<MHDFieldLoop>::x2Momentum_index) = rho0 * vy;
		state_cc(i, j, k, HydroSystem<MHDFieldLoop>::x3Momentum_index) = rho0 * vz;
		state_cc(i, j, k, HydroSystem<MHDFieldLoop>::internalEnergy_index) = Eint;
		state_cc(i, j, k, HydroSystem<MHDFieldLoop>::energy_index) = Eint + Ekin + Emag;
	});
}

template <> void QuokkaSimulation<MHDFieldLoop>::setInitialConditionsOnGridFaceVars(quokka::grid const &grid_elem)
{
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = grid_elem.dx_;
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = grid_elem.prob_lo_;
	const amrex::Array4<double> &state_fc = grid_elem.array_;
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const quokka::direction dir = grid_elem.dir_;

	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		const double xL = prob_lo[0] + (i * dx[0]);
		const double yL = prob_lo[1] + (j * dx[1]);

		// Az = MAX([A ( R0 - r )],0)
		auto A_z = [=](double x, double y) {
			const double R = std::sqrt(x * x + y * y);
			return std::max(A * (R_0 - R), 0.);
		};
		auto B_x = [=](double xL, double yL) { return (A_z(xL, yL + dx[1]) - A_z(xL, yL)) / dx[1]; };
		auto B_y = [=](double xL, double yL) { return -(A_z(xL + dx[0], yL) - A_z(xL, yL)) / dx[0]; };
		const double bx = B_x(xL, yL);
		const double by = B_y(xL, yL);

		if (dir == quokka::direction::x) {
			state_fc(i, j, k, Physics_Indices<MHDFieldLoop>::mhdFirstIndex) = bx;
		} else if (dir == quokka::direction::y) {
			state_fc(i, j, k, Physics_Indices<MHDFieldLoop>::mhdFirstIndex) = by;
		} else if (dir == quokka::direction::z) {
			state_fc(i, j, k, Physics_Indices<MHDFieldLoop>::mhdFirstIndex) = 0;
		}
	});
}

template <> void QuokkaSimulation<MHDFieldLoop>::refineGrid(int lev, amrex::TagBoxArray &tags, amrex::Real /*time*/, int /*ngrow*/)
{
	RefineOn refine_based_on{};
	amrex::ParmParse const pp("field_loop");
	pp.query("refine_based_on", refine_based_on);

	auto const &dx = geom[lev].CellSizeArray();
	auto const &plo = geom[lev].ProbLoArray();
	auto const &phi = geom[lev].ProbHiArray();

	for (amrex::MFIter mfi(state_new_cc_[lev]); mfi.isValid(); ++mfi) {
		const amrex::Box &box = mfi.validbox();
		const auto tag = tags.array(mfi);
		const auto &Bx_fc = state_new_fc_[lev][0].const_array(mfi);
		const auto &By_fc = state_new_fc_[lev][1].const_array(mfi);
		const auto &Bz_fc = state_new_fc_[lev][2].const_array(mfi);

		if (refine_based_on == RefineOn::Region) {
			// static mesh refinement
			amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
				const double x_frac = ((i + 0.5) * dx[0]) / (phi[0] - plo[0]);
				const double y_frac = ((j + 0.5) * dx[1]) / (phi[1] - plo[1]);

				if (x_frac >= 0.7 && x_frac <= 0.8 && y_frac >= 0.3 && y_frac <= 0.7) {
					tag(i, j, k) = amrex::TagBox::SET;
				}
			});
		} else if (refine_based_on == RefineOn::MagneticEnergy) {
			// refine on magnetic energy density
			constexpr int idx = Physics_Indices<MHDFieldLoop>::mhdFirstIndex;
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

template <> void QuokkaSimulation<MHDFieldLoop>::ComputeDerivedVar(int lev, std::string const &dname, amrex::MultiFab &mf, const int ncomp) const
{
	// compute derived variables and save in 'mf'
	if (dname == "magnetic_divergence") {
		const amrex::Geometry &geom_lev = geom[lev];
		const auto dx = geom_lev.CellSizeArray();
		auto const &state_fc = state_new_fc_[lev];
		auto output = mf.arrays();

		// Get the face-centered magnetic field arrays
		auto const &Bx_arr = state_fc[0].const_arrays();
		auto const &By_arr = state_fc[1].const_arrays();
		auto const &Bz_arr = state_fc[2].const_arrays();

		amrex::ParallelFor(mf, {0, 0, 0}, [=] AMREX_GPU_DEVICE(int box, int i, int j, int k) noexcept {
			// Compute divergence using finite differences
			constexpr int idx = Physics_Indices<MHDFieldLoop>::mhdFirstIndex;
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
	QuokkaSimulation<MHDFieldLoop> sim;
	sim.setInitialConditions();
	sim.evolve();
	return 0;
}
