//==============================================================================
// Copyright 2025 Ben Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file test_field_loop.cpp
/// \brief
///   This problem is based on the test described here:
///   https://www.astro.princeton.edu/~jstone/Athena/tests/field-loop/Field-loop.html
///

#include <cmath>

#include "AMReX_Array.H"
#include "AMReX_Array4.H"
#include "AMReX_REAL.H"

#include "QuokkaSimulation.hpp"
#include "grid.hpp"
#include "hydro/EOS.hpp"
#include "hydro/hydro_system.hpp"
#include "physics_info.hpp"
#include "util/BC.hpp"

struct FieldLoop {
};

template <> struct quokka::EOS_Traits<FieldLoop> {
	static constexpr double gamma = 5. / 3.;
	static constexpr double mean_molecular_weight = C::m_u;
	static constexpr double boltzmann_constant = C::k_B;
};

template <> struct Physics_Traits<FieldLoop> {
	static constexpr bool is_hydro_enabled = true;
	static constexpr int numMassScalars = 0;		     // number of mass scalars
	static constexpr int numPassiveScalars = numMassScalars + 0; // number of passive scalars
	static constexpr bool is_self_gravity_enabled = false;
	static constexpr bool is_radiation_enabled = false;
	static constexpr bool is_mhd_enabled = true;
	static constexpr int nGroups = 1; // number of radiation groups
	static constexpr UnitSystem unit_system = UnitSystem::CGS;
};

constexpr double A = 1.0e-3;
constexpr double R_0 = 0.3;

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

		state_cc(i, j, k, HydroSystem<FieldLoop>::density_index) = rho0;
		state_cc(i, j, k, HydroSystem<FieldLoop>::x1Momentum_index) = rho0 * vx;
		state_cc(i, j, k, HydroSystem<FieldLoop>::x2Momentum_index) = rho0 * vy;
		state_cc(i, j, k, HydroSystem<FieldLoop>::x3Momentum_index) = rho0 * vz;
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
			state_fc(i, j, k, Physics_Indices<FieldLoop>::mhdFirstIndex) = bx;
		} else if (dir == quokka::direction::y) {
			state_fc(i, j, k, Physics_Indices<FieldLoop>::mhdFirstIndex) = by;
		} else if (dir == quokka::direction::z) {
			state_fc(i, j, k, Physics_Indices<FieldLoop>::mhdFirstIndex) = 0;
		}
	});
}

auto problem_main() -> int
{
	auto BCs_cc = quokka::BC<FieldLoop>(amrex::BCType::int_dir); // periodic

	const int nvars_fc = Physics_Indices<FieldLoop>::nvarTotal_fc;
	amrex::Vector<amrex::BCRec> BCs_fc(nvars_fc);
	for (int icomp = 0; icomp < nvars_fc; ++icomp) {
		for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
			BCs_fc[icomp].setLo(idim, amrex::BCType::int_dir); // periodic
			BCs_fc[icomp].setHi(idim, amrex::BCType::int_dir);
		}
	}

	QuokkaSimulation<FieldLoop> sim(BCs_cc, BCs_fc);
	sim.setInitialConditions();
	sim.evolve();
	return 0;
}
