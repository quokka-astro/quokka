//==============================================================================
// Copyright 2026 Neco Kriel.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file testSmallScaleDynamo.cpp
/// \brief Small Scale Dynamo simulation.
///

#include "AMReX_Print.H"
#include "QuokkaSimulation.hpp"
#include "hydro/hydro_system.hpp"
#include "hydro/mhd_system.hpp"
#include "turbulence/TurbulentDriving.hpp"
#include "util/BC.hpp"

#include "AMReX_FabArray.H"
#include "AMReX_Geometry.H"
#include "AMReX_GpuDevice.H"
#include "AMReX_MultiFab.H"
#include "AMReX_REAL.H"
#include <cmath>

struct SmallScaleDynamo {
};

// isothermal EOS: pressure = cs^2 * rho (no thermal energy equation)
template <> struct quokka::EOS_Traits<SmallScaleDynamo> {
	static constexpr double gamma = 1.0;
	static constexpr double cs_isothermal = 1.0; // dimensionless sound speed
	static constexpr double mean_molecular_weight = C::m_u;
};

template <> struct Physics_Traits<SmallScaleDynamo> {
	static constexpr bool is_hydro_enabled = true;
	static constexpr bool is_radiation_enabled = false;
	static constexpr bool is_mhd_enabled = true;
	static constexpr bool is_self_gravity_enabled = false;
	static constexpr bool is_dust_enabled = false;
	static constexpr int nDustGroups = 0;
	static constexpr int numMassScalars = 0;
	static constexpr int numPassiveScalars = 0;
	static constexpr int nGroups = 1;
	static constexpr UnitSystem unit_system = UnitSystem::CONSTANTS;
	static constexpr double boltzmann_constant = C::k_B;
	static constexpr amrex::Real gravitational_constant = 1.0;
};

// isothermal EOS has no internal energy to reconstruct; pressure computed directly from rho
template <> struct HydroSystem_Traits<SmallScaleDynamo> {
	static constexpr bool reconstruct_eint = false;
};

// uniform density, zero velocity; turbulent driving (configured in input file) stirs the gas
template <> void QuokkaSimulation<SmallScaleDynamo>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const amrex::Array4<double> &state_cc = grid_elem.array_;

	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		state_cc(i, j, k, HydroSystem<SmallScaleDynamo>::density_index) = 1.0;
		state_cc(i, j, k, HydroSystem<SmallScaleDynamo>::x1Momentum_index) = 0.0;
		state_cc(i, j, k, HydroSystem<SmallScaleDynamo>::x2Momentum_index) = 0.0;
		state_cc(i, j, k, HydroSystem<SmallScaleDynamo>::x3Momentum_index) = 0.0;
		state_cc(i, j, k, HydroSystem<SmallScaleDynamo>::energy_index) = 0.0;
		state_cc(i, j, k, HydroSystem<SmallScaleDynamo>::internalEnergy_index) = 0.0;
	});
}

// ABC seed field via discrete curl of the vector potential A = A0*[sin(k0*z)+cos(k0*y), sin(k0*x)+cos(k0*z), sin(k0*y)+cos(k0*x)].
// B = curl(A) is computed via finite differences at face positions, ensuring div(B) = 0 to machine precision.
// k0 = 2*pi*n/L where n = problem.seed_b_wavenumber (default 2).
// B0 = sqrt(seed_b_fraction) * target_vdisp, where seed_b_fraction = E_mag/E_kin (default 0, i.e. no seed field).
template <> void QuokkaSimulation<SmallScaleDynamo>::setInitialConditionsOnGridFaceVars(quokka::grid const &grid_elem)
{
	amrex::ParmParse const pp("problem");
	int seed_b_wavenumber = 2;
	double seed_b_fraction = 0.0;
	pp.query("seed_b_wavenumber", seed_b_wavenumber);
	pp.query("seed_b_fraction", seed_b_fraction);

	double target_vdisp = 0.0;
	amrex::ParmParse const pp_turb("turbulence");
	pp_turb.query("target_vdisp", target_vdisp);

	const amrex::Array4<double> &state_fc = grid_elem.array_;
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const quokka::direction dir = grid_elem.dir_;
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = grid_elem.dx_;
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = grid_elem.prob_lo_;
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_hi = grid_elem.prob_hi_;

	const double Lx = prob_hi[0] - prob_lo[0];
	const double k0 = 2.0 * M_PI * seed_b_wavenumber / Lx;
	const double B0 = std::sqrt(seed_b_fraction) * target_vdisp;
	const double A0 = B0 / k0;

	constexpr int mhd_index = Physics_Indices<SmallScaleDynamo>::mhdFirstIndex;

	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		if (dir == quokka::direction::x) {
			// x-face: B_x = dA_z/dy - dA_y/dz
			const double x_f = prob_lo[0] + i * dx[0];
			const double y_hi = prob_lo[1] + (j + 1) * dx[1];
			const double y_lo = prob_lo[1] + j * dx[1];
			const double z_hi = prob_lo[2] + (k + 1) * dx[2];
			const double z_lo = prob_lo[2] + k * dx[2];
			const double Az_yhi = A0 * (std::sin(k0 * y_hi) + std::cos(k0 * x_f));
			const double Az_ylo = A0 * (std::sin(k0 * y_lo) + std::cos(k0 * x_f));
			const double Ay_zhi = A0 * (std::sin(k0 * x_f) + std::cos(k0 * z_hi));
			const double Ay_zlo = A0 * (std::sin(k0 * x_f) + std::cos(k0 * z_lo));
			state_fc(i, j, k, mhd_index) = (Az_yhi - Az_ylo) / dx[1] - (Ay_zhi - Ay_zlo) / dx[2];
		} else if (dir == quokka::direction::y) {
			// y-face: B_y = dA_x/dz - dA_z/dx
			const double y_f = prob_lo[1] + j * dx[1];
			const double z_hi = prob_lo[2] + (k + 1) * dx[2];
			const double z_lo = prob_lo[2] + k * dx[2];
			const double x_hi = prob_lo[0] + (i + 1) * dx[0];
			const double x_lo = prob_lo[0] + i * dx[0];
			const double Ax_zhi = A0 * (std::sin(k0 * z_hi) + std::cos(k0 * y_f));
			const double Ax_zlo = A0 * (std::sin(k0 * z_lo) + std::cos(k0 * y_f));
			const double Az_xhi = A0 * (std::sin(k0 * y_f) + std::cos(k0 * x_hi));
			const double Az_xlo = A0 * (std::sin(k0 * y_f) + std::cos(k0 * x_lo));
			state_fc(i, j, k, mhd_index) = (Ax_zhi - Ax_zlo) / dx[2] - (Az_xhi - Az_xlo) / dx[0];
		} else if (dir == quokka::direction::z) {
			// z-face: B_z = dA_y/dx - dA_x/dy
			const double z_f = prob_lo[2] + k * dx[2];
			const double x_hi = prob_lo[0] + (i + 1) * dx[0];
			const double x_lo = prob_lo[0] + i * dx[0];
			const double y_hi = prob_lo[1] + (j + 1) * dx[1];
			const double y_lo = prob_lo[1] + j * dx[1];
			const double Ay_xhi = A0 * (std::sin(k0 * x_hi) + std::cos(k0 * z_f));
			const double Ay_xlo = A0 * (std::sin(k0 * x_lo) + std::cos(k0 * z_f));
			const double Ax_yhi = A0 * (std::sin(k0 * z_f) + std::cos(k0 * y_hi));
			const double Ax_ylo = A0 * (std::sin(k0 * z_f) + std::cos(k0 * y_lo));
			state_fc(i, j, k, mhd_index) = (Ay_xhi - Ay_xlo) / dx[0] - (Ax_yhi - Ax_ylo) / dx[1];
		}
	});
}

auto problem_main() -> int
{
	amrex::ParmParse const pp("problem");
	int seed_b_wavenumber = 2;
	pp.query("seed_b_wavenumber", seed_b_wavenumber);
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(seed_b_wavenumber > 0, "problem.seed_b_wavenumber must be a positive integer");

	int ampl_auto_adjust = 0;
	amrex::ParmParse const pp_turb("turbulence");
	pp_turb.query("ampl_auto_adjust", ampl_auto_adjust);
	if (ampl_auto_adjust == 1) {
		amrex::Print() << "WARNING: turbulence.ampl_auto_adjust = 1 is set. The driving amplitude will vary over time, "
			          "which makes it harder to interpret the dynamo growth rate. Consider setting ampl_auto_adjust = 0 "
			          "and tuning ampl_factor manually to reach the desired velocity dispersion.\n";
	}

	auto BCs_cc = quokka::BC<SmallScaleDynamo>(quokka::BCType::int_dir);

	const int nvars_fc = Physics_Indices<SmallScaleDynamo>::nvarTotal_fc;
	amrex::Vector<amrex::BCRec> BCs_fc(nvars_fc);
	for (int icomp = 0; icomp < nvars_fc; ++icomp) {
		for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
			BCs_fc[icomp].setLo(idim, amrex::BCType::int_dir);
			BCs_fc[icomp].setHi(idim, amrex::BCType::int_dir);
		}
	}

	QuokkaSimulation<SmallScaleDynamo> sim(BCs_cc, BCs_fc);

	sim.setInitialConditions();

	sim.evolve();

	return 0;
}
