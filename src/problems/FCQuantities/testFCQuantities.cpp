//==============================================================================
// Copyright 2022 Neco Kriel.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file testFCQuantities.cpp
/// \brief Defines a test problem to make sure face-centred quantities are created correctly.
///

#include "hydro/hydro_system.hpp"
#include "hydro/mhd_system.hpp"
#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <format>
#include <limits>
#include <ostream>
#include <stdexcept>
#include <string>
#include <valarray>

#include "AMReX_Array.H"
#include "AMReX_Array4.H"
#include "AMReX_BoxArray.H"
#include "AMReX_DistributionMapping.H"
#include "AMReX_Geometry.H"
#include "AMReX_GpuDevice.H"
#include "AMReX_MultiFab.H"
#include "AMReX_MultiFabUtil.H"
#include "AMReX_Print.H"
#include "AMReX_REAL.H"

#include "QuokkaSimulation.hpp"
#include "grid.hpp"
#include "physics_info.hpp"
#include "util/BC.hpp"

struct FCQuantities {
};

template <> struct quokka::EOS_Traits<FCQuantities> {
	static constexpr double gamma = 5. / 3.;
	static constexpr double mean_molecular_weight = C::m_u;
};

template <> struct Physics_Traits<FCQuantities> : DefaultPhysicsTraits {
	// cell-centred
	static constexpr bool is_hydro_enabled = true;
	// face-centred
	static constexpr bool is_mhd_enabled = true;
};

constexpr double rho0 = 1.0;					     // background density
constexpr double P0 = 1.0 / quokka::EOS_Traits<FCQuantities>::gamma; // background pressure
constexpr double v0 = 0.;					     // background velocity
constexpr double amp = 1.0e-6;					     // perturbation amplitude

AMREX_GPU_DEVICE void computeWaveSolution(int i, int j, int k, amrex::Array4<amrex::Real> const &state, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx,
					  amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_lo)
{
	const amrex::Real x_L = prob_lo[0] + (i + static_cast<amrex::Real>(0.0)) * dx[0];
	const amrex::Real x_R = prob_lo[0] + (i + static_cast<amrex::Real>(1.0)) * dx[0];
	const amrex::Real A = amp;

	const quokka::valarray<double, 3> R = {1.0, -1.0, 1.5}; // right eigenvector of sound wave
	const quokka::valarray<double, 3> U_0 = {rho0, rho0 * v0, P0 / (quokka::EOS_Traits<FCQuantities>::gamma - 1.0) + 0.5 * rho0 * std::pow(v0, 2)};
	const quokka::valarray<double, 3> dU = (A * R / (2.0 * M_PI * dx[0])) * (std::cos(2.0 * M_PI * x_L) - std::cos(2.0 * M_PI * x_R));

	double const rho = U_0[0] + dU[0];
	double const xmom = U_0[1] + dU[1];
	double const Etot = U_0[2] + dU[2];
	double const Eint = Etot - 0.5 * (xmom * xmom) / rho;

	state(i, j, k, HydroSystem<FCQuantities>::density_index) = rho;
	state(i, j, k, HydroSystem<FCQuantities>::x1Momentum_index) = xmom;
	state(i, j, k, HydroSystem<FCQuantities>::x2Momentum_index) = 0;
	state(i, j, k, HydroSystem<FCQuantities>::x3Momentum_index) = 0;
	state(i, j, k, HydroSystem<FCQuantities>::energy_index) = Etot;
	state(i, j, k, HydroSystem<FCQuantities>::internalEnergy_index) = Eint;
}

template <> void QuokkaSimulation<FCQuantities>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	// extract grid information
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const dx = grid_elem.dx_;
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const prob_lo = grid_elem.prob_lo_;
	const amrex::Array4<double> &state = grid_elem.array_;
	const amrex::Box &indexRange = grid_elem.indexRange_;

	const int ncomp_cc = Physics_Indices<FCQuantities>::nvarTotal_cc;
	// loop over the grid and set the initial condition
	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		for (int n = 0; n < ncomp_cc; ++n) {
			state(i, j, k, n) = 0; // fill unused quantities with zeros
		}
		computeWaveSolution(i, j, k, state, dx, prob_lo);
	});
}

template <> void QuokkaSimulation<FCQuantities>::setInitialConditionsOnGridFaceVars(quokka::grid const &grid_elem)
{
	// extract grid information
	const amrex::Array4<double> &state = grid_elem.array_;
	const amrex::Box &indexRange = grid_elem.indexRange_;
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const dx = grid_elem.dx_;
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const prob_lo = grid_elem.prob_lo_;
	const quokka::direction dir = grid_elem.dir_;

	// Use a nodal magnetic vector potential (A_z) so the discrete divergence is identically zero.
	auto const psi = [=] AMREX_GPU_DEVICE(amrex::Real x, amrex::Real y, amrex::Real z) noexcept -> amrex::Real {
		return std::sin(2.0 * M_PI * x) * std::sin(2.0 * M_PI * y) * std::sin(2.0 * M_PI * z);
	};

	if (dir == quokka::direction::x) {
		amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
			amrex::Real const x = prob_lo[0] + static_cast<amrex::Real>(i) * dx[0];
			amrex::Real const y_lo = prob_lo[1] + static_cast<amrex::Real>(j) * dx[1];
			amrex::Real const y_hi = prob_lo[1] + static_cast<amrex::Real>(j + 1) * dx[1];
			amrex::Real const z = prob_lo[2] + static_cast<amrex::Real>(k) * dx[2];
			state(i, j, k, MHDSystem<FCQuantities>::bfield_index) = (psi(x, y_hi, z) - psi(x, y_lo, z)) / dx[1];
		});
	} else if (dir == quokka::direction::y) {
		amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
			amrex::Real const x_lo = prob_lo[0] + static_cast<amrex::Real>(i) * dx[0];
			amrex::Real const x_hi = prob_lo[0] + static_cast<amrex::Real>(i + 1) * dx[0];
			amrex::Real const y = prob_lo[1] + static_cast<amrex::Real>(j) * dx[1];
			amrex::Real const z = prob_lo[2] + static_cast<amrex::Real>(k) * dx[2];
			state(i, j, k, MHDSystem<FCQuantities>::bfield_index) = -(psi(x_hi, y, z) - psi(x_lo, y, z)) / dx[0];
		});
	} else if (dir == quokka::direction::z) {
		amrex::ParallelFor(indexRange,
				   [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept { state(i, j, k, MHDSystem<FCQuantities>::bfield_index) = 0.0; });
	}
}

void setAmrNCell(amrex::Vector<int> const &n_cell)
{
	amrex::ParmParse pp("amr");
	pp.addarr("n_cell", n_cell);
}

void setPlotfileParams(std::string const &prefix)
{
	amrex::ParmParse pp;
	pp.add("plotfile_interval", 1);
	pp.add("plotfile_prefix", prefix);
	pp.add("skip_initial_plotfile", 0);
}

void checkEnforceLimitsPreservesMagneticEnergy()
{
	amrex::Print() << "Checking MHD temperature floor magnetic-energy accounting...\n";

	// create domain
	amrex::Box const domain(amrex::IntVect(AMREX_D_DECL(0, 0, 0)), amrex::IntVect(AMREX_D_DECL(0, 0, 0)));
	amrex::BoxArray const ba(domain);
	amrex::DistributionMapping const dm(ba);
	amrex::RealBox const real_box({AMREX_D_DECL(0.0, 0.0, 0.0)}, {AMREX_D_DECL(1.0, 1.0, 1.0)});
	amrex::Vector<int> is_periodic(AMREX_SPACEDIM, 0);
	amrex::Geometry const geom(domain, &real_box, amrex::CoordSys::cartesian, is_periodic.data());

	// create state_cc
	amrex::MultiFab state_cc(ba, dm, Physics_Indices<FCQuantities>::nvarTotal_cc, 0);
	state_cc.setVal(0.0);

	// create state_fc
	std::array<amrex::MultiFab, AMREX_SPACEDIM> state_fc;
	for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
		state_fc[dir].define(amrex::convert(ba, amrex::IntVect::TheDimensionVector(dir)), dm, Physics_Indices<FCQuantities>::nvarPerDim_fc, 0);
		state_fc[dir].setVal(0.0);
	}

	// set initial conditions
	constexpr amrex::Real rho = 1.0;
	constexpr amrex::Real temp_cold = 1.0;
	constexpr amrex::Real temp_floor = 10.0;
	amrex::Real const Eint_cold = quokka::EOS<FCQuantities>::ComputeEintFromTgas(rho, temp_cold);
	amrex::Real const Eint_floor = quokka::EOS<FCQuantities>::ComputeEintFromTgas(rho, temp_floor);
	amrex::Real const Emag = 10.0 * Eint_floor;
	amrex::Real const Bx = std::sqrt(2.0 * Emag); // By, Bz are zero

	state_cc.setVal(rho, HydroSystem<FCQuantities>::density_index, 1, 0);
	state_cc.setVal(Eint_cold + Emag, HydroSystem<FCQuantities>::energy_index, 1, 0);
	state_cc.setVal(Eint_cold, HydroSystem<FCQuantities>::internalEnergy_index, 1, 0);
	state_fc[0].setVal(Bx, Physics_Indices<FCQuantities>::mhdFirstIndex, 1, 0);

	// Enforce temperature floor:
	// this should raise the temperature from temp_cold to temp_floor,
	// while leaving the magnetic energy unchanged
	HydroSystem<FCQuantities>::EnforceLimits(0.0, 0.0, temp_floor, state_cc, state_fc, geom,
						 [] AMREX_GPU_DEVICE(amrex::Real /*x*/, amrex::Real /*y*/, amrex::Real /*z*/,
								     amrex::Real base_density_floor) noexcept -> amrex::Real { return base_density_floor; });
	amrex::Gpu::streamSynchronizeAll();

	amrex::Real const expected_Etot = Eint_floor + Emag;
	amrex::Real const actual_Etot = state_cc.min(HydroSystem<FCQuantities>::energy_index);
	amrex::Real const Etot_rel_error = std::abs(actual_Etot - expected_Etot) / std::max(std::abs(expected_Etot), 1.0);
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(Etot_rel_error <= 1000.0 * std::numeric_limits<amrex::Real>::epsilon(),
					 std::format("MHD temperature floor total energy mismatch: got {}, expected {}", actual_Etot, expected_Etot));

	amrex::Real const actual_aux_eint = state_cc.min(HydroSystem<FCQuantities>::internalEnergy_index);
	amrex::Real const aux_rel_error = std::abs(actual_aux_eint - Eint_floor) / std::max(std::abs(Eint_floor), 1.0);
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(aux_rel_error <= 1000.0 * std::numeric_limits<amrex::Real>::epsilon(),
					 std::format("MHD auxiliary internal energy floor mismatch: got {}, expected {}", actual_aux_eint, Eint_floor));
}

void checkDivFreeRestart(QuokkaSimulation<FCQuantities> const &sim)
{
	auto const &state_fc = sim.getNewMF_fc();
	auto const &state_cc = sim.getNewMF_cc();

	amrex::Real max_div_ratio = 0.0;
	for (int lev = 0; lev < state_fc.size(); ++lev) {
		amrex::Array<amrex::MultiFab const *, AMREX_SPACEDIM> face_ptrs;
		for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
			face_ptrs[dir] = &state_fc[lev][dir];
		}

		amrex::MultiFab divB(state_cc[lev].boxArray(), state_cc[lev].DistributionMap(), 1, 0);
		amrex::computeDivergence(divB, face_ptrs, sim.geom[lev]);
		amrex::Real const max_div = divB.norm0(0, 0, false);

		amrex::Real max_b = 0.0;
		for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
			max_b = std::max(max_b, state_fc[lev][dir].norm0(MHDSystem<FCQuantities>::bfield_index, 0, false));
		}

		auto const &dx = sim.geom[lev].CellSizeArray();
		amrex::Real const dx_min = std::min({dx[0], dx[1], dx[2]});
		amrex::Real const scale_b = std::max(max_b, static_cast<amrex::Real>(1.0e-30));
		max_div_ratio = std::max(max_div_ratio, dx_min * max_div / scale_b);
	}

	amrex::Real const tolerance = 1000.0 * std::numeric_limits<amrex::Real>::epsilon();
	amrex::Print() << "Max |div B| * dx / |B| = " << max_div_ratio << "\n\n";
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(max_div_ratio <= tolerance,
					 std::format("Face-centered divergence exceeds tolerance: {} > {}", max_div_ratio, tolerance));
}

auto problem_main() -> int
{
	amrex::Vector<int> const coarse_ncells = {64, 32, 16};
	amrex::Vector<int> const fine_ncells = {128, 64, 32};

	setAmrNCell(coarse_ncells);
	setPlotfileParams("fcq_pre");
	QuokkaSimulation<FCQuantities> sim_write;
	checkEnforceLimitsPreservesMagneticEnergy();
	sim_write.setInitialConditions();
	amrex::Print() << "\n";

	setAmrNCell(fine_ncells);
	setPlotfileParams("fcq_post");
	QuokkaSimulation<FCQuantities> sim_restart;
	sim_restart.setChkFile("chk0000000");
	sim_restart.setInitialConditions();
	amrex::Print() << "\n";

	amrex::Print() << "Checking face-centered divergence after restart refinement...\n";
	checkDivFreeRestart(sim_restart);

	return 0;
}
