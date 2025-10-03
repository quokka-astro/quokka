//==============================================================================
// Copyright 2025 Neco Kriel.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file test_balsara_vortex_hllc.cpp
/// \brief Hydrodynamic (HLLC) Balsara vortex (B = 0; constant-ρ, prescribed p).
///

#include <cassert>
#include <cmath>
#include <gcem.hpp>

#include "AMReX_Array.H"
#include "AMReX_Array4.H"
#include "AMReX_REAL.H"

#include "QuokkaSimulation.hpp"
#include "grid.hpp"
#include "hydro/EOS.hpp"
#include "physics_info.hpp"
#include "util/BC.hpp"

struct HydroBalsaraVortex {};

template <> struct quokka::EOS_Traits<HydroBalsaraVortex> {
	static constexpr double gamma = 5.0 / 3.0;
	static constexpr double mean_molecular_weight = C::m_u;
	static constexpr double boltzmann_constant = C::k_B;
};

template <> struct Physics_Traits<HydroBalsaraVortex> {
	static constexpr bool is_hydro_enabled = true;
	static constexpr int numMassScalars = 0;
	static constexpr int numPassiveScalars = numMassScalars + 0;
	static constexpr bool is_self_gravity_enabled = false;
	static constexpr bool is_radiation_enabled = false;
	static constexpr bool is_mhd_enabled = false;
	static constexpr int nGroups = 1;
	static constexpr UnitSystem unit_system = UnitSystem::CGS;
};

// background state
constexpr double gamma_gas = quokka::EOS_Traits<HydroBalsaraVortex>::gamma;
constexpr double bg_density = 1.0;
constexpr double bg_pressure = 1.0;
// vortex parameters
constexpr double vortex_speed = 5.0 / (2.0 * M_PI);
constexpr double vortex_x0 = 0.5;
constexpr double vortex_y0 = 0.5;
// drift is off by default
constexpr double vortex_drift_x1 = 0.0;
constexpr double vortex_drift_x2 = 0.0;
constexpr double vortex_drift_x3 = 0.0;

AMREX_GPU_DEVICE
inline void computeVortexSolution(
	int i, int j, int k,
	amrex::Array4<amrex::Real> const &state,
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx,
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_lo)
{
	const amrex::Real x1_L = prob_lo[0] + i * dx[0];
	const amrex::Real x2_L = prob_lo[1] + j * dx[1];
	const amrex::Real x1_C = x1_L + 0.5 * dx[0];
	const amrex::Real x2_C = x2_L + 0.5 * dx[1];

	const double rel_x1 = static_cast<double>(x1_C) - vortex_x0;
	const double rel_x2 = static_cast<double>(x2_C) - vortex_y0;
	const double radius_sq = rel_x1 * rel_x1 + rel_x2 * rel_x2;
	
	const double density  = bg_density;
	const double pressure = bg_pressure - 0.5 * vortex_speed * vortex_speed * gcem::exp(1.0 - radius_sq);

	const double delta_vel_x1 = -rel_x2 * vortex_speed * gcem::exp(0.5 * (1.0 - radius_sq));
	const double delta_vel_x2 = rel_x1 * vortex_speed * gcem::exp(0.5 * (1.0 - radius_sq));
	const double vel_x1 = vortex_drift_x1 + delta_vel_x1;
	const double vel_x2 = vortex_drift_x2 + delta_vel_x2;
	const double vel_x3 = vortex_drift_x3;

	const double mom_x1 = density * vel_x1;
	const double mom_x2 = density * vel_x2;
	const double mom_x3 = density * vel_x3;

	const double Eint = pressure / (gamma_gas - 1.0);
	const double Ekin = 0.5 * density * (vel_x1 * vel_x1 + vel_x2 * vel_x2 + vel_x3 * vel_x3);
	const double Etot = Eint + Ekin;

	state(i, j, k, HydroSystem<HydroBalsaraVortex>::density_index) = density;
	state(i, j, k, HydroSystem<HydroBalsaraVortex>::x1Momentum_index) = mom_x1;
	state(i, j, k, HydroSystem<HydroBalsaraVortex>::x2Momentum_index) = mom_x2;
	state(i, j, k, HydroSystem<HydroBalsaraVortex>::x3Momentum_index) = mom_x3;
	state(i, j, k, HydroSystem<HydroBalsaraVortex>::energy_index) = Etot;
	state(i, j, k, HydroSystem<HydroBalsaraVortex>::internalEnergy_index) = Eint;
}

template <>
void QuokkaSimulation<HydroBalsaraVortex>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = grid_elem.dx_;
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = grid_elem.prob_lo_;
	const amrex::Array4<double> &state_cc = grid_elem.array_;
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const int ncomp_cc = Physics_Indices<HydroBalsaraVortex>::nvarTotal_cc;

	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		for (int icomp = 0; icomp < ncomp_cc; ++icomp) {
			state_cc(i, j, k, icomp) = 0.0;
		}
		computeVortexSolution(i, j, k, state_cc, dx, prob_lo);
	});
}

template <>
void QuokkaSimulation<HydroBalsaraVortex>::computeReferenceSolution(
	amrex::MultiFab &ref,
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx,
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_lo)
{
	for (amrex::MFIter iter(ref); iter.isValid(); ++iter) {
		const amrex::Box &indexRange = iter.validbox();
		auto const &stateExact = ref.array(iter);
		auto const ncomp = ref.nComp();

		amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
			for (int icomp = 0; icomp < ncomp; ++icomp) {
				stateExact(i, j, k, icomp) = 0.0;
			}
			computeVortexSolution(i, j, k, stateExact, dx, prob_lo);
		});
	}
}

auto problem_main() -> int
{
	auto BCs_cc = quokka::BC<HydroBalsaraVortex>(quokka::BCType::int_dir); // periodic

	QuokkaSimulation<HydroBalsaraVortex> sim(BCs_cc);
	sim.computeReferenceSolution_ = true;

	sim.setInitialConditions();
	sim.evolve();

	int status = 0;
	const double error_tol = 0.002;
	if (sim.errorNorm_ > error_tol) {
		status = 1;
	}
	return status;
}
