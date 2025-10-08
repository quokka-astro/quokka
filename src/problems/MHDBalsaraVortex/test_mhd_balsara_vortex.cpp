//==============================================================================
// Copyright 2025 Neco Kriel.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file test_hydro_balsara_vortex.cpp
/// \brief hydro (HLLC) Balsara vortex (constant-ρ, prescribed p).
///

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

struct MHDBalsaraVortex {
};

template <> struct quokka::EOS_Traits<MHDBalsaraVortex> {
	static constexpr double gamma = 5.0 / 3.0;
	static constexpr double mean_molecular_weight = C::m_u;
	static constexpr double boltzmann_constant = C::k_B;
};

template <> struct Physics_Traits<MHDBalsaraVortex> {
	static constexpr bool is_hydro_enabled = true;
	static constexpr int numMassScalars = 0;
	static constexpr int numPassiveScalars = numMassScalars + 0;
	static constexpr bool is_self_gravity_enabled = false;
	static constexpr bool is_radiation_enabled = false;
	static constexpr bool is_mhd_enabled = true;
	static constexpr int nGroups = 1;
	static constexpr UnitSystem unit_system = UnitSystem::CGS;
};

// background state
constexpr double gamma_gas = quokka::EOS_Traits<MHDBalsaraVortex>::gamma;
constexpr double bg_density = 1.0;
constexpr double bg_pressure = 1.0;
constexpr double sound_speed = gcem::sqrt(gamma_gas * bg_pressure / bg_density);
// vortex parameters
AMREX_GPU_MANAGED double vortex_Mach = 0.01; // NOLINT
// domain extends over [-5, 5] by default
constexpr double vortex_center_x1 = 0.0;
constexpr double vortex_center_x2 = 0.0;
// drift is off by default
AMREX_GPU_MANAGED double vortex_drift_x1 = 0.0; // NOLINT
AMREX_GPU_MANAGED double vortex_drift_x2 = 0.0; // NOLINT

AMREX_GPU_DEVICE
inline void computeVortexSolution(int i, int j, int k, amrex::Array4<amrex::Real> const &state, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx,
				  amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_lo)
{
	const amrex::Real x1_L = prob_lo[0] + i * dx[0];
	const amrex::Real x2_L = prob_lo[1] + j * dx[1];
	const amrex::Real x1_C = x1_L + 0.5 * dx[0];
	const amrex::Real x2_C = x2_L + 0.5 * dx[1];

	const double delta_x1_from_center = static_cast<double>(x1_C) - vortex_center_x1;
	const double delta_x2_from_center = static_cast<double>(x2_C) - vortex_center_x2;
	const double radius_sq = delta_x1_from_center * delta_x1_from_center + delta_x2_from_center * delta_x2_from_center;
	const double radial_profile = std::exp(0.5 * (1.0 - radius_sq));
	const double radial_profile_sq = radial_profile * radial_profile;

	const double density = bg_density;
	const double vortex_speed = vortex_Mach * sound_speed;
	const double pressure = bg_pressure - 0.5 * density * vortex_speed * vortex_speed * radial_profile_sq;

	const double delta_vel_x1 = -delta_x2_from_center * vortex_speed * radial_profile;
	const double delta_vel_x2 = delta_x1_from_center * vortex_speed * radial_profile;
	const double vel_x1 = vortex_drift_x1 + delta_vel_x1;
	const double vel_x2 = vortex_drift_x2 + delta_vel_x2;
	const double vel_x3 = 0.0;

	const double mom_x1 = density * vel_x1;
	const double mom_x2 = density * vel_x2;
	const double mom_x3 = density * vel_x3;

	const double Eint = pressure / (gamma_gas - 1.0);
	const double Ekin = 0.5 * density * (vel_x1 * vel_x1 + vel_x2 * vel_x2 + vel_x3 * vel_x3);
	const double Etot = Eint + Ekin;

	state(i, j, k, HydroSystem<MHDBalsaraVortex>::density_index) = density;
	state(i, j, k, HydroSystem<MHDBalsaraVortex>::x1Momentum_index) = mom_x1;
	state(i, j, k, HydroSystem<MHDBalsaraVortex>::x2Momentum_index) = mom_x2;
	state(i, j, k, HydroSystem<MHDBalsaraVortex>::x3Momentum_index) = mom_x3;
	state(i, j, k, HydroSystem<MHDBalsaraVortex>::energy_index) = Etot;
	state(i, j, k, HydroSystem<MHDBalsaraVortex>::internalEnergy_index) = Eint;
}

template <> void QuokkaSimulation<MHDBalsaraVortex>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = grid_elem.dx_;
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = grid_elem.prob_lo_;
	const amrex::Array4<double> &state_cc = grid_elem.array_;
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const int ncomp_cc = Physics_Indices<MHDBalsaraVortex>::nvarTotal_cc;

	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		for (int icomp = 0; icomp < ncomp_cc; ++icomp) {
			state_cc(i, j, k, icomp) = 0.0;
		}
		computeVortexSolution(i, j, k, state_cc, dx, prob_lo);
	});
}

template <> void QuokkaSimulation<MHDBalsaraVortex>::setInitialConditionsOnGridFaceVars(quokka::grid const &grid_elem)
{
	// extract grid information
	const amrex::Array4<double> &state_fc = grid_elem.array_;
	const amrex::Box &indexRange = grid_elem.indexRange_;

	const int ncomp_fc = Physics_Indices<MHDBalsaraVortex>::nvarPerDim_fc;
	// loop over the grid and set the initial condition
	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		for (int n = 0; n < ncomp_fc; ++n) {
			state_fc(i, j, k, n) = 0; // fill all b-field quantities with zeros
		}
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
				stateExact(i, j, k, icomp) = 0.0;
			}
			computeVortexSolution(i, j, k, stateExact, dx, prob_lo);
		});
	}
}

auto problem_main() -> int
{
	amrex::ParmParse const hpp("setup");
	
	int advection_int = 0;
	int num_orbits = 1;
	hpp.query("vortex_Mach", vortex_Mach);
	hpp.query("advection", advection_int);
	hpp.query("num_orbits", num_orbits);
	const double vortex_speed = vortex_Mach * sound_speed;
	const bool is_advection_enabled = (advection_int != 0);

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
		const double advection_speed = vortex_speed;
		vortex_drift_x2 = vortex_drift_x1 = advection_speed / std::sqrt(2.0);
		const double length_x1 = sim.geom[0].ProbLength(0);
		const double length_x2 = sim.geom[0].ProbLength(1);
		if (std::abs(length_x1 - length_x2) > 1e-12) {
			amrex::Abort("The domain must be square for advection.");
		}
		const double advection_distance = std::sqrt(length_x1 * length_x1 + length_x2 * length_x2);
		const double advection_duration = advection_distance / vortex_speed;
		stop_time = static_cast<double>(num_orbits) * advection_duration;
	} else {
		vortex_drift_x1 = vortex_drift_x2 = 0.0;
		const double orbital_duration = 2 * M_PI / vortex_speed;
		stop_time = static_cast<double>(num_orbits) * orbital_duration;
	}

	sim.stopTime_ = stop_time;
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
