//==============================================================================
// Copyright 2026 Neco Kriel.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file testViscousSedovShock.cpp
/// \brief Setup a test problem for a spherical blast wave with bulk viscosity.
///

#include <cmath>

#include "AMReX_MultiFab.H"
#include "AMReX_Print.H"

#include "QuokkaSimulation.hpp"
#include "hydro/hydro_system.hpp"

struct ViscousSedovShock {
};

// isothermal EOS by default; set gamma != 1.0 for gamma-law (cs_isothermal is then ignored)
template <> struct quokka::EOS_Traits<ViscousSedovShock> {
	static constexpr double gamma = 1.0;
	static constexpr double cs_isothermal = 1.0; // only used when gamma = 1.0
	static constexpr double mean_molecular_weight = C::m_u;
};

template <> struct Physics_Traits<ViscousSedovShock> {
	static constexpr bool is_self_gravity_enabled = false;
	static constexpr bool is_hydro_enabled = true;
	static constexpr int numMassScalars = 0;
	static constexpr int numPassiveScalars = numMassScalars + 0;
	static constexpr bool is_radiation_enabled = false;
	static constexpr bool is_dust_enabled = false;
	static constexpr int nDustGroups = 1;
	static constexpr bool is_mhd_enabled = false;
	static constexpr int nGroups = 1;
	static constexpr UnitSystem unit_system = UnitSystem::CGS;
};

template <> void QuokkaSimulation<ViscousSedovShock>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = grid_elem.dx_;
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = grid_elem.prob_lo_;
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const amrex::Array4<double> &state_cc = grid_elem.array_;

	constexpr double background_density = 1.0;
	constexpr double blast_energy = 1.0;
	constexpr double gamma = quokka::EOS_Traits<ViscousSedovShock>::gamma;
	const double blast_radius = 3.5 * dx[0];
	const double blast_volume = (4.0 / 3.0) * M_PI * blast_radius * blast_radius * blast_radius;

	// isothermal: blast energy -> density enhancement
	// gamma-law: blast energy -> pressure enhancement
	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		const double x_mid = prob_lo[0] + (i + 0.5) * dx[0];
		const double y_mid = prob_lo[1] + (j + 0.5) * dx[1];
		const double z_mid = prob_lo[2] + (k + 0.5) * dx[2];
		const double radius = std::sqrt(x_mid * x_mid + y_mid * y_mid + z_mid * z_mid);
		const bool inside_blast = radius < blast_radius;

		double density = NAN;
		double internal_energy = NAN;
		if constexpr (gamma == 1.0) {
			// isothermal: pressure = rho * cs^2, so energy state is unused
			constexpr double sound_speed = quokka::EOS_Traits<ViscousSedovShock>::cs_isothermal;
			const double blast_density = blast_energy / (sound_speed * sound_speed * blast_volume);
			density = inside_blast ? blast_density : background_density;
			internal_energy = 0.0;
		} else {
			// gamma-law: uniform density, pressure jump encodes blast energy
			const double blast_pressure = blast_energy * (gamma - 1.0) / blast_volume;
			constexpr double background_pressure = 1.0e-4; // low background pressure for gamma-law
			const double pressure = inside_blast ? blast_pressure : background_pressure;
			density = background_density;
			internal_energy = pressure / (gamma - 1.0);
		}

		state_cc(i, j, k, HydroSystem<ViscousSedovShock>::density_index) = density;
		state_cc(i, j, k, HydroSystem<ViscousSedovShock>::x1Momentum_index) = 0.0;
		state_cc(i, j, k, HydroSystem<ViscousSedovShock>::x2Momentum_index) = 0.0;
		state_cc(i, j, k, HydroSystem<ViscousSedovShock>::x3Momentum_index) = 0.0;
		// total energy = internal energy since velocity is zero (no kinetic energy)
		state_cc(i, j, k, HydroSystem<ViscousSedovShock>::energy_index) = internal_energy;
		state_cc(i, j, k, HydroSystem<ViscousSedovShock>::internalEnergy_index) = internal_energy;
	});
}

auto problem_main() -> int
{
	QuokkaSimulation<ViscousSedovShock> sim;

	sim.setInitialConditions();
	const amrex::Real mass_init = sim.state_new_cc_[0].sum(HydroSystem<ViscousSedovShock>::density_index);

	sim.evolve();
	const amrex::Real mass_final = sim.state_new_cc_[0].sum(HydroSystem<ViscousSedovShock>::density_index);

	// bulk viscosity is in conservative form: total mass must be exactly preserved
	const amrex::Real mass_error = std::abs(mass_final - mass_init) / mass_init;
	amrex::Print() << "Mass conservation error: " << mass_error << "\n";

	constexpr double mass_tol = 1.0e-10;
	const bool test_failed = mass_error > mass_tol;
	if (test_failed) {
		amrex::Print() << "FAILED: mass conservation error " << mass_error << " exceeds tolerance " << mass_tol << "\n";
	} else {
		amrex::Print() << "PASSED.\n";
	}

	return static_cast<int>(test_failed);
}
