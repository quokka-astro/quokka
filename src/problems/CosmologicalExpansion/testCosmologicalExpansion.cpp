//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file testCosmologicalExpansion.cpp
/// \brief Defines a test problem for pure hydrodynamics with diagonal velocity
/// undergoing just the cosmological expansion.
///

#include "QuokkaSimulation.hpp"
#include "cosmology/Cosmology.hpp"
#include "hydro/hydro_system.hpp"
#include "physics_info.hpp"
#include <array>
#include <cmath>
#include <tuple>

struct ExpansionProblem {};

template <> struct quokka::EOS_Traits<ExpansionProblem> {
	static constexpr amrex::Real gamma = 5.0 / 3.0;
	static constexpr amrex::Real mean_molecular_weight = C::m_u;
};

template <> struct Physics_Traits<ExpansionProblem> : DefaultPhysicsTraits {
	static constexpr bool is_hydro_enabled = true;
	static constexpr bool is_cosmology_enabled = true;

	// Cosmology: flat EdS universe (only gas, no dm)
	static constexpr amrex::Real omega_m = 1.0;
	static constexpr amrex::Real omega_b = 1.0;
	static constexpr amrex::Real omega_dm = 0.0;
	static constexpr amrex::Real omega_r = 0.0;
	static constexpr amrex::Real omega_lambda = 0.0;
	static constexpr amrex::Real hubble_constant = 0.7; // H0 = 70 km/s/Mpc
	static constexpr amrex::Real a_init = 1e-3;
	static constexpr amrex::Real cosmology_dt_limit = 1e-2; // = default; exact source terms, dt only sets the round-off accumulation
};

/// Initial state of the uniform gas.
struct ExpansionIC {		   // default
	amrex::Real rho = 1.0e-30; // comoving density [g cm^-3]
	amrex::Real P = 1.0e-25;   // pressure [erg cm^-3]
	amrex::Real vx = 5.77e4;   // 10^5/sqrt(3): |v| = 1 km/s, equal on the three axes
	amrex::Real vy = 5.77e4;
	amrex::Real vz = 5.77e4;
};

auto readIC() -> ExpansionIC
{
	ExpansionIC ic{}; // all members take their default values
	amrex::ParmParse const pp("problem");
	pp.query("rho0", ic.rho);
	pp.query("P0", ic.P);
	pp.query("vx0", ic.vx);
	pp.query("vy0", ic.vy);
	pp.query("vz0", ic.vz);
	return ic;
}

template <> void QuokkaSimulation<ExpansionProblem>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	const amrex::Real gamma = quokka::EOS_Traits<ExpansionProblem>::gamma;
	const ExpansionIC ic = readIC();

	const amrex::Box &indexRange = grid_elem.indexRange_;
	const amrex::Array4<amrex::Real> &state_cc = grid_elem.array_;

	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		state_cc(i, j, k, HydroSystem<ExpansionProblem>::density_index) = ic.rho;
		state_cc(i, j, k, HydroSystem<ExpansionProblem>::x1Momentum_index) = ic.rho * ic.vx;
		state_cc(i, j, k, HydroSystem<ExpansionProblem>::x2Momentum_index) = ic.rho * ic.vy;
		state_cc(i, j, k, HydroSystem<ExpansionProblem>::x3Momentum_index) = ic.rho * ic.vz;
		state_cc(i, j, k, HydroSystem<ExpansionProblem>::internalEnergy_index) = ic.P / (gamma - 1.0);
		state_cc(i, j, k, HydroSystem<ExpansionProblem>::energy_index) =
		    ic.P / (gamma - 1.0) + 0.5 * ic.rho * (ic.vx * ic.vx + ic.vy * ic.vy + ic.vz * ic.vz); // eint + ekin
	});
}

// Analytical solution (comoving density is constant)
auto get_analytic_energy(amrex::Real e_init, amrex::Real a0, amrex::Real a_fin, amrex::Real gamma) -> amrex::Real
{
	return e_init * std::pow(a0 / a_fin, 3.0 * (gamma - 1.0)); // internal energy scaling a^(-3*(gamma-1))
}

template <>
void QuokkaSimulation<ExpansionProblem>::computeReferenceSolution(amrex::MultiFab &ref, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const & /*dx*/,
								  amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const & /*prob_lo*/)
// commented the dimension of the cells (dx) and the coordinate origin (prob_lo), since the problem is uniform (no v, P or rho gradients)
{
	const amrex::Real gamma = quokka::EOS_Traits<ExpansionProblem>::gamma;
	amrex::Real a_init = Physics_Traits<ExpansionProblem>::a_init;
	amrex::ParmParse const pp_cosmo("cosmology");
	pp_cosmo.query("a_init", a_init);
	const amrex::Real a_now = this->a_now_;
	const amrex::Real a_ratio = a_init / a_now;
	const ExpansionIC ic = readIC();

	// Analytic solutions
	const amrex::Real e0_int = ic.P / (gamma - 1);
	const amrex::Real eint_sol = get_analytic_energy(e0_int, a_init, a_now, gamma);
	const amrex::Real rho_sol = ic.rho; // comoving density doesn't change

	// Peculiar velocity and momentum
	const amrex::Real vx_sol = ic.vx * a_ratio; // peculiar velocity scales as a^-1
	const amrex::Real vy_sol = ic.vy * a_ratio;
	const amrex::Real vz_sol = ic.vz * a_ratio;
	const amrex::Real momx_sol = rho_sol * vx_sol;
	const amrex::Real momy_sol = rho_sol * vy_sol;
	const amrex::Real momz_sol = rho_sol * vz_sol;
	const amrex::Real v2_sol = (vx_sol * vx_sol + vy_sol * vy_sol + vz_sol * vz_sol); // squared velocity modulus
	const amrex::Real e_kin_sol = 0.5 * rho_sol * v2_sol;				  // kinetic energy
	const amrex::Real e_tot_sol = eint_sol + e_kin_sol;				  // Etot = Eint + Ek

	auto const &ref_arrays = ref.arrays(); // pointers (Array4) of the grid collection MultiFab ref

	// Kernel GPU to fill the reference solution captured by [=]
	amrex::ParallelFor(ref, [=] AMREX_GPU_DEVICE(int box_no, int i, int j, int k) noexcept {
		ref_arrays[box_no](i, j, k, HydroSystem<ExpansionProblem>::density_index) = rho_sol;
		ref_arrays[box_no](i, j, k, HydroSystem<ExpansionProblem>::x1Momentum_index) = momx_sol;
		ref_arrays[box_no](i, j, k, HydroSystem<ExpansionProblem>::x2Momentum_index) = momy_sol;
		ref_arrays[box_no](i, j, k, HydroSystem<ExpansionProblem>::x3Momentum_index) = momz_sol;
		ref_arrays[box_no](i, j, k, HydroSystem<ExpansionProblem>::energy_index) = e_tot_sol;
		ref_arrays[box_no](i, j, k, HydroSystem<ExpansionProblem>::internalEnergy_index) = eint_sol;
	});

	// GPU synchronization: CPU waits until the GPU has written all the values in the MultiFabs
	amrex::Gpu::streamSynchronize();
}

auto problem_main() -> int
{
	// Instantiate the simulation object for the expansion problem
	QuokkaSimulation<ExpansionProblem> sim;

	// Read parameters from the input file
	sim.readParameters();

	// Assert EdS cosmology
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(sim.cosmology_params_.Omega_m == 1.0 && sim.cosmology_params_.Omega_L == 0.0 && sim.cosmology_params_.Omega_r == 0.0,
					 "CosmologicalExpansion assumes an Einstein-de Sitter universe (t(a) = 2/(3 H0) a^1.5).");

	// Simulation stop time based on target scale factors
	amrex::Real a_final = 10; // default in the future
	amrex::ParmParse const pp_cosmo("cosmology");
	pp_cosmo.query("a_final", a_final);
	const amrex::Real a_init = sim.a_now_;
	const amrex::Real H0 = sim.cosmology_params_.H0;

	// EdS analytical
	const amrex::Real t_init = (2.0 / 3.0) * (1.0 / H0) * std::pow(a_init, 1.5);
	const amrex::Real t_final = (2.0 / 3.0) * (1.0 / H0) * std::pow(a_final, 1.5);
	const amrex::Real sim_time = t_final - t_init;
	amrex::ParmParse const pp;
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(!pp.contains("stop_time"),
					 "CosmologicalExpansion sets the stop time from cosmology.a_final: remove stop_time from the input file.");
	sim.stopTime_ = sim_time;

	const amrex::Real s_to_Gyr = 1.0 / 3.15576e16;
	amrex::Print() << "  Evolving from a = " << a_init << " to a = " << a_final << "\n";
	amrex::Print() << "  Simulation duration: " << sim_time * s_to_Gyr << " Gyr (" << sim.stopTime_ << " s)\n";

	// Retrieve density, energy and momentum indices
	const int rho_idx = HydroSystem<ExpansionProblem>::density_index;
	const int eint_idx = HydroSystem<ExpansionProblem>::internalEnergy_index;
	const int etot_idx = HydroSystem<ExpansionProblem>::energy_index;
	const int momx_idx = HydroSystem<ExpansionProblem>::x1Momentum_index;
	const int momy_idx = HydroSystem<ExpansionProblem>::x2Momentum_index;
	const int momz_idx = HydroSystem<ExpansionProblem>::x3Momentum_index;

	// Set initial conditions and extract initial-state averages for error analysis
	sim.setInitialConditions();
	const amrex::Real total_rho = sim.state_new_cc_[0].sum(rho_idx); // global sum over all MPI ranks (collective), ghost cells excluded
	const amrex::Real total_eint = sim.state_new_cc_[0].sum(eint_idx);
	const auto num_pts = static_cast<amrex::Real>(sim.boxArray(0).numPts()); // cast since numPts is a Long

	const amrex::Real rho0_avg = total_rho / num_pts;
	const amrex::Real eint0_avg = total_eint / num_pts;

	amrex::Print() << "Initial Average Density           : " << rho0_avg << "\n";
	amrex::Print() << "Initial Average Internal Energy   : " << eint0_avg << "\n";

	sim.evolve();
	amrex::Print() << "\nExpansion Test Results:\n";
	const amrex::Real a_f = sim.a_now_; // final simulation scale factor (read AFTER evolve, otherwise it would still be the initial value)
	const amrex::Real a_rel_err = std::abs(a_f - a_final) / a_final;
	amrex::Print() << "  Final a = " << a_f << "\n";
	amrex::Print() << "  a_final relative error (a_sim_fin - a_fin) / a_fin  = " << a_rel_err << "\n";

	// Relative L1 error of every component against computeReferenceSolution().
	// The field is uniform, so L1, L2 and Linf coincide
	const auto comp_errors = sim.computeComponentErrors();
	// each element is a tuple (name, abs_err, rel_err, ref_norm); index = component index
	// get<2> extract third component of the tuple, the rel err = |sim − ref| / |ref|
	const auto rel_L1 = [&](int idx) { return std::get<2>(comp_errors[idx]); };

	struct Check {
		const char *name;
		int idx;
		amrex::Real tol;
	};
	const std::array<Check, 6> checks{{{.name = "density", .idx = rho_idx, .tol = 1e-12},	       // reference run: 0.0
					   {.name = "internal energy", .idx = eint_idx, .tol = 1e-10}, // reference run: 3.0e-14
					   {.name = "total energy", .idx = etot_idx, .tol = 1e-10},    // reference run: 6.2e-15
					   {.name = "momentum x", .idx = momx_idx, .tol = 1e-10},      // reference run: 3.0e-15
					   {.name = "momentum y", .idx = momy_idx, .tol = 1e-10},      // reference run: 3.0e-15
					   {.name = "momentum z", .idx = momz_idx, .tol = 1e-10}}};    // reference run: 3.0e-15

	const amrex::Real tol_fried = 1e-8;	    // = rtol of evolveScaleFactor; measured 5.4e-10
	const bool a_ok = (a_rel_err <= tol_fried); // false also when a_rel_err is NaN
	amrex::Print() << "  scale factor a(t_f): relative error = " << a_rel_err << " (tol " << tol_fried << ") " << (a_ok ? "PASS" : "FAIL") << "\n";
	bool passed = a_ok;
	for (const auto &c : checks) {		       // range-based for on the elements of checks
		const amrex::Real err = rel_L1(c.idx); // call the lambda that captures the relative error via its index
		const bool ok = (err <= c.tol);	       // false also when err is NaN
		passed = passed && ok;
		amrex::Print() << "  " << c.name << ": relative L1 = " << err << " (tol " << c.tol << ") " << (ok ? "PASS" : "FAIL") << "\n";
	}
	amrex::Print() << (passed ? "TEST PASSED.\n" : "TEST FAILED: Error exceeds tolerances!\n");
	return passed ? 0 : 1;
}
