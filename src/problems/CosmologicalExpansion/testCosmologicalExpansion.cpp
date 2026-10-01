//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file testCosmologicalExpansion.cpp
/// \brief Defines a test problem for pure hydrodynamics with digonal velocity
/// undergoing just the cosmological expansion.
///

#include "QuokkaSimulation.hpp"
#include "cosmology/Cosmology.hpp"
#include "hydro/hydro_system.hpp"
#include "physics_info.hpp"
#include <cmath>

struct ExpansionProblem {
	static constexpr amrex::Real rho0_default = 1.0e-30;   // low density
	static constexpr amrex::Real P0_default   = 1.0e-25;   // low pressure (enough to have low sound speed and thus small dt)
	static constexpr amrex::Real vx0_default  = 5.77e4    // (5*10^5)/3;	       
	static constexpr amrex::Real vy0_default  = 5.77e4;
	static constexpr amrex::Real vz0_default  = 5.77e4;
};

template <> struct quokka::EOS_Traits<ExpansionProblem> {
	static constexpr amrex::Real gamma = 5.0 / 3.0;
	static constexpr amrex::Real mean_molecular_weight = C::m_u;
};

template <> struct Physics_Traits<ExpansionProblem> : DefaultPhysicsTraits {
	static constexpr bool is_hydro_enabled = true;
	static constexpr bool is_cosmology_enabled = true;

	// Cosmology: flat EdS universe (only gas, no dm)
	static constexpr amrex::Real omega_m  = 1.0;
	static constexpr amrex::Real omega_b  = 1.0;
	static constexpr amrex::Real omega_dm = 0.0;
	static constexpr amrex::Real omega_r  = 0.0;
	static constexpr amrex::Real omega_lambda = 0.0;
	static constexpr amrex::Real hubble_constant = 0.7;	    // H0 = 70 km/s/Mpc
	static constexpr amrex::Real a_init = 1.0;		        // start at z=0
	static constexpr amrex::Real cosmology_dt_limit = 1e-4; // very small for accuracy
};

template <> void QuokkaSimulation<ExpansionProblem>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	// Initial conditions: default of the problem
	const amrex::Real gamma = quokka::EOS_Traits<ExpansionProblem>::gamma;
	amrex::Real rho = ExpansionProblem::rho0_default;
	amrex::Real P   = ExpansionProblem::P0_default;
	amrex::Real vx  = ExpansionProblem::vx0_default;
	amrex::Real vy  = ExpansionProblem::vy0_default;
	amrex::Real vz  = ExpansionProblem::vz0_default;

	// Initial conditions: precedence to user-entered values (override if present in the .in file)
	amrex::ParmParse pp("problem");
	pp.query("rho0", rho);
	pp.query("P0", P);
	pp.query("vx0", vx);
	pp.query("vy0", vy);
	pp.query("vz0", vz);

	const amrex::Box &indexRange = grid_elem.indexRange_;	  // set of the indices of the grid patch (e.g. from 0 to 31 in x, y, z) 
	const amrex::Array4<amrex::Real> &state_cc = grid_elem.array_; // Array4 is a pointer to the data

	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		state_cc(i, j, k, HydroSystem<ExpansionProblem>::density_index) = rho;
		state_cc(i, j, k, HydroSystem<ExpansionProblem>::x1Momentum_index) = rho * vx;
		state_cc(i, j, k, HydroSystem<ExpansionProblem>::x2Momentum_index) = rho * vy;
		state_cc(i, j, k, HydroSystem<ExpansionProblem>::x3Momentum_index) = rho * vz;
		state_cc(i, j, k, HydroSystem<ExpansionProblem>::internalEnergy_index) = P / (gamma - 1.0);
		state_cc(i, j, k, HydroSystem<ExpansionProblem>::energy_index) = P / (gamma - 1.0) + 0.5 * rho * (vx * vx + vy * vy + vz * vz); // eint + ekin
	});
}

// Analytical solution (comoving density is constant)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto get_analytic_energy(amrex::Real e_init, amrex::Real a0, amrex::Real a_fin, amrex::Real gamma) -> amrex::Real
{
	return e_init * std::pow(a0 / a_fin, 3.0 * (gamma - 1.0));  // internal energy scaling a^(-3*(gamma-1))
}

template <>
void QuokkaSimulation<ExpansionProblem>::computeReferenceSolution(amrex::MultiFab &ref, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const & /*dx*/,
								  amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const & /*prob_lo*/)
// commented the dimension of the cells (dx) and the coordinate origin (prob_lo), since the problem is uniform (no v, P or rho gradients)
{
	const amrex::Real gamma = quokka::EOS_Traits<ExpansionProblem>::gamma;
	amrex::Real a_init = Physics_Traits<ExpansionProblem>::a_init;
	amrex::ParmParse pp_cosmo("cosmology");
	pp_cosmo.query("a_init", a_init);
	amrex::Real a_now = this->a_now_;	         
	const amrex::Real a_ratio = a_init / a_now;  

	amrex::Real rho = ExpansionProblem::rho0_default;
	amrex::Real P = ExpansionProblem::P0_default;
	amrex::Real vx = ExpansionProblem::vx0_default;
	amrex::Real vy = ExpansionProblem::vy0_default;
	amrex::Real vz = ExpansionProblem::vz0_default;
	amrex::ParmParse pp("problem");
	pp.query("rho0", rho);
	pp.query("P0", P);
	pp.query("vx0", vx);
	pp.query("vy0", vy);
	pp.query("vz0", vz);

	// Analytic solutions
	const amrex::Real e0_int = P / (gamma - 1);
	const amrex::Real eint_sol = get_analytic_energy(e0_int, a_init, a_now, gamma);
	const amrex::Real rho_sol = rho;         // comoving density doesn't change

	// Peculiar velocity and momentum
	const amrex::Real vx_sol = vx * a_ratio; // peculiar velocity scales as a^-1
	const amrex::Real vy_sol = vy * a_ratio;
	const amrex::Real vz_sol = vz * a_ratio;
	const amrex::Real momx_sol = rho * vx_sol;
	const amrex::Real momy_sol = rho * vy_sol;
	const amrex::Real momz_sol = rho * vz_sol;

	const amrex::Real v2_sol = (vx_sol * vx_sol + vy_sol * vy_sol + vz_sol * vz_sol);   // velocity module
	const amrex::Real e_kin_sol = 0.5 * rho * v2_sol;    // kinetic energy
	const amrex::Real e_tot_sol = eint_sol + e_kin_sol;  // Etot = Eint + Ek

	
	auto const &ref_arrays = ref.arrays();   // pointers (Array4) of the grid collection MultiFab ref

	// Kernel GPU to fill the reference solution captured by [=]
	amrex::ParallelFor(ref, [=] AMREX_GPU_DEVICE(int box_no, int i, int j, int k) noexcept {
		ref_arrays[box_no](i, j, k, HydroSystem<ExpansionProblem>::density_index) = rho;
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

	// Read parameters from the .in file
	sim.readParameters();

	// Set simulation parameters
	const amrex::Real yr_to_s = 3.15576e7;
	sim.stopTime_     = 1.0e8 * yr_to_s; // 100 Myr
	sim.maxTimesteps_ = 1000;
	sim.cflNumber_    = 0.3;

	// Allow overrides from input file
	amrex::ParmParse pp_amr("amr");
	pp_amr.query("max_timesteps", sim.maxTimesteps_);
	if (pp_amr.query("stop_time", sim.stopTime_)) {
		sim.stopTime_ *= yr_to_s;
	}
	amrex::ParmParse pp_quokka("quokka");
	pp_quokka.query("cfl", sim.cflNumber_);

	// Retrive density, energy and momentum index
	const int rho_idx  = HydroSystem<ExpansionProblem>::density_index;
	const int eint_idx = HydroSystem<ExpansionProblem>::internalEnergy_index;
	const int momx_idx = HydroSystem<ExpansionProblem>::x1Momentum_index;
	const int momy_idx = HydroSystem<ExpansionProblem>::x2Momentum_index;
	const int momz_idx = HydroSystem<ExpansionProblem>::x3Momentum_index;

	// Set initial conditions and extract initial-state averages for error analysis
	sim.setInitialConditions();
	const amrex::Real total_rho  = sim.state_new_cc_[0].sum(rho_idx, 0); // sum(idx, 0) in AMReX is collective for MPI, 0 is the number of ghost cells
	const amrex::Real total_eint = sim.state_new_cc_[0].sum(eint_idx, 0);
	const amrex::Real num_pts    = sim.boxArray(0).numPts();

	const amrex::Real rho0_avg  = total_rho / num_pts;
	const amrex::Real eint0_avg = total_eint / num_pts;

	amrex::Print() << "Initial Average Density           : " << rho0_avg << "\n";
	amrex::Print() << "Initial Average Internal Energy   : " << eint0_avg << "\n";

	// Initial scale factor (may be overridden in .in file)
	amrex::Real a0 = Physics_Traits<ExpansionProblem>::a_init;
	amrex::ParmParse pp_cosmo("cosmology");
	pp_cosmo.query("a_init", a0);

	sim.evolve();

	// Final scale factor (read AFTER evolve, otherwise it would still be the initial value)
	amrex::Real a_f = sim.a_now_;

	// Error calculation : L1 and Linf norms
	auto const &mf_sim = sim.state_new_cc_[0]; // the MultiFab of the simulation at the final time
	// Allocation of the new MultiFab mf_ref for the analytical solution, copying the spatial structure and division between the simulation processors
	// (boxArray(0) and DistributionMap(0)), the same number of components ( mf_sim.nComp()), but 0 ghost cells
	amrex::MultiFab mf_ref(mf_sim.boxArray(), mf_sim.DistributionMap(), mf_sim.nComp(), 0); 

	// Calculation of the current analytical solution
	sim.computeReferenceSolution(mf_ref, sim.Geom(0).CellSizeArray(),
				     sim.Geom(0).ProbLoArray()); 

	// Error calculation: simulation - reference
	amrex::MultiFab mf_err(mf_sim.boxArray(), mf_sim.DistributionMap(), mf_sim.nComp(), 0); // allocate a new MultiFab to store the error of each cell
	amrex::MultiFab::Copy(mf_err, mf_sim, 0, 0, mf_sim.nComp(), 0);				            // copy all the simulationd data (mf_sim) in mf_err
	amrex::MultiFab::Subtract(mf_err, mf_ref, 0, 0, mf_sim.nComp(), 0);			            // cell error = sim cell - ref cell

	// Calculation of the norms
	struct Norms {
		amrex::Real L1, L2, Linf;
	};

	// Lambda to calculate the norms
	auto get_norms = [&](int idx) -> Norms
	{ 
		amrex::Real norm_ref = mf_ref.norm1(idx);  // mf_ref is the analytic solution
		if (norm_ref > 0) {                        // avoid division by 0
			// return the relative errors {L1, L2, Linf}
			return {mf_err.norm1(idx) / norm_ref, mf_err.norm2(idx) / mf_ref.norm2(idx), mf_err.norminf(idx) / mf_ref.norminf(idx)};
		} 
		else {                               
			// return the absolute errors {L1, L2, Linf}
			return {mf_err.norm1(idx), mf_err.norm2(idx), mf_err.norminf(idx)};
		}
	};

	// Calculation of the norms
	Norms rho_norm  = get_norms(rho_idx);
	Norms eint_norm = get_norms(eint_idx);
	Norms momx_norm = get_norms(momx_idx);
	Norms momy_norm = get_norms(momy_idx);
	Norms momz_norm = get_norms(momz_idx);

	
	// Final print
	amrex::Print() << "\nVerification Norms \n";
	amrex::Print() << "  Density   : L1 = " << rho_norm.L1 << " | L2 = " << rho_norm.L2 << " | Linf = " << rho_norm.Linf << "\n";
	amrex::Print() << "  Energy    : L1 = " << eint_norm.L1 << " | L2 = " << eint_norm.L2 << " | Linf = " << eint_norm.Linf << "\n";
	amrex::Print() << "  Momentum x: L1 = " << momx_norm.L1 << " | L2 = " << momx_norm.L2 << " | Linf = " << momx_norm.Linf << "\n";
	amrex::Print() << "  Momentum y: L1 = " << momy_norm.L1 << " | L2 = " << momy_norm.L2 << " | Linf = " << momy_norm.Linf << "\n";
	amrex::Print() << "  Momentum z: L1 = " << momz_norm.L1 << " | L2 = " << momz_norm.L2 << " | Linf = " << momz_norm.Linf << "\n";

	
	amrex::Print() << "\nExpansion Test Results:\n";
	amrex::Print() << "  Final a = " << a_f << "\n";

	// Test status
	int status = 0; // success

	// Tolerances
	const amrex::Real tol_rho = 1e-12; // density is constant, so the only error is floating point
	const amrex::Real tol_eint = 1e-4; // numerical integration involved, so more permissive treshold
	const amrex::Real tol_mom = 1e-4;

	// L1 norm as a reference for the error
	if (rho_norm.L1 > tol_rho || eint_norm.L1 > tol_eint || 
	    momx_norm.L1 > tol_mom || momy_norm.L1 > tol_mom || momz_norm.L1 > tol_mom) {
		amrex::Print() << "TEST FAILED: Error exceeds tolerances!\n";
		status = 1;
	} else {
		amrex::Print() << "TEST PASSED.\n";
	}
	return status;
}

