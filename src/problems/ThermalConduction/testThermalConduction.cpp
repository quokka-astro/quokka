//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file testHydroBlast3D.cpp
/// \brief Defines a test problem for a 3D explosion.
///
#include "AMReX.H"
#include "AMReX_BLassert.H"
#include "AMReX_MultiFab.H"
#include "AMReX_ParmParse.H"
#include "AMReX_Print.H"
#include "AMReX_SPACE.H"
#include "hydro/hydro_system.hpp"
#include "math/interpolate.hpp"
#include <fstream>

#include "QuokkaSimulation.hpp"
#include "radiation/radiation_system.hpp"
#include "util/BC.hpp"
#include "util/fextract.hpp"
#include "util/richardson.hpp"

double Eint0 = 2.505e-8; //equivalent to T = 2.e8 K
double Efloor = 5.674216387016754e-11; //equivalent tp T = 2.e6 K
double rho0 = 0.1 ; // g/cm^3
double D = 7.743518128921141e+27; // cm^2/s
double sigma = 6.0267140390625e+16; // conduction timescale in s

struct ThermalConductionProblem {
};

bool test_passes = false; // if one of the energy checks fails, set to false. NOLINT

template <> struct quokka::EOS_Traits<ThermalConductionProblem> {
	static constexpr double gamma = 2.;
	static constexpr double mean_molecular_weight = C::m_u;
};

template <> struct HydroSystem_Traits<ThermalConductionProblem> {
	static constexpr bool reconstruct_eint = false;
};

template <> struct Physics_Traits<ThermalConductionProblem> {
	static constexpr bool is_self_gravity_enabled = false;
	// cell-centred
	static constexpr bool is_hydro_enabled = true;
	static constexpr int numMassScalars = 0;		     // number of mass scalars
	static constexpr int numPassiveScalars = numMassScalars + 0; // number of passive scalars
	static constexpr bool is_radiation_enabled = false;
	static constexpr bool is_dust_enabled = false;
	static constexpr int nDustGroups = 1; // number of dust groups
	// face-centred
	static constexpr bool is_mhd_enabled = false;
	static constexpr int nGroups = 1; // number of radiation groups
	static constexpr UnitSystem unit_system = UnitSystem::CGS;
};

template <> void QuokkaSimulation<ThermalConductionProblem>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	// initialize a ThermalConduction test problem using parameters from

	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const dx = grid_elem.dx_;
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const prob_lo = grid_elem.prob_lo_;
	const amrex::Box &indexRange = grid_elem.indexRange_;

	const amrex::Array4<double> &state_cc = grid_elem.array_;
	auto tables = resampledTables_.const_tables();
	// loop over the grid and set the initial condition
	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		const amrex::Real x = prob_lo[0] + (i + 0.5) * dx[0];
		
		/*-------------------------------*/
		// Problem 1----> Gaussian temperature profile
		const amrex::Real rho = rho0 * C::m_p;	   // g/cm^3
		const amrex::Real sigma2 = sigma*sigma; // width of the Gaussian
		const amrex::Real Eint = Eint0 * std::exp(-x * x / sigma2 / 2.) + Efloor;
		/*-------------------------------*/
		// Problem 2----> Step Function temperature profile
		//  const amrex::Real rho = rho0 * C::m_p; // g/cm^3s
		//  amrex::Real Eint;
		//  if(x<0.0) {
		//  	Eint= Eint0; // higher temperature in the left half of the domain
		//  }
		//  else {
		//  	Eint = Efloor; // lower temperature outside the center
		//  }

		/*-------------------------------*/
		// Problem 3----> Spherical temperature profile with sharp boundary
		// double const R0 = 0.2 * C::parsec;
		//  const amrex::Real rho = C::m_p; // g/cm^3
		//  const amrex::Real Tout = 10.0;
		//  const amrex::Real Tin  = 100.0;
		//  double R = std::sqrt(x*x + y*y + z*z);
		//  amrex::Real Eint;
		//  if(R<=R0){
		//  	Eint = quokka::EOS<ThermalConductionProblem>::ComputeEintFromTgas(rho, Tout) ;
		//  }
		//  else{
		//  	Eint = quokka::EOS<ThermalConductionProblem>::ComputeEintFromTgas(rho, Tin);
		//  }

		/*------------------------------------------------*/
		// Problem 4----> Spherical temperature profile with smooth boundary

		// const amrex::Real rho = C::m_p; // g/cm^3
		// double R = std::sqrt(x*x + y*y + z*z);
		// const amrex::Real Tout = 10.0;
		// const amrex::Real Tin  = 100.0;
		// const amrex::Real D = 3.e23 ;
		// const amrex::Real chit = 4.0 * D * 1.e10;
		// const amrex::Real sqrt_chit = std::sqrt(chit);
		// const amrex::Real term1 = 0.5 * (std::erf((R0 + R) / sqrt_chit) + std::erf((R0 - R) / sqrt_chit));
		// const amrex::Real term2 = (sqrt_chit / (2.0 * R * std::sqrt(M_PI))) * (std::exp(-((R0 + R) * (R0 + R)) / chit)
		//                                                                 - std::exp(-((R0 - R) * (R0 - R)) / chit));

		// const amrex::Real T = Tout + (Tin - Tout) * (term1 + term2);
		/*-------------------------------------------------*/

		for (int n = 0; n < state_cc.nComp(); ++n) {
			state_cc(i, j, k, n) = 0.; // zero fill all components
		}

		// const amrex::Real Eint = quokka::EOS<ThermalConductionProblem>::ComputeEintFromTgas(rho, T);
		if(i==0){
			amrex::Real Tgas1 =quokka::ResampledCooling::ComputeTgasFromEgas(rho, Eint0, tables);
			amrex::Real cs = quokka::ResampledCooling::ComputeSoundSpeedFromRhoEint(rho, Eint0, tables);
			amrex::Print() << "Debug1: " << ", Eint = " <<  Eint0 << ", Tmax = " << Tgas1 << ", cs = " << cs/1.e5 << "\n";
			Tgas1 =quokka::ResampledCooling::ComputeTgasFromEgas(rho, Efloor, tables);
			cs = quokka::ResampledCooling::ComputeSoundSpeedFromRhoEint(rho, Efloor, tables);
			amrex::Print() << "Debug2: " << ", Eint = " << Efloor << ", Tfloor = " << Tgas1 << ", cs = " << cs/1.e5 << "\n";
		}
		state_cc(i, j, k, HydroSystem<ThermalConductionProblem>::density_index) = rho;
		state_cc(i, j, k, HydroSystem<ThermalConductionProblem>::energy_index) = Eint; // total energy = internal energy + kinetic energy
		state_cc(i, j, k, HydroSystem<ThermalConductionProblem>::internalEnergy_index) = Eint;
	});
}

template <>
void QuokkaSimulation<ThermalConductionProblem>::computeReferenceSolution(amrex::MultiFab &ref, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx,
									  amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_lo)
{

	const amrex::Real t = tNew_[0];

	for (amrex::MFIter iter(ref); iter.isValid(); ++iter) {
		const amrex::Box &indexRange = iter.validbox();
		auto const &stateExact = ref.array(iter);
		auto const ncomp = ref.nComp();

		amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
			amrex::Real const x = prob_lo[0] + (i + 0.5) * dx[0];


			//Solution for the Gaussian temperature profile
			const amrex::Real rho = rho0 * C::m_p; // g/cm^3
			const amrex::Real sigma2_0 = sigma*sigma; // initial width of the Gaussian
			const amrex::Real sigma2_t = sigma2_0 + 2.0 * D * t * C::m_u/rho; // width of the Gaussian at time t
			const amrex::Real norm = Eint0 * (std::sqrt(sigma2_0 / sigma2_t));
			const amrex::Real Eint_exact = norm *  std::exp(-x * x / sigma2_t / 2.) + Efloor ;
			
			

			//Solution for the Step function temperature profile
			// const amrex::Real rho = rho0 * C::m_p; // g/cm^3
			// const amrex::Real Dt = D * 0.6 * C::m_u/rho0; // effective diffusivity
			// // const amrex::Real Eint_exact = Efloor + (Eint0 - Efloor) * 0.5 * (1.0 - std::erf(-x / std::sqrt(4.0 * Dt * t)));
			// const amrex::Real Eint_exact = 0.5 * (Eint0 + Efloor) + 0.5 * (Eint0 - Efloor) * std::erf(-x / std::sqrt(4.0 * Dt * t));

			// clear all components
			for (int n = 0; n < ncomp; ++n) {
				stateExact(i, j, k, n) = 0.;
			}
			if(i==0){
				amrex::Print() << "t=" << t << ", Sigma2_t: " << sigma2_t << "\n";
			}
			// fill gas components
			stateExact(i, j, k, HydroSystem<ThermalConductionProblem>::density_index) = rho;
			stateExact(i, j, k, HydroSystem<ThermalConductionProblem>::energy_index) = Eint_exact;
			stateExact(i, j, k, HydroSystem<ThermalConductionProblem>::internalEnergy_index) = Eint_exact;
			stateExact(i, j, k, HydroSystem<ThermalConductionProblem>::x1Momentum_index) = 0.0;
			stateExact(i, j, k, HydroSystem<ThermalConductionProblem>::x2Momentum_index) = 0.;
			stateExact(i, j, k, HydroSystem<ThermalConductionProblem>::x3Momentum_index) = 0.;
		});
	}
}

auto runConductionTest(int nx) -> double
{
	// Read problem parameters
	const double max_time = 15.e10; //15 conduction times

	// amrex::ParmParse const hpp("conduction");
	// hpp.query("enabled", enableElectronConduction_);
	// hpp.query("conductivity_prefactor", electronConductionKappa0_);
	// hpp.query("conduction_cfl", conductionCFL);
	// hpp.query("flux_limiter_phi", electronConductionFluxLimiterPhi_);
	// hpp.query("saturation_factor", electronConductionSaturationFactor_);

	const double CFL_number = 0.3;
	const int max_timesteps = std::max(2000, nx * 100);

	// Set grid dimensions using AMReX parameter system
	amrex::ParmParse pp("amr");
	amrex::Vector<int> const ncells = {nx, nx, nx};
	pp.add("max_level", 0);
	pp.addarr("n_cell", ncells);

	// Set domain bounds using AMReX parameter system
	amrex::ParmParse pp_geom("geometry");
	amrex::Vector<double> const prob_lo = {-1.5428e+18, -1.5428e+18 , -1.5428e+18};
	amrex::Vector<double> const prob_hi = {1.5428e+18, 1.5428e+18, 1.5428e+18};
	amrex::Vector<int> const is_periodic = {0, 0, 0};
	pp_geom.addarr("prob_lo", prob_lo);
	pp_geom.addarr("prob_hi", prob_hi);
	pp_geom.addarr("is_periodic", is_periodic);


	// Setup boundary conditions
	constexpr int ncomp_cc = Physics_Indices<ThermalConductionProblem>::nvarTotal_cc;
	amrex::Vector<amrex::BCRec> BCs_cc(ncomp_cc);
	for (int n = 0; n < ncomp_cc; ++n) {
		for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
			BCs_cc[n].setLo(dir, amrex::BCType::foextrap);
			BCs_cc[n].setHi(dir, amrex::BCType::foextrap);
		}
	}

	// Run simulation
	QuokkaSimulation<ThermalConductionProblem> sim(BCs_cc);

	sim.cflNumber_ = CFL_number;
	sim.stopTime_ = max_time;
	sim.maxTimesteps_ = max_timesteps;

	// set initial conditions
	sim.setInitialConditions();

	sim.evolve();
	return sim.computeErrorNorm();
}


auto problem_main() -> int
{
	// boundary conditions
	// constexpr int ncomp_cc = Physics_Indices<ThermalConductionProblem>::nvarTotal_cc;
	// amrex::Vector<amrex::BCRec> BCs_cc(ncomp_cc);
	// for (int n = 0; n < ncomp_cc; ++n) {
	// 	for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
	// 		BCs_cc[n].setLo(dir, amrex::BCType::foextrap);
	// 		BCs_cc[n].setHi(dir, amrex::BCType::foextrap);
	// 	}
	// }

	// QuokkaSimulation<ThermalConductionProblem> sim(BCs_cc);	
	// sim.setInitialConditions();
	// sim.evolve();

	// const double rel_err_tol = 0.03;
	// int status = 0;
	// const double error_norm = sim.computeErrorNorm();

	// if (error_norm > rel_err_tol) {
	// 	status = 1;
	// }

	// amrex::Print() << "Finished." << '\n';
	// return status;

	/***Richardson Extrapolation ****/

	quokka::richardson::applyQuietDefaults();
	quokka::richardson::Parameters params{};
	params.machine_precision_target = 2.0e-9; // limit based on delta_b_magn, smaller values can be used if this is decreased
	params.nx_initial = 128;
	params.nx_max = 1024; 
	params.expected_rate = 2.0;
	params.tolerance = 0.3;
	params.test_name = "Thermal Conduction";
	params.csv_filename = "thermal_conduction_convergence.csv";

	return quokka::richardson::run(params, [](int nx) { return runConductionTest(nx); });


}
