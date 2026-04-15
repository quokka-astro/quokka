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

// declare global variables

double temperature = 0.0; // K
double R0 = 0.2 *  C::parsec; 
template <> void QuokkaSimulation<ThermalConductionProblem>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	// initialize a ThermalConduction test problem using parameters from
	
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = grid_elem.dx_;
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const prob_lo = grid_elem.prob_lo_;
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const prob_hi = grid_elem.prob_hi_;
	const amrex::Box &indexRange = grid_elem.indexRange_;
	
	const amrex::Array4<double> &state_cc = grid_elem.array_;
	const Real cell_vol = AMREX_D_TERM(dx[0], *dx[1], *dx[2]);
	const int mid_index =indexRange.bigEnd(0)/2; 
	
	// loop over the grid and set the initial condition
	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		const amrex::Real x = prob_lo[0] + (i + 0.5) * dx[0];
		const amrex::Real y = prob_lo[1] + (j + 0.5) * dx[1];
		const amrex::Real z = prob_lo[2] + (k + 0.5) * dx[2];
		
		
		/*-------------------------------*/
		// Problem 1----> Gaussian temperature profile
		// const amrex::Real rho = C::m_p; // g/cm^3
		// const amrex::Real T0 = 100.0; // peak temperature at the center
		// const amrex::Real Tfloor = 1.e-7; // floor temperature at the edges
		// const amrex::Real D = 3.e23 ; 
		// const amrex::Real sigma2 = 2. * D * 1.e10; // width of the Gaussian
		// amrex::Real T = T0 * std::exp(-x*x/sigma2/2.) * std::exp(-y*y/sigma2/2.) * std::exp(-z*z/sigma2/2.) + Tfloor;

		/*-------------------------------*/
		//Problem 2----> Step Function temperature profile
		// const amrex::Real rho = C::m_p; // g/cm^3s
		// amrex::Real T ;
		// if(x<0.0) {
		// 	T = 100.; // higher temperature in the left half of the domain
		// }
		// else {
		// 	T = 1.0; // lower temperature outside the center
		// }
		// for (int n = 0; n < state_cc.nComp(); ++n) {
		// 	state_cc(i, j, k, n) = 0.; // zero fill all components
		// }
		//  amrex::Real Eint = quokka::EOS<ThermalConductionProblem>::ComputeEintFromTgas(rho, T); 

		/*-------------------------------*/
		//Problem 3----> Spherical temperature profile with sharp boundary
		// const amrex::Real rho = C::m_p; // g/cm^3
		// const amrex::Real Tout = 10.0;
		// const amrex::Real Tin  = 100.0;
		// double R = std::sqrt(x*x + y*y + z*z);
		// amrex::Real Eint;
		// if(R<=R0){
		// 	Eint = quokka::EOS<ThermalConductionProblem>::ComputeEintFromTgas(rho, Tout) ; 
		// }
		// else{
		// 	Eint = quokka::EOS<ThermalConductionProblem>::ComputeEintFromTgas(rho, Tin);
		// }

		/*------------------------------------------------*/
		//Problem 4----> Spherical temperature profile with smooth boundary
		
		// const amrex::Real rho = C::m_p; // g/cm^3
		double R = std::sqrt(x*x + y*y + z*z);
		// const amrex::Real Tout = 10.0;
		// const amrex::Real Tin  = 100.0;
		// const amrex::Real D = 3.e23 ; 
		// const amrex::Real chit = 4.0 * D * 1.e10;
		// const amrex::Real sqrt_chit = std::sqrt(chit);
		// const amrex::Real term1 = 0.5 * (std::erf((R0 + R) / sqrt_chit) + std::erf((R0 - R) / sqrt_chit));
		// const amrex::Real term2 = (sqrt_chit / (2.0 * R * std::sqrt(M_PI))) * (std::exp(-((R0 + R) * (R0 + R)) / chit) 
		//                                                                 - std::exp(-((R0 - R) * (R0 - R)) / chit));

		// const amrex::Real T = Tout + (Tin - Tout) * (term1 + term2);	
		// amrex::Real Eint =  quokka::EOS<ThermalConductionProblem>::ComputeEintFromTgas(rho, T) ; 
		// if(i==127 & j==127 & k==127){
		// 	amrex::Print() << "i, j, k: " << i << " " << j << " " << k << " T: " << T  << std::endl;
		// 	amrex::Print() << "Eint: " << quokka::EOS<ThermalConductionProblem>::ComputeEintFromTgas(rho, T) << std::endl;
		// 	amrex::Print() << "x: " << x << std::endl;
		// 	amrex::Print() << "y: " << y << std::endl;
		// 	amrex::Print() << "z: " << z << std::endl;
		// }
		/*-------------------------------------------------*/
		/*------------------------------------------------*/
		//Problem 5----> Top hat temperature profile with sharp boundary in wind
		amrex::Real rho ; 
		
		const amrex::Real Twind = 2.e6;
		const amrex::Real Tcloud  = 1.e4;
		amrex::Real rho_cloud = C::m_p; // g/cm^3
		const amrex::Real Mach = 4.0; // Mach number of the wind
		
		amrex::Real T;
		amrex::Real vz;
		if(R < R0){
			T = Tcloud;
			rho = rho_cloud; // g/cm^3
			vz = 0.0; // cloud is stationary
		}
		else{
			T = Twind; // g/cm^3
			rho = rho_cloud * Twind / T; // g/cm^3
			amrex::Real pressure = rho * Twind / C::k_B / C::m_u;
			const amrex::Real cs_wind = quokka::EOS<ThermalConductionProblem>::ComputeSoundSpeed(rho, pressure);
			vz = Mach * cs_wind; // 100 km/s
			
		}
		
		if(i==0 & j==0 & k==0){
			amrex::Print() << "Parameters of the cloud-wind problem: " << std::endl;
			amrex::Print() << "Twind: " << Twind << std::endl;
			amrex::Print() << "Tcloud: " << Tcloud << std::endl;
			amrex::Print() << "Mach: " << Mach << std::endl;
		}
		/*-------------------------------------------------*/
		amrex::Real Eint = quokka::EOS<ThermalConductionProblem>::ComputeEintFromTgas(rho, T) ; 

		state_cc(i, j, k, HydroSystem<ThermalConductionProblem>::density_index) = rho;
		state_cc(i, j, k, HydroSystem<ThermalConductionProblem>::x3Momentum_index) = rho * vz;
		state_cc(i, j, k, HydroSystem<ThermalConductionProblem>::energy_index) = Eint + 0.5 * rho * vz * vz; // total energy = internal energy + kinetic energy
		state_cc(i, j, k, HydroSystem<ThermalConductionProblem>::internalEnergy_index) = Eint;
	});
}

auto problem_main() -> int
{
	// boundary conditions
	constexpr int ncomp_cc = Physics_Indices<ThermalConductionProblem>::nvarTotal_cc;
	amrex::Vector<amrex::BCRec> BCs_cc(ncomp_cc);
	for (int n = 0; n < ncomp_cc; ++n) {
		for (int dir = 0; dir<3; ++dir) {
		BCs_cc[n].setLo(dir, amrex::BCType::foextrap);  
		BCs_cc[n].setHi(dir, amrex::BCType::foextrap); 
		}
	}

	// Problem initialization
	QuokkaSimulation<ThermalConductionProblem> sim(BCs_cc);
	
	
	// initialize
	sim.setInitialConditions();

	// evolve
	sim.evolve();

	// Cleanup and exit
	amrex::Print() << "Finished." << '\n';
	return 0;
}
