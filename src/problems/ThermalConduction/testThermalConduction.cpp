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

template <> void QuokkaSimulation<ThermalConductionProblem>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	// initialize a ThermalConduction test problem using parameters from

	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const dx = grid_elem.dx_;
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const prob_lo = grid_elem.prob_lo_;
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const prob_hi = grid_elem.prob_hi_;
	const amrex::Box &indexRange = grid_elem.indexRange_;

	const amrex::Array4<double> &state_cc = grid_elem.array_;

	// loop over the grid and set the initial condition
	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		const amrex::Real x = prob_lo[0] + (i + 0.5) * dx[0];

		/*-------------------------------*/
		// Problem 1----> Gaussian temperature profile
		const amrex::Real rho = C::m_p;	   // g/cm^3
		const amrex::Real T0 = 100.0;	   // peak temperature at the center
		const amrex::Real Tfloor = 1.e-10; // floor temperature at the edges
		const amrex::Real D = 3.e23;
		const amrex::Real sigma2 = 2. * D * 1.e10; // width of the Gaussian
		const amrex::Real T = T0 * std::exp(-x * x / sigma2 / 2.) + Tfloor;

		/*-------------------------------*/
		// Problem 2----> Step Function temperature profile
		//  const amrex::Real rho = C::m_p; // g/cm^3s
		//  amrex::Real T ;
		//  if(x<0.0) {
		//  	T = 100.; // higher temperature in the left half of the domain
		//  }
		//  else {
		//  	T = 1.0; // lower temperature outside the center
		//  }

		//   amrex::Real Eint = quokka::EOS<ThermalConductionProblem>::ComputeEintFromTgas(rho, T);

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

		const amrex::Real Eint = quokka::EOS<ThermalConductionProblem>::ComputeEintFromTgas(rho, T);
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

			const amrex::Real T0 = 100.0;	   // peak temperature at the center
			const amrex::Real Tfloor = 1.e-10; // floor temperature at the edges
			const amrex::Real D = 3.e23;
			const amrex::Real sigma2_0 = 2. * D * 1.e10;	   // initial width of the Gaussian
			const amrex::Real sigma2 = sigma2_0 + 2.0 * D * t; // width of the Gaussian at time t
			const amrex::Real norm = T0 * (std::sqrt(sigma2_0 / sigma2));

			const amrex::Real T_exact = norm * std::exp(-x * x / sigma2 / 2.) + Tfloor;
			const amrex::Real rho = C::m_p; // g/cm^3
			const amrex::Real Eint = quokka::EOS<ThermalConductionProblem>::ComputeEintFromTgas(rho, T_exact);

			// clear all components
			for (int n = 0; n < ncomp; ++n) {
				stateExact(i, j, k, n) = 0.;
			}

			// fill gas components
			stateExact(i, j, k, HydroSystem<ThermalConductionProblem>::density_index) = rho;
			stateExact(i, j, k, HydroSystem<ThermalConductionProblem>::energy_index) = Eint;
			stateExact(i, j, k, HydroSystem<ThermalConductionProblem>::internalEnergy_index) = Eint;
			stateExact(i, j, k, HydroSystem<ThermalConductionProblem>::x1Momentum_index) = 0.0;
			stateExact(i, j, k, HydroSystem<ThermalConductionProblem>::x2Momentum_index) = 0.;
			stateExact(i, j, k, HydroSystem<ThermalConductionProblem>::x3Momentum_index) = 0.;
		});
	}
}

auto problem_main() -> int
{
	// boundary conditions
	constexpr int ncomp_cc = Physics_Indices<ThermalConductionProblem>::nvarTotal_cc;
	amrex::Vector<amrex::BCRec> BCs_cc(ncomp_cc);
	for (int n = 0; n < ncomp_cc; ++n) {
		for (int dir = 0; dir < 3; ++dir) {
			BCs_cc[n].setLo(dir, amrex::BCType::foextrap);
			BCs_cc[n].setHi(dir, amrex::BCType::foextrap);
		}
	}

	// // Problem initialization
	QuokkaSimulation<ThermalConductionProblem> sim(BCs_cc);

	// initialize
	sim.setInitialConditions();

	// evolve
	sim.evolve();

	const double rel_err_tol = 0.03;
	int status = 0;
	const double error_norm = sim.computeErrorNorm();

	if (error_norm > rel_err_tol) {
		status = 1;
	}

	amrex::Print() << "Finished." << '\n';
	return status;
}
