//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file testThermalConduction.cpp
/// \brief Defines a test problem for thermal conduction.
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

/** Thermal conduction test problem
The initial condition for the test problem for running a wind-cloud problem. */



const double Twind = 3.e6;
const double Tcloud  = 1.e4;
const double rho_cloud = 0.006 * C::m_p; // g/cm^3
const double Mach = 4.0; // Mach number of the wind
const double R0 = 545 * C::parsec; // radius of the cloud		

struct ThermalConductionProblem {
};

template <> struct quokka::EOS_Traits<ThermalConductionProblem> {
	static constexpr double gamma = 5./3.;
	static constexpr double mean_molecular_weight = C::m_u;
};

template <> struct HydroSystem_Traits<ThermalConductionProblem> {
	static constexpr bool reconstruct_eint = false;
};

template <> struct Physics_Traits<ThermalConductionProblem> : DefaultPhysicsTraits {
	// cell-centred
	static constexpr bool is_hydro_enabled = true;
	static constexpr bool is_mhd_enabled = true;
};

template <> void QuokkaSimulation<ThermalConductionProblem>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	// initialize a ThermalConduction test problem using parameters from

	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const dx = grid_elem.dx_;
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const prob_lo = grid_elem.prob_lo_;
	const amrex::Box &indexRange = grid_elem.indexRange_;

	const amrex::Array4<double> &state_cc = grid_elem.array_;
	// loop over the grid and set the initial condition
	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		const amrex::Real x = prob_lo[0] + (i + 0.5) * dx[0];
		const amrex::Real y = prob_lo[1] + (j + 0.5) * dx[1];
		const amrex::Real z = prob_lo[2] + (k + 0.5) * dx[2];

		amrex::Real rho;	  // g/cm^3
		amrex::Real T;
		amrex::Real vz;
		amrex::Real cs_wind;
		double R = std::sqrt((x)*(x) + (y)*(y) + (z-R0)*(z-R0));
		if(R < R0){
			T = Tcloud;
			rho = rho_cloud; // g/cm^3
			vz = 0.0; // cloud is stationary
		}
		else{
			T = Twind;
			rho = rho_cloud * Tcloud / Twind; // g/cm^3
			amrex::Real pressure = rho * T * C::k_B / C::m_u;
			cs_wind = quokka::EOS<ThermalConductionProblem>::ComputeSoundSpeed(rho, pressure);
			vz = Mach * cs_wind; // 100 km/s
		}
		const amrex::Real Eint = quokka::EOS<ThermalConductionProblem>::ComputeEintFromTgas(rho, T);
		/*-------------------------------------------------*/

		for (int n = 0; n < state_cc.nComp(); ++n) {
			state_cc(i, j, k, n) = 0.; // zero fill all components
		}
		if(i==0 & j==0 & k==0 ){
			amrex::Print() << "Initial conditions at the center of the domain: " << std::endl;
			amrex::Print() << "Density: " << rho << std::endl;
			amrex::Print() << "Temperature: " << T << std::endl;
			amrex::Print() << "Internal Energy: " << Eint << std::endl;
			amrex::Print() << "cs: " << cs_wind << ", vz:" << vz << std::endl;
		}
		state_cc(i, j, k, HydroSystem<ThermalConductionProblem>::density_index) = rho;
		state_cc(i, j, k, HydroSystem<ThermalConductionProblem>::x3Momentum_index) = rho * vz;
		state_cc(i, j, k, HydroSystem<ThermalConductionProblem>::energy_index) = Eint + 0.5 * (rho * vz * vz);
		state_cc(i, j, k, HydroSystem<ThermalConductionProblem>::internalEnergy_index) = Eint;
	});
}


template <> void QuokkaSimulation<ThermalConductionProblem>::setInitialConditionsOnGridFaceVars(quokka::grid const &grid_elem)
{
	const amrex::Array4<double> &state_fc = grid_elem.array_;
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const quokka::direction dir = grid_elem.dir_;

	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		constexpr double bx = 0.0;
		constexpr double by = 0.0;
		constexpr double bz = 1.;

		if (dir == quokka::direction::x) {
			state_fc(i, j, k, Physics_Indices<ThermalConductionProblem>::mhdFirstIndex) = bx;
		} else if (dir == quokka::direction::y) {
			state_fc(i, j, k, Physics_Indices<ThermalConductionProblem>::mhdFirstIndex) = by;
		} else if (dir == quokka::direction::z) {
			state_fc(i, j, k, Physics_Indices<ThermalConductionProblem>::mhdFirstIndex) = bz;
		}
	});
}


template <> void QuokkaSimulation<ThermalConductionProblem>::refineGrid(int lev, amrex::TagBoxArray &tags, amrex::Real /*time*/, int /*ngrow*/)
{
	// geometrical refinement
	// tag cells within one-sigma of the initial Gaussian profile for refinement
	const double refine_Lmax = 1.5 * R0; 

	const auto prob_lo = geom[lev].ProbLoArray();
	const auto dx = geom[lev].CellSizeArray();
	const auto tag = tags.arrays();

	amrex::ParallelFor(tags, [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept {
		// NOTE: must check all nodes of the cell!
		// Otherwise, cells that are too big can completely prevent refinement.
		amrex::Real const x0 = prob_lo[0] + (i * dx[0]);
		amrex::Real const x1 = prob_lo[0] + ((i + 1) * dx[0]);
		amrex::Real const y0 = prob_lo[1] + (j * dx[1]);
		amrex::Real const y1 = prob_lo[1] + ((j + 1) * dx[1]);
		amrex::Real const z0 = prob_lo[2] + (k * dx[2]);
		amrex::Real const z1 = prob_lo[2] + ((k + 1) * dx[2]);

		auto tagIfPointInRegion = [=](amrex::Real x, amrex::Real y, amrex::Real /*z*/) {
			if ((std::abs(x) < refine_Lmax) && (std::abs(y) < refine_Lmax) ) {
				tag[bx](i, j, k) = amrex::TagBox::SET;
			}
		};

		for (auto const &x : {x0, x1}) {
			for (auto const &y : {y0, y1}) {
				for (auto const &z : {z0, z1}) {
					tagIfPointInRegion(x, y, z);
				}
			}
		}
	});
	amrex::Gpu::streamSynchronize();
}

auto problem_main() -> int
{
	// boundary conditions
	constexpr int ncomp_cc = Physics_Indices<ThermalConductionProblem>::nvarTotal_cc;
	amrex::Vector<amrex::BCRec> BCs_cc(ncomp_cc);
	// for (int n = 0; n < ncomp_cc; ++n) {
	// 	for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
	// 	BCs_cc[n].setLo(dir, amrex::BCType::foextrap);  
	// 	BCs_cc[n].setHi(dir, amrex::BCType::foextrap); 
	// 	}
	// }
    	for (int n = 0; n < ncomp_cc; ++n) {
		for (int i = 0; i < AMREX_SPACEDIM; ++i) {
			// diode boundary conditions
			if (i == 2) {
				BCs_cc[n].setLo(i, amrex::BCType::ext_dir); // diode
				BCs_cc[n].setHi(i, amrex::BCType::foextrap);
			} else {
				BCs_cc[n].setLo(i, amrex::BCType::foextrap); // periodic
				BCs_cc[n].setHi(i, amrex::BCType::foextrap); // periodic
			}
		}
	} 
	const int nvars_fc = Physics_Indices<ThermalConductionProblem>::nvarTotal_fc;
	amrex::Vector<amrex::BCRec> BCs_fc(nvars_fc);
	for (int icomp = 0; icomp < nvars_fc; ++icomp) {
		for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
			BCs_fc[icomp].setLo(idim, amrex::BCType::foextrap); // periodic
			BCs_fc[icomp].setHi(idim, amrex::BCType::foextrap);
		}
	}
	// Problem initialization
	QuokkaSimulation<ThermalConductionProblem> sim(BCs_cc, BCs_fc);


	// initialize
	sim.setInitialConditions();

	// evolve
	sim.evolve();

	// Cleanup and exit
	amrex::Print() << "Finished." << '\n';
	return 0;

	/***Richardson Extrapolation ****/

	// quokka::richardson::applyQuietDefaults();
	// quokka::richardson::Parameters params{};
	// params.machine_precision_target = 2.0e-9; // limit based on delta_b_magn, smaller values can be used if this is decreased
	// params.nx_initial = 128;
	// params.nx_max = 512;
	// params.expected_rate = 2.0;
	// params.tolerance = 0.3;
	// params.test_name = "Thermal Conduction";
	// params.csv_filename = "thermal_conduction_convergence.csv";

	// return quokka::richardson::run(params, [](int nx) { return runConductionTest(nx); });
}
