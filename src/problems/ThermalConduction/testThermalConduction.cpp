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

/** Thermal conduction test problem
The initial condition for the test problem is a Gaussian temperature profile with a constant density.
The solution is also a Gaussian profile with an increasing (decreasing) width (peak) with time.
We run the test for one conduction timescale and check that the numerical solution matches the analytic solution.
Physical parameters for the test problem are chosen to satisfy t_hydro / t_conduction >> 1, so that the gas does not have time to move
and the energy evolution is purely due to conduction.
How to choose the parameters for the thermal conduction test problem
1. Fix a box length L and a grid resolution nx which will set the resolution dx.
2. Width of the gaussian = sigma = 5 * dx.
3. Choose a peak temperature T0 and estimate the sound speed cs.
4. Diffusion coefficient D = 1.e3 * sigma * cs. This will ensure that t_hydro / t_conduction = 1.e3.
5. Conductivity prefactor = D * rho * c_v should be supplied in the input file. */

const double Eint0 = 2.505e-8;		     // equivalent to T = 2.e8 K
const double Efloor = 5.674216387016754e-11; // equivalent tp T = 2.e6 K
const double rho0 = 0.1;		     // g/cm^3
const double D = 2.1981515823750267e+28;     // diffusion coefficient, in units of cm^2/s
const double sigma = 1.2053428078125e+17;    // conduction timescale in s

const double Twind = 2.e6;
const double Tcloud  = 1.e4;
const double rho_cloud = C::m_p; // g/cm^3
double cs_wind = 0.0;
const double Mach = 4.0; // Mach number of the wind
const double R0 = 0.1 * C::parsec; // radius of the cloud		

struct ThermalConductionProblem {
};

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
	// loop over the grid and set the initial condition
	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		const amrex::Real x = prob_lo[0] + (i + 0.5) * dx[0];
		const amrex::Real y = prob_lo[1] + (j + 0.5) * dx[1];
		const amrex::Real z = prob_lo[2] + (k + 0.5) * dx[2];

		/*-------------------------------*/
		// Problem ----> Gaussian temperature profile
		// const amrex::Real rho = rho0 * C::m_p;	  // g/cm^3
		// const amrex::Real sigma2 = sigma * sigma; // width of the Gaussian
		// const amrex::Real Eint = Eint0 * std::exp(-x * x / sigma2 / 2.) + Efloor;
		/*-------------------------------*/

		//Problem 5----> Top hat temperature profile with sharp boundary in wind
		amrex::Real rho;	  // g/cm^3
		amrex::Real T;
		amrex::Real vz;
		double R = std::sqrt(x*x + y*y + z*z);
		if(R < R0){
			T = Tcloud;
			rho = rho_cloud; // g/cm^3
			vz = 0.0; // cloud is stationary
		}
		else{
			T = Twind; // g/cm^3
			rho = rho_cloud * Tcloud / T; // g/cm^3
			amrex::Real pressure = rho * Twind * C::k_B / C::m_u;
			cs_wind = quokka::EOS<ThermalConductionProblem>::ComputeSoundSpeed(rho, pressure);
			vz = Mach * cs_wind; // 100 km/s
			
		}
		const amrex::Real Eint = quokka::EOS<ThermalConductionProblem>::ComputeEintFromTgas(rho, T);
		if(i==0 & j==0 & k==0){
			amrex::Print() << "Parameters of the cloud-wind problem: " << std::endl;
			amrex::Print() << "Twind: " << Twind << std::endl;
			amrex::Print() << "Tcloud: " << Tcloud << std::endl;
			amrex::Print() << "Mach: " << Mach << std::endl;
			amrex::Print() << "Wind velocity: " << vz << std::endl;
			amrex::Print() << "Sound speed in the wind: " << cs_wind << std::endl;
		}
		/*-------------------------------------------------*/

		for (int n = 0; n < state_cc.nComp(); ++n) {
			state_cc(i, j, k, n) = 0.; // zero fill all components
		}

		state_cc(i, j, k, HydroSystem<ThermalConductionProblem>::density_index) = rho;
		state_cc(i, j, k, HydroSystem<ThermalConductionProblem>::x3Momentum_index) = rho * vz;
		state_cc(i, j, k, HydroSystem<ThermalConductionProblem>::energy_index) = Eint +  rho * vz * vz / 2.;
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

			// Solution for the Gaussian temperature profile
			const amrex::Real rho = rho0 * C::m_p;		     // g/cm^3
			const amrex::Real sigma2_0 = sigma * sigma;	     // initial width of the Gaussian
			const amrex::Real sigma2_t = sigma2_0 + 2.0 * D * t; // width of the Gaussian at time t
			const amrex::Real norm = Eint0 * (std::sqrt(sigma2_0 / sigma2_t));
			const amrex::Real Eint_exact = norm * std::exp(-x * x / sigma2_t / 2.) + Efloor;

			// clear all components
			for (int n = 0; n < ncomp; ++n) {
				stateExact(i, j, k, n) = 0.;
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

//Refinement 

template <> void QuokkaSimulation<ThermalConductionProblem>::refineGrid(int lev, amrex::TagBoxArray &tags, amrex::Real /*time*/, int /*ngrow*/)
{
	// geometrical refinement
	// tag cells within one-sigma of the initial Gaussian profile for refinement
	const double refine_Lmax = 1.1 * R0 ; // 0.2 pc
	
	const auto prob_lo = geom[lev].ProbLoArray();
	const auto dx = geom[lev].CellSizeArray();
	const auto tag = tags.arrays();

	amrex::ParallelFor(tags, [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept {
		// NOTE: must check all nodes of the cell!
		// Otherwise, cells that are too big can completely prevent refinement.
		amrex::Real const x0 = prob_lo[0] + (i * dx[0]);
		amrex::Real const y0 = prob_lo[1] + (j * dx[1]);
		amrex::Real const z0 = prob_lo[2] + (k * dx[2]);

		amrex::Real const x1 = prob_lo[0] + ((i + 1) * dx[0]);
		amrex::Real const y1 = prob_lo[1] + ((j + 1) * dx[1]);
		amrex::Real const z1 = prob_lo[2] + ((k + 1) * dx[2]);

		auto tagIfPointInRegion = [=](amrex::Real x, amrex::Real y, amrex::Real z) {
			if ((std::abs(x) < refine_Lmax)) {
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

template <>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE void AMRSimulation<ThermalConductionProblem>::setCustomBoundaryConditions(const amrex::IntVect &iv, amrex::Array4<Real> const &consVar,
                          int /*dcomp*/ , int /*numcomp*/, amrex::GeometryData const &geom,
                           const Real /*time*/, const amrex::BCRec * /*bcr*/, int /*bcomp*/,
                           int /*orig_comp*/ )
{
  auto [i, j, k] = iv.dim3();
  amrex::Box const &box = geom.Domain();
  const auto &domain_lo = box.loVect3d();
  const auto &domain_hi = box.hiVect3d();
  const int klo = domain_lo[2];
  const int khi = domain_hi[2];
  int kedge, normal;

//    if (k < klo) {
//       kedge = klo;
//       normal = -1;
// 	  consVar(i, j, k, HydroSystem<ThermalConductionProblem>::density_index)    =  rho_cloud * Tcloud / T;
// 	consVar(i, j, k, HydroSystem<ThermalConductionProblem>::x1Momentum_index) =  x1Mom_edge;
// 	consVar(i, j, k, HydroSystem<ThermalConductionProblem>::x2Momentum_index) =  x2Mom_edge;
// 	consVar(i, j, k, HydroSystem<ThermalConductionProblem>::x3Momentum_index) =  x3Mom_edge;
// 	consVar(i, j, k, HydroSystem<ThermalConductionProblem>::energy_index)     = etot_edge;
// 	consVar(i, j, k, HydroSystem<ThermalConductionProblem>::internalEnergy_index) = eint_edge;
// }
//     const double rho_edge   = consVar(i, j, kedge, HydroSystem<ThermalConductionProblem>::density_index);
//     const double x1Mom_edge = consVar(i, j, kedge, HydroSystem<ThermalConductionProblem>::x1Momentum_index);
//     const double x2Mom_edge = consVar(i, j, kedge, HydroSystem<ThermalConductionProblem>::x2Momentum_index);
//           double x3Mom_edge = consVar(i, j, kedge, HydroSystem<ThermalConductionProblem>::x3Momentum_index);
//     const double etot_edge  = consVar(i, j, kedge, HydroSystem<ThermalConductionProblem>::energy_index);
//     const double eint_edge  = consVar(i, j, kedge, HydroSystem<ThermalConductionProblem>::internalEnergy_index);


//     if((x3Mom_edge*normal)<0){//gas is inflowing
//       x3Mom_edge = -1. *consVar(i, j, kedge, HydroSystem<ThermalConductionProblem>::x3Momentum_index);
//     }

        // consVar(i, j, k, HydroSystem<ThermalConductionProblem>::density_index)    = rho_edge ;
        // consVar(i, j, k, HydroSystem<ThermalConductionProblem>::x1Momentum_index) =  x1Mom_edge;
        // consVar(i, j, k, HydroSystem<ThermalConductionProblem>::x2Momentum_index) =  x2Mom_edge;
        // consVar(i, j, k, HydroSystem<ThermalConductionProblem>::x3Momentum_index) =  x3Mom_edge;
        // consVar(i, j, k, HydroSystem<ThermalConductionProblem>::energy_index)     = etot_edge;
        // consVar(i, j, k, HydroSystem<ThermalConductionProblem>::internalEnergy_index) = eint_edge;

}

auto problem_main() -> int
{
	// boundary conditions
	constexpr int ncomp_cc = Physics_Indices<ThermalConductionProblem>::nvarTotal_cc;
	amrex::Vector<amrex::BCRec> BCs_cc(ncomp_cc);
	for (int n = 0; n < ncomp_cc; ++n) {
		for (int i = 0; i < AMREX_SPACEDIM; ++i) {
			// constant inflow from zlow
			if (i == 2) {
				BCs_cc[n].setLo(i, amrex::BCType::foextrap); // inflow
				BCs_cc[n].setHi(i, amrex::BCType::foextrap); // outflow
			} else {
				BCs_cc[n].setLo(i, amrex::BCType::foextrap); // periodic
				BCs_cc[n].setHi(i, amrex::BCType::foextrap); // periodic
			}
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
