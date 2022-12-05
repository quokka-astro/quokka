//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file test_hydro3d_blast.cpp
/// \brief Defines a test problem for a 3D explosion.
///

#include <iostream>
#include <limits>
#include <math.h>
#include <random>

#include "AMReX.H"
#include "AMReX_BC_TYPES.H"
#include "AMReX_BLassert.H"
#include "AMReX_Config.H"
#include "AMReX_FabArrayUtility.H"
#include "AMReX_GpuDevice.H"
#include "AMReX_MultiFab.H"
#include "AMReX_ParallelContext.H"
#include "AMReX_ParallelDescriptor.H"
#include "AMReX_ParmParse.H"
#include "AMReX_Print.H"
#include "AMReX_Random.H"
#include "AMReX_RandomEngine.H"
#include "AMReX_SPACE.H"
#include "AMReX_TableData.H"

#include "CloudyCooling.hpp"
#include "ODEIntegrate.hpp"
#include "RadhydroSimulation.hpp"
#include "hydro_system.hpp"
#include "radiation_system.hpp"
#include "test_sne.hpp"

using amrex::Real;
using namespace amrex;

#define MAX 100

struct NewProblem {
};

template <> struct SimulationData<NewProblem> {
	cloudy_tables cloudyTables;
	std::unique_ptr<amrex::TableData<Real, 3>> table_data;
};

template <> struct HydroSystem_Traits<NewProblem> {
	static constexpr double gamma = 5. / 3.;
	static constexpr bool reconstruct_eint = true; // Set to true - temperature
};
template <> struct Physics_Traits<NewProblem> {
	static constexpr bool is_hydro_enabled = true;
	static constexpr bool is_radiation_enabled = false;
	static constexpr bool is_chemistry_enabled = false;
	static constexpr bool is_mhd_enabled = false;
	static constexpr int numPassiveScalars = 3; // number of passive scalars
};

template <> void RadhydroSimulation<NewProblem>::setInitialConditionsOnGrid(quokka::grid grid_elem)
{

	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = grid_elem.dx_;
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = grid_elem.prob_lo_;
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_hi = grid_elem.prob_hi_;
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const amrex::Array4<double> &state_cc = grid_elem.array_;

	double vol = AMREX_D_TERM(dx[0], *dx[1], *dx[2]);

	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		amrex::Real const x = prob_lo[0] + (i + amrex::Real(0.5)) * dx[0];
		amrex::Real const y = prob_lo[1] + (j + amrex::Real(0.5)) * dx[1];
		amrex::Real const z = prob_lo[2] + (k + amrex::Real(0.5)) * dx[2];

		double z_star = 245.0 * pc;
		double Sigma_star = 42.0 * Msun / pc / pc;
		double rho_dm = 0.0064 * Msun / pc / pc / pc;
		double R0 = 8.e3 * pc;

		double sigma1 = 7. * kmps;
		double sigma2 = 70. * kmps;
		double rho01 = 2.85 * Const_mH;
		double rho02 = 1.e-5 * 2.85 * Const_mH;

		/*Calculate DM Potential*/
		double prefac;
		prefac = 2. * 3.1415 * Const_G * rho_dm * std::pow(R0, 2);
		double Phidm = (prefac * std::log(1. + std::pow(z / R0, 2)));

		/*Calculate Stellar Disk Potential*/
		double prefac2;
		prefac2 = 2. * 3.1415 * Const_G * Sigma_star * z_star;
		double Phist = prefac2 * (std::pow(1. + z * z / z_star / z_star, 0.5) - 1.);

		double Phitot = Phist + Phidm;

		double rho, rho_disk, rho_halo;
		rho_disk = rho01 * std::exp(-Phitot / std::pow(sigma1, 2.0));
		rho_halo = rho02 * std::exp(-Phitot / std::pow(sigma2, 2.0)); // in g/cc
		rho = (rho_disk + rho_halo);

		double P = rho_disk * std::pow(sigma1, 2.0) + rho_halo * std::pow(sigma2, 2.0);

		AMREX_ASSERT(!std::isnan(rho));

		const auto gamma = HydroSystem<NewProblem>::gamma_;

		if (std::sqrt(z * z) < 0.25 * kpc) {
			state_cc(i, j, k, Physics_Indices<NewProblem>::pscalarFirstIndex) = 1.e2 / vol;	     // Disk tracer
			state_cc(i, j, k, Physics_Indices<NewProblem>::pscalarFirstIndex + 1) = 1.e-5 / vol; // Halo tracer
			state_cc(i, j, k, Physics_Indices<NewProblem>::pscalarFirstIndex + 2) = 1.e-5 / vol; // Injected tracer
		}

		else {
			state_cc(i, j, k, Physics_Indices<NewProblem>::pscalarFirstIndex) = 1.e-5 / vol;     // Disk tracer
			state_cc(i, j, k, Physics_Indices<NewProblem>::pscalarFirstIndex + 1) = 1.e2 / vol;  // Halo tracer
			state_cc(i, j, k, Physics_Indices<NewProblem>::pscalarFirstIndex + 2) = 1.e-5 / vol; // Injected tracer
		}

		state_cc(i, j, k, HydroSystem<NewProblem>::density_index) = rho;
		state_cc(i, j, k, HydroSystem<NewProblem>::x1Momentum_index) = 0.0;
		state_cc(i, j, k, HydroSystem<NewProblem>::x2Momentum_index) = 0.0;
		state_cc(i, j, k, HydroSystem<NewProblem>::x3Momentum_index) = 0.0;
		state_cc(i, j, k, HydroSystem<NewProblem>::internalEnergy_index) = P / (gamma - 1.);
		state_cc(i, j, k, HydroSystem<NewProblem>::energy_index) = P / (gamma - 1.);
	});
}

template <> void RadhydroSimulation<NewProblem>::ErrorEst(int lev, amrex::TagBoxArray &tags, amrex::Real /*time*/, int /*ngrow*/)
{
	// tag cells for refinement

	const amrex::Real eta_threshold = 3.5; // gradient refinement threshold
	const amrex::Real P_min = 1.0e-3;      // minimum pressure for refinement

	for (amrex::MFIter mfi(state_new_cc_[lev]); mfi.isValid(); ++mfi) {
		const amrex::Box &box = mfi.validbox();
		const auto state = state_new_cc_[lev].const_array(mfi);
		const auto tag = tags.array(mfi);

		amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
			/*amrex::Real scal_xyz    = state(i, j, k, Physics_Indices<NewProblem>::pscalarFirstIndex+2) ;

			amrex::Real scal_xplus  = state(i+1, j, k, Physics_Indices<NewProblem>::pscalarFirstIndex+2) ;
			amrex::Real scal_xminus = state(i-1, j, k, Physics_Indices<NewProblem>::pscalarFirstIndex+2) ;
			amrex::Real del_scalx   = std::max(std::abs(scal_xplus - scal_xyz), std::abs(scal_xminus - scal_xyz));

			amrex::Real scal_yplus  = state(i, j+1, k, Physics_Indices<NewProblem>::pscalarFirstIndex+2) ;
			amrex::Real scal_yminus = state(i, j-1, k, Physics_Indices<NewProblem>::pscalarFirstIndex+2) ;
			amrex::Real del_scaly   = std::max(std::abs(scal_yplus - scal_xyz), std::abs(scal_yminus - scal_xyz));

			amrex::Real scal_zplus  = state(i, j, k+1, Physics_Indices<NewProblem>::pscalarFirstIndex+2) ;
			amrex::Real scal_zminus = state(i, j, k-1, Physics_Indices<NewProblem>::pscalarFirstIndex+2) ;
			amrex::Real del_scalz   = std::max(std::abs(scal_zplus - scal_xyz), std::abs(scal_zminus - scal_xyz));


			amrex::Real const gradient_indicator =
			    std::max({del_scalx, del_scaly, del_scalz}) / scal_xyz;*/

			// if ((gradient_indicator > eta_threshold)) {
			tag(i, j, k) = amrex::TagBox::SET;
			// printf("Reached here=%d, %d, %d, %.2e\n", i, j, k, gradient_indicator);
			// }
		});
	}
}

// template <>
// void RadhydroSimulation<NewProblem>::ErrorEst(int lev,
//                                                 amrex::TagBoxArray &tags,
//                                                 amrex::Real /*time*/ ,
//                                                int /*ngrow*/) {
//   // tag cells for refinement
//   // /*amrex::Print() << "tagging cells for refinement...\n";

//   for (amrex::MFIter mfi(state_new_cc_[lev]); mfi.isValid(); ++mfi) {
//     const amrex::Box &box = mfi.validbox();
//     const auto state = state_new_cc_[lev].const_array(mfi);
//     const auto tag = tags.array(mfi);
//     const int nidx = HydroSystem<NewProblem>::density_index;

//     amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
//       Real const q = state(i, j, k, nidx);
//       if(q>1.e2*Const_mH){
//       tag(i, j, k) = amrex::TagBox::SET; }
//     });
//   }
// }

/*****Adding Cooling Terms*****/

struct ODEUserData {
	amrex::Real rho;
	cloudyGpuConstTables tables;
};

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto user_rhs(Real /*t*/, quokka::valarray<Real, 1> &y_data, quokka::valarray<Real, 1> &y_rhs, void *user_data) -> int
{
	// unpack user_data
	auto *udata = static_cast<ODEUserData *>(user_data);
	Real rho = udata->rho;
	cloudyGpuConstTables &tables = udata->tables;

	// compute temperature (implicit solve, depends on composition)
	Real Eint = y_data[0];
	Real T = ComputeTgasFromEgas(rho, Eint, HydroSystem<NewProblem>::gamma_, tables);

	// compute cooling function
	y_rhs[0] = cloudy_cooling_function(rho, T, tables);
	return 0;
}

void computeCooling(amrex::MultiFab &mf, const Real dt_in, cloudy_tables &cloudyTables)
{
	BL_PROFILE("RadhydroSimulation::computeCooling()")

	const Real dt = dt_in;
	const Real reltol_floor = 0.01;
	const Real rtol = 1.0e-4; // not recommended to change this

	auto tables = cloudyTables.const_tables();

	// loop over all cells in MultiFab mf
	for (amrex::MFIter iter(mf); iter.isValid(); ++iter) {
		const amrex::Box &indexRange = iter.validbox();
		auto const &state = mf.array(iter);

		amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
			const Real rho = state(i, j, k, HydroSystem<NewProblem>::density_index);
			const Real x1Mom = state(i, j, k, HydroSystem<NewProblem>::x1Momentum_index);
			const Real x2Mom = state(i, j, k, HydroSystem<NewProblem>::x2Momentum_index);
			const Real x3Mom = state(i, j, k, HydroSystem<NewProblem>::x3Momentum_index);
			const Real Egas = state(i, j, k, HydroSystem<NewProblem>::energy_index);

			Real Eint = RadSystem<NewProblem>::ComputeEintFromEgas(rho, x1Mom, x2Mom, x3Mom, Egas);

			ODEUserData user_data{rho, tables};
			quokka::valarray<Real, 1> y = {Eint};
			quokka::valarray<Real, 1> abstol = {reltol_floor * ComputeEgasFromTgas(rho, T_floor, HydroSystem<NewProblem>::gamma_, tables)};

			// do integration with RK2 (Heun's method)
			int steps_taken = 0;

			rk_adaptive_integrate(user_rhs, 0, y, dt, &user_data, rtol, abstol, steps_taken);

			const Real Egas_new = RadSystem<NewProblem>::ComputeEgasFromEint(rho, x1Mom, x2Mom, x3Mom, y[0]);

			const Real Eint_new = y[0];
			const Real dEint = Eint_new - Eint;
			Real Temp = ComputeTgasFromEgas(rho, Eint_new, HydroSystem<NewProblem>::gamma_, tables);

			state(i, j, k, HydroSystem<NewProblem>::energy_index) += dEint;
			state(i, j, k, HydroSystem<NewProblem>::internalEnergy_index) += dEint;
		});

		AMREX_ASSERT(!state.contains_nan(0, state.nComp()));
	}
}

void AddSupernova(amrex::MultiFab &mf, const Real dt_in, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo,
		  const Real time, int level)
{
	BL_PROFILE("HydroSimulation::AddSupernova()")
	const Real dt = dt_in;

	double Mass_source = 8. * Msun;
	double Metal_source = 150. * Mass_source;
	double Energy_source = 1.e51;
	double vol = AMREX_D_TERM(dx[0], *dx[1], *dx[2]);

	for (amrex::MFIter iter(mf); iter.isValid(); ++iter) {
		const amrex::Box &indexRange = iter.validbox();
		auto const &state = mf.array(iter);

		amrex::ParallelForRNG(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k, RandomEngine const &engine) noexcept {
			double z = prob_lo[2] + (k + amrex::Real(0.5)) * dx[2];
			double fz = std::exp(-z * z / (hscale * hscale));
			double nsn_expect = probSN_prefac * fz * vol * dt;

			int n_sn = 1;
			double random = Random(engine);
			if (random < nsn_expect) {
				state(i, j, k, HydroSystem<NewProblem>::energy_index) += n_sn * Energy_source / vol;
				state(i, j, k, HydroSystem<NewProblem>::internalEnergy_index) += n_sn * Energy_source / vol;
				state(i, j, k, Physics_Indices<NewProblem>::pscalarFirstIndex + 2) += 1.e3 / vol;
				printf("The location of SN=%d,%d,%d, at time=%.2e\n", i, j, k, time);
				printf("Added at level =%d\n", level);
			}
		});
	}
}

template <> void RadhydroSimulation<NewProblem>::computeAfterLevelAdvance(int lev, amrex::Real time, amrex::Real dt_lev, int ncycle)
{
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = geom[lev].ProbLoArray();
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx = geom[lev].CellSizeArray();
	// amrex::Vector<std::unique_ptr<amrex::AmrexLevel>> *levels ; //= getAmrLevels();
	// std::vector<std::vector<n_t*>> *masks;
	// InSituUtils::GenerateMasks(levels, masks);
	if (lev == finestLevel()) {
		AddSupernova(state_new_cc_[lev], dt_lev, dx, prob_lo, time, lev);
	}
	computeCooling(state_new_cc_[lev], dt_lev, userData_.cloudyTables);
}

template <>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto RadhydroSimulation<NewProblem>::GetGradFixedPotential(amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> posvec)
    -> amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>
{

	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> grad_potential;

	double z_star = 245.0 * pc;
	double Sigma_star = 42.0 * Msun / pc / pc;
	double rho_dm = 0.0064 * Msun / pc / pc / pc;
	double R0 = 8.e3 * pc;

	double x = posvec[0];

	grad_potential[0] = 0.0;

#if (AMREX_SPACEDIM >= 2)
	double y = posvec[1];
	grad_potential[1] = 0.0;
#endif
#if (AMREX_SPACEDIM >= 3)
	double z = posvec[2];
	grad_potential[2] = 2. * 3.1415 * Const_G * rho_dm * std::pow(R0, 2) * (2. * z / std::pow(R0, 2)) / (1. + std::pow(z, 2) / std::pow(R0, 2));
	grad_potential[2] += 2. * 3.1415 * Const_G * Sigma_star * (z / z_star) * (std::pow(1. + z * z / (z_star * z_star), -0.5));
#endif

	return grad_potential;
}

/* Add Strang Split Source Term for External Fixed Potential Here */
template <> void RadhydroSimulation<NewProblem>::addStrangSplitSources(amrex::MultiFab &mf, int lev, amrex::Real time, amrex::Real dt_lev)
{
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = geom[lev].ProbLoArray();
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx = geom[lev].CellSizeArray();
	const Real dt = dt_lev;

	for (amrex::MFIter iter(mf); iter.isValid(); ++iter) {
		const amrex::Box &indexRange = iter.validbox();
		auto const &state = mf.array(iter);

		amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
			amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> posvec, GradPhi;
			double x1mom_new, x2mom_new, x3mom_new;

			const Real rho = state(i, j, k, HydroSystem<NewProblem>::density_index);
			const Real x1mom = state(i, j, k, HydroSystem<NewProblem>::x1Momentum_index);
			const Real x2mom = state(i, j, k, HydroSystem<NewProblem>::x2Momentum_index);
			const Real x3mom = state(i, j, k, HydroSystem<NewProblem>::x3Momentum_index);
			const Real Egas = state(i, j, k, HydroSystem<NewProblem>::energy_index);

			const auto vx = x1mom / rho;
			const auto vy = x2mom / rho;
			const auto vz = x3mom / rho;
			const double vel_mag = std::sqrt(vx * vx + vy * vy + vz * vz);

			Real Eint = RadSystem<NewProblem>::ComputeEintFromEgas(rho, x1mom, x2mom, x3mom, Egas);

			posvec[0] = prob_lo[0] + (i + 0.5) * dx[0];

#if (AMREX_SPACEDIM >= 2)
			posvec[1] = prob_lo[1] + (j + 0.5) * dx[1];
#endif

#if (AMREX_SPACEDIM >= 3)
			posvec[2] = prob_lo[2] + (k + 0.5) * dx[2];
#endif

			GradPhi = RadhydroSimulation<NewProblem>::GetGradFixedPotential(posvec);

			x1mom_new = state(i, j, k, HydroSystem<NewProblem>::x1Momentum_index) + dt * (-rho * GradPhi[0]);
			x2mom_new = state(i, j, k, HydroSystem<NewProblem>::x2Momentum_index) + dt * (-rho * GradPhi[1]);
			x3mom_new = state(i, j, k, HydroSystem<NewProblem>::x3Momentum_index) + dt * (-rho * GradPhi[2]);

			state(i, j, k, HydroSystem<NewProblem>::x1Momentum_index) = x1mom_new;
			state(i, j, k, HydroSystem<NewProblem>::x2Momentum_index) = x2mom_new;
			state(i, j, k, HydroSystem<NewProblem>::x3Momentum_index) = x3mom_new;

			state(i, j, k, HydroSystem<NewProblem>::energy_index) =
			    RadSystem<NewProblem>::ComputeEgasFromEint(rho, x1mom_new, x2mom_new, x3mom_new, Eint);
		});
	}
}

/**************************End Adding Strang Split Source Term *****************/

/**************************Adding User Defined Limits on Variables **************/

template <>
void HydroSystem<NewProblem>::EnforceLimits(amrex::Real const densityFloor, amrex::Real const pressureFloor, amrex::Real const speedCeiling,
					    amrex::Real const tempCeiling, amrex::Real const tempFloor, amrex::MultiFab &state_mf)
{
	// prevent vacuum creation
	amrex::Real const rho_floor = densityFloor; // workaround nvcc bug
	amrex::Real const P_floor = pressureFloor;
	auto state = state_mf.arrays();

	amrex::ParallelFor(state_mf, [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept {
		amrex::Real const rho = state[bx](i, j, k, density_index);
		amrex::Real const vx1 = state[bx](i, j, k, x1Momentum_index) / rho;
		amrex::Real const vx2 = state[bx](i, j, k, x2Momentum_index) / rho;
		amrex::Real const vx3 = state[bx](i, j, k, x3Momentum_index) / rho;
		amrex::Real const vsq = (vx1 * vx1 + vx2 * vx2 + vx3 * vx3);
		amrex::Real const Etot = state[bx](i, j, k, energy_index);
		amrex::Real const Eint = state[bx](i, j, k, internalEnergy_index);
		amrex::Real const Temp = (gamma_ - 1.) * Eint / (kb * (rho / Const_mH / 0.6)); // Assuming all gas in ionized.

		amrex::Real inj_scalar = state[bx](i, j, k, Physics_Indices<NewProblem>::pscalarFirstIndex + 2);

		amrex::Real rho_new = rho;
		if (rho < rho_floor) {
			rho_new = rho_floor;
			state[bx](i, j, k, density_index) = rho_new;
		}

		if (std::abs(vx1) > speedCeiling) {
			amrex::Real dummy = Etot - state[bx](i, j, k, density_index) * vx1 * vx1 / 2.;
			state[bx](i, j, k, x1Momentum_index) = (std::abs(vx1) / vx1) * speedCeiling * state[bx](i, j, k, density_index);
			state[bx](i, j, k, energy_index) = dummy + std::pow(state[bx](i, j, k, x1Momentum_index), 2.) / state[bx](i, j, k, density_index) / 2.;
		}

		if (std::abs(vx2) > speedCeiling) {
			amrex::Real dummy = Etot - state[bx](i, j, k, density_index) * vx2 * vx2 / 2.;
			state[bx](i, j, k, x2Momentum_index) = (std::abs(vx2) / vx2) * speedCeiling * state[bx](i, j, k, density_index);
			state[bx](i, j, k, energy_index) = dummy + std::pow(state[bx](i, j, k, x2Momentum_index), 2.) / state[bx](i, j, k, density_index) / 2.;
		}

		if (std::abs(vx3) > speedCeiling) {
			amrex::Real dummy = Etot - state[bx](i, j, k, density_index) * vx3 * vx3 / 2.;
			state[bx](i, j, k, x3Momentum_index) = (std::abs(vx3) / vx3) * speedCeiling * state[bx](i, j, k, density_index);
			state[bx](i, j, k, energy_index) = dummy + std::pow(state[bx](i, j, k, x3Momentum_index), 2.) / state[bx](i, j, k, density_index) / 2.;
		}

		if (Temp > tempCeiling) {
			amrex::Real dummy = Etot - state[bx](i, j, k, internalEnergy_index);
			state[bx](i, j, k, internalEnergy_index) = tempCeiling * (kb * (state[bx](i, j, k, density_index) / Const_mH / 0.6)) / (gamma_ - 1.0);
			state[bx](i, j, k, energy_index) = dummy + state[bx](i, j, k, internalEnergy_index);
		}

		if (Temp < tempFloor) {
			amrex::Real dummy = Etot - state[bx](i, j, k, internalEnergy_index);
			state[bx](i, j, k, internalEnergy_index) = tempFloor * (kb * (state[bx](i, j, k, density_index) / Const_mH / 0.6)) / (gamma_ - 1.0);
			state[bx](i, j, k, energy_index) = dummy + state[bx](i, j, k, internalEnergy_index);
		}

		/*if(inj_scalar<0.0) {
		  state(i, j, k, Physics_Indices<NewProblem>::pscalarFirstIndex+2) = 0.0;
		}*/

		if (!is_eos_isothermal()) {
			// recompute gas energy (to prevent P < 0)
			amrex::Real const Eint_star = Etot - 0.5 * rho_new * vsq;
			amrex::Real const P_star = Eint_star * (gamma_ - 1.);
			amrex::Real P_new = P_star;
			if (P_star < P_floor) {
				P_new = P_floor;
#pragma nv_diag_suppress divide_by_zero
				amrex::Real const Etot_new = P_new / (gamma_ - 1.) + 0.5 * rho_new * vsq;
				state[bx](i, j, k, energy_index) = Etot_new;
			}
		}
	});
}
/**************************End Adding User Defined Limits on Variables **************/

auto problem_main() -> int
{

	const int nvars = RadhydroSimulation<NewProblem>::nvarTotal_cc_;
	amrex::Vector<amrex::BCRec> boundaryConditions(nvars);

	/*Implementing Outflowing Boundary Conditions in the Z-direction*/

	for (int n = 0; n < nvars; ++n) {
		for (int i = 0; i < AMREX_SPACEDIM; ++i) {
			// outflowing boundary conditions
			if (i == 2) {
				boundaryConditions[n].setLo(i, amrex::BCType::foextrap);
				boundaryConditions[n].setHi(i, amrex::BCType::foextrap);
			} else {
				boundaryConditions[n].setLo(i, amrex::BCType::int_dir); // periodic
				boundaryConditions[n].setHi(i, amrex::BCType::int_dir); // periodic
			}
		}
	}

	/**For a Fully Periodic Box*/
	// for (int n = 0; n < nvars; ++n) {
	// 	for (int i = 0; i < AMREX_SPACEDIM; ++i) {

	//         boundaryConditions[n].setLo(i, amrex::BCType::int_dir); // periodic
	//         boundaryConditions[n].setHi(i, amrex::BCType::int_dir); // periodic

	//       }}

	// Problem initialization
	RadhydroSimulation<NewProblem> sim(boundaryConditions);
	sim.reconstructionOrder_ = 3; // 2=PLM, 3=PPM
	sim.cflNumber_ = 0.2;	      // *must* be less than 1/3 in 3D!

	readCloudyData(sim.userData_.cloudyTables);
	// initialize
	sim.setInitialConditions();

	// evolve
	sim.evolve();

	// Cleanup and exit
	amrex::Print() << "Finished." << std::endl;
	return 0;
}
