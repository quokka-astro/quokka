
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

#include "AMReX_TableData.H"
#include "QuokkaSimulation.hpp"
#include "hydro/hydro_system.hpp"
#include "math/FastMath.hpp"
#include "math/quadrature.hpp"
#include "radiation/radiation_system.hpp"
#include "test_sne.hpp"

// global variables needed for Dirichlet boundary condition and initial conditions
#if 0 // workaround AMDGPU compiler bug
namespace
{
#endif
Real rho0 = NAN;			// NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
AMREX_GPU_MANAGED Real Tgas0 = NAN;	// NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
AMREX_GPU_MANAGED Real P_outflow = NAN; // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
std::string input_data_file;		//
AMREX_GPU_MANAGED amrex::GpuArray<amrex::Real, 64> logphi_data{
    0.83638381, 4.50705067, 5.10271383, 5.45268878, 5.70140736, 5.89447928, 6.05229308, 6.1857468,  6.30135334, 6.40331919, 6.49451684, 6.57699822, 6.65227803,
    6.72150811, 6.78558362, 6.84521466, 6.90097381, 6.95332936, 7.00266958, 7.04931937, 7.09355406, 7.1356083,	7.17568429, 7.21395706, 7.25057937, 7.28568514,
    7.31939257, 7.35180662, 7.38302073, 7.41311872, 7.44217572, 7.47025967, 7.49743193, 7.52374829, 7.54925951, 7.57401195, 7.5980481,	7.62140688, 7.64412421,
    7.66623309, 7.68776413, 7.70874559, 7.72920373, 7.74916295, 7.76864596, 7.78767394, 7.80626666, 7.82444266, 7.84221924, 7.85961268, 7.87663823, 7.89331028,
    7.90964238, 7.92564731, 7.94133718, 7.95672339, 7.9718168,	7.98662762, 8.00116561, 8.01543998, 8.02945951, 8.04323254, 8.056767,	8.07007045};
AMREX_GPU_MANAGED amrex::GpuArray<amrex::Real, 64> logg_data{
    -11.24587667, -10.68390187, -10.45405356, -10.31158848, -10.21082455, -10.13474898, -10.0750678, -10.02714739, -9.98806155, -9.95589857, -9.92931504,
    -9.90722171,  -9.88888792,	-9.87375864,  -9.86122421,  -9.85088355,  -9.84247276,	-9.83565399, -9.83007266,  -9.8256095,	-9.82214665, -9.81939435,
    -9.81722964,  -9.81559821,	-9.81433642,  -9.8133616,   -9.81265555,  -9.81214602,	-9.81177066, -9.81149643,  -9.81129813, -9.81115635, -9.8110556,
    -9.81098525,  -9.81093618,	-9.81090247,  -9.81087945,  -9.81086374,  -9.81085319,	-9.81084604, -9.8108412,   -9.81083792, -9.81083566, -9.81083407,
    -9.81083293,  -9.81083206,	-9.81083139,  -9.81083083,  -9.81083035,  -9.81082991,	-9.81082951, -9.81082913,  -9.81082876, -9.8108284,  -9.81082804,
    -9.81082769,  -9.81082735,	-9.810827,    -9.81082666,  -9.81082632,  -9.81082599,	-9.81082565, -9.81082532,  -9.81082499};
AMREX_GPU_MANAGED amrex::GpuArray<amrex::Real, 64> z_data{
    1.50900000e+20, 5.55695238e+20, 9.60490476e+20, 1.36528571e+21, 1.77008095e+21, 2.17487619e+21, 2.57967143e+21, 2.98446667e+21,
    3.38926190e+21, 3.79405714e+21, 4.19885238e+21, 4.60364762e+21, 5.00844286e+21, 5.41323810e+21, 5.81803333e+21, 6.22282857e+21,
    6.62762381e+21, 7.03241905e+21, 7.43721429e+21, 7.84200952e+21, 8.24680476e+21, 8.65160000e+21, 9.05639524e+21, 9.46119048e+21,
    9.86598571e+21, 1.02707810e+22, 1.06755762e+22, 1.10803714e+22, 1.14851667e+22, 1.18899619e+22, 1.22947571e+22, 1.26995524e+22,
    1.31043476e+22, 1.35091429e+22, 1.39139381e+22, 1.43187333e+22, 1.47235286e+22, 1.51283238e+22, 1.55331190e+22, 1.59379143e+22,
    1.63427095e+22, 1.67475048e+22, 1.71523000e+22, 1.75570952e+22, 1.79618905e+22, 1.83666857e+22, 1.87714810e+22, 1.91762762e+22,
    1.95810714e+22, 1.99858667e+22, 2.03906619e+22, 2.07954571e+22, 2.12002524e+22, 2.16050476e+22, 2.20098429e+22, 2.24146381e+22,
    2.28194333e+22, 2.32242286e+22, 2.36290238e+22, 2.40338190e+22, 2.44386143e+22, 2.48434095e+22, 2.52482048e+22, 2.56530000e+22};

AMREX_GPU_MANAGED Real z_star = 245.0 * pc;
AMREX_GPU_MANAGED Real Sigma_star = 1.71 * Msun / pc / pc;
AMREX_GPU_MANAGED Real rho_dm = 1.4e-3 * Msun / pc / pc / pc;
AMREX_GPU_MANAGED Real R0 = 16.e3 * pc;
AMREX_GPU_MANAGED Real ks_sigma_sfr = 5.499927024044233e-57;
AMREX_GPU_MANAGED Real hscale = 300. * pc;
AMREX_GPU_MANAGED Real sigma1 = 11. * kmps;
AMREX_GPU_MANAGED Real sigma2 = 10. * 11. * kmps;
AMREX_GPU_MANAGED Real rho01 = 0.0268988 * Const_mH;
AMREX_GPU_MANAGED Real rho02 = 1.e-5 * 0.0268988 * Const_mH;
;
#if 0 // workaround AMDGPU compiler bug
};                       // namespace
#endif

using amrex::Real;
using namespace amrex;

#define MAX 100

struct NewProblem {
	amrex::Real dummy;
};

template <> struct HydroSystem_Traits<NewProblem> {
	static constexpr double gamma = 5. / 3.;
	static constexpr bool reconstruct_eint = true; // Set to true - temperature
};

template <> struct quokka::EOS_Traits<NewProblem> {
	static constexpr double gamma = 5. / 3.;
	static constexpr double mean_molecular_weight = C::m_u;
	static constexpr double boltzmann_constant = C::k_B;
};

template <> struct Physics_Traits<NewProblem> {
	static constexpr bool is_hydro_enabled = true;
	static constexpr bool is_radiation_enabled = false;
	static constexpr bool is_chemistry_enabled = false;
	static constexpr bool is_mhd_enabled = false;
	static constexpr int numMassScalars = 0;    // number of mass scalars
	static constexpr int numPassiveScalars = 1; // number of passive scalars
	static constexpr int nGroups = 1;	    // number of radiation groups
};

template <> struct SimulationData<NewProblem> {

	// cloudy_tables cloudyTables;
	std::unique_ptr<amrex::TableData<Real, 3>> table_data;

	std::unique_ptr<amrex::TableData<Real, 1>> blast_x;
	std::unique_ptr<amrex::TableData<Real, 1>> blast_y;
	std::unique_ptr<amrex::TableData<Real, 1>> blast_z;

	int nblast = 0;
	int SN_counter_cumulative = 0;
	Real SN_rate_per_vol = NAN;  // rate per unit time per unit volume
	Real E_blast = 1.0e51;	     // ergs
	Real M_ejecta = 5.0 * Msun;  // 5.0 * Msun; // g
	Real refine_threshold = 1.0; // gradient refinement threshold
};

template <> void QuokkaSimulation<NewProblem>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{

	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = grid_elem.dx_;
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = grid_elem.prob_lo_;
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_hi = grid_elem.prob_hi_;
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const amrex::Array4<double> &state_cc = grid_elem.array_;

	double vol = AMREX_D_TERM(dx[0], *dx[1], *dx[2]);

	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		amrex::Real const z = prob_lo[2] + (k + amrex::Real(0.5)) * dx[2];

		//Calculate DM Potential
		double prefac;
		prefac = 2. * 3.1415 * Const_G * rho_dm * std::pow(R0, 2);
		double Phidm = (prefac * std::log(1. + std::pow(z / R0, 2)));

		//Calculate Stellar Disk Potential
		double prefac2;
		prefac2 = 2. * 3.1415 * Const_G * Sigma_star * z_star;
		double Phist = prefac2 * (std::pow(1. + z * z / z_star / z_star, 0.5) - 1.);

		//Calculate Gas Disk Potential

		double Phigas;
		// Interpolate to find the accurate g-value from array-- because linterp doesn't work on Setonix
		size_t ii = 0;
		double x_interp = std::abs(z);
		amrex::GpuArray<amrex::Real, 64> xx = z_data;
		amrex::GpuArray<amrex::Real, 64> yy = logphi_data;
		while (ii < xx.size() - 1 && x_interp > xx[ii + 1]) {
			ii++;
		}

		// Perform linear interpolation
		const Real x1 = xx[ii];
		const Real x2 = xx[ii + 1];
		const Real y1 = yy[ii];
		const Real y2 = yy[ii + 1];
		amrex::Real phi_interp = (y1 + (y2 - y1) * (x_interp - x1) / (x2 - x1));
		Phigas = FastMath::pow10(phi_interp);

		double Phitot = Phist + Phidm + Phigas;

		double rho, rho_disk, rho_halo;
		rho_disk = rho01 * std::exp(-Phitot / std::pow(sigma1, 2.0));
		rho_halo = rho02 * std::exp(-Phitot / std::pow(sigma2, 2.0)); // in g/cc
		rho = (rho_disk + rho_halo);

		double P = rho_disk * std::pow(sigma1, 2.0) + rho_halo * std::pow(sigma2, 2.0);

		AMREX_ASSERT(!std::isnan(rho));

		const auto gamma = HydroSystem<NewProblem>::gamma_;

		state_cc(i, j, k, HydroSystem<NewProblem>::density_index) = rho;
		state_cc(i, j, k, HydroSystem<NewProblem>::x1Momentum_index) = 0.0;
		state_cc(i, j, k, HydroSystem<NewProblem>::x2Momentum_index) = 0.0;
		state_cc(i, j, k, HydroSystem<NewProblem>::x3Momentum_index) = 0.0;
		state_cc(i, j, k, HydroSystem<NewProblem>::internalEnergy_index) = P / (gamma - 1.);
		state_cc(i, j, k, HydroSystem<NewProblem>::energy_index) = P / (gamma - 1.);
		state_cc(i, j, k, Physics_Indices<NewProblem>::pscalarFirstIndex) = 1.e-5 / vol; // Injected tracer
	});
}

void AddSupernova(amrex::MultiFab &mf, amrex::GpuArray<Real, AMREX_SPACEDIM> prob_lo, amrex::GpuArray<Real, AMREX_SPACEDIM> prob_hi,
		  amrex::GpuArray<Real, AMREX_SPACEDIM> dx, SimulationData<NewProblem> const &userData, int level)
{
	// inject energy into cells with stochastic sampling
	BL_PROFILE("QuokkaSimulation::Addsupernova()")

	const Real cell_vol = AMREX_D_TERM(dx[0], *dx[1], *dx[2]); // cm^3
	const Real rho_eint_blast = userData.E_blast / cell_vol;   // ergs cm^-3
	const Real rho_blast = userData.M_ejecta / cell_vol;	   // g cm^-3
	const int cum_sn = userData.SN_counter_cumulative;

	for (amrex::MFIter iter(mf); iter.isValid(); ++iter) {
		const amrex::Box &box = iter.validbox();
		auto const &state = mf.array(iter);
		auto const &px = userData.blast_x->table();
		auto const &py = userData.blast_y->table();
		auto const &pz = userData.blast_z->table();
		const int np = userData.nblast;

		amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
			const Real xc = prob_lo[0] + static_cast<Real>(i) * dx[0] + 0.5 * dx[0];
			const Real yc = prob_lo[1] + static_cast<Real>(j) * dx[1] + 0.5 * dx[1];
			const Real zc = prob_lo[2] + static_cast<Real>(k) * dx[2] + 0.5 * dx[2];

			for (int n = 0; n < np; ++n) {
				Real x0 = NAN;
				Real y0 = NAN;
				Real z0 = NAN;
				Real Rpds = 0.0;

				x0 = std::abs(xc - px(n));
				y0 = std::abs(yc - py(n));
				z0 = std::abs(zc - pz(n));

				if (x0 < 0.5 * dx[0] && y0 < 0.5 * dx[1] && z0 < 0.5 * dx[2]) {
					state(i, j, k, HydroSystem<NewProblem>::density_index) += rho_blast;
					state(i, j, k, HydroSystem<NewProblem>::energy_index) += rho_eint_blast;
					state(i, j, k, HydroSystem<NewProblem>::internalEnergy_index) += rho_eint_blast;
					state(i, j, k, Physics_Indices<NewProblem>::pscalarFirstIndex) += 1.e3 / cell_vol;

					printf("The total number of SN gone off=%d\n", cum_sn);
					Rpds = 14. * std::pow(state(i, j, k, HydroSystem<NewProblem>::density_index) / Const_mH, -3. / 7.);
					printf("Rpds = %.2e pc\n", Rpds);
				}
			}
		});
	}
}

template <> void QuokkaSimulation<NewProblem>::computeBeforeTimestep()
{
	// compute how many (and where) SNe will go off on the this coarse timestep
	// sample from Poisson distribution

	const Real dt_coarse = dt_[0];
	const Real domain_area = geom[0].ProbLength(0) * geom[0].ProbLength(1);
	const Real mean = 0.0;
	const Real stddev = hscale / geom[0].ProbLength(2) / 2.;

	const Real expectation_value = ks_sigma_sfr * domain_area * dt_coarse;

	const int count = static_cast<int>(amrex::RandomPoisson(expectation_value));

	// resize particle arrays
	amrex::Array<int, 1> const lo{0};
	amrex::Array<int, 1> const hi{count};
	userData_.blast_x = std::make_unique<amrex::TableData<Real, 1>>(lo, hi, amrex::The_Pinned_Arena());
	userData_.blast_y = std::make_unique<amrex::TableData<Real, 1>>(lo, hi, amrex::The_Pinned_Arena());
	userData_.blast_z = std::make_unique<amrex::TableData<Real, 1>>(lo, hi, amrex::The_Pinned_Arena());
	userData_.nblast = count;
	userData_.SN_counter_cumulative += count;

	// for each, sample location at random
	auto const &px = userData_.blast_x->table();
	auto const &py = userData_.blast_y->table();
	auto const &pz = userData_.blast_z->table();
	for (int i = 0; i < count; ++i) {
		px(i) = geom[0].ProbLength(0) * amrex::Random();
		py(i) = geom[0].ProbLength(1) * amrex::Random();
		pz(i) = geom[0].ProbLength(2) * amrex::RandomNormal(mean, stddev);
	}
}

template <> void QuokkaSimulation<NewProblem>::computeAfterLevelAdvance(int lev, amrex::Real time, amrex::Real dt_lev, int ncycle)
{
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = geom[lev].ProbLoArray();
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_hi = geom[lev].ProbHiArray();
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx = geom[lev].CellSizeArray();

	AddSupernova(state_new_cc_[lev], prob_lo, prob_hi, dx, userData_, lev);
}

template <>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE auto
HydroSystem<NewProblem>::GetGradFixedPotential(amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> posvec) -> amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>
{

	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> grad_potential;

	double x = posvec[0];

	grad_potential[0] = 0.0;

#if (AMREX_SPACEDIM >= 2)
	double y = posvec[1];
	grad_potential[1] = 0.0;
#endif
#if (AMREX_SPACEDIM >= 3)
	double z = posvec[2];

	// Interpolate to find the accurate g-value from array-- because linterp doesn't work on Setonix
	size_t i = 0;
	double x_interp = std::abs(z);
	amrex::GpuArray<amrex::Real, 64> xx = z_data;
	amrex::GpuArray<amrex::Real, 64> yy = logg_data;
	while (i < xx.size() - 1 && x_interp > xx[i + 1]) {
		i++;
	}

	// Perform linear interpolation
	const Real x1 = xx[i];
	const Real x2 = xx[i + 1];
	const Real y1 = yy[i];
	const Real y2 = yy[i + 1];

	amrex::Real ginterp = (y1 + (y2 - y1) * (x_interp - x1) / (x2 - x1));

	grad_potential[2] = 2. * 3.1415 * Const_G * rho_dm * std::pow(R0, 2) * (2. * z / std::pow(R0, 2)) / (1. + std::pow(z, 2) / std::pow(R0, 2));
	grad_potential[2] += 2. * 3.1415 * Const_G * Sigma_star * (z / z_star) * (std::pow(1. + z * z / (z_star * z_star), -0.5));
	grad_potential[2] += (z / std::abs(z)) * FastMath::pow10(ginterp);

#endif

	return grad_potential;
}

// Add Strang Split Source Term for External Fixed Potential Here 
template <> void QuokkaSimulation<NewProblem>::addStrangSplitSources(amrex::MultiFab &mf, int lev, amrex::Real time, amrex::Real dt_lev)
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

			Real Eint = RadSystem<NewProblem>::ComputeEintFromEgas(rho, x1mom, x2mom, x3mom, Egas);

			posvec[0] = prob_lo[0] + (i + 0.5) * dx[0];

#if (AMREX_SPACEDIM >= 2)
			posvec[1] = prob_lo[1] + (j + 0.5) * dx[1];
#endif

#if (AMREX_SPACEDIM >= 3)
			posvec[2] = prob_lo[2] + (k + 0.5) * dx[2];
#endif

			GradPhi = HydroSystem<NewProblem>::GetGradFixedPotential(posvec);

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



//Code for producing inistu Projection plots
template <> auto QuokkaSimulation<NewProblem>::ComputeProjections(const int dir) const -> std::unordered_map<std::string, amrex::BaseFab<amrex::Real>>
{
	// compute density projection
	std::unordered_map<std::string, amrex::BaseFab<amrex::Real>> proj;

	proj["mass_outflow"] = computePlaneProjection<amrex::ReduceOpSum>(
	    [=] AMREX_GPU_DEVICE(int i, int j, int k, amrex::Array4<const Real> const &state) noexcept {
		    // int nmscalars = Physics_Traits<NewProblem>::numMassScalars;
		    Real const rho = state(i, j, k, HydroSystem<NewProblem>::density_index);
		    Real const vx3 = state(i, j, k, HydroSystem<NewProblem>::x3Momentum_index) / rho;

		    amrex::Real Eint = state(i, j, k, HydroSystem<NewProblem>::internalEnergy_index);
		    amrex::GpuArray<Real, 0> massScalars = RadSystem<NewProblem>::ComputeMassScalars(state, i, j, k);
		    return (rho * vx3);
	    },
	    dir);

	proj["hot_mass_outflow"] = computePlaneProjection<amrex::ReduceOpSum>(
	    [=] AMREX_GPU_DEVICE(int i, int j, int k, amrex::Array4<const Real> const &state) noexcept {
		    double flux;
		    Real const rho = state(i, j, k, HydroSystem<NewProblem>::density_index);
		    Real const vx3 = state(i, j, k, HydroSystem<NewProblem>::x3Momentum_index) / rho;
		    Real const Eint = state(i, j, k, HydroSystem<NewProblem>::internalEnergy_index);
		    amrex::GpuArray<Real, 0> massScalars = RadSystem<NewProblem>::ComputeMassScalars(state, i, j, k);
		    Real const primTemp = quokka::EOS<NewProblem>::ComputeTgasFromEint(rho, Eint, massScalars);
		    if (primTemp > 1.e6) {
			    flux = rho * vx3;
		    } else {
			    flux = 0.0;
		    }
		    return flux;
	    },
	    dir);

	proj["warm_mass_outflow"] = computePlaneProjection<amrex::ReduceOpSum>(
	    [=] AMREX_GPU_DEVICE(int i, int j, int k, amrex::Array4<const Real> const &state) noexcept {
		    double flux;
		    Real const rho = state(i, j, k, HydroSystem<NewProblem>::density_index);
		    Real const vx3 = state(i, j, k, HydroSystem<NewProblem>::x3Momentum_index) / rho;
		    Real const Eint = state(i, j, k, HydroSystem<NewProblem>::internalEnergy_index);
		    amrex::GpuArray<Real, 0> massScalars = RadSystem<NewProblem>::ComputeMassScalars(state, i, j, k);
		    Real const primTemp = quokka::EOS<NewProblem>::ComputeTgasFromEint(rho, Eint, massScalars);
		    if (primTemp < 2.e4) {
			    flux = rho * vx3;
		    } else {
			    flux = 0.0;
		    }
		    return flux;
	    },
	    dir);

	proj["scalar_outflow"] = computePlaneProjection<amrex::ReduceOpSum>(
	    [=] AMREX_GPU_DEVICE(int i, int j, int k, amrex::Array4<const Real> const &state) noexcept {
		    Real const rho = state(i, j, k, HydroSystem<NewProblem>::density_index);
		    Real const rhoZ = state(i, j, k, Physics_Indices<NewProblem>::pscalarFirstIndex);
		    Real const vz = state(i, j, k, HydroSystem<NewProblem>::x3Momentum_index) / rho;
		    return (rhoZ * vz);
	    },
	    dir);

	proj["warm_scalar_outflow"] = computePlaneProjection<amrex::ReduceOpSum>(
	    [=] AMREX_GPU_DEVICE(int i, int j, int k, amrex::Array4<const Real> const &state) noexcept {
		    double flux;
		    Real const rho = state(i, j, k, HydroSystem<NewProblem>::density_index);
		    Real const rhoZ = state(i, j, k, Physics_Indices<NewProblem>::pscalarFirstIndex);
		    Real const vx3 = state(i, j, k, HydroSystem<NewProblem>::x3Momentum_index) / rho;
		    Real const Eint = state(i, j, k, HydroSystem<NewProblem>::internalEnergy_index);
		    amrex::GpuArray<Real, 0> massScalars = RadSystem<NewProblem>::ComputeMassScalars(state, i, j, k);
		    Real const primTemp = quokka::EOS<NewProblem>::ComputeTgasFromEint(rho, Eint, massScalars);
		    if (primTemp < 2.e4) {
			    flux = rhoZ * vx3;
		    } else {
			    flux = 0.0;
		    }
		    return flux;
	    },
	    dir);

	proj["hot_scalar_outflow"] = computePlaneProjection<amrex::ReduceOpSum>(
	    [=] AMREX_GPU_DEVICE(int i, int j, int k, amrex::Array4<const Real> const &state) noexcept {
		    double flux;
		    Real const rho = state(i, j, k, HydroSystem<NewProblem>::density_index);
		    Real const rhoZ = state(i, j, k, Physics_Indices<NewProblem>::pscalarFirstIndex);
		    Real const vx3 = state(i, j, k, HydroSystem<NewProblem>::x3Momentum_index) / rho;
		    Real const Eint = state(i, j, k, HydroSystem<NewProblem>::internalEnergy_index);
		    amrex::GpuArray<Real, 0> massScalars = RadSystem<NewProblem>::ComputeMassScalars(state, i, j, k);
		    Real const primTemp = quokka::EOS<NewProblem>::ComputeTgasFromEint(rho, Eint, massScalars);
		    if (primTemp > 1.e6) {
			    flux = rhoZ * vx3;
		    } else {
			    flux = 0.0;
		    }
		    return flux;
	    },
	    dir);

	proj["rho"] = computePlaneProjection<amrex::ReduceOpSum>(
	    [=] AMREX_GPU_DEVICE(int i, int j, int k, amrex::Array4<const Real> const &state) noexcept {
		    Real const rho = state(i, j, k, HydroSystem<NewProblem>::density_index);
		    return (rho);
	    },
	    dir);

	proj["scalar"] = computePlaneProjection<amrex::ReduceOpSum>(
	    [=] AMREX_GPU_DEVICE(int i, int j, int k, amrex::Array4<const Real> const &state) noexcept {
		    Real const rhoZ = state(i, j, k, Physics_Indices<NewProblem>::pscalarFirstIndex);
		    return (rhoZ);
	    },
	    dir);
	return proj;
}

//Implement User-defined diode BC 
template <>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE void AMRSimulation<NewProblem>::setCustomBoundaryConditions(const amrex::IntVect &iv, amrex::Array4<Real> const &consVar,
												int /*dcomp*/, int /*numcomp*/, amrex::GeometryData const &geom,
												const Real /*time*/, const amrex::BCRec * /*bcr*/,
												int /*bcomp*/, int /*orig_comp*/)
{
	auto [i, j, k] = iv.dim3();
	amrex::Box const &box = geom.Domain();
	const auto &domain_lo = box.loVect3d();
	const auto &domain_hi = box.hiVect3d();
	const int klo = domain_lo[2];
	const int khi = domain_hi[2];
	int kedge = 0;
	int normal =0;

	if (k < klo) {
		kedge = klo;
		normal = -1;
	} else if (k > khi) {
		kedge = khi;
		normal = 1.0;
	}

	const double rho_edge = consVar(i, j, kedge, HydroSystem<NewProblem>::density_index);
	const double x1Mom_edge = consVar(i, j, kedge, HydroSystem<NewProblem>::x1Momentum_index);
	const double x2Mom_edge = consVar(i, j, kedge, HydroSystem<NewProblem>::x2Momentum_index);
	double x3Mom_edge = consVar(i, j, kedge, HydroSystem<NewProblem>::x3Momentum_index);
	const double etot_edge = consVar(i, j, kedge, HydroSystem<NewProblem>::energy_index);
	const double eint_edge = consVar(i, j, kedge, HydroSystem<NewProblem>::internalEnergy_index);

	if ((x3Mom_edge * normal) < 0) { // gas is inflowing
		x3Mom_edge = -1. * consVar(i, j, kedge, HydroSystem<NewProblem>::x3Momentum_index);
	}

	consVar(i, j, k, HydroSystem<NewProblem>::density_index) = rho_edge;
	consVar(i, j, k, HydroSystem<NewProblem>::x1Momentum_index) = x1Mom_edge;
	consVar(i, j, k, HydroSystem<NewProblem>::x2Momentum_index) = x2Mom_edge;
	consVar(i, j, k, HydroSystem<NewProblem>::x3Momentum_index) = x3Mom_edge;
	consVar(i, j, k, HydroSystem<NewProblem>::energy_index) = etot_edge;
	consVar(i, j, k, HydroSystem<NewProblem>::internalEnergy_index) = eint_edge;
}

auto problem_main() -> int
{

	const int ncomp_cc = Physics_Indices<NewProblem>::nvarTotal_cc;
	amrex::Vector<amrex::BCRec> BCs_cc(ncomp_cc);

	for (int n = 0; n < ncomp_cc; ++n) {
		for (int i = 0; i < AMREX_SPACEDIM; ++i) {
			// diode boundary conditions
			if (i == 2) {
				BCs_cc[n].setLo(i, amrex::BCType::ext_dir);
				BCs_cc[n].setHi(i, amrex::BCType::ext_dir);
			} else {
				BCs_cc[n].setLo(i, amrex::BCType::int_dir); // periodic
				BCs_cc[n].setHi(i, amrex::BCType::int_dir); // periodic
			}
		}
	}

	// Problem initialization
	QuokkaSimulation<NewProblem> sim(BCs_cc);

	sim.reconstructionOrder_ = 3; // 2=PLM, 3=PPM
	sim.cflNumber_ = 0.3;	      // *must* be less than 1/3 in 3D!

	sim.setInitialConditions();

	// evolve
	sim.evolve();

	// Cleanup and exit
	amrex::Print() << "Finished." << std::endl;
	return 0;
}
