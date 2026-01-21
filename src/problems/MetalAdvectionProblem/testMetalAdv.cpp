
//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file test_sne.cpp
/// \brief Defines a problem for disk galaxy ISM.
///

#include <cmath>
#include <iostream>

#include "AMReX.H"
#include "AMReX_BC_TYPES.H"
#include "AMReX_BLassert.H"
#include "AMReX_MultiFab.H"
#include "AMReX_ParmParse.H"
#include "AMReX_Print.H"
#include "AMReX_Random.H"
#include "AMReX_SPACE.H"
#include "AMReX_TableData.H"

#include "QuokkaSimulation.hpp"
#include "hydro/hydro_system.hpp"
#include "radiation/radiation_system.hpp"

// global variables needed for Dirichlet boundary condition and initial conditions


//########----Values for R8 Model-------################

AMREX_GPU_MANAGED amrex::GpuArray<amrex::Real, 100> logphi_data{9.729282, 10.623358, 11.041888, 11.309771, 11.501713,
						11.647868, 11.763742, 11.858439, 11.937747, 12.005546,
						12.064526, 12.116602, 12.163168, 12.205251, 12.243627,
						12.278891, 12.311509, 12.341847, 12.370204, 12.396823,
						12.421904, 12.445615, 12.468099, 12.489475, 12.509849,
						12.529309, 12.547934, 12.565794, 12.582948, 12.599450,
						12.615348, 12.630684, 12.645497, 12.659822, 12.673689,
						12.687127, 12.700161, 12.712816, 12.725112, 12.737070,
						12.748708, 12.760041, 12.771086, 12.781858, 12.792368,
						12.802631, 12.812656, 12.822455, 12.832038, 12.841414,
						12.850592, 12.859579, 12.868385, 12.877016, 12.885478,
						12.893779, 12.901924, 12.909919, 12.917769, 12.925480,
						12.933057, 12.940503, 12.947824, 12.955024, 12.962106,
						12.969075, 12.975934, 12.982685, 12.989334, 12.995882,
						13.002333, 13.008690, 13.014955, 13.021131, 13.027220,
						13.033225, 13.039148, 13.044991, 13.050757, 13.056447,
						13.062064, 13.067609, 13.073084, 13.078491, 13.083831,
						13.089107, 13.094319, 13.099469, 13.104559, 13.109590,
						13.114564, 13.119481, 13.124343, 13.129151, 13.133907,
						13.138611, 13.143265, 13.147869, 13.152425, 13.156934
								};


AMREX_GPU_MANAGED amrex::GpuArray<amrex::Real, 100> logg_data{-9.821986, -9.373092, -9.181660, -9.074428, -9.010522,
					-8.972159, -8.949421, -8.936419, -8.929211, -8.925341,
					-8.923377, -8.922390, -8.921909, -8.921679, -8.921572,
					-8.921523, -8.921500, -8.921489, -8.921484, -8.921482,
					-8.921480, -8.921479, -8.921478, -8.921477, -8.921476,
					-8.921476, -8.921475, -8.921474, -8.921473, -8.921473,
					-8.921472, -8.921471, -8.921471, -8.921470, -8.921469,
					-8.921469, -8.921468, -8.921467, -8.921467, -8.921466,
					-8.921465, -8.921465, -8.921464, -8.921464, -8.921463,
					-8.921462, -8.921462, -8.921461, -8.921461, -8.921460,
					-8.921460, -8.921459, -8.921459, -8.921458, -8.921458,
					-8.921457, -8.921457, -8.921457, -8.921456, -8.921456,
					-8.921455, -8.921455, -8.921454, -8.921454, -8.921454,
					-8.921453, -8.921453, -8.921453, -8.921452, -8.921452,
					-8.921451, -8.921451, -8.921451, -8.921450, -8.921450,
					-8.921450, -8.921449, -8.921449, -8.921449, -8.921449,
					-8.921448, -8.921448, -8.921448, -8.921448, -8.921447,
					-8.921447, -8.921447, -8.921447, -8.921446, -8.921446,
					-8.921446, -8.921446, -8.921445, -8.921445, -8.921445,
					-8.921445, -8.921445, -8.921444, -8.921444, -8.921444
};
AMREX_GPU_MANAGED amrex::GpuArray<amrex::Real, 100> z_data{6.186900e+19, 1.856070e+20, 3.093450e+20, 4.330830e+20, 5.568210e+20,
				6.805590e+20, 8.042970e+20, 9.280350e+20, 1.051773e+21, 1.175511e+21,
				1.299249e+21, 1.422987e+21, 1.546725e+21, 1.670463e+21, 1.794201e+21,
				1.917939e+21, 2.041677e+21, 2.165415e+21, 2.289153e+21, 2.412891e+21,
				2.536629e+21, 2.660367e+21, 2.784105e+21, 2.907843e+21, 3.031581e+21,
				3.155319e+21, 3.279057e+21, 3.402795e+21, 3.526533e+21, 3.650271e+21,
				3.774009e+21, 3.897747e+21, 4.021485e+21, 4.145223e+21, 4.268961e+21,
				4.392699e+21, 4.516437e+21, 4.640175e+21, 4.763913e+21, 4.887651e+21,
				5.011389e+21, 5.135127e+21, 5.258865e+21, 5.382603e+21, 5.506341e+21,
				5.630079e+21, 5.753817e+21, 5.877555e+21, 6.001293e+21, 6.125031e+21,
				6.248769e+21, 6.372507e+21, 6.496245e+21, 6.619983e+21, 6.743721e+21,
				6.867459e+21, 6.991197e+21, 7.114935e+21, 7.238673e+21, 7.362411e+21,
				7.486149e+21, 7.609887e+21, 7.733625e+21, 7.857363e+21, 7.981101e+21,
				8.104839e+21, 8.228577e+21, 8.352315e+21, 8.476053e+21, 8.599791e+21,
				8.723529e+21, 8.847267e+21, 8.971005e+21, 9.094743e+21, 9.218481e+21,
				9.342219e+21, 9.465957e+21, 9.589695e+21, 9.713433e+21, 9.837171e+21,
				9.960909e+21, 1.008465e+22, 1.020838e+22, 1.033212e+22, 1.045586e+22,
				1.057960e+22, 1.070334e+22, 1.082707e+22, 1.095081e+22, 1.107455e+22,
				1.119829e+22, 1.132203e+22, 1.144576e+22, 1.156950e+22, 1.169324e+22,
				1.181698e+22, 1.194072e+22, 1.206445e+22, 1.218819e+22, 1.231193e+22,

      };


AMREX_GPU_MANAGED Real z_star = 245.0 * C::parsec;
AMREX_GPU_MANAGED Real Sigma_star = 42.0 * C::M_solar/C::parsec/C::parsec;
AMREX_GPU_MANAGED Real rho_dm = 0.0064 * C::M_solar/C::parsec/C::parsec/C::parsec;
AMREX_GPU_MANAGED Real R0 = 8.e3 * C::parsec; 
AMREX_GPU_MANAGED Real ks_sigma_sfr = 2.088579882548443e-55; 
AMREX_GPU_MANAGED Real hscale= 150. * C::parsec;
AMREX_GPU_MANAGED Real sigma1 = 700000.0;
AMREX_GPU_MANAGED Real sigma2 = 7000000.0;
AMREX_GPU_MANAGED Real rho01 = 2.78556e-24;
AMREX_GPU_MANAGED Real rho02 = 2.7855600000000006e-29;;

//################----------------------##################################

AMREX_GPU_MANAGED Real hscaleIa= 2. * 150. * C::parsec; //should be twice of typeII scale height
AMREX_GPU_MANAGED Real hscaleAGB= 2. * 150. * C::parsec; //should 300 pc.

AMREX_GPU_MANAGED Real Tgas0 = 1.e4 ; //Temperature of gas ejected by AGB
AMREX_GPU_MANAGED Real kpc = 1.e3 * C::parsec;
struct NewProblem {
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
	static constexpr bool is_self_gravity_enabled = false;
	static constexpr bool is_radiation_enabled = false;
	static constexpr bool is_dust_enabled = false;
	static constexpr int nDustGroups = 1; // number of dust groups
	static constexpr bool is_chemistry_enabled = false;
	static constexpr bool is_mhd_enabled = true;
	static constexpr int numMassScalars = 0;    // number of mass scalars
	static constexpr int numPassiveScalars = 3; // number of passive scalars
	static constexpr int nGroups = 1;	    // number of radiation groups
	static constexpr UnitSystem unit_system = UnitSystem::CGS;
};

template <> struct SimulationData<NewProblem> {

	std::unique_ptr<amrex::TableData<Real, 1>> blast_x;
	std::unique_ptr<amrex::TableData<Real, 1>> blast_y;
	std::unique_ptr<amrex::TableData<Real, 1>> blast_z;

	std::unique_ptr<amrex::TableData<Real, 1>> blast_x1a;
	std::unique_ptr<amrex::TableData<Real, 1>> blast_y1a;
	std::unique_ptr<amrex::TableData<Real, 1>> blast_z1a;

	std::unique_ptr<amrex::TableData<Real, 1>> blast_xAGB;
	std::unique_ptr<amrex::TableData<Real, 1>> blast_yAGB;
	std::unique_ptr<amrex::TableData<Real, 1>> blast_zAGB;

	int nblast = 0;
	int nblast1a = 0;
	int nblastAGB = 0;
	int SN_counter_cumulative = 0;
	Real SN_rate_per_vol = NAN;  // rate per unit time per unit volume
	Real E_blast = 1.0e51;	     // ergs
	Real M_ejecta = 5.0 * C::M_solar;  // 5.0 * Msun; // g
	Real refine_threshold = 1.0; // gradient refinement threshold
	Real M_ejecta_AGB = 1.3 * C::M_solar;  // 5.0 * Msun; // g
};

template <> void QuokkaSimulation<NewProblem>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	double magnetic_field_microgauss = 2.6;
	double disk_zscale_kpc = 1.0;
	
	amrex::ParmParse const pp("magnetic_field");
	pp.query("magnetic_field_microgauss", magnetic_field_microgauss);
	pp.query("disk_zscale_kpc", disk_zscale_kpc);
	
	const double z_d = disk_zscale_kpc * (1.0e3 * C::parsec);
	const double B_0 = magnetic_field_microgauss * 1.0e-6 / std::sqrt(4.0 * M_PI);
	
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = grid_elem.dx_;
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = grid_elem.prob_lo_;
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const amrex::Array4<double> &state_cc = grid_elem.array_;

	double vol = AMREX_D_TERM(dx[0], *dx[1], *dx[2]);

	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		amrex::Real const z = prob_lo[2] + (k + amrex::Real(0.5)) * dx[2];

		//Get xmid and ymid
		amrex::Real const xlow = prob_lo[0] + (i * dx[0]);
		amrex::Real const ylow = prob_lo[1] + (j * dx[1]);
		amrex::Real const zlow = prob_lo[2] + (k * dx[2]);
		
		amrex::Real const xhigh = prob_lo[0] + ((i + 1) * dx[0]);
		amrex::Real const yhigh = prob_lo[1] + ((j + 1) * dx[1]);
		amrex::Real const zhigh = prob_lo[2] + ((k + 1) * dx[2]);

		amrex::Real const x_mid = 0.5 * (xlow + xhigh);
		amrex::Real const y_mid = 0.5 * (ylow + yhigh);
		amrex::Real const z_mid = 0.5 * (zlow + zhigh);
		
		// Calculate DM Potential
		double prefac;
		prefac = 2. * M_PI * C::Gconst * rho_dm * std::pow(R0, 2);
		double Phidm = (prefac * std::log(1. + std::pow(z_mid / R0, 2)));

		// Calculate Stellar Disk Potential
		double prefac2;
		prefac2 = 2. * M_PI * C::Gconst * Sigma_star * z_star;
		double Phist = prefac2 * (std::pow(1. + z_mid * z_mid / z_star / z_star, 0.5) - 1.);

		// Calculate Gas Disk Potential

		double Phigas;
		// Interpolate to find the accurate g-value from array-- because linterp doesn't work on Setonix
		// TODO - AV to find out why linterp doesn't work
		size_t ii = 0;
		double x_interp = std::abs(z);
		while (ii < z_data.size() - 1 && x_interp > z_data[ii + 1]) {
			ii++;
		}

		// Perform linear interpolation
		const Real x1 = z_data[ii];
		const Real x2 = z_data[ii + 1];
		const Real y1 = logphi_data[ii];
		const Real y2 = logphi_data[ii + 1];
		amrex::Real phi_interp = (y1 + (y2 - y1) * (x_interp - x1) / (x2 - x1));
		Phigas = std::pow(10., phi_interp);

		double Phitot = Phist + Phidm + Phigas;

		double rho, rho_disk, rho_halo;
		rho_disk = rho01 * std::exp(-Phitot / std::pow(sigma1, 2.0));
		rho_halo = rho02 * std::exp(-Phitot / std::pow(sigma2, 2.0)); // in g/cc
		rho = (rho_disk + rho_halo);

		double P = rho_disk * std::pow(sigma1, 2.0) + rho_halo * std::pow(sigma2, 2.0);
		
		//Set magnetic field components
		amrex::Real const B_y = B_0 * std::exp(-std::abs(z_mid) / z_d);

		amrex::Real const magnetic_energy_density = 0.5 * (B_y * B_y);
		amrex::Real const total_internal_energy = P / (HydroSystem<NewProblem>::gamma_ - 1.0) + magnetic_energy_density;
		AMREX_ASSERT(!std::isnan(rho));

		const auto gamma = HydroSystem<NewProblem>::gamma_;

		state_cc(i, j, k, HydroSystem<NewProblem>::density_index) = rho;
		state_cc(i, j, k, HydroSystem<NewProblem>::x1Momentum_index) = 0.0;
		state_cc(i, j, k, HydroSystem<NewProblem>::x2Momentum_index) = 0.0;
		state_cc(i, j, k, HydroSystem<NewProblem>::x3Momentum_index) = 0.0;
		state_cc(i, j, k, HydroSystem<NewProblem>::internalEnergy_index) = total_internal_energy;
		state_cc(i, j, k, HydroSystem<NewProblem>::energy_index) = total_internal_energy;
		state_cc(i, j, k, Physics_Indices<NewProblem>::pscalarFirstIndex) = 1.e-5 / vol; // Injected tracer
		state_cc(i, j, k, Physics_Indices<NewProblem>::pscalarFirstIndex+1) = 1.e-5 / vol; // Injected tracer 2
		state_cc(i, j, k, Physics_Indices<NewProblem>::pscalarFirstIndex+2) = 1.e-5 / vol; // Injected tracer 3
	});
}

void AddSupernova(amrex::MultiFab &mf, amrex::GpuArray<Real, AMREX_SPACEDIM> prob_lo, amrex::GpuArray<Real, AMREX_SPACEDIM> prob_hi,
		  amrex::GpuArray<Real, AMREX_SPACEDIM> dx, SimulationData<NewProblem> const &userData, int level)
{
	// TODO for AV - ave (and restore) the RNG state in the metadata.yaml file
	//  inject energy into cells with stochastic sampling
	BL_PROFILE("QuokkaSimulation::Addsupernova()")

	const Real cell_vol = AMREX_D_TERM(dx[0], *dx[1], *dx[2]); // cm^3
	const Real rho_eint_blast = userData.E_blast / cell_vol;   // ergs cm^-3
	const Real rho_blast = userData.M_ejecta / cell_vol;	   // g cm^-3
	const Real rho_blast_AGB = userData.M_ejecta_AGB / cell_vol;	   // g cm^-3
	const Real scalar_blast = 1.e3 / cell_vol;		   // g cm^-3
	const int cum_sn = userData.SN_counter_cumulative;

	for (amrex::MFIter iter(mf); iter.isValid(); ++iter) {
		const amrex::Box &box = iter.validbox();
		auto const &state = mf.array(iter);
		auto const &px = userData.blast_x->table();
		auto const &py = userData.blast_y->table();
		auto const &pz = userData.blast_z->table();
		const int np = userData.nblast;

		auto const &px1a = userData.blast_x1a->table();
		auto const &py1a = userData.blast_y1a->table();
		auto const &pz1a = userData.blast_z1a->table();
		const int np1a = userData.nblast1a;
		
		auto const &pxAGB = userData.blast_xAGB->table();
		auto const &pyAGB = userData.blast_yAGB->table();
		auto const &pzAGB = userData.blast_zAGB->table();
		const int npAGB = userData.nblastAGB;

		amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
			const Real xc = prob_lo[0] + static_cast<Real>(i) * dx[0] + 0.5 * dx[0];
			const Real yc = prob_lo[1] + static_cast<Real>(j) * dx[1] + 0.5 * dx[1];
			const Real zc = prob_lo[2] + static_cast<Real>(k) * dx[2] + 0.5 * dx[2];

			for (int n = 0; n < np; ++n) {
				Real x0 = std::abs(xc - px(n));
				Real y0 = std::abs(yc - py(n));
				Real z0 = std::abs(zc - pz(n));

				if (x0 < 0.5 * dx[0] && y0 < 0.5 * dx[1] && z0 < 0.5 * dx[2]) {
					state(i, j, k, HydroSystem<NewProblem>::density_index) += rho_blast;
					state(i, j, k, HydroSystem<NewProblem>::energy_index) += rho_eint_blast;
					state(i, j, k, HydroSystem<NewProblem>::internalEnergy_index) += rho_eint_blast;
					state(i, j, k, Physics_Indices<NewProblem>::pscalarFirstIndex) += scalar_blast;
				}
			}
				//Add SN1a
				for (int n = 0; n < np1a; ++n) {
				Real x0 = std::abs(xc - px1a(n));
				Real y0 = std::abs(yc - py1a(n));
				Real z0 = std::abs(zc - pz1a(n));

				if (x0 < 0.5 * dx[0] && y0 < 0.5 * dx[1] && z0 < 0.5 * dx[2]) {

					state(i, j, k, HydroSystem<NewProblem>::density_index) += rho_blast;
					state(i, j, k, HydroSystem<NewProblem>::energy_index) += rho_eint_blast;
					state(i, j, k, HydroSystem<NewProblem>::internalEnergy_index) += rho_eint_blast;
					state(i, j, k, Physics_Indices<NewProblem>::pscalarFirstIndex+1) += scalar_blast;

				}
			}

			//Add AGB
				for (int n = 0; n < npAGB; ++n) {
				Real x0 = std::abs(xc - pxAGB(n));
				Real y0 = std::abs(yc - pyAGB(n));
				Real z0 = std::abs(zc - pzAGB(n));

				if (x0 < 0.5 * dx[0] && y0 < 0.5 * dx[1] && z0 < 0.5 * dx[2]) {
					double rho0 = state(i, j, k, HydroSystem<NewProblem>::density_index) ;

					Real const Eint = quokka::EOS<NewProblem>::ComputeEintFromTgas(rho0, Tgas0);

					state(i, j, k, HydroSystem<NewProblem>::density_index) += rho_blast_AGB;
					state(i, j, k, HydroSystem<NewProblem>::energy_index) += Eint;
					state(i, j, k, HydroSystem<NewProblem>::internalEnergy_index) += Eint;
					state(i, j, k, Physics_Indices<NewProblem>::pscalarFirstIndex+2) += scalar_blast;
				}
			}

		});
		amrex::Print() << "The total number of TypeII+TypeIa SN gone off="<<  userData.SN_counter_cumulative << "\n";
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
	const Real stddev1a = hscaleIa / geom[0].ProbLength(2) / 2.;
	const Real stddevAGB = hscaleAGB / geom[0].ProbLength(2) / 2.;
	const Real frac_rate_typeII = 0.6; // fraction of SN that are type II
	const Real expectation_value = ks_sigma_sfr*frac_rate_typeII * domain_area * dt_coarse;

	const Real expectation_value1a = (ks_sigma_sfr*(1.-frac_rate_typeII)) * domain_area * dt_coarse; 

	const Real expectation_valueAGB = (ks_sigma_sfr*16.0) * domain_area * dt_coarse;

	const int count = static_cast<int>(amrex::RandomPoisson(expectation_value));
	const int count1a = static_cast<int>(amrex::RandomPoisson(expectation_value1a));
	      int countAGB = static_cast<int>(amrex::RandomPoisson(expectation_valueAGB));

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
	amrex::Print() << "Number of Type II SN going off in this step:" << count <<"\n";
	for (int i = 0; i < count; ++i) {
		px(i) = geom[0].ProbLength(0) * amrex::Random();
		py(i) = geom[0].ProbLength(1) * amrex::Random();
		pz(i) = 2.*kpc;;
		while(1.*kpc < pz(i)){
			pz(i) = geom[0].ProbLength(2) * amrex::RandomNormal(mean, stddev);
		}
		amrex::Print() << "The location of "<< i<< " Type II SN: (x,y,z) : ("  << px(i) <<"," << py(i) << "," << pz(i) <<")" <<"\n"; 
	}
	
	//Get probablities for Type Ias
	amrex::Array<int, 1> const hi1a{count1a};
	userData_.blast_x1a = std::make_unique<amrex::TableData<Real, 1>>(lo, hi1a, amrex::The_Pinned_Arena());
	userData_.blast_y1a = std::make_unique<amrex::TableData<Real, 1>>(lo, hi1a, amrex::The_Pinned_Arena());
	userData_.blast_z1a = std::make_unique<amrex::TableData<Real, 1>>(lo, hi1a, amrex::The_Pinned_Arena());
	userData_.nblast1a = count1a;
	userData_.SN_counter_cumulative +=  count1a;

	auto const &px1a = userData_.blast_x1a->table();
	auto const &py1a = userData_.blast_y1a->table();
	auto const &pz1a = userData_.blast_z1a->table();

	amrex::Print() << "Number of Type Ia SN going off in this step:" << count1a <<"\n"; 
	for (int i = 0; i < count1a; ++i) {
		px1a(i) = geom[0].ProbLength(0) * amrex::Random();
		py1a(i) = geom[0].ProbLength(1) * amrex::Random();
		pz1a(i) = 2.*kpc;
		while(1.*kpc < pz1a(i)){
			pz1a(i) = geom[0].ProbLength(2) * amrex::RandomNormal(mean, stddev1a);
		}	
		amrex::Print() << "The location of " << i << " Type Ia SN: (x,y,z) : ("  << px1a(i) <<"," << py1a(i) << "," << pz1a(i) <<")" <<"\n"; 
	}

	//Get probablities for AGBs
	amrex::Array<int, 1> const hiAGB{countAGB};
	userData_.blast_xAGB = std::make_unique<amrex::TableData<Real, 1>>(lo, hiAGB, amrex::The_Pinned_Arena());
	userData_.blast_yAGB = std::make_unique<amrex::TableData<Real, 1>>(lo, hiAGB, amrex::The_Pinned_Arena());
	userData_.blast_zAGB = std::make_unique<amrex::TableData<Real, 1>>(lo, hiAGB, amrex::The_Pinned_Arena());
	userData_.nblastAGB = countAGB;
	userData_.SN_counter_cumulative +=  countAGB ;

	auto const &pxAGB = userData_.blast_xAGB->table();
	auto const &pyAGB = userData_.blast_yAGB->table();
	auto const &pzAGB = userData_.blast_zAGB->table();

	for (int i = 0; i < countAGB; ++i) {
		pxAGB(i) = geom[0].ProbLength(0) * amrex::Random();
		pyAGB(i) = geom[0].ProbLength(1) * amrex::Random();
		pzAGB(i) = 2.*kpc;
		while(1.*kpc < pzAGB(i)){
			pzAGB(i) = geom[0].ProbLength(2) * amrex::RandomNormal(mean, stddevAGB);
		}

	}

}

template <> void QuokkaSimulation<NewProblem>::computeAfterLevelAdvance(int lev, amrex::Real time, amrex::Real dt_lev, int ncycle)
{
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = geom[lev].ProbLoArray();
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_hi = geom[lev].ProbHiArray();
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx = geom[lev].CellSizeArray();

	AddSupernova(state_new_cc_[lev], prob_lo, prob_hi, dx, userData_, lev);
}


template <> void QuokkaSimulation<NewProblem>::ComputeDerivedVar(int lev, std::string const &dname, amrex::MultiFab &mf, const int ncomp_cc_in) const
{
	// compute derived variables and save in 'mf'
	if (dname == "temperature") {
		const int ncomp = ncomp_cc_in;
		auto tables = resampledTables_.const_tables();

		for (amrex::MFIter iter(mf); iter.isValid(); ++iter) {
			const amrex::Box &indexRange = iter.validbox();
			auto const &output = mf.array(iter);
			auto const &state = state_new_cc_[lev].const_array(iter);

			amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
				Real const rho = state(i, j, k, HydroSystem<NewProblem>::density_index);
				Real const x1Mom = state(i, j, k, HydroSystem<NewProblem>::x1Momentum_index);
				Real const x2Mom = state(i, j, k, HydroSystem<NewProblem>::x2Momentum_index);
				Real const x3Mom = state(i, j, k, HydroSystem<NewProblem>::x3Momentum_index);
				Real const Egas = state(i, j, k, HydroSystem<NewProblem>::energy_index);
				Real const Eint = RadSystem<NewProblem>::ComputeEintFromEgas(rho, x1Mom, x2Mom, x3Mom, Egas);
				Real const Tgas = ComputeTgasFromEgas(rho, Eint, tables);

				output(i, j, k, ncomp) = Tgas;
			});
		}
	}
}

template <>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE auto
HydroSystem<NewProblem>::GetGradFixedPotential(amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> posvec) -> amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>
{

	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> grad_potential;
	grad_potential[0] = 0.0;
	grad_potential[1] = 0.0;

	double z = posvec[2];

	// Interpolate to find the accurate g-value from array-- because linterp doesn't work on Setonix
	size_t i = 0;
	double x_interp = std::abs(z);
	while (i < z_data.size() - 1 && x_interp > z_data[i + 1]) {
		i++;
	}

	// Perform linear interpolation
	const Real x1 = z_data[i];
	const Real x2 = z_data[i + 1];
	const Real y1 = logg_data[i];
	const Real y2 = logg_data[i + 1];

	amrex::Real ginterp = (y1 + (y2 - y1) * (x_interp - x1) / (x2 - x1));

	grad_potential[2] = 2. * M_PI * C::Gconst * rho_dm * std::pow(R0, 2) * (2. * z / std::pow(R0, 2)) / (1. + std::pow(z, 2) / std::pow(R0, 2));
	grad_potential[2] += 2. * M_PI * C::Gconst * Sigma_star * (z / z_star) * (std::pow(1. + z * z / (z_star * z_star), -0.5));
	grad_potential[2] += (z / std::abs(z)) * std::pow(10., ginterp);

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

			Real Eint = RadSystem<NewProblem>::ComputeEintFromEgas(rho, x1mom, x2mom, x3mom, Egas);

			posvec[0] = prob_lo[0] + (i + 0.5) * dx[0];

#if (AMREX_SPACEDIM >= 2)
			posvec[1] = prob_lo[1] + (j + 0.5) * dx[1];
#endif

#if (AMREX_SPACEDIM >= 3)
			posvec[2] = prob_lo[2] + (k + 0.5) * dx[2];
#endif

			GradPhi = HydroSystem<NewProblem>::GetGradFixedPotential(posvec);

			x1mom_new = x1mom + dt * (-rho * GradPhi[0]);
			x2mom_new = x2mom + dt * (-rho * GradPhi[1]);
			x3mom_new = x3mom + dt * (-rho * GradPhi[2]);

			// State momentum values need to be updated this way.
			state(i, j, k, HydroSystem<NewProblem>::x1Momentum_index) = x1mom_new;
			state(i, j, k, HydroSystem<NewProblem>::x2Momentum_index) = x2mom_new;
			state(i, j, k, HydroSystem<NewProblem>::x3Momentum_index) = x3mom_new;

			state(i, j, k, HydroSystem<NewProblem>::energy_index) =
			    RadSystem<NewProblem>::ComputeEgasFromEint(rho, x1mom_new, x2mom_new, x3mom_new, Eint);
		});
	}
}

//Add magnetic field initial conditions

template <> void QuokkaSimulation<NewProblem>::setInitialConditionsOnGridFaceVars(quokka::grid const &grid_elem)
{
	double magnetic_field_microgauss = 2.6;
	double disk_zscale_kpc = 1.0;

	amrex::ParmParse const pp("magnetic_field");
	pp.query("magnetic_field_microgauss", magnetic_field_microgauss);
	pp.query("disk_zscale_kpc", disk_zscale_kpc);
	 
	const double z_d = disk_zscale_kpc * (1.0e3 * C::parsec);
	const double B_0 = magnetic_field_microgauss * 1.0e-6 / std::sqrt(4.0 * M_PI);

	const amrex::Array4<double> &state_fc = grid_elem.array_;
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const quokka::direction dir = grid_elem.dir_;
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = grid_elem.dx_;
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = grid_elem.prob_lo_;

	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		// Cartesian coordinates at this face
		amrex::Real const dx_cen = (dir == quokka::direction::x) ? 0.0 : 0.5 * dx[0];
		amrex::Real const dy_cen = (dir == quokka::direction::y) ? 0.0 : 0.5 * dx[1];
		amrex::Real const dz_cen = (dir == quokka::direction::z) ? 0.0 : 0.5 * dx[2];
		amrex::Real const x = prob_lo[0] + (i * dx[0]) + dx_cen;
		amrex::Real const y = prob_lo[1] + (j * dx[1]) + dy_cen;
		amrex::Real const z = prob_lo[2] + (k * dx[2]) + dz_cen;

		amrex::Real Bx = 0.0;
		amrex::Real By = B_0 *  std::exp(-std::abs(z) / z_d);
		amrex::Real Bz = 0.0;

		constexpr int mhd_index = Physics_Indices<NewProblem>::mhdFirstIndex;
		if (dir == quokka::direction::x) {
			state_fc(i, j, k, mhd_index) = Bx;
		} else if (dir == quokka::direction::y) {
			state_fc(i, j, k, mhd_index) = By;
		} else if (dir == quokka::direction::z) {
			state_fc(i, j, k, mhd_index) = Bz;
		}
	});
}


// Code for producing in-situ Projection plots
template <> auto QuokkaSimulation<NewProblem>::ComputeProjections(const amrex::Direction dir) const -> std::unordered_map<std::string, amrex::BaseFab<amrex::Real>>
{
	// compute density projection
	std::unordered_map<std::string, amrex::BaseFab<amrex::Real>> proj;

	// compute (total) density projection
	proj["mass_outflow"] = quokka::diagnostics::ComputePlaneProjection<amrex::ReduceOpSum>(
	    state_new_cc_, finestLevel(), geom, ref_ratio, dir, [=] AMREX_GPU_DEVICE(int i, int j, int k, amrex::Array4<const Real> const &state) noexcept {
		    Real const rho = state(i, j, k, HydroSystem<NewProblem>::density_index);
			Real const vx3 = state(i, j, k, HydroSystem<NewProblem>::x3Momentum_index) / rho;
		    return (rho * vx3);
	    });

	proj["hot_mass_outflow"] = quokka::diagnostics::ComputePlaneProjection<amrex::ReduceOpSum>(
	    state_new_cc_, finestLevel(), geom, ref_ratio, dir, [=] AMREX_GPU_DEVICE(int i, int j, int k, amrex::Array4<const Real> const &state) noexcept {
		    double flux;
		    Real const rho = state(i, j, k, HydroSystem<NewProblem>::density_index);
		    Real const vx3 = state(i, j, k, HydroSystem<NewProblem>::x3Momentum_index) / rho;
		    Real const Eint = state(i, j, k, HydroSystem<NewProblem>::internalEnergy_index);
		    amrex::GpuArray<Real, 0> massScalars = RadSystem<NewProblem>::ComputeMassScalars(state, i, j, k);
		    Real const primTemp = quokka::EOS<NewProblem>::ComputeTgasFromEint(rho, Eint, massScalars);
		    if (primTemp > 5.e5) {
			    flux = rho * vx3;
		    } else {
			    flux = 0.0;
		    }
		    return flux;
	    });

	proj["warm_mass_outflow"] = quokka::diagnostics::ComputePlaneProjection<amrex::ReduceOpSum>(
	    state_new_cc_, finestLevel(), geom, ref_ratio, dir, [=] AMREX_GPU_DEVICE(int i, int j, int k, amrex::Array4<const Real> const &state) noexcept {
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
	    });

	proj["scalar0_outflow"] = quokka::diagnostics::ComputePlaneProjection<amrex::ReduceOpSum>(
	    state_new_cc_, finestLevel(), geom, ref_ratio, dir, [=] AMREX_GPU_DEVICE(int i, int j, int k, amrex::Array4<const Real> const &state) noexcept {
		    Real const rho = state(i, j, k, HydroSystem<NewProblem>::density_index);
		    Real const rhoZ = state(i, j, k, Physics_Indices<NewProblem>::pscalarFirstIndex);
		    Real const vz = state(i, j, k, HydroSystem<NewProblem>::x3Momentum_index) / rho;
		    return (rhoZ * vz);
	    });

	proj["warm_scalar0_outflow"] = quokka::diagnostics::ComputePlaneProjection<amrex::ReduceOpSum>(
	    state_new_cc_, finestLevel(), geom, ref_ratio, dir, [=] AMREX_GPU_DEVICE(int i, int j, int k, amrex::Array4<const Real> const &state) noexcept {
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
	    });

	proj["hot_scalar0_outflow"] = quokka::diagnostics::ComputePlaneProjection<amrex::ReduceOpSum>(
	    state_new_cc_, finestLevel(), geom, ref_ratio, dir, [=] AMREX_GPU_DEVICE(int i, int j, int k, amrex::Array4<const Real> const &state) noexcept {
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
	    });

	proj["scalar1_outflow"] = quokka::diagnostics::ComputePlaneProjection<amrex::ReduceOpSum>(
	    state_new_cc_, finestLevel(), geom, ref_ratio, dir, [=] AMREX_GPU_DEVICE(int i, int j, int k, amrex::Array4<const Real> const &state) noexcept {
		    Real const rho = state(i, j, k, HydroSystem<NewProblem>::density_index);
		    Real const rhoZ = state(i, j, k, Physics_Indices<NewProblem>::pscalarFirstIndex+1);
		    Real const vz = state(i, j, k, HydroSystem<NewProblem>::x3Momentum_index) / rho;
		    return (rhoZ * vz);
	    });

	proj["warm_scalar1_outflow"] = quokka::diagnostics::ComputePlaneProjection<amrex::ReduceOpSum>(
	    state_new_cc_, finestLevel(), geom, ref_ratio, dir, [=] AMREX_GPU_DEVICE(int i, int j, int k, amrex::Array4<const Real> const &state) noexcept {
		    double flux;
		    Real const rho = state(i, j, k, HydroSystem<NewProblem>::density_index);
		    Real const rhoZ = state(i, j, k, Physics_Indices<NewProblem>::pscalarFirstIndex+1);
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
	    });

	proj["hot_scalar1_outflow"] = quokka::diagnostics::ComputePlaneProjection<amrex::ReduceOpSum>(
	    state_new_cc_, finestLevel(), geom, ref_ratio, dir, [=] AMREX_GPU_DEVICE(int i, int j, int k, amrex::Array4<const Real> const &state) noexcept {
		    double flux;
		    Real const rho = state(i, j, k, HydroSystem<NewProblem>::density_index);
		    Real const rhoZ = state(i, j, k, Physics_Indices<NewProblem>::pscalarFirstIndex+1);
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
	    });


	proj["scalar2_outflow"] = quokka::diagnostics::ComputePlaneProjection<amrex::ReduceOpSum>(
	    state_new_cc_, finestLevel(), geom, ref_ratio, dir, [=] AMREX_GPU_DEVICE(int i, int j, int k, amrex::Array4<const Real> const &state) noexcept {
		    Real const rho = state(i, j, k, HydroSystem<NewProblem>::density_index);
		    Real const rhoZ = state(i, j, k, Physics_Indices<NewProblem>::pscalarFirstIndex+2);
		    Real const vz = state(i, j, k, HydroSystem<NewProblem>::x3Momentum_index) / rho;
		    return (rhoZ * vz);
	    });

	proj["warm_scalar2_outflow"] = quokka::diagnostics::ComputePlaneProjection<amrex::ReduceOpSum>(
	    state_new_cc_, finestLevel(), geom, ref_ratio, dir, [=] AMREX_GPU_DEVICE(int i, int j, int k, amrex::Array4<const Real> const &state) noexcept {
		    double flux;
		    Real const rho = state(i, j, k, HydroSystem<NewProblem>::density_index);
		    Real const rhoZ = state(i, j, k, Physics_Indices<NewProblem>::pscalarFirstIndex+2);
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
	    });

	proj["hot_scalar2_outflow"] = quokka::diagnostics::ComputePlaneProjection<amrex::ReduceOpSum>(
	    state_new_cc_, finestLevel(), geom, ref_ratio, dir, [=] AMREX_GPU_DEVICE(int i, int j, int k, amrex::Array4<const Real> const &state) noexcept {
		    double flux;
		    Real const rho = state(i, j, k, HydroSystem<NewProblem>::density_index);
		    Real const rhoZ = state(i, j, k, Physics_Indices<NewProblem>::pscalarFirstIndex+2);
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
	    });


	proj["rho"] = quokka::diagnostics::ComputePlaneProjection<amrex::ReduceOpSum>(
	    state_new_cc_, finestLevel(), geom, ref_ratio, dir, [=] AMREX_GPU_DEVICE(int i, int j, int k, amrex::Array4<const Real> const &state) noexcept {
		    Real const rho = state(i, j, k, HydroSystem<NewProblem>::density_index);
		    return (rho);
	    });

	proj["scalar0"] = quokka::diagnostics::ComputePlaneProjection<amrex::ReduceOpSum>(
	    state_new_cc_, finestLevel(), geom, ref_ratio, dir, [=] AMREX_GPU_DEVICE(int i, int j, int k, amrex::Array4<const Real> const &state) noexcept {
		    Real const rhoZ = state(i, j, k, Physics_Indices<NewProblem>::pscalarFirstIndex);
		    return (rhoZ);
	    });

	proj["hot_scalar0"] = quokka::diagnostics::ComputePlaneProjection<amrex::ReduceOpSum>(
	    state_new_cc_, finestLevel(), geom, ref_ratio, dir, [=] AMREX_GPU_DEVICE(int i, int j, int k, amrex::Array4<const Real> const &state) noexcept {
		    double scal;
			Real const rho = state(i, j, k, HydroSystem<NewProblem>::density_index);
		    Real const rhoZ = state(i, j, k, Physics_Indices<NewProblem>::pscalarFirstIndex);
		    Real const Eint = state(i, j, k, HydroSystem<NewProblem>::internalEnergy_index);
		    amrex::GpuArray<Real, 0> massScalars = RadSystem<NewProblem>::ComputeMassScalars(state, i, j, k);
		    Real const primTemp = quokka::EOS<NewProblem>::ComputeTgasFromEint(rho, Eint, massScalars);
		    if (primTemp > 1.e6) {
			    scal = rhoZ;
		    } else {
			    scal = 0.0;
		    }
		    return scal;
	    });

	proj["warm_scalar0"] = quokka::diagnostics::ComputePlaneProjection<amrex::ReduceOpSum>(
	    state_new_cc_, finestLevel(), geom, ref_ratio, dir, [=] AMREX_GPU_DEVICE(int i, int j, int k, amrex::Array4<const Real> const &state) noexcept {
		    double scal;
			Real const rho = state(i, j, k, HydroSystem<NewProblem>::density_index);
		    Real const rhoZ = state(i, j, k, Physics_Indices<NewProblem>::pscalarFirstIndex);
		    Real const Eint = state(i, j, k, HydroSystem<NewProblem>::internalEnergy_index);
		    amrex::GpuArray<Real, 0> massScalars = RadSystem<NewProblem>::ComputeMassScalars(state, i, j, k);
		    Real const primTemp = quokka::EOS<NewProblem>::ComputeTgasFromEint(rho, Eint, massScalars);
		    if (primTemp < 2.e4) {
			    scal = rhoZ;
		    } else {
			    scal = 0.0;
		    }
		    return scal;
	    });	

	proj["scalar1"] = quokka::diagnostics::ComputePlaneProjection<amrex::ReduceOpSum>(
	    state_new_cc_, finestLevel(), geom, ref_ratio, dir, [=] AMREX_GPU_DEVICE(int i, int j, int k, amrex::Array4<const Real> const &state) noexcept {
		    Real const rhoZ = state(i, j, k, Physics_Indices<NewProblem>::pscalarFirstIndex+1);
		    return (rhoZ);
	    });

	proj["hot_scalar1"] = quokka::diagnostics::ComputePlaneProjection<amrex::ReduceOpSum>(
	    state_new_cc_, finestLevel(), geom, ref_ratio, dir, [=] AMREX_GPU_DEVICE(int i, int j, int k, amrex::Array4<const Real> const &state) noexcept {
		    double scal;
			Real const rho = state(i, j, k, HydroSystem<NewProblem>::density_index);
		    Real const rhoZ = state(i, j, k, Physics_Indices<NewProblem>::pscalarFirstIndex+1);
		    Real const Eint = state(i, j, k, HydroSystem<NewProblem>::internalEnergy_index);
		    amrex::GpuArray<Real, 0> massScalars = RadSystem<NewProblem>::ComputeMassScalars(state, i, j, k);
		    Real const primTemp = quokka::EOS<NewProblem>::ComputeTgasFromEint(rho, Eint, massScalars);
		    if (primTemp > 1.e6) {
			    scal = rhoZ;
		    } else {
			    scal = 0.0;
		    }
		    return scal;
	    });

	proj["warm_scalar1"] = quokka::diagnostics::ComputePlaneProjection<amrex::ReduceOpSum>(
	    state_new_cc_, finestLevel(), geom, ref_ratio, dir, [=] AMREX_GPU_DEVICE(int i, int j, int k, amrex::Array4<const Real> const &state) noexcept {
		    double scal;
			Real const rho = state(i, j, k, HydroSystem<NewProblem>::density_index);
		    Real const rhoZ = state(i, j, k, Physics_Indices<NewProblem>::pscalarFirstIndex+1);
		    Real const Eint = state(i, j, k, HydroSystem<NewProblem>::internalEnergy_index);
		    amrex::GpuArray<Real, 0> massScalars = RadSystem<NewProblem>::ComputeMassScalars(state, i, j, k);
		    Real const primTemp = quokka::EOS<NewProblem>::ComputeTgasFromEint(rho, Eint, massScalars);
		    if (primTemp < 2.e4) {
			    scal = rhoZ;
		    } else {
			    scal = 0.0;
		    }
		    return scal;
	    });	

	proj["scalar2"] = quokka::diagnostics::ComputePlaneProjection<amrex::ReduceOpSum>(
	    state_new_cc_, finestLevel(), geom, ref_ratio, dir, [=] AMREX_GPU_DEVICE(int i, int j, int k, amrex::Array4<const Real> const &state) noexcept {
		    Real const rhoZ = state(i, j, k, Physics_Indices<NewProblem>::pscalarFirstIndex+2);
		    return (rhoZ);
	    });	

	proj["hot_scalar2"] = quokka::diagnostics::ComputePlaneProjection<amrex::ReduceOpSum>(
	    state_new_cc_, finestLevel(), geom, ref_ratio, dir, [=] AMREX_GPU_DEVICE(int i, int j, int k, amrex::Array4<const Real> const &state) noexcept {
		    double scal;
			Real const rho = state(i, j, k, HydroSystem<NewProblem>::density_index);
		    Real const rhoZ = state(i, j, k, Physics_Indices<NewProblem>::pscalarFirstIndex+2);
		    Real const Eint = state(i, j, k, HydroSystem<NewProblem>::internalEnergy_index);
		    amrex::GpuArray<Real, 0> massScalars = RadSystem<NewProblem>::ComputeMassScalars(state, i, j, k);
		    Real const primTemp = quokka::EOS<NewProblem>::ComputeTgasFromEint(rho, Eint, massScalars);
		    if (primTemp > 1.e6) {
			    scal = rhoZ;
		    } else {
			    scal = 0.0;
		    }
		    return scal;
	    });

	proj["warm_scalar2"] = quokka::diagnostics::ComputePlaneProjection<amrex::ReduceOpSum>(
	    state_new_cc_, finestLevel(), geom, ref_ratio, dir, [=] AMREX_GPU_DEVICE(int i, int j, int k, amrex::Array4<const Real> const &state) noexcept {
		    double scal;
			Real const rho = state(i, j, k, HydroSystem<NewProblem>::density_index);
		    Real const rhoZ = state(i, j, k, Physics_Indices<NewProblem>::pscalarFirstIndex+2);
		    Real const Eint = state(i, j, k, HydroSystem<NewProblem>::internalEnergy_index);
		    amrex::GpuArray<Real, 0> massScalars = RadSystem<NewProblem>::ComputeMassScalars(state, i, j, k);
		    Real const primTemp = quokka::EOS<NewProblem>::ComputeTgasFromEint(rho, Eint, massScalars);
		    if (primTemp < 2.e4) {
			    scal = rhoZ;
		    } else {
			    scal = 0.0;
		    }
		    return scal;
	    });	
	return proj;
}

// Implement User-defined diode BC
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
	int normal = 0;

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
	const double pscalar0_edge = consVar(i, j, kedge, HydroSystem<NewProblem>::scalar0_index);
	const double pscalar1_edge = consVar(i, j, kedge, HydroSystem<NewProblem>::scalar0_index+1);
	const double pscalar2_edge = consVar(i, j, kedge, HydroSystem<NewProblem>::scalar0_index+2);

	if ((x3Mom_edge * normal) < 0) { // gas is inflowing
		x3Mom_edge = -1. * consVar(i, j, kedge, HydroSystem<NewProblem>::x3Momentum_index);
	}

	consVar(i, j, k, HydroSystem<NewProblem>::density_index) = rho_edge;
	consVar(i, j, k, HydroSystem<NewProblem>::x1Momentum_index) = x1Mom_edge;
	consVar(i, j, k, HydroSystem<NewProblem>::x2Momentum_index) = x2Mom_edge;
	consVar(i, j, k, HydroSystem<NewProblem>::x3Momentum_index) = x3Mom_edge;
	consVar(i, j, k, HydroSystem<NewProblem>::energy_index) = etot_edge;
	consVar(i, j, k, HydroSystem<NewProblem>::internalEnergy_index) = eint_edge;
	consVar(i, j, k, HydroSystem<NewProblem>::scalar0_index) = pscalar0_edge;
	consVar(i, j, k, HydroSystem<NewProblem>::scalar0_index+1) = pscalar1_edge;
	consVar(i, j, k, HydroSystem<NewProblem>::scalar0_index+2) = pscalar2_edge;
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

	const int nvars_fc = Physics_Indices<NewProblem>::nvarTotal_fc;
	const int nvars_per_dim_fc = Physics_Indices<NewProblem>::nvarPerDim_fc;
	amrex::Vector<amrex::BCRec> BCs_fc(nvars_fc);

	for (int icomp = 0; icomp < nvars_fc; ++icomp) {
		int const component_dir = (nvars_per_dim_fc > 0) ? (icomp / nvars_per_dim_fc) : 0;
		for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
			if (idim == 2) {
				BCs_fc[icomp].setLo(idim, amrex::BCType::ext_dir);
				BCs_fc[icomp].setHi(idim, amrex::BCType::ext_dir);
			} else {
				BCs_fc[icomp].setLo(idim, amrex::BCType::int_dir); // periodic
				BCs_fc[icomp].setHi(idim, amrex::BCType::int_dir); // periodic
			}
		}
	}

	// set random state
	const int seed = 42;
	amrex::InitRandom(seed, 1); // all ranks should produce the same values

	// Problem initialization
	QuokkaSimulation<NewProblem> sim(BCs_cc, BCs_fc);

	sim.reconstructionOrder_ = 3; // 2=PLM, 3=PPM
	sim.cflNumber_ = 0.3;	      // *must* be less than 1/3 in 3D!

	sim.setInitialConditions();

	// evolve
	sim.evolve();

	// Cleanup and exit
	amrex::Print() << "Finished." << std::endl;
	return 0;
}
