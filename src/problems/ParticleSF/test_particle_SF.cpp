/// \file test_particle_SF.cpp
/// \brief Defines a test problem for stochastic star formation.
///

#include "AMReX.H"
#include "AMReX_BC_TYPES.H"
#include "AMReX_MultiFab.H"
#include "AMReX_ParmParse.H"
#include "AMReX_Print.H"
#include "AMReX_SPACE.H"
#include "util/fextract.hpp"

#include "QuokkaSimulation.hpp"
#include "fundamental_constants.H"
#include "hydro/hydro_system.hpp"
#include "particles/particle_types.hpp"
#include "test_particle_SF.hpp"

#ifdef HAVE_PYTHON
#include "util/matplotlibcpp.h"
#endif

struct ParticleSFProblem {
};

constexpr double M_sol = C::M_solar;
constexpr double mu = 1.0 * C::m_p;
constexpr double gamma_ = 5. / 3.;
AMREX_GPU_MANAGED Real rho0 = NAN;  // NOLINT
const double year = 3.15576e+07;    // in seconds
AMREX_GPU_MANAGED Real Tamb = 10.0; // NOLINT
// AMREX_GPU_MANAGED Real sigma1 = 700000.0;
;

template <> struct Particle_Traits<ParticleSFProblem> {
	// static constexpr ParticleSwitch particle_switch = ParticleSwitch::None;
	static constexpr ParticleSwitch particle_switch = ParticleSwitch::StochasticStellarPop;
};

template <> struct quokka::EOS_Traits<ParticleSFProblem> {
	static constexpr double gamma = gamma_;
	static constexpr double mean_molecular_weight = mu;
};

template <> struct HydroSystem_Traits<ParticleSFProblem> {
	static constexpr bool reconstruct_eint = true; // need to reconstruct temperature
};

template <> struct Physics_Traits<ParticleSFProblem> {
	// cell-centred
	static constexpr bool is_hydro_enabled = true;
	static constexpr int numMassScalars = 0;		     // number of mass scalars
	static constexpr int numPassiveScalars = numMassScalars + 0; // number of passive scalars
	static constexpr bool is_radiation_enabled = false;
	// face-centred
	static constexpr bool is_mhd_enabled = false;
	static constexpr int nGroups = 1; // number of radiation groups
	static constexpr UnitSystem unit_system = UnitSystem::CGS;
};

template <> void QuokkaSimulation<ParticleSFProblem>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const amrex::Array4<double> &state_cc = grid_elem.array_;
	// const double rho_e = CV * T0 * rho0;
	// const auto prob_lo = geom[0].ProbLoArray();
	// const auto prob_hi = geom[0].ProbHiArray();

	const auto dx = geom[0].CellSizeArray();

	// loop over the grid and set the initial condition
	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		// All cells are Jeans unstable
		double P = NAN;
		double rho = NAN;
		const auto gamma = HydroSystem<ParticleSFProblem>::gamma_;
		const double cs = std::sqrt(C::k_B * Tamb / C::m_u);
		rho = 5.0 * cs * cs / (dx[0] * dx[0] * Gconst_);
		rho0 = rho;
		P = rho * std::pow(cs, 2.0) / gamma;
		state_cc(i, j, k, HydroSystem<ParticleSFProblem>::density_index) = rho;
		state_cc(i, j, k, HydroSystem<ParticleSFProblem>::x1Momentum_index) = 0;
		state_cc(i, j, k, HydroSystem<ParticleSFProblem>::x2Momentum_index) = 0;
		state_cc(i, j, k, HydroSystem<ParticleSFProblem>::x3Momentum_index) = 0;
		state_cc(i, j, k, HydroSystem<ParticleSFProblem>::energy_index) = P / (gamma - 1.);
		;
		state_cc(i, j, k, HydroSystem<ParticleSFProblem>::internalEnergy_index) = P / (gamma - 1.);
		;
	});
}

template <> void QuokkaSimulation<ParticleSFProblem>::ErrorEst(int lev, amrex::TagBoxArray &tags, amrex::Real /*time*/, int /*ngrow*/)
{
	// tag cells for refinement: static mesh refinement for the whole domain

	for (amrex::MFIter mfi(state_new_cc_[lev]); mfi.isValid(); ++mfi) {
		const amrex::Box &box = mfi.validbox();
		const auto tag = tags.array(mfi);

		amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept { tag(i, j, k) = amrex::TagBox::SET; });
	}
}

auto problem_main() -> int
{
	
	const int ncomp_cc = Physics_Indices<ParticleSFProblem>::nvarTotal_cc;
	amrex::Vector<amrex::BCRec> BCs_cc(ncomp_cc);
	for (int n = 0; n < ncomp_cc; ++n) {
		for (int i = 0; i < AMREX_SPACEDIM; ++i) {
			// periodic boundaries
			BCs_cc[n].setLo(i, amrex::BCType::int_dir);
			BCs_cc[n].setHi(i, amrex::BCType::int_dir);
		}
	}


	// Problem initialization
	QuokkaSimulation<ParticleSFProblem> sim(BCs_cc);

	sim.reconstructionOrder_ = 3; // 2=PLM, 3=PPM
	sim.cflNumber_ = 0.3;	      // *must* be less than 1/3 in 3D!
	sim.stopTime_ = 1.0e6 * year; // 1 Myr
	sim.initDt_ = 1.0e5 * year;   // 0.1 Myr

	// initialize
	sim.setInitialConditions();

	// evolve
	sim.evolve();

	amrex::Real eps_ff = 0.0;
	amrex::ParmParse const pp("particles");
	pp.query("eps_ff", eps_ff);

	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx0 = sim.geom[0].CellSizeArray();
	const amrex::Real cell_volume = AMREX_D_TERM(dx0[0], *dx0[1], *dx0[2]);
	const auto prob_lo = sim.geom[0].ProbLoArray();
	const auto prob_hi = sim.geom[0].ProbHiArray();

	const int nx = static_cast<int>((prob_hi[0] - prob_lo[0]) / dx0[0]);
	const int ny = static_cast<int>((prob_hi[1] - prob_lo[1]) / dx0[1]);
	const int nz = static_cast<int>((prob_hi[2] - prob_lo[2]) / dx0[2]);


	//Check particle stats

	const amrex::Real eps_star = 0.5;
	const double exp_Mstar_high_mean = 19.39;
	const double exp_fstar_high = 0.220;
	const amrex::Real t_ff = std::sqrt(3.0 * M_PI / (32.0 * C::Gconst * rho0));
	const amrex::Real prob_star_formation = eps_ff * sim.initDt_ / eps_star / t_ff;

	const amrex::Real particle_mass = rho0 * cell_volume * eps_star * sim.initDt_ / t_ff;
	const amrex::Real m_high_tot = particle_mass * exp_fstar_high;
	const amrex::Real num_high_mass_stars_exp = m_high_tot / (exp_Mstar_high_mean * C::M_solar);
	const amrex::Real exp_num_stars = prob_star_formation * (1 + num_high_mass_stars_exp) * nx * ny * nz;

	// get total mass of the final particles
	const auto [real_data_final, idata_final] =
	    sim.particleRegister_.getParticleDescriptor(quokka::ParticleType::StochasticStellarPop)->getParticleDataAtLevel(0);
	const int mass_idx = 3;
	double high_mass_stars_total_mass = 0.0;
	double all_stars_total_mass = 0.0;
	int num_high_mass_stars = 0;
	const int num_all_stars = static_cast<int>(real_data_final.size());
	for (int i = 0; i < num_all_stars; ++i) {
		if (idata_final[i][0] != static_cast<int>(quokka::StellarEvolutionStage::LowMassComposite)) {
			high_mass_stars_total_mass += real_data_final[i][mass_idx];
			num_high_mass_stars++;
		}
		all_stars_total_mass += real_data_final[i][mass_idx];
	}
	const double mean_mass_high_mass_stars = high_mass_stars_total_mass / num_high_mass_stars;
	const double mass_fraction_high_mass_stars = high_mass_stars_total_mass / all_stars_total_mass;
	const double mean_mass_high_mass_stars_Msun = mean_mass_high_mass_stars / M_sol;
	amrex::Print() << "Total particle mass = " << all_stars_total_mass / M_sol << " Msun\n";
	amrex::Print() << "Number of high mass stars = " << num_high_mass_stars << "\n";
	amrex::Print() << "Number of all stars = " << num_all_stars << "\n";
	amrex::Print() << "Mstar_high_mean = " << mean_mass_high_mass_stars_Msun << " Msun\n";
	amrex::Print() << "fstar_high = " << mass_fraction_high_mass_stars << "\n";

	// expectations

	amrex::Print() << "\nExpected values:\n";
	amrex::Print() << "Expected number of stars = " << exp_num_stars << "\n";
	amrex::Print() << "Mstar_high_mean = " << exp_Mstar_high_mean << " Msun\n";
	amrex::Print() << "fstar_high = " << exp_fstar_high << "\n";

	// relative error
	const double rel_error_Mstar_high_mean = std::abs(mean_mass_high_mass_stars_Msun - exp_Mstar_high_mean) / exp_Mstar_high_mean;
	const double rel_error_fstar_high = std::abs(mass_fraction_high_mass_stars - exp_fstar_high) / exp_fstar_high;
	const double rel_error_num_stars = std::abs(num_all_stars - exp_num_stars) / exp_num_stars;

	//Check gas mass 
	const double initial_gas_mass = rho0 * cell_volume * nx * ny * nz;
	const double final_gas_mass = sim.state_new_cc_[0].sum(HydroSystem<ParticleSFProblem>::density_index) * cell_volume; ;
	const double change_gas_mass = initial_gas_mass - (all_stars_total_mass - high_mass_stars_total_mass);
	
	amrex::Print() << "\nRelative error:\n";
	amrex::Print() << "rel_err(num_stars) = " << rel_error_num_stars << "\n";
	amrex::Print() << "rel_err(Mstar_high_mean) = " << rel_error_Mstar_high_mean << "\n";
	amrex::Print() << "rel_err(fstar_high) = " << rel_error_fstar_high << "\n";
	amrex::Print() << "Initial (gas mass) - Mass of Low Mass Stars =  " << change_gas_mass/M_sol << " Msun \n";
	amrex::Print() << "Rel error wrt final gas mass = " << std::abs(final_gas_mass - change_gas_mass)/final_gas_mass  << "\n";

	return 0;
}
