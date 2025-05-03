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

struct SinkProblem {
};

constexpr double M_sol = C::M_solar;
constexpr double mu = 1.0 * C::m_p;
constexpr double gamma_ = 5. / 3.;
const double rho0 = 1.0 * C::m_p; // g cm^-3
const double T0 = 10.0;		  // K
const double CV = 1. / (gamma_ - 1.) / mu * C::k_B;
const double year = 3.15576e+07; // in seconds

const double sf_cell_density = 1.0e6 * C::m_p; // g cm^-3
const double sf_cell_loc = 1.0; // in x,y,z direction, cm

template <> struct Particle_Traits<SinkProblem> {
	// static constexpr ParticleSwitch particle_switch = ParticleSwitch::None;
	static constexpr ParticleSwitch particle_switch = ParticleSwitch::StochasticStellarPop;
};

template <> struct quokka::EOS_Traits<SinkProblem> {
	static constexpr double gamma = gamma_;
	static constexpr double mean_molecular_weight = mu;
};

template <> struct HydroSystem_Traits<SinkProblem> {
	static constexpr bool reconstruct_eint = true; // need to reconstruct temperature
};

template <> struct Physics_Traits<SinkProblem> {
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

template <> void QuokkaSimulation<SinkProblem>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const amrex::Array4<double> &state_cc = grid_elem.array_;
	const double rho_e = CV * T0 * rho0;
	const auto prob_lo = geom[0].ProbLoArray();
	const auto dx = geom[0].CellSizeArray();

	// loop over the grid and set the initial condition
	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		const double x = prob_lo[0] + (i * dx[0]);
		const double y = prob_lo[1] + (j * dx[1]);
		const double z = prob_lo[2] + (k * dx[2]);
		if (x < sf_cell_loc && x + dx[0] > sf_cell_loc && y < sf_cell_loc && y + dx[1] > sf_cell_loc && z < sf_cell_loc && z + dx[2] > sf_cell_loc) {
			state_cc(i, j, k, HydroSystem<SinkProblem>::density_index) = sf_cell_density;
		} else {
			state_cc(i, j, k, HydroSystem<SinkProblem>::density_index) = rho0;
		}
		state_cc(i, j, k, HydroSystem<SinkProblem>::x1Momentum_index) = 0.0;
		state_cc(i, j, k, HydroSystem<SinkProblem>::x2Momentum_index) = 0.0;
		state_cc(i, j, k, HydroSystem<SinkProblem>::x3Momentum_index) = 0.0;
		state_cc(i, j, k, HydroSystem<SinkProblem>::energy_index) = rho_e;
		state_cc(i, j, k, HydroSystem<SinkProblem>::internalEnergy_index) = rho_e;
	});
}

template <> void QuokkaSimulation<SinkProblem>::ErrorEst(int lev, amrex::TagBoxArray &tags, amrex::Real /*time*/, int /*ngrow*/)
{
	// tag cells for refinement: static mesh refinement for the whole domain

	auto const &dx = geom[lev].CellSizeArray();
	auto const &plo = geom[lev].ProbLoArray();
	auto const &phi = geom[lev].ProbHiArray();

	for (amrex::MFIter mfi(state_new_cc_[lev]); mfi.isValid(); ++mfi) {
		const amrex::Box &box = mfi.validbox();
		const auto tag = tags.array(mfi);

		amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
			tag(i, j, k) = amrex::TagBox::SET;
		});
	}
}

auto problem_main() -> int
{
	auto isNormalComp = [=](int n, int dim) {
		if ((n == HydroSystem<SinkProblem>::x1Momentum_index) && (dim == 0)) {
			return true;
		}
		if ((n == HydroSystem<SinkProblem>::x2Momentum_index) && (dim == 1)) {
			return true;
		}
		if ((n == HydroSystem<SinkProblem>::x3Momentum_index) && (dim == 2)) {
			return true;
		}
		return false;
	};

	const int ncomp_cc = Physics_Indices<SinkProblem>::nvarTotal_cc;
	amrex::Vector<amrex::BCRec> BCs_cc(ncomp_cc);
	for (int n = 0; n < ncomp_cc; ++n) {
		for (int i = 0; i < AMREX_SPACEDIM; ++i) {
			// periodic boundaries
			BCs_cc[n].setLo(i, amrex::BCType::int_dir);
			BCs_cc[n].setHi(i, amrex::BCType::int_dir);
		}
	}

	// amrex::ParmParse const pp("problem");

	// Problem initialization
	QuokkaSimulation<SinkProblem> sim(BCs_cc);

	sim.reconstructionOrder_ = 3; // 2=PLM, 3=PPM
	sim.cflNumber_ = 0.3;	      // *must* be less than 1/3 in 3D!
	sim.stopTime_ = 1.0e7 * year; // 10 Myr
	sim.initDt_ = 1.0e5 * year; // 0.1 Myr

	// initialize
	sim.setInitialConditions();

	// // get total mass of the initial gas
	// amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx0 = sim.geom[0].CellSizeArray();
	// amrex::Real const vol = AMREX_D_TERM(dx0[0], *dx0[1], *dx0[2]);
	// amrex::Real const total_mass_init = sim.state_new_cc_[0].sum(HydroSystem<SinkProblem>::density_index) * vol;
	// double total_total_mass = NAN;
	// double total_total_mass_final = NAN;
	// double total_particle_mass = 0.0;

	// // get total particle mass
	// const auto &real_data = sim.particleRegister_.getParticleDescriptor(quokka::ParticleType::Sink)->getParticleDataAtLevel(0).first;
	// if (amrex::ParallelDescriptor::IOProcessor()) {
	// 	// const double total_particle_mass = std::accumulate(real_data.begin(), real_data.end(), 0.0, [](double sum, const auto &d) { return sum +
	// 	// d[3]; });
	// 	for (const auto &p : real_data) {
	// 		total_particle_mass += p[3];
	// 	}
	// 	total_total_mass = total_mass_init + total_particle_mass;
	// 	amrex::Print() << "\nBefore evolution:\n";
	// 	amrex::Print() << "Total gas mass = " << total_mass_init << "\n";
	// 	amrex::Print() << "Total particle mass = " << total_particle_mass << "\n";
	// 	amrex::Print() << "Total total mass = " << total_total_mass << "\n";
	// }

	// evolve
	sim.evolve();

	// get total mass of the final particles
	const auto [real_data_final, idata_final] = sim.particleRegister_.getParticleDescriptor(quokka::ParticleType::StochasticStellarPop)->getParticleDataAtLevel(0);
	if (amrex::ParallelDescriptor::IOProcessor()) {
		const int mass_idx = 3;
		double high_mass_stars_total_mass = 0.0;
		double all_stars_total_mass = 0.0;
		int num_high_mass_stars = 0;
		for (int i = 0; i < real_data_final.size(); ++i) {
			if (idata_final[i][0] == static_cast<int>(quokka::StellarEvolutionStage::SNProgenitor)) {
				high_mass_stars_total_mass += real_data_final[i][mass_idx];
				num_high_mass_stars++;
			}
			all_stars_total_mass += real_data_final[i][mass_idx];
		}
		const double mean_mass_high_mass_stars = high_mass_stars_total_mass / num_high_mass_stars;
		const double mass_fraction_high_mass_stars = high_mass_stars_total_mass / all_stars_total_mass;
		const double mean_mass_high_mass_stars_Msun = mean_mass_high_mass_stars / M_sol;
		amrex::Print() << "Total particle mass = " << all_stars_total_mass / M_sol << " Msun\n";
		amrex::Print() << "Mstar_high_mean = " << mean_mass_high_mass_stars_Msun << " Msun\n";
		amrex::Print() << "fstar_high = " << mass_fraction_high_mass_stars << "\n";

		// expectations
		const double exp_Mstar_high_mean = 19.39;
		const double exp_fstar_high = 0.220;
		amrex::Print() << "\nExpected values:\n";
		amrex::Print() << "Mstar_high_mean = " << exp_Mstar_high_mean << " Msun\n";
		amrex::Print() << "fstar_high = " << exp_fstar_high << "\n";

		// relative error
		const double rel_error_Mstar_high_mean = std::abs(mean_mass_high_mass_stars_Msun - exp_Mstar_high_mean) / exp_Mstar_high_mean;
		const double rel_error_fstar_high = std::abs(mass_fraction_high_mass_stars - exp_fstar_high) / exp_fstar_high;
		amrex::Print() << "\nRelative error:\n";
		amrex::Print() << "rel_err(Mstar_high_mean) = " << rel_error_Mstar_high_mean << "\n";
		amrex::Print() << "rel_err(fstar_high) = " << rel_error_fstar_high << "\n";
	}

	return 0;
}
