/// \file test_particle_sink_formation.cpp
/// \brief Defines a test problem for sink particle formation.
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
#include "test_particle_sink_formation.hpp"

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

const double sf_cell_density = 1.0e5 * C::m_p; // g cm^-3
const double sf_cell_loc = 1.0;		       // in x,y,z direction, cm

template <> struct Particle_Traits<SinkProblem> {
	// static constexpr ParticleSwitch particle_switch = ParticleSwitch::None;
	static constexpr ParticleSwitch particle_switch = ParticleSwitch::Sink;
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
		if (x <= sf_cell_loc && x + dx[0] > sf_cell_loc && y <= sf_cell_loc && y + dx[1] > sf_cell_loc && z <= sf_cell_loc && z + dx[2] > sf_cell_loc) {
			// the cell at sf_cell_loc
			state_cc(i, j, k, HydroSystem<SinkProblem>::density_index) = sf_cell_density;
		} else if (x - 2 * dx[0] <= sf_cell_loc && x - dx[0] > sf_cell_loc && y <= sf_cell_loc && y + dx[1] > sf_cell_loc && z <= sf_cell_loc &&
			   z + dx[2] > sf_cell_loc) {
			// the cell that is 2 cells left of sf_cell_loc
			state_cc(i, j, k, HydroSystem<SinkProblem>::density_index) = sf_cell_density * 0.999;
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

	// auto const &dx = geom[lev].CellSizeArray();
	// auto const &plo = geom[lev].ProbLoArray();
	// auto const &phi = geom[lev].ProbHiArray();

	for (amrex::MFIter mfi(state_new_cc_[lev]); mfi.isValid(); ++mfi) {
		const amrex::Box &box = mfi.validbox();
		const auto tag = tags.array(mfi);

		amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept { tag(i, j, k) = amrex::TagBox::SET; });
	}
}

auto problem_main() -> int
{
	const int ncomp_cc = Physics_Indices<SinkProblem>::nvarTotal_cc;
	amrex::Vector<amrex::BCRec> BCs_cc(ncomp_cc);
	for (int n = 0; n < ncomp_cc; ++n) {
		for (int i = 0; i < AMREX_SPACEDIM; ++i) {
			// periodic boundaries
			BCs_cc[n].setLo(i, amrex::BCType::int_dir);
			BCs_cc[n].setHi(i, amrex::BCType::int_dir);
		}
	}

	// Problem initialization
	QuokkaSimulation<SinkProblem> sim(BCs_cc);

	sim.reconstructionOrder_ = 3; // 2=PLM, 3=PPM
	sim.cflNumber_ = 0.3;	      // *must* be less than 1/3 in 3D!
	sim.stopTime_ = 1.0e6 * year; // 1 Myr
	sim.initDt_ = 1.0e5 * year;   // 0.1 Myr

	// initialize
	sim.setInitialConditions();

	const auto [position0, values0] = fextract(sim.state_new_cc_[0], sim.Geom(0), 0, 0.0, true);
	const int nx = static_cast<int>(position0.size());

	// get total gas mass of the initial state
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx0 = sim.geom[0].CellSizeArray();
	amrex::Real const vol = AMREX_D_TERM(dx0[0], *dx0[1], *dx0[2]);
	amrex::Real const m_gas_init = sim.state_new_cc_[0].sum(HydroSystem<SinkProblem>::density_index) * vol;

	// evolve
	sim.evolve();

	auto [position, values] = fextract(sim.state_new_cc_[0], sim.Geom(0), 0, 0.0, true);

	// get total gas mass of the final state
	amrex::Real const m_gas_final = sim.state_new_cc_[0].sum(HydroSystem<SinkProblem>::density_index) * vol;

	int status = 0;

	// get total particle mass of the final state
	const auto &real_data_final = sim.particleRegister_.getParticleDescriptor(quokka::ParticleType::Sink)->getParticleDataAtLevelZero().first;
	if (amrex::ParallelDescriptor::IOProcessor()) {
		amrex::Print() << "Initial state:\n";
		amrex::Print() << "Gas mass = " << m_gas_init / M_sol << " Msun\n";

		amrex::Print() << "Final state:\n";

		const int mass_idx = 3;
		double m_stars_final = 0.0;
		const int num_stars = static_cast<int>(real_data_final.size());
		for (int i = 0; i < num_stars; ++i) {
			m_stars_final += real_data_final[i][mass_idx];
		}
		amrex::Print() << "Gas mass = " << m_gas_final / M_sol << " Msun\n";
		amrex::Print() << "Particle mass = " << m_stars_final / M_sol << " Msun\n";
		amrex::Print() << "Number of particles = " << num_stars << "\n";

		// get gas+particle mass
		const double m_final = m_gas_final + m_stars_final;
		amrex::Print() << "gas+particle mass = " << m_final / M_sol << " Msun\n";

		// relative error
		const double rel_error_gas_mass = std::abs(m_gas_init - m_final) / m_gas_init;
		amrex::Print() << "\nRelative error:\n";
		amrex::Print() << "rel_err(gas_mass) = " << rel_error_gas_mass << "\n";

		if (num_stars == 0) {
			status = 1;
			amrex::Print() << "Test failed: no particles created !!!\n";
		} else if (std::isnan(rel_error_gas_mass) || rel_error_gas_mass > 1.0e-10) {
			status = 1;
			amrex::Print() << "Test failed: mass not conserved !!!\n";
		} else {
			amrex::Print() << "Test passed\n";
		}

		// plot
		std::vector<double> xs(nx);
		std::vector<double> rho_x(nx);
		std::vector<double> rho0_x(nx);
		for (int i = 0; i < nx; ++i) {
			xs[i] = position[i];
			rho_x[i] = values.at(HydroSystem<SinkProblem>::density_index)[i];
			rho0_x[i] = values0.at(HydroSystem<SinkProblem>::density_index)[i];
		}

#ifdef HAVE_PYTHON
		matplotlibcpp::clf();
		std::map<std::string, std::string> rho0_args;
		rho0_args["label"] = "rho0";
		rho0_args["color"] = "blue";
		matplotlibcpp::plot(xs, rho0_x, rho0_args);
		std::map<std::string, std::string> rho_args;
		rho_args["label"] = "rho";
		rho_args["color"] = "red";
		rho_args["linestyle"] = "--";
		matplotlibcpp::plot(xs, rho_x, rho_args);
		matplotlibcpp::xlabel("x (cm)");
		matplotlibcpp::ylabel("rho (g cm^-3)");
		matplotlibcpp::legend();
		matplotlibcpp::save("./sink_formation_density.pdf");
#endif
	}

	return status;
}
