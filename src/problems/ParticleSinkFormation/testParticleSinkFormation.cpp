/// \file testParticleSinkFormation.cpp
/// \brief Defines a test problem for sink particle formation.
///

#include "AMReX.H"
#include "AMReX_BC_TYPES.H"
#include "AMReX_MultiFab.H"
#include "AMReX_ParmParse.H"
#include "AMReX_Print.H"
#include "AMReX_SPACE.H"
#include "util/fextract.hpp"
#include <exception>
#include <format>

#include "QuokkaSimulation.hpp"
#include "fundamental_constants.H"
#include "hydro/hydro_system.hpp"
#include "particles/particle_types.hpp"
#include "util/BC.hpp"

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
const double year = 3.15576e+07;			// in seconds
const double cs = std::sqrt(gamma_ * C::k_B * T0 / mu); // NOLINT
constexpr double B0 = 1.0e-7;				// uniform background field

template <> struct Particle_Traits<SinkProblem> : DefaultParticleTraits {
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

template <> struct Physics_Traits<SinkProblem> : DefaultPhysicsTraits {
	// cell-centred
	static constexpr bool is_hydro_enabled = true;
	static constexpr bool is_self_gravity_enabled = true;
	// face-centred
	static constexpr bool is_mhd_enabled = true;
};

template <> void QuokkaSimulation<SinkProblem>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const amrex::Array4<double> &state_cc = grid_elem.array_;
	const auto prob_lo = geom[0].ProbLoArray();
	const auto dx = geom[0].CellSizeArray();

	const double jeans_density = quokka::ParticleUtils::computeJeansDensity(cs, dx[0]);
	const double Emag = 0.5 * B0 * B0;

	// loop over the grid and set the initial condition
	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		const double x = prob_lo[0] + (i * dx[0]);
		const double y = prob_lo[1] + (j * dx[1]);
		const double z = prob_lo[2] + (k * dx[2]);
		double rho = rho0;
		const double sf_cell_loc = 0.1 * dx[0]; // The cell right next to the origin
		if (x <= sf_cell_loc && x + dx[0] > sf_cell_loc && y <= sf_cell_loc && y + dx[1] > sf_cell_loc && z <= sf_cell_loc && z + dx[2] > sf_cell_loc) {
			// this is the cell with peak density
			rho = 1.2 * jeans_density;
		} else if (x - 2 * dx[0] <= sf_cell_loc && x - dx[0] > sf_cell_loc && y <= sf_cell_loc && y + dx[1] > sf_cell_loc && z <= sf_cell_loc &&
			   z + dx[2] > sf_cell_loc) {
			// this is the cell that is 2 cells left of the peak density cell
			rho = 1.1 * jeans_density;
		}
		const double rho_e = CV * T0 * rho;
		state_cc(i, j, k, HydroSystem<SinkProblem>::density_index) = rho;
		state_cc(i, j, k, HydroSystem<SinkProblem>::x1Momentum_index) = 0.0;
		state_cc(i, j, k, HydroSystem<SinkProblem>::x2Momentum_index) = 0.0;
		state_cc(i, j, k, HydroSystem<SinkProblem>::x3Momentum_index) = 0.0;
		state_cc(i, j, k, HydroSystem<SinkProblem>::energy_index) = rho_e + Emag;
		state_cc(i, j, k, HydroSystem<SinkProblem>::internalEnergy_index) = rho_e;
	});
}

template <> void QuokkaSimulation<SinkProblem>::setInitialConditionsOnGridFaceVars(quokka::grid const &grid_elem)
{
	const amrex::Array4<double> &state_fc = grid_elem.array_;
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const quokka::direction dir = grid_elem.dir_;
	const double B_val = (dir == quokka::direction::x) ? B0 : 0.0;

	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) { state_fc(i, j, k, Physics_Indices<SinkProblem>::mhdFirstIndex) = B_val; });
}

template <> void QuokkaSimulation<SinkProblem>::refineGrid(int lev, amrex::TagBoxArray &tags, amrex::Real /*time*/, int /*ngrow*/)
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
	// Problem initialization
	QuokkaSimulation<SinkProblem> sim;

	sim.reconstructionOrder_ = 3; // 2=PLM, 3=PPM
	sim.cflNumber_ = 0.3;	      // *must* be less than 1/3 in 3D!
	sim.stopTime_ = 1.0e7 * year; // 1 Myr

	// initialize
	sim.setInitialConditions();

	const auto [position0, values0] = fextract(sim.state_new_cc_[0], sim.Geom(0), 0, 0.0, true);
	const int nx = static_cast<int>(position0.size());

	// get total gas mass of the initial state
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx0 = sim.geom[0].CellSizeArray();
	amrex::Real const vol = AMREX_D_TERM(dx0[0], *dx0[1], *dx0[2]);
	amrex::Real const m_gas_init = sim.state_new_cc_[0].sum(HydroSystem<SinkProblem>::density_index) * vol;

	// get total particle mass of the initial state
	const int mass_index = 3;
	[[maybe_unused]] const auto [ids_init, real_data_init, int_data_init] =
	    sim.particleRegister_.getParticleDescriptor(quokka::ParticleType::Sink)->getParticleDataAtAllLevels();
	amrex::Real const m_stars_init = std::accumulate(real_data_init.begin(), real_data_init.end(), 0.0,
							 [mass_index](double sum, const auto &particle) { return sum + particle[mass_index]; });
	const double m_tot_init = m_gas_init + m_stars_init;

	int status = 0;

	// evolve to step 1
	sim.maxTimesteps_ = 1;
	sim.evolve();

	// get total gas mass after step 1
	amrex::Real const m_gas_step1 = sim.state_new_cc_[0].sum(HydroSystem<SinkProblem>::density_index) * vol;

	// get total particle mass after step 1
	[[maybe_unused]] const auto [ids_step1, real_data_step1, int_data_step1] =
	    sim.particleRegister_.getParticleDescriptor(quokka::ParticleType::Sink)->getParticleDataAtAllLevels();
	const int n_stars_step1 = static_cast<int>(real_data_step1.size());

	if (n_stars_step1 != 1) {
		status = 1;
		amrex::Print() << "Test failed: expected number of particles created in the formation step is 1, but got " << n_stars_step1 << "\n";
		return status;
	}

	amrex::Real const m_stars_step1 = std::accumulate(real_data_step1.begin(), real_data_step1.end(), 0.0,
							  [mass_index](double sum, const auto &particle) { return sum + particle[mass_index]; });
	const double m_tot_step1 = m_gas_step1 + m_stars_step1;

	amrex::Print() << "Initial gas mass = " << m_gas_init << "\n";
	amrex::Print() << "Initial particle mass = " << m_stars_init << "\n";
	amrex::Print() << "Initial total mass = " << m_tot_init << "\n";

	// Check relative error in the formation step and confirm mass is conserved to machine precision
	const double rel_error_gas_mass_step1 = std::abs(m_tot_init - m_tot_step1) / m_tot_init;
	amrex::Print() << "Step 1: total mass = " << m_tot_step1 << "\n";
	amrex::Print() << "Step 1: (total_mass - initial_total_mass) / initial_total_mass = " << rel_error_gas_mass_step1 << "\n";
	const double rel_error_total_mass_step1 = 1.0e-14;
	if (!(rel_error_gas_mass_step1 < rel_error_total_mass_step1)) {
		status = 1;
		amrex::Print() << "Test failed: mass not conserved to machine precision in the formation step !!!\n";
	}

	// return status;

	// evolve to the end
	sim.maxTimesteps_ = 20;
	sim.evolve();

	// get total gas mass after the end
	amrex::Real const m_gas_final = sim.state_new_cc_[0].sum(HydroSystem<SinkProblem>::density_index) * vol;

	// get total particle mass after the end
	[[maybe_unused]] const auto [ids_final, real_data_final, int_data_final] =
	    sim.particleRegister_.getParticleDescriptor(quokka::ParticleType::Sink)->getParticleDataAtAllLevels();
	amrex::Real const m_stars_final = std::accumulate(real_data_final.begin(), real_data_final.end(), 0.0,
							  [mass_index](double sum, const auto &particle) { return sum + particle[mass_index]; });
	const double m_tot_final = m_gas_final + m_stars_final;

	// Check relative error in the accretion step and confirm mass is conserved to machine precision
	const double rel_error_gas_mass_final = std::abs(m_tot_init - m_tot_final) / m_tot_init;
	amrex::Print() << "Final: rel_err(total_mass) = " << rel_error_gas_mass_final << "\n";
	const double rel_error_total_mass_final = 1.0e-13;
	int status_final = 1;
	if (rel_error_gas_mass_final < rel_error_total_mass_final) {
		status_final = 0;
	}
	status += status_final;

	// find ratio of particle mass to gas mass
	const double mass_ratio = m_stars_final / m_gas_final;
	amrex::Print() << "particle mass / gas mass = " << mass_ratio << "\n";

	if (status > 0) {
		amrex::Print() << "Test failed: mass not conserved to machine precision in the accretion step !!!\n";
	} else {
		amrex::Print() << "Test passed\n";
	}

	auto [position, values] = fextract(sim.state_new_cc_[0], sim.Geom(0), 0, 0.0, true);

	if (amrex::ParallelDescriptor::IOProcessor()) {
		std::vector<double> xs(nx);
		std::vector<double> rho_x(nx);
		std::vector<double> rho0_x(nx);
		for (int i = 0; i < nx; ++i) {
			xs[i] = position[i];
			rho_x[i] = values.at(HydroSystem<SinkProblem>::density_index)[i];
			rho0_x[i] = values0.at(HydroSystem<SinkProblem>::density_index)[i];
		}

#ifdef HAVE_PYTHON
		try {
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
			matplotlibcpp::title(std::format("t = {:.2e}", sim.tNew_[0]));
			matplotlibcpp::legend();
			matplotlibcpp::save("./sink_formation_density.pdf");
		} catch (std::exception const &e) {
			amrex::Print() << "Skipping sink-formation plot: " << e.what() << "\n";
		}
#endif
	}

	return status;
}
