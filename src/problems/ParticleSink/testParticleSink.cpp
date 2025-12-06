/// \file testParticleSink.cpp
/// \brief Defines a test problem for sink particles.
///

#include "AMReX.H"
#include "AMReX_BC_TYPES.H"
#include "AMReX_MultiFab.H"
#include "AMReX_ParmParse.H"
#include "AMReX_Print.H"
#include "AMReX_SPACE.H"
#include "math/interpolate.hpp"
#include "util/fextract.hpp"

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

static bool refine_half_domain = false; // NOLINT

constexpr double mu = 1.0 * C::m_p;
constexpr double gamma_ = 5. / 3.;
const double rho0 = 1.0 * C::m_p; // g cm^-3
const double T0 = 10.0;		  // K
const double CV = 1. / (gamma_ - 1.) / mu * C::k_B;
const double year = 3.15576e+07; // in seconds
const double dt_init = 3.0 * year;

static std::string particles_file = "sink4.txt"; // NOLINT

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
	static constexpr bool is_self_gravity_enabled = true;
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

template <> void QuokkaSimulation<SinkProblem>::createInitialSinkParticles()
{
	// read particles from ASCII file
	const int nreal_extra = 4; // mass vx vy vz
	SinkParticles->SetVerbose(1);
	SinkParticles->InitFromAsciiFile(particles_file, nreal_extra, nullptr);
}

template <> void QuokkaSimulation<SinkProblem>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const amrex::Array4<double> &state_cc = grid_elem.array_;
	const double rho_e = CV * T0 * rho0;

	// loop over the grid and set the initial condition
	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		state_cc(i, j, k, HydroSystem<SinkProblem>::density_index) = rho0;
		state_cc(i, j, k, HydroSystem<SinkProblem>::x1Momentum_index) = 0.0;
		state_cc(i, j, k, HydroSystem<SinkProblem>::x2Momentum_index) = 0.0;
		state_cc(i, j, k, HydroSystem<SinkProblem>::x3Momentum_index) = 0.0;
		state_cc(i, j, k, HydroSystem<SinkProblem>::energy_index) = rho_e;
		state_cc(i, j, k, HydroSystem<SinkProblem>::internalEnergy_index) = rho_e;
	});
}

template <> void QuokkaSimulation<SinkProblem>::refineGrid(int lev, amrex::TagBoxArray &tags, amrex::Real /*time*/, int /*ngrow*/)
{
	// tag cells for refinement: static mesh refinement for the whole domain (if refine_half_domain is false) or for x > 0 (if refine_half_domain is true)

	auto const &dx = geom[lev].CellSizeArray();
	auto const &plo = geom[lev].ProbLoArray();
	auto const &phi = geom[lev].ProbHiArray();
	const bool refine_half_domain_ = refine_half_domain;

	for (amrex::MFIter mfi(state_new_cc_[lev]); mfi.isValid(); ++mfi) {
		const amrex::Box &box = mfi.validbox();
		const auto tag = tags.array(mfi);

		amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
			const double x_frac = ((i + 0.5) * dx[0]) / (phi[0] - plo[0]);
			const double y_frac = ((j + 0.5) * dx[1]) / (phi[1] - plo[1]);
			const double z_frac = ((k + 0.5) * dx[2]) / (phi[2] - plo[2]);
			if (!refine_half_domain_ || (x_frac >= 0.7 && x_frac <= 0.8 && y_frac >= 0.3 && y_frac <= 0.7 && z_frac >= 0.3 && z_frac <= 0.7)) {
				tag(i, j, k) = amrex::TagBox::SET;
			}
		});
	}
}

auto problem_main() -> int
{
	auto BCs_cc = quokka::BC<SinkProblem>(quokka::BCType::reflecting);

	amrex::ParmParse const pp("problem");
	pp.query("particles_file", particles_file);
	pp.query("refine_half_domain", refine_half_domain);

	// Problem initialization
	QuokkaSimulation<SinkProblem> sim(BCs_cc);

	sim.reconstructionOrder_ = 3; // 2=PLM, 3=PPM
	sim.cflNumber_ = 0.3;	      // *must* be less than 1/3 in 3D!
	sim.stopTime_ = 10.0 * dt_init;
	sim.initDt_ = dt_init;
	sim.tempFloor_ = 10.0; // K

	// initialize
	sim.setInitialConditions();

	// get total gas mass in the initial state
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx0 = sim.geom[0].CellSizeArray();
	amrex::Real const vol = AMREX_D_TERM(dx0[0], *dx0[1], *dx0[2]);
	amrex::Real const total_mass_init = sim.state_new_cc_[0].sum(HydroSystem<SinkProblem>::density_index) * vol;
	double total_particle_mass = 0.0;

	// get total particle mass
	const auto &real_data = sim.particleRegister_.getParticleDescriptor(quokka::ParticleType::Sink)->getParticleDataAtLevel(0).first;
	if (amrex::ParallelDescriptor::IOProcessor()) {
		// const double total_particle_mass = std::accumulate(real_data.begin(), real_data.end(), 0.0, [](double sum, const auto &d) { return sum +
		// d[3]; });
		for (const auto &p : real_data) {
			total_particle_mass += p[3];
		}
		amrex::Print() << "\nBefore evolution:\n";
		amrex::Print() << "Total gas mass = " << total_mass_init << "\n";
		amrex::Print() << "Total particle mass = " << total_particle_mass << "\n";
	}

	const double total_total_mass_init = total_mass_init + total_particle_mass;

	// evolve
	sim.maxTimesteps_ = 0;
	sim.evolve();

	// get total gas mass in the final state
	amrex::Real const total_mass_step1 = sim.state_new_cc_[0].sum(HydroSystem<SinkProblem>::density_index) * vol;

	auto [position, values] = fextract(sim.state_new_cc_[0], sim.Geom(0), 0, 0.0, true);
	const int nx = static_cast<int>(position.size());

	// get dx
	const double overlap_loc = 1.5001 * dx0[0];
	const double outer_radius = 5.0001 * dx0[0];

	int status = 0;

	const auto &real_data_ste1 = sim.particleRegister_.getParticleDescriptor(quokka::ParticleType::Sink)->getParticleDataAtLevel(0).first;

	if (amrex::ParallelDescriptor::IOProcessor()) {
		// compute total particle mass and error
		double total_particle_mass_step1 = 0.0;
		for (const auto &p : real_data_ste1) {
			total_particle_mass_step1 += p[3];
		}
		const double total_total_mass_step1 = total_mass_step1 + total_particle_mass_step1;

		// compute difference in mass changes
		const double gas_mass_change = total_mass_step1 - total_mass_init;
		const double particle_mass_change = total_particle_mass_step1 - total_particle_mass;
		const double rel_mass_error = gas_mass_change == 0.0 ? 0.0 : std::abs(gas_mass_change + particle_mass_change) / std::abs(gas_mass_change);
		amrex::Print() << "\nAfter evolution:\n";
		amrex::Print() << "Gas mass change = " << gas_mass_change << "\n";
		amrex::Print() << "Particle mass change = " << particle_mass_change << "\n";
		amrex::Print() << "Total mass change = " << gas_mass_change + particle_mass_change << "\n";
		amrex::Print() << "Relative error in change of mass = " << rel_mass_error << "\n";

		// compute relative error in the change of total mass
		const double rel_error_total_mass = std::abs(total_total_mass_step1 - total_total_mass_init) / total_total_mass_init;
		amrex::Print() << "Relative error in change of total mass = " << rel_error_total_mass << "\n";

		// The total mass (gas + particles) should be conserved within machine precision (1e-14)
		const double mass_rel_error_tol = 1.0e-14;
		if (!(rel_error_total_mass < mass_rel_error_tol)) {
			status = 1;
			amrex::Print() << "Test failed: total mass is not conserved at step 1\n";
		}

		// exact solution
		const double rhodot = 7.078494865e-34; // g / cm3 / s
		const double drho = rhodot * dt_init;

		// compute density error
		std::vector<double> xs(nx);
		std::vector<double> xs_over_dx(nx);
		std::vector<double> rho(nx);
		std::vector<double> num_den(nx);
		std::vector<double> exact_den(nx);
		std::vector<double> exact_num_den(nx);
		double err_norm = 0.0;
		double sol_norm = 0.0;
		for (int i = 0; i < nx; ++i) {
			xs[i] = position[i];
			xs_over_dx[i] = position[i] / dx0[0];
			rho[i] = values.at(HydroSystem<SinkProblem>::density_index)[i];
			num_den[i] = rho[i] / C::m_p; // cm^-3

			// exact solution
			if (std::abs(xs[i]) <= overlap_loc) {
				exact_den[i] = rho0 - 4 * drho; // two particles at a position; overlapping
			} else if (std::abs(xs[i]) <= outer_radius) {
				exact_den[i] = rho0 - 2 * drho; // two particles at a position; non-overlapping
			} else {
				exact_den[i] = rho0;
			}
			exact_num_den[i] = exact_den[i] / C::m_p; // cm^-3

			sol_norm += exact_num_den[i];
			err_norm += std::abs(num_den[i] - exact_num_den[i]);
		}

		const double rel_error = err_norm / sol_norm;
		amrex::Print() << "\nCheck density profile:\n";
		amrex::Print() << "Error norm = " << err_norm << "\n";
		amrex::Print() << "Solution norm = " << sol_norm << "\n";
		amrex::Print() << "Relative L1 error norm = " << rel_error << "\n";

		// The relative L1 error norm with respect to the exact solution could be large because there is a hydro update after sink accretion.
		const double rel_error_tol = 3.0e-6;
		if (!(rel_error < rel_error_tol)) {
			status = 1;
			amrex::Print() << "Test failed: density profile is not correct\n";
		}

#ifdef HAVE_PYTHON
		matplotlibcpp::clf();
		matplotlibcpp::ylim(0.0, 1.1);
		std::map<std::string, std::string> exact_num_den_args;
		exact_num_den_args["label"] = "exact";
		exact_num_den_args["color"] = "black";
		matplotlibcpp::plot(xs, exact_num_den, exact_num_den_args);
		std::map<std::string, std::string> num_den_args;
		num_den_args["label"] = "simulation";
		num_den_args["color"] = "red";
		num_den_args["linestyle"] = "--";
		matplotlibcpp::plot(xs, num_den, num_den_args);
		matplotlibcpp::xlabel("x (cm)");
		matplotlibcpp::ylabel("n (cm^-3)");
		matplotlibcpp::legend();
		matplotlibcpp::save("./sink_density.png");

		matplotlibcpp::clf();
		matplotlibcpp::ylim(0.0, 1.1);
		matplotlibcpp::xlim(-12, 12);
		num_den_args["label"] = "simulation";
		num_den_args["color"] = "red";
		num_den_args["linestyle"] = "--";
		matplotlibcpp::plot(xs_over_dx, num_den, num_den_args);
		matplotlibcpp::xlabel("x / dx");
		matplotlibcpp::ylabel("n (cm^-3)");
		matplotlibcpp::legend();
		matplotlibcpp::save("./sink_density_vs_x_over_dx.png");
#endif
	}

	// evolve
	sim.maxTimesteps_ = 10;
	sim.evolve();

	// get total particle mass in the final state
	const auto &real_data_final = sim.particleRegister_.getParticleDescriptor(quokka::ParticleType::Sink)->getParticleDataAtLevel(0).first;
	double total_particle_mass_final = 0.0;
	for (const auto &p : real_data_final) {
		total_particle_mass_final += p[3];
	}
	amrex::Print() << "Total particle mass = " << total_particle_mass_final << "\n";

	// get total gas mass in the final state
	amrex::Real const total_mass_final = sim.state_new_cc_[0].sum(HydroSystem<SinkProblem>::density_index) * vol;

	if (amrex::ParallelDescriptor::IOProcessor()) {
		amrex::Print() << "Total gas mass = " << total_mass_final << "\n";
		const double total_total_mass_final = total_mass_final + total_particle_mass_final;

		// compute relative error in the change of total mass
		const double rel_error_total_mass_final = std::abs(total_total_mass_final - total_total_mass_init) / total_total_mass_init;
		amrex::Print() << "Relative error in change of total mass = " << rel_error_total_mass_final << "\n";

		// Total mass should be conserved to machine precision
		const double mass_rel_error_tol = 1.0e-13;
		if (!(rel_error_total_mass_final < mass_rel_error_tol)) {
			status = 1;
			amrex::Print() << "Test failed: total mass is not conserved at the end of the simulation\n";
		}

		if (status == 0) {
			amrex::Print() << "Test passed\n";
		}
	}

	return status;
}

