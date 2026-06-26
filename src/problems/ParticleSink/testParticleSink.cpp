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
#include <format>
#include <numeric>
#include <utility>

#include "QuokkaSimulation.hpp"
#include "fundamental_constants.H"
#include "hydro/hydro_system.hpp"
#include "particles/particle_types.hpp"
#include "particles/particle_utils.hpp"
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
const double dt_init = 2.0 * year;
constexpr double B0 = 3.715708546e-08; // constant background field [Gauss-equivalent units]. Set a precise value so that
				       // beta = 2.0

static std::string particles_file = "sink4.txt"; // NOLINT

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

template <> struct SimulationData<SinkProblem> {
	AMREX_GPU_MANAGED amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> boost_velocity{0.0, 0.0, 0.0};
};

template <> void QuokkaSimulation<SinkProblem>::createInitialSinkParticles()
{
	// read particles from ASCII file
	const int nreal_extra = 4; // mass vx vy vz
	SinkParticles->SetVerbose(1);
	SinkParticles->InitFromAsciiFile(particles_file, nreal_extra, nullptr);

	// Apply boost velocity to particles if needed
	for (int lev = 0; lev <= SinkParticles->finestLevel(); ++lev) {
		auto &particles = SinkParticles->GetParticles(lev);

		for (auto &kv : particles) {
			auto &particle_array = kv.second.GetArrayOfStructs();
			const int np = particle_array.numParticles();
			auto *pdata = particle_array().data();
			const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> boost_velocity = userData_.boost_velocity;

			// Launch GPU kernel to apply boost velocity to particles
			amrex::ParallelFor(np, [=] AMREX_GPU_DEVICE(int i) {
				auto &p = pdata[i]; // NOLINT
				p.rdata(quokka::SinkParticleVxIdx) += boost_velocity[0];
				p.rdata(quokka::SinkParticleVyIdx) += boost_velocity[1];
				p.rdata(quokka::SinkParticleVzIdx) += boost_velocity[2];
			});
		}
	}

	// Ensure GPU operations are complete
	amrex::Gpu::streamSynchronize();
}

template <> void QuokkaSimulation<SinkProblem>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const amrex::Array4<double> &state_cc = grid_elem.array_;
	const double rho_e = CV * T0 * rho0;
	const double Emag = 0.5 * B0 * B0;
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> boost_velocity = userData_.boost_velocity;
	const double v2 = (userData_.boost_velocity[0] * userData_.boost_velocity[0]) + (userData_.boost_velocity[1] * userData_.boost_velocity[1]) +
			  (userData_.boost_velocity[2] * userData_.boost_velocity[2]);

	// loop over the grid and set the initial condition
	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		state_cc(i, j, k, HydroSystem<SinkProblem>::density_index) = rho0;
		state_cc(i, j, k, HydroSystem<SinkProblem>::x1Momentum_index) = rho0 * boost_velocity[0];
		state_cc(i, j, k, HydroSystem<SinkProblem>::x2Momentum_index) = rho0 * boost_velocity[1];
		state_cc(i, j, k, HydroSystem<SinkProblem>::x3Momentum_index) = rho0 * boost_velocity[2];
		state_cc(i, j, k, HydroSystem<SinkProblem>::energy_index) = rho_e + Emag + 0.5 * rho0 * v2;
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
	amrex::ParmParse const pp("problem");
	pp.query("particles_file", particles_file);
	pp.query("refine_half_domain", refine_half_domain);
	double boost_vel_x = NAN;
	pp.query("boost_vel_x", boost_vel_x);
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(std::isfinite(boost_vel_x), "boost_vel_x must be set in the input file");

	// Problem initialization
	QuokkaSimulation<SinkProblem> sim;

	sim.reconstructionOrder_ = 3;  // 2=PLM, 3=PPM
	sim.cflNumber_ = 0.3;	       // *must* be less than 1/3 in 3D!
	sim.stopTime_ = 1000.0 * year; // 1000 years
	sim.tempFloor_ = 10.0;	       // K

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

	// ============================================================
	// Phase 1: Run base simulation for 1 timestep and validate against analytic solution
	// ============================================================
	amrex::Print() << "\n=== Phase 1: Base simulation (1 timestep) ===\n";
	sim.maxTimesteps_ = 1;
	sim.initDt_ = dt_init; // set a small initial dt to limit the accreted mass to a small fraction of the total mass
	sim.evolve();

	// get total gas mass in the final state
	amrex::Real const total_mass_step1 = sim.state_new_cc_[0].sum(HydroSystem<SinkProblem>::density_index) * vol;

	auto [position, values] = fextract(sim.state_new_cc_[0], sim.Geom(0), 0, 0.0, true);
	const int nx = static_cast<int>(position.size());

	// get dx
	const double overlap_loc = 1.5001 * dx0[0];
	const double outer_radius = 5.0001 * dx0[0];

	int status = 0;
	Real rho_dot_exact = 0.0;

	const auto &real_data_ste1 = sim.particleRegister_.getParticleDescriptor(quokka::ParticleType::Sink)->getParticleDataAtLevel(0).first;

	// Store Phase 1 particle positions and angular momentum for later Galilean invariance comparison
	std::vector<std::array<double, 3>> ang_mom_phase1;
	std::vector<std::array<double, 3>> pos_phase1;

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

		// Print and store angular momentum for each particle (data layout: x,y,z, mass,vx,vy,vz,mdot,Lx,Ly,Lz)
		constexpr int Lx_offset = AMREX_SPACEDIM + quokka::SinkParticleLxIdx;
		for (int pi = 0; std::cmp_less(pi, real_data_ste1.size()); ++pi) {
			const auto &p = real_data_ste1[pi];
			amrex::Print() << std::format("Particle {} angular momentum: Lx={:.4e} Ly={:.4e} Lz={:.4e}\n", pi, p[Lx_offset], p[Lx_offset + 1],
						      p[Lx_offset + 2]);
			ang_mom_phase1.push_back({p[Lx_offset], p[Lx_offset + 1], p[Lx_offset + 2]});
			pos_phase1.push_back({p[0], p[1], p[2]});
		}

		// compute relative error in the change of total mass
		const double rel_error_total_mass = std::abs(total_total_mass_step1 - total_total_mass_init) / total_total_mass_init;
		amrex::Print() << "Relative error in change of total mass = " << rel_error_total_mass << "\n";

		// The total mass (gas + particles) should be conserved within machine precision (1e-14)
		const double mass_rel_error_tol = 1.0e-14;
		if (!(rel_error_total_mass < mass_rel_error_tol)) {
			status = 1;
			amrex::Print() << "Test failed: total mass is not conserved at step 1\n";
		}

		// compute exact accretion rate (MHD-aware Bondi-Hoyle formula)
		{
			const Real magnetic_pressure = 0.5 * B0 * B0;
			const Real beta = (rho0 / mu) * C::k_B * T0 / magnetic_pressure;
			const Real cs_iso = std::sqrt(C::k_B * T0 / mu);
			// MHD-aware fast magnetosonic speed: cf^2 = cs^2 * (1 + 2/beta) (isothermal)
			const Real cf_sqr = cs_iso * cs_iso * (1.0 + 2.0 / beta);
			const Real v_infty_sqr = 0.0;
			const Real par_mass = 10.0 * C::M_solar;
			const Real r_BH = C::Gconst * par_mass / (v_infty_sqr + cf_sqr);
			const Real lambda = gcem::exp(1.5) / 4.0;
			// M_dot = 4 pi rho_infty r_BH^2 * sqrt(v_infty^2 + lambda^2 cf^2), where lambda = exp(3/2) / 4
			const Real M_dot_exact = 4.0 * M_PI * rho0 * r_BH * r_BH * std::sqrt(v_infty_sqr + lambda * lambda * cf_sqr);
			rho_dot_exact = M_dot_exact / std::pow(7 * dx0[0], 3);
			amrex::Print() << "Exact rhodot = " << rho_dot_exact << "\n";
		}

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
			const Real drho = rho_dot_exact * sim.tNew_[0];
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
		amrex::Print() << "\nCheck density profile vs analytic solution:\n";
		amrex::Print() << "Error norm = " << err_norm << "\n";
		amrex::Print() << "Solution norm = " << sol_norm << "\n";
		amrex::Print() << "Relative L1 error norm = " << rel_error << "\n";

		// The relative L1 error norm with respect to the exact solution could be large because there is a hydro update after sink accretion.
		const double rel_error_tol = 1.0e-5;
		if (!(std::abs(rel_error) < rel_error_tol)) {
			status = 1;
			amrex::Print() << "Test failed: density profile does not match analytic solution\n";
		} else {
			amrex::Print() << "Phase 1 passed: density profile matches analytic solution\n";
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
		matplotlibcpp::title(std::format("t = {:.2e}", sim.tNew_[0]));
		matplotlibcpp::legend();
		matplotlibcpp::save("./sink_density.pdf");
#endif
	}

	// ============================================================
	// Phase 2: Run boosted simulation for 1 timestep and validate Galilean invariance
	// ============================================================
	amrex::Print() << "\n=== Phase 2: Boosted simulation (1 timestep) - Galilean invariance test ===\n";

	QuokkaSimulation<SinkProblem> sim2;
	sim2.userData_.boost_velocity = {boost_vel_x, 0.0, 0.0};

	sim2.reconstructionOrder_ = 3;
	sim2.cflNumber_ = 0.3;
	sim2.stopTime_ = 1000.0 * year; // 1000 years
	sim2.initDt_ = dt_init;		// set a small initial dt to limit the accreted mass to a small fraction of the total mass
	sim2.tempFloor_ = 10.0;

	// initialize
	sim2.setInitialConditions();

	// evolve for 1 timestep to match sim's Phase 1
	sim2.maxTimesteps_ = 1;
	sim2.evolve();

	const auto &real_data_phase2 = sim2.particleRegister_.getParticleDescriptor(quokka::ParticleType::Sink)->getParticleDataAtLevel(0).first;

	// Extract density profile from boosted simulation
	auto [position2, values2] = fextract(sim2.state_new_cc_[0], sim2.Geom(0), 0, 0.0, true);

	// Validate boosted simulation against analytical solution (Galilean invariance test)
	// If the physics is Galilean invariant, the boosted simulation should match its analytical
	// solution with the same accuracy as the base simulation matches its analytical solution
	if (amrex::ParallelDescriptor::IOProcessor()) {
		// Compare angular momentum vs Phase 1 for Galilean invariance validation.
		// Angular momentum is computed in the particle's rest frame (using relative velocity), so
		// the accreted L must be identical regardless of the boost.
		// Sort both Phase 1 and Phase 2 particles by position before comparing to ensure
		// consistent ordering, since particle redistribution may change the array order.
		constexpr int Lx_offset = AMREX_SPACEDIM + quokka::SinkParticleLxIdx;
		const double ang_mom_rel_error_tol = 0.03; // 3% tolerance

		// Build sorted index arrays for Phase 1 and Phase 2 by (x, y, z)
		const int npar = static_cast<int>(ang_mom_phase1.size());
		std::vector<int> idx1(npar);
		std::iota(idx1.begin(), idx1.end(), 0);
		std::sort(idx1.begin(), idx1.end(), [&](int a, int b) {
			return std::tie(pos_phase1[a][0], pos_phase1[a][1], pos_phase1[a][2]) < std::tie(pos_phase1[b][0], pos_phase1[b][1], pos_phase1[b][2]);
		});

		std::vector<int> idx2(npar);
		std::iota(idx2.begin(), idx2.end(), 0);
		std::sort(idx2.begin(), idx2.end(), [&](int a, int b) {
			return std::tie(real_data_phase2[a][0], real_data_phase2[a][1], real_data_phase2[a][2]) <
			       std::tie(real_data_phase2[b][0], real_data_phase2[b][1], real_data_phase2[b][2]);
		});

		amrex::Print() << "\nAngular momentum Galilean invariance check:\n";
		for (int pi = 0; pi < npar; ++pi) {
			const auto &p2 = real_data_phase2[idx2[pi]];
			const double Lx2 = p2[Lx_offset];
			const double Ly2 = p2[Lx_offset + 1];
			const double Lz2 = p2[Lx_offset + 2];
			amrex::Print() << std::format("Particle {} (Phase2) angular momentum: Lx={:.4e} Ly={:.4e} Lz={:.4e}\n", pi, Lx2, Ly2, Lz2);

			const double Lx1 = ang_mom_phase1[idx1[pi]][0];
			const double Ly1 = ang_mom_phase1[idx1[pi]][1];
			const double Lz1 = ang_mom_phase1[idx1[pi]][2];
			const double L1_norm = std::sqrt(Lx1 * Lx1 + Ly1 * Ly1 + Lz1 * Lz1);
			const double dL_norm = std::sqrt((Lx2 - Lx1) * (Lx2 - Lx1) + (Ly2 - Ly1) * (Ly2 - Ly1) + (Lz2 - Lz1) * (Lz2 - Lz1));
			const double ang_mom_rel_error = (L1_norm > 0.0) ? dL_norm / L1_norm : dL_norm;
			amrex::Print() << std::format("Particle {} angular momentum relative error (Phase1 vs Phase2): {:.4e}\n", pi, ang_mom_rel_error);
			if (!(ang_mom_rel_error < ang_mom_rel_error_tol)) {
				status = 1;
				amrex::Print() << std::format("Test failed: angular momentum not Galilean invariant for particle {}\n", pi);
			}
		}

		// Compute analytical solution for boosted case based on its actual evolution time
		const Real drho2 = rho_dot_exact * sim2.tNew_[0];

		// Compute density error for boosted simulation vs analytical solution
		std::vector<double> rho2(nx);
		std::vector<double> num_den2(nx);
		std::vector<double> exact_den2(nx);
		std::vector<double> exact_num_den2(nx);
		double err_norm2 = 0.0;
		double sol_norm2 = 0.0;

		for (int i = 0; i < nx; ++i) {
			const double x = position2[i];
			rho2[i] = values2.at(HydroSystem<SinkProblem>::density_index)[i];
			num_den2[i] = rho2[i] / C::m_p; // cm^-3
			// Analytical solution (same formula as Phase 1, but with drho2)
			if (std::abs(x) <= overlap_loc) {
				exact_den2[i] = rho0 - 4 * drho2; // two particles overlapping
			} else if (std::abs(x) <= outer_radius) {
				exact_den2[i] = rho0 - 2 * drho2; // two particles non-overlapping
			} else {
				exact_den2[i] = rho0;
			}
			exact_num_den2[i] = exact_den2[i] / C::m_p; // cm^-3
			sol_norm2 += exact_num_den2[i];
			err_norm2 += std::abs(num_den2[i] - exact_num_den2[i]);
		}

		const double rel_error2 = err_norm2 / sol_norm2;
		amrex::Print() << "\nCheck boosted density profile vs analytic solution:\n";
		amrex::Print() << "Error norm = " << err_norm2 << "\n";
		amrex::Print() << "Solution norm = " << sol_norm2 << "\n";
		amrex::Print() << "Relative L1 error norm = " << rel_error2 << "\n";

		// Compare error in boosted case to error in base case to validate Galilean invariance
		// Both should have similar accuracy relative to their respective analytical solutions
		const double rel_error_tol = 1.0e-5;
		if (!(std::abs(rel_error2) < rel_error_tol)) {
			status = 1;
			amrex::Print() << "Test failed: boosted simulation does not match analytic solution\n";
		} else {
			amrex::Print() << "Phase 2 passed: boosted simulation matches analytic solution (Galilean invariance validated)\n";
		}

#ifdef HAVE_PYTHON
		matplotlibcpp::clf();
		matplotlibcpp::ylim(0.0, 1.1);
		std::map<std::string, std::string> exact_num_den_args;
		exact_num_den_args["label"] = "exact";
		exact_num_den_args["color"] = "black";
		matplotlibcpp::plot(position2, exact_num_den2, exact_num_den_args);
		std::map<std::string, std::string> num_den_args;
		num_den_args["label"] = "simulation";
		num_den_args["color"] = "red";
		num_den_args["linestyle"] = "--";
		matplotlibcpp::plot(position2, num_den2, num_den_args);
		matplotlibcpp::xlabel("x (cm)");
		matplotlibcpp::ylabel("n (cm^-3)");
		matplotlibcpp::title(std::format("t = {:.2e}", sim2.tNew_[0]));
		matplotlibcpp::legend();
		matplotlibcpp::save("./sink_density_boosted.pdf");
#endif
	}

	// ============================================================
	// Phase 3: Continue boosted simulation for 10 more timesteps and check mass conservation
	// ============================================================
	amrex::Print() << "\n=== Phase 3: Boosted simulation (10 more timesteps) - mass conservation test ===\n";

	// Get initial mass for Phase 3
	amrex::Real const total_mass_phase3_init = sim2.state_new_cc_[0].sum(HydroSystem<SinkProblem>::density_index) * vol;
	const auto &real_data_phase3_init = sim2.particleRegister_.getParticleDescriptor(quokka::ParticleType::Sink)->getParticleDataAtLevel(0).first;
	double total_particle_mass_phase3_init = 0.0;
	for (const auto &p : real_data_phase3_init) {
		total_particle_mass_phase3_init += p[3];
	}
	const double total_total_mass_phase3_init = total_mass_phase3_init + total_particle_mass_phase3_init;

	// Continue evolution for 10 more timesteps
	sim2.maxTimesteps_ = 11; // already did 1, so total will be 11
	sim2.evolve();

	// Get final mass for Phase 3
	amrex::Real const total_mass_phase3_final = sim2.state_new_cc_[0].sum(HydroSystem<SinkProblem>::density_index) * vol;
	const auto &real_data_phase3_final = sim2.particleRegister_.getParticleDescriptor(quokka::ParticleType::Sink)->getParticleDataAtLevel(0).first;
	double total_particle_mass_phase3_final = 0.0;
	for (const auto &p : real_data_phase3_final) {
		total_particle_mass_phase3_final += p[3];
	}
	const double total_total_mass_phase3_final = total_mass_phase3_final + total_particle_mass_phase3_final;

	const auto &real_data_phase3 = sim2.particleRegister_.getParticleDescriptor(quokka::ParticleType::Sink)->getParticleDataAtLevel(0).first;

	if (amrex::ParallelDescriptor::IOProcessor()) {
		// Print angular momentum for each particle (data layout: x,y,z, mass,vx,vy,vz,mdot,Lx,Ly,Lz)
		constexpr int Lx_offset = AMREX_SPACEDIM + quokka::SinkParticleLxIdx;
		for (int pi = 0; std::cmp_less(pi, real_data_phase3.size()); ++pi) {
			const auto &p = real_data_phase3[pi];
			amrex::Print() << std::format("Particle {} angular momentum: Lx={:.4e} Ly={:.4e} Lz={:.4e}\n", pi, p[Lx_offset], p[Lx_offset + 1],
						      p[Lx_offset + 2]);
		}

		amrex::Print() << "\nPhase 3 mass conservation check:\n";
		amrex::Print() << "Initial total mass = " << total_total_mass_phase3_init << "\n";
		amrex::Print() << "Final total mass = " << total_total_mass_phase3_final << "\n";

		// compute relative error in the change of total mass
		const double rel_error_total_mass_phase3 =
		    std::abs(total_total_mass_phase3_final - total_total_mass_phase3_init) / total_total_mass_phase3_init;
		amrex::Print() << "Relative error in change of total mass = " << rel_error_total_mass_phase3 << "\n";

		// Total mass should be conserved to machine precision
		const double mass_rel_error_tol = 1.0e-13;
		if (!(rel_error_total_mass_phase3 < mass_rel_error_tol)) {
			status = 1;
			amrex::Print() << "Test failed: total mass is not conserved in Phase 3\n";
		} else {
			amrex::Print() << "Phase 3 passed: mass conservation satisfied\n";
		}

		if (status == 0) {
			amrex::Print() << "\n=== All phases passed ===\n";
		} else {
			amrex::Print() << "\n=== One of the 3 phases failed ===\n";
		}
	}

	return status;
}
