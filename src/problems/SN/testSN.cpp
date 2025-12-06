/// \file testSN.cpp
/// \brief Defines a test problem for supernova feedback.
///

#include "AMReX.H"
#include "AMReX_BC_TYPES.H"
#include "AMReX_MultiFab.H"
#include "AMReX_ParmParse.H"
#include "AMReX_Print.H"
#include "AMReX_SPACE.H"
#include <fstream>
#include <iomanip>

#include "QuokkaSimulation.hpp"
#include "fundamental_constants.H"
#include "hydro/hydro_system.hpp"
#include "particles/particle_types.hpp"
#include "util/BC.hpp"

struct SNProblem {
};

static bool refine_half_domain = false; // NOLINT

static double max_Eint_global = 0.0;	       // NOLINT
static double max_Eint_last = 0.0;	       // NOLINT
static std::vector<double> max_Eint_history{}; // NOLINT
static std::vector<double> t_history{};	       // NOLINT

static std::string SN_particles_file = "SN_particles.txt"; // NOLINT
static std::string coolingTableType_ = "resampled";	   // NOLINT

constexpr double mu = 1.0 * C::m_u;
// constexpr double mu = 1.295 * C::m_u; // neutral gas
constexpr double gamma_ = 5. / 3.;
const double CV = 1. / (gamma_ - 1.) / mu * C::k_B;
const double cloudy_H_mass_fraction = 1.0 / (1.0 + 0.1 * 3.971);
const double year = 3.15576e+07; // in seconds
const double mass_SNR = 10.0 * C::M_solar;
const int n_SNR = 2;

static double n_amb = 1.0;    // ambient density (g cm^-3) // NOLINT
static double T_amb = 100.0;  // ambient temperature (K) // NOLINT
static double t_stop = 3.0e5; // stop time (yr) // NOLINT

template <> struct Particle_Traits<SNProblem> {
	// static constexpr ParticleSwitch particle_switch = ParticleSwitch::None;
	static constexpr ParticleSwitch particle_switch = ParticleSwitch::Test;
};

template <> struct quokka::EOS_Traits<SNProblem> {
	static constexpr double gamma = gamma_;
	static constexpr double mean_molecular_weight = mu;
};

template <> struct HydroSystem_Traits<SNProblem> {
	static constexpr bool reconstruct_eint = true; // need to reconstruct temperature
};

template <> struct Physics_Traits<SNProblem> {
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

template <> void QuokkaSimulation<SNProblem>::createInitialTestParticles()
{
	// read particles from ASCII file
	const int nreal_extra = 7; // mass vx vy vz birth_time death_time lum
	TestParticles->SetVerbose(1);
	TestParticles->InitFromAsciiFile(SN_particles_file, nreal_extra, nullptr);

	// Loop over all particle at all levels and set first integer component to SNProgenitor
	for (int lev = 0; lev <= TestParticles->finestLevel(); ++lev) {
		auto &particles = TestParticles->GetParticles(lev);

		for (auto &kv : particles) {
			auto &particle_array = kv.second.GetArrayOfStructs();
			const int np = particle_array.numParticles();
			auto *pdata = particle_array().data();

			// Launch GPU kernel to set integer components
			amrex::ParallelFor(np, [=] AMREX_GPU_DEVICE(int i) {
				auto &p = pdata[i]; // NOLINT
				p.idata(0) = static_cast<int>(quokka::StellarEvolutionStage::SNProgenitor);
			});
		}
	}

	// Ensure GPU operations are complete
	amrex::Gpu::streamSynchronize();
}

template <> void QuokkaSimulation<SNProblem>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const amrex::Array4<double> &state_cc = grid_elem.array_;
	const double rho_bg = n_amb * C::m_u / cloudy_H_mass_fraction;
	const double E0 = CV * T_amb * rho_bg;
	const double rho = rho_bg;
	const double rho_e = E0;

	// loop over the grid and set the initial condition
	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		state_cc(i, j, k, HydroSystem<SNProblem>::density_index) = rho;
		state_cc(i, j, k, HydroSystem<SNProblem>::x1Momentum_index) = 0.0;
		state_cc(i, j, k, HydroSystem<SNProblem>::x2Momentum_index) = 0.0;
		state_cc(i, j, k, HydroSystem<SNProblem>::x3Momentum_index) = 0.0;
		state_cc(i, j, k, HydroSystem<SNProblem>::energy_index) = rho_e;
		state_cc(i, j, k, HydroSystem<SNProblem>::internalEnergy_index) = rho_e;
	});
}

template <> void QuokkaSimulation<SNProblem>::refineGrid(int lev, amrex::TagBoxArray &tags, amrex::Real /*time*/, int /*ngrow*/)
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

template <> void QuokkaSimulation<SNProblem>::computeAfterTimestep()
{
	// find the maximum temperature in the state_new_cc_[0]
	const double max_internal_energy_density = state_new_cc_[0].max(HydroSystem<SNProblem>::internalEnergy_index);
	max_Eint_global = std::max(max_Eint_global, max_internal_energy_density);
	max_Eint_last = max_internal_energy_density;
	max_Eint_history.push_back(max_internal_energy_density);
	t_history.push_back(tNew_[0]);
}

auto problem_main() -> int
{
	// Set boundary conditions - octant symmetry (reflecting)
	auto BCs_cc = quokka::BC<SNProblem>(quokka::BCType::reflecting);

	// get n_amb from the input file
	amrex::ParmParse const pp("problem");
	pp.query("n_amb", n_amb);
	pp.query("T_amb", T_amb);
	pp.query("t_stop", t_stop);
	pp.query("SN_particles_file", SN_particles_file);
	pp.query("refine_half_domain", refine_half_domain);

	amrex::ParmParse const cpp("cooling");
	cpp.query("cooling_table_type", coolingTableType_);

	// Problem initialization
	QuokkaSimulation<SNProblem> sim(BCs_cc);

	sim.reconstructionOrder_ = 3; // 2=PLM, 3=PPM
	sim.stopTime_ = t_stop * year;
	sim.cflNumber_ = 0.3; // *must* be less than 1/3 in 3D!
	sim.initDt_ = 0.001 * year;

	// initialize
	sim.setInitialConditions();

	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx0 = sim.geom[0].CellSizeArray();
	amrex::Real const vol = AMREX_D_TERM(dx0[0], *dx0[1], *dx0[2]);
	amrex::Real const total_mass_init = sim.state_new_cc_[0].sum(HydroSystem<SNProblem>::density_index) * vol;

	// evolve
	sim.evolve();

	amrex::Real const total_mass_final = sim.state_new_cc_[0].sum(HydroSystem<SNProblem>::density_index) * vol;
	const amrex::Real mass_increase = total_mass_final - total_mass_init;
	amrex::Print() << "----------------- Problem diagnostics -----------------" << "\n";
	amrex::Print() << "Total mass increase: " << mass_increase << "\n";
	amrex::Print() << "Expected total mass increase: " << n_SNR * mass_SNR << "\n";
	const double mass_increase_rel_err = std::abs(mass_increase - n_SNR * mass_SNR) / (n_SNR * mass_SNR);
	amrex::Print() << "Mass increase relative error: " << mass_increase_rel_err << "\n";
	const double mass_increase_rel_err_tol = 1.0e-7;

	const amrex::Real max_internal_energy = max_Eint_global * vol;
	const amrex::Real expected_minimum_max_internal_energy = 1.0e51 / (7 * 7 * 7); // 1e51 erg energy into (2 * 3 + 1)^3 cells
	int status = 1;
	const bool pass_max_internal_energy = max_internal_energy > expected_minimum_max_internal_energy;
	const bool pass_mass = mass_increase_rel_err < mass_increase_rel_err_tol;
	bool is_pass = pass_max_internal_energy;
	if (sim.maxTimesteps_ < 20) {
		is_pass = is_pass && pass_mass;
	} else {
		// Write data to CSV file
		std::ofstream csv_file("sn_energy_history_" + coolingTableType_ + ".csv");
		if (csv_file.is_open()) {
			// Set precision to 13 significant digits
			csv_file << std::scientific << std::setprecision(13);

			// Write header
			csv_file << "step, Time_yr, Max_Internal_Energy_erg\n";

			// Write data
			for (int i = 0; i < static_cast<int>(max_Eint_history.size()); ++i) {
				csv_file << i << ", " << t_history[i] / year << ", " << max_Eint_history[i] * vol << "\n";
			}

			csv_file.close();
			amrex::Print() << "Energy history data written to sn_energy_history.csv\n";
		} else {
			amrex::Print() << "Error: Could not open CSV file for writing\n";
		}
	}

	if (is_pass) {
		status = 0;
		amrex::Print() << "Test passed. Max internal energy in cells: " << max_internal_energy << "\n";
		amrex::Print() << "Max internal energy last timestep: " << max_Eint_last * vol << "\n";
	} else {
		status = 1;
		amrex::Print() << "Test failed. Max internal energy in cells too low: " << max_internal_energy << "\n";
		amrex::Print() << "Expected minimum max internal energy: " << expected_minimum_max_internal_energy << "\n";
		amrex::Print() << "OR\n";
		amrex::Print() << "Mass increase relative error: " << mass_increase_rel_err << "\n";
	}
	amrex::Print() << "---------------------------------------------------------" << "\n";

	return status;
}

