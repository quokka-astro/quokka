/// \file testSN.cpp
/// \brief Defines a test problem for supernova feedback.
///

#include "AMReX.H"
#include "AMReX_BC_TYPES.H"
#include "AMReX_GpuQualifiers.H"
#include "AMReX_MultiFab.H"
#include "AMReX_ParmParse.H"
#include "AMReX_Print.H"
#include "AMReX_SPACE.H"
#include "util/fextract.hpp"
#include <fstream>
#include <iomanip>

#include "QuokkaSimulation.hpp"
#include "fundamental_constants.H"
#include "hydro/hydro_system.hpp"
#include "particles/particle_types.hpp"
#include "util/BC.hpp"

#ifdef HAVE_PYTHON
#include "util/matplotlibcpp.h"
#endif

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
const int n_SNR = 1;
constexpr double B0 = 1.0e-7; // uniform background field for MHD variant

static double n_amb = 1.0;    // ambient density (g cm^-3) // NOLINT
static double T_amb = 100.0;  // ambient temperature (K) // NOLINT
static double t_stop = 3.0e5; // stop time (yr) // NOLINT
// static amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> boost_velocity{0.0, 0.0, 0.0}; // NOLINT
static bool skip_checks = false; // NOLINT

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
	static constexpr bool is_mhd_enabled = true;
	static constexpr int nGroups = 1; // number of radiation groups
	static constexpr UnitSystem unit_system = UnitSystem::CGS;
};

template <> struct SimulationData<SNProblem> {
	AMREX_GPU_MANAGED amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> boost_velocity{0.0, 0.0, 0.0};
};

template <> void QuokkaSimulation<SNProblem>::createInitialTestParticles()
{
	// read particles from ASCII file
	const int nreal_extra = 7; // mass vx vy vz birth_time death_time lum
	TestParticles->SetVerbose(0);
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
				p.rdata(quokka::TestParticleVxIdx) += userData_.boost_velocity[0];
				p.rdata(quokka::TestParticleVyIdx) += userData_.boost_velocity[1];
				p.rdata(quokka::TestParticleVzIdx) += userData_.boost_velocity[2];
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
	const double Emag = 0.5 * B0 * B0;
	const double v2 = (userData_.boost_velocity[0] * userData_.boost_velocity[0]) + (userData_.boost_velocity[1] * userData_.boost_velocity[1]) +
			  (userData_.boost_velocity[2] * userData_.boost_velocity[2]);

	// loop over the grid and set the initial condition
	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		state_cc(i, j, k, HydroSystem<SNProblem>::density_index) = rho;
		state_cc(i, j, k, HydroSystem<SNProblem>::x1Momentum_index) = rho * userData_.boost_velocity[0];
		state_cc(i, j, k, HydroSystem<SNProblem>::x2Momentum_index) = rho * userData_.boost_velocity[1];
		state_cc(i, j, k, HydroSystem<SNProblem>::x3Momentum_index) = rho * userData_.boost_velocity[2];
		state_cc(i, j, k, HydroSystem<SNProblem>::energy_index) = rho_e + Emag + 0.5 * rho * v2;
		state_cc(i, j, k, HydroSystem<SNProblem>::internalEnergy_index) = rho_e;
	});
}

template <> void QuokkaSimulation<SNProblem>::setInitialConditionsOnGridFaceVars(quokka::grid const &grid_elem)
{
	const amrex::Array4<double> &state_fc = grid_elem.array_;
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const quokka::direction dir = grid_elem.dir_;
	const double B_val = (dir == quokka::direction::x) ? B0 : 0.0;

	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) { state_fc(i, j, k, Physics_Indices<SNProblem>::mhdFirstIndex) = B_val; });
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
	// get n_amb from the input file
	amrex::ParmParse const pp("problem");
	pp.query("n_amb", n_amb);
	pp.query("T_amb", T_amb);
	pp.query("t_stop", t_stop);
	pp.query("SN_particles_file", SN_particles_file);
	pp.query("refine_half_domain", refine_half_domain);
	pp.query("skip_checks", skip_checks);

	amrex::ParmParse const cpp("cooling");
	cpp.query("cooling_table_type", coolingTableType_);

	// Problem initialization
	QuokkaSimulation<SNProblem> sim;

	sim.stopTime_ = t_stop * year;

	// initialize
	sim.setInitialConditions();

	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx0 = sim.geom[0].CellSizeArray();
	amrex::Real const vol = AMREX_D_TERM(dx0[0], *dx0[1], *dx0[2]);
	amrex::Real const total_mass_init = sim.state_new_cc_[0].sum(HydroSystem<SNProblem>::density_index) * vol;

	// evolve
	sim.evolve();

	auto [position, values] = fextract(sim.state_new_cc_[0], sim.Geom(0), 0, 0, true);
	const int nx = static_cast<int>(position.size());

	std::vector<double> T(nx);
	std::vector<double> x(nx);
	std::vector<double> vx(nx);

	// plot the temperature and vx profile along the x axis at the center
	if (amrex::ParallelDescriptor::IOProcessor()) {
#ifdef HAVE_PYTHON
		for (int i = 0; i < nx; ++i) {
			const double rho = values.at(HydroSystem<SNProblem>::density_index)[i];
			const double Eint = values.at(HydroSystem<SNProblem>::internalEnergy_index)[i];
			const double vx_val = values.at(HydroSystem<SNProblem>::x1Momentum_index)[i] / rho;
			T[i] = Eint / (rho * CV); // simplified, but good enough for the purpose
			x[i] = position[i];
			vx[i] = vx_val;
		}
#endif
	}

	QuokkaSimulation<SNProblem> sim2;
	// set boosted velocity
	const double boost_vel_x = 1.0e8;
	sim2.userData_.boost_velocity = {boost_vel_x, 0.0, 0.0};

	sim2.stopTime_ = t_stop * year;

	// initialize
	sim2.setInitialConditions();

	// evolve
	sim2.evolve();

	auto [position2, values2] = fextract(sim2.state_new_cc_[0], sim2.Geom(0), 0, 0, true);

	std::vector<double> T2(nx);
	std::vector<double> x2(nx);
	std::vector<double> vx2_rel(nx);

	int status = 0;

	// plot the temperature and vx profile along the x axis at the center
	if (amrex::ParallelDescriptor::IOProcessor()) {
#ifdef HAVE_PYTHON
		double v_value_norm = 0.0;
		double v_err_norm = 0.0;
		double T_value_norm = 0.0;
		double T_err_norm = 0.0;
		for (int i = 0; i < nx; ++i) {
			const double rho = values2.at(HydroSystem<SNProblem>::density_index)[i];
			const double Eint = values2.at(HydroSystem<SNProblem>::internalEnergy_index)[i];
			const double vx_val = values2.at(HydroSystem<SNProblem>::x1Momentum_index)[i] / rho;
			T2[i] = Eint / (rho * CV); // simplified, but good enough for the purpose
			x2[i] = position2[i];
			vx2_rel[i] = vx_val - boost_vel_x;
			v_value_norm += std::abs(vx_val); // use raw vx to account for the large boost velocity
			v_err_norm += std::abs(vx2_rel[i] - vx[i]);
			T_value_norm += std::abs(T[i]);
			T_err_norm += std::abs(T2[i] - T[i]);
		}
		const double v_rel_err_norm = v_err_norm / v_value_norm;
		const double T_rel_err_norm = T_err_norm / T_value_norm;
		const double v_rel_err_tol = 0.05;
		const double T_rel_err_tol = 0.001;
		amrex::Print() << fmt::format("Relative L1 norm for vx = {}, tolerence = {}\n", v_rel_err_norm, v_rel_err_tol);
		amrex::Print() << fmt::format("Relative L1 norm for T = {}, tolerence = {}\n", T_rel_err_norm, T_rel_err_tol);
		if (!(v_rel_err_norm < v_rel_err_tol) || !(T_rel_err_norm < T_rel_err_tol)) {
			status = 1;
		}

		matplotlibcpp::clf();
		matplotlibcpp::plot(x, T, {{"label", "base"}, {"color", "C0"}});
		matplotlibcpp::plot(x2, T2, {{"label", "boosted"}, {"color", "C1"}, {"linestyle", "--"}});
		matplotlibcpp::legend();
		matplotlibcpp::xlabel("x (cm)");
		matplotlibcpp::ylabel("T (K)");
		matplotlibcpp::tight_layout();
		matplotlibcpp::save("sn_temperature_profile.pdf");

		matplotlibcpp::clf();
		matplotlibcpp::plot(x, vx, {{"label", "base"}, {"color", "C0"}});
		matplotlibcpp::plot(x2, vx2_rel, {{"label", "boosted"}, {"color", "C1"}, {"linestyle", "--"}});
		matplotlibcpp::legend();
		matplotlibcpp::xlabel("x (cm)");
		matplotlibcpp::ylabel("vx (cm/s)");
		matplotlibcpp::save("sn_velocity_profile.pdf");
#endif
	}

	return status;
}
