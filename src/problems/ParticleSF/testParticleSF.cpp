/// \file testParticleSF.cpp
/// \brief Defines a test problem for stochastic star formation.
///

#include "AMReX.H"
#include "AMReX_BC_TYPES.H"
#include "AMReX_MultiFab.H"
#include "AMReX_ParallelDescriptor.H"
#include "AMReX_ParmParse.H"
#include "AMReX_Print.H"
#include "AMReX_SPACE.H"
#include "util/BC.hpp"
#include <algorithm>
#include <format>

#include "QuokkaSimulation.hpp"
#include "fundamental_constants.H"
#include "hydro/hydro_system.hpp"
#include "particles/particle_types.hpp"
#include "radiation/radiation_system.hpp"

struct ParticleSFProblem {};

constexpr Real mu = 1.0 * C::m_p;
constexpr Real gamma_ = 5. / 3.;
constexpr Real year = 3.15576e+07;	       // in seconds
static Real n0 = 1.0e4;			       // NOLINT
static Real Tamb = 10.0;		       // NOLINT
static bool validate_initial_imf_stats = true; // NOLINT

template <> struct Particle_Traits<ParticleSFProblem> : DefaultParticleTraits {
	// static constexpr ParticleSwitch particle_switch = ParticleSwitch::None;
	static constexpr ParticleSwitch particle_switch = ParticleSwitch::StochasticStellarPop;
};

template <> struct quokka::EOS_Traits<ParticleSFProblem> {
	static constexpr double gamma = gamma_;
	static constexpr double mean_molecular_weight = mu;
	using EOSBackend = quokka::EOSTabulated<ParticleSFProblem>;
};

template <> struct HydroSystem_Traits<ParticleSFProblem> {
	static constexpr bool reconstruct_eint = true; // need to reconstruct temperature
};

template <> struct Physics_Traits<ParticleSFProblem> : DefaultPhysicsTraits {
	// cell-centred
	static constexpr bool is_hydro_enabled = true;
};

template <> struct SimulationData<ParticleSFProblem> {
	Real m_gas_init;
};

// The same problem with radiation enabled, used to restart from a hydro-only checkpoint of ParticleSFProblem
struct ParticleSFRadProblem {};

template <> struct Particle_Traits<ParticleSFRadProblem> : DefaultParticleTraits {
	static constexpr ParticleSwitch particle_switch = ParticleSwitch::StochasticStellarPop;
};

template <> struct quokka::EOS_Traits<ParticleSFRadProblem> {
	static constexpr double gamma = gamma_;
	static constexpr double mean_molecular_weight = mu;
	using EOSBackend = quokka::EOSTabulated<ParticleSFRadProblem>;
};

template <> struct HydroSystem_Traits<ParticleSFRadProblem> {
	static constexpr bool reconstruct_eint = true;
};

template <> struct Physics_Traits<ParticleSFRadProblem> : DefaultPhysicsTraits {
	static constexpr bool is_hydro_enabled = true;
	static constexpr bool is_radiation_enabled = true;
};

template <> struct RadSystem_Traits<ParticleSFRadProblem> {
	static constexpr double c_hat_over_c = 1.0e-4; // keep the number of radiation substeps small
	static constexpr double Erad_floor = 1.0e-30;  // erg cm^-3; the radiation energy is set to this floor on restart
	static constexpr int beta_order = 0;
};

template <> AMREX_GPU_HOST_DEVICE auto RadSystem<ParticleSFRadProblem>::ComputePlanckOpacity(const double /*rho*/, const double /*Tgas*/) -> amrex::Real
{
	return 1.0e-20; // optically thin
}

template <> AMREX_GPU_HOST_DEVICE auto RadSystem<ParticleSFRadProblem>::ComputeFluxMeanOpacity(const double /*rho*/, const double /*Tgas*/) -> amrex::Real
{
	return 1.0e-20; // optically thin
}

// The same problem with two radiation groups. The particles of the hydro-only checkpoint have one luminosity component,
// so the restart has to add a component to every particle.
struct ParticleSFRad2Problem {};

template <> struct Particle_Traits<ParticleSFRad2Problem> : DefaultParticleTraits {
	static constexpr ParticleSwitch particle_switch = ParticleSwitch::StochasticStellarPop;
};

template <> struct quokka::EOS_Traits<ParticleSFRad2Problem> {
	static constexpr double gamma = gamma_;
	static constexpr double mean_molecular_weight = mu;
	using EOSBackend = quokka::EOSTabulated<ParticleSFRad2Problem>;
};

template <> struct HydroSystem_Traits<ParticleSFRad2Problem> {
	static constexpr bool reconstruct_eint = true;
};

template <> struct Physics_Traits<ParticleSFRad2Problem> : DefaultPhysicsTraits {
	static constexpr bool is_hydro_enabled = true;
	static constexpr bool is_radiation_enabled = true;
	static constexpr int nGroups = 2;
};

template <> struct RadSystem_Traits<ParticleSFRad2Problem> {
	static constexpr double c_hat_over_c = 1.0e-4;
	static constexpr double Erad_floor = 1.0e-30;
	static constexpr int beta_order = 0;
	static constexpr double energy_unit = C::ev2erg;
	// Group 0: FUV, 6 eV to 11.2 eV; group 1: Lyman-Werner, 11.2 eV to 13.6 eV
	static constexpr amrex::GpuArray<double, Physics_Traits<ParticleSFRad2Problem>::nGroups + 1> radBoundaries{6.0, 11.2, 13.6};
	static constexpr OpacityModel opacity_model = OpacityModel::piecewise_constant_opacity;
};

template <>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto
RadSystem<ParticleSFRad2Problem>::DefineOpacityExponentsAndLowerValues(amrex::GpuArray<double, nGroups_ + 1> /*rad_boundaries*/, const double /*rho*/,
								       const double /*Tgas*/) -> amrex::GpuArray<amrex::GpuArray<double, nGroups_ + 1>, 2>
{
	amrex::GpuArray<amrex::GpuArray<double, nGroups_ + 1>, 2> exponents_and_values{};
	for (int i = 0; i < nGroups_ + 1; ++i) {
		exponents_and_values[0][i] = 0.0;     // constant opacity
		exponents_and_values[1][i] = 1.0e-20; // optically thin
	}
	return exponents_and_values;
}

// Sorted records (id, position, non-luminosity reals, ints) of all StochasticStellarPop particles. Collective; the records are
// returned on the IO rank only.
template <typename problem_t> auto getParticleRecords(QuokkaSimulation<problem_t> &sim) -> std::vector<std::vector<double>>
{
	constexpr int nGroups = Physics_Traits<problem_t>::nGroups;
	constexpr int lum_start = AMREX_SPACEDIM + quokka::StochasticStellarPopParticleLumIdx; // the first AMREX_SPACEDIM entries are the position
	const auto [ids, real_data, int_data] =
	    sim.particleRegister_.getParticleDescriptor(quokka::ParticleType::StochasticStellarPop)->getParticleDataAtAllLevels();
	std::vector<std::vector<double>> records(ids.size());
	for (std::size_t i = 0; i < ids.size(); ++i) {
		records[i].push_back(static_cast<double>(ids[i]));
		for (int n = 0; n < static_cast<int>(real_data[i].size()); ++n) {
			if (n < lum_start || n >= lum_start + nGroups) {
				records[i].push_back(real_data[i][n]);
			}
		}
		for (const int v : int_data[i]) {
			records[i].push_back(static_cast<double>(v));
		}
	}
	std::sort(records.begin(), records.end());
	return records;
}

template <> void QuokkaSimulation<ParticleSFProblem>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const amrex::Array4<double> &state_cc = grid_elem.array_;

	const double rho = n0 * mu;
	const double e_int = 1.0 / (gamma_ - 1.0) * rho * C::k_B * Tamb / mu;

	// loop over the grid and set the initial condition
	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		// All cells are Jeans unstable
		state_cc(i, j, k, HydroSystem<ParticleSFProblem>::density_index) = rho;
		state_cc(i, j, k, HydroSystem<ParticleSFProblem>::x1Momentum_index) = 0;
		state_cc(i, j, k, HydroSystem<ParticleSFProblem>::x2Momentum_index) = 0;
		state_cc(i, j, k, HydroSystem<ParticleSFProblem>::x3Momentum_index) = 0;
		state_cc(i, j, k, HydroSystem<ParticleSFProblem>::energy_index) = e_int;
		state_cc(i, j, k, HydroSystem<ParticleSFProblem>::internalEnergy_index) = e_int;
	});
}

template <> void QuokkaSimulation<ParticleSFProblem>::refineGrid(int lev, amrex::TagBoxArray &tags, amrex::Real /*time*/, int /*ngrow*/)
{
	// tag cells for refinement: static mesh refinement for the whole domain

	for (amrex::MFIter mfi(state_new_cc_[lev]); mfi.isValid(); ++mfi) {
		const amrex::Box &box = mfi.validbox();
		const auto tag = tags.array(mfi);

		amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept { tag(i, j, k) = amrex::TagBox::SET; });
	}
}

template <> void QuokkaSimulation<ParticleSFProblem>::computeAfterTimestep()
{
	const int step = istep[0];
	const bool use_default_low_mass_cap = (quokka::low_mass_composite_max_mass >= 0.99 * std::numeric_limits<amrex::Real>::max());
	if (step == 1 && validate_initial_imf_stats && use_default_low_mass_cap) {
		amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx0 = geom[0].CellSizeArray();
		const amrex::Real cell_volume = AMREX_D_TERM(dx0[0], *dx0[1], *dx0[2]);

		amrex::Real eps_ff = 0.5;
		amrex::ParmParse const pp("particles");
		pp.query("eps_ff", eps_ff);

		const amrex::Real eps_star = 0.5;
		const amrex::Real rho0 = n0 * mu;
		const amrex::Real t_ff = std::sqrt(3.0 * M_PI / (32.0 * C::Gconst * rho0));
		const amrex::Real prob_star_formation = (eps_ff / eps_star) * (initDt_ / t_ff);
		amrex::Print() << "Probability of star formation = " << prob_star_formation << "\n";
		AMREX_ALWAYS_ASSERT_WITH_MESSAGE(prob_star_formation < 1.0,
						 "Probability of star formation must be less than 1.0, adjust Tamb, dx, or rho to ensure this is the case");

		const auto n_cells = CountCells(0);
		const auto [real_data_final, idata_final] =
		    particleRegister_.getParticleDescriptor(quokka::ParticleType::StochasticStellarPop)->getParticleDataAtLevel(0);
		const double m_gas_final = state_new_cc_[0].sum(HydroSystem<ParticleSFProblem>::density_index) * cell_volume;

		const int mass_idx = 3;

		int status = 0;

		if (amrex::ParallelDescriptor::IOProcessor()) {
			const double exp_Mstar_high_mean = 19.39; // Msun
			const double exp_fstar_high = 0.220;	  // fraction of mass in high mass stars

			const amrex::Real exp_m_star_per_cell = rho0 * cell_volume * eps_star;
			const amrex::Real exp_m_star_high_per_cell = exp_m_star_per_cell * exp_fstar_high;
			const amrex::Real exp_m_star_high_total = exp_m_star_high_per_cell * static_cast<amrex::Real>(n_cells) * prob_star_formation;
			const amrex::Real exp_n_star_high_total = exp_m_star_high_total / (exp_Mstar_high_mean * C::M_solar);
			// one low-mass star per cell
			const amrex::Real exp_n_star_low_total = static_cast<amrex::Real>(n_cells) * prob_star_formation;

			// statistics from the simulation

			double m_star_high_tot = 0.0;
			double m_star_tot = 0.0;
			int n_star_high = 0;
			const int n_star_tot = static_cast<int>(real_data_final.size());

			if (n_star_tot == 0) {
				amrex::Abort("Test failed: No stars formed in step 1");
			}

			for (int i = 0; i < n_star_tot; ++i) {
				if (idata_final[i][0] != static_cast<int>(quokka::StellarEvolutionStage::LowMassComposite)) {
					m_star_high_tot += real_data_final[i][mass_idx];
					n_star_high++;
				}
				m_star_tot += real_data_final[i][mass_idx];
			}
			const double mean_mass_high_mass_stars = m_star_high_tot / n_star_high;
			const int n_star_low = n_star_tot - n_star_high;

			double log_vel = NAN;
			double vx = NAN;
			double vy = NAN;
			double vz = NAN;
			double vtot = NAN;
			double vmin = 3.e5;  // minimum velocity in km/s
			double vmax = -3.e5; // maximum velocity in km/s
			const int n_bins = 20;
			const double log_v_min = std::log(3.0);	  // minimum velocity of the input distribution
			const double log_v_max = std::log(385.0); // maximum velocity of the input distributions
			const double bin_width = (log_v_max - log_v_min) / n_bins;
			std::vector<int> hist(n_bins, 0);
			for (int i = 0; i < n_star_tot; ++i) {
				if (idata_final[i][0] != static_cast<int>(quokka::StellarEvolutionStage::LowMassComposite)) {
					vx = real_data_final[i][mass_idx + 1] / 1.e5;
					vy = real_data_final[i][mass_idx + 2] / 1.e5;
					vz = real_data_final[i][mass_idx + 3] / 1.e5;
					vtot = std::sqrt(vx * vx + vy * vy + vz * vz);
					if (vtot < vmin) {
						vmin = vtot; // update minimum velocity
					} else if (vtot > vmax) {
						vmax = vtot; // update maximum velocity
					}
					log_vel = std::log(vtot); // store log of velocity in km/s
					int const bin_index = static_cast<int>((log_vel - log_v_min) / bin_width);
					if (bin_index >= 0 && bin_index < n_bins) {
						hist[bin_index]++;
					}
				}
			}

			double const slope_predicted = 1. - ((std::log(hist[n_bins - 1]) - std::log(hist[0])) / (log_v_max - log_v_min));
			amrex::Print() << "Slope of velocity distribution = " << slope_predicted << "\n";
			amrex::Print() << "Minimum velocity = " << vmin << " km/s\n";
			amrex::Print() << "Maximum velocity = " << vmax << " km/s\n";

			// get total mass in gas
			const double m_gas_change = userData_.m_gas_init - m_gas_final;

			amrex::Print() << "Mass of high-mass stars [expected]   = " << m_star_high_tot / C::M_solar << " ["
				       << exp_m_star_high_total / C::M_solar << "] M_sol \n";
			amrex::Print() << "Number of high-mass stars [expected]   = " << n_star_high << " [" << exp_n_star_high_total << "] \n";
			amrex::Print() << "Mean mass of high-mass stars [expected]   = " << mean_mass_high_mass_stars / C::M_solar << " ["
				       << exp_Mstar_high_mean << "] M_sol \n";
			amrex::Print() << "Number of low-mass stars [expected]   = " << n_star_low << " [" << exp_n_star_low_total << "] \n";
			amrex::Print() << "Mass of all stars [expected]   = " << m_star_tot / C::M_solar << " [" << m_gas_change / C::M_solar << "] M_sol \n";

			const Real tol_m_star_high_tot = 0.1;
			const Real tol_m_star_tot = 0.1;
			const Real tol_n_star_high = 0.1;
			const Real sigma_over_expectation_n_star_low = 1.0 / std::sqrt(exp_n_star_low_total);
			const Real tol_n_star_low = 3.0 * sigma_over_expectation_n_star_low;

			if (!((m_star_high_tot - exp_m_star_high_total) / exp_m_star_high_total < tol_m_star_high_tot)) {
				status = 1;
				amrex::Print() << "Test failed: Mass of high-mass stars does not match expectation\n";
			}
			if (!((n_star_high - exp_n_star_high_total) / exp_n_star_high_total < tol_n_star_high)) {
				status = 1;
				amrex::Print() << "Test failed: Number of high-mass stars does not match expectation\n";
			}
			if (!((n_star_low - exp_n_star_low_total) / exp_n_star_low_total < tol_n_star_low)) {
				status = 1;
				amrex::Print() << "Test failed: Number of low-mass stars does not match expectation\n";
			}
			if (!((m_star_tot - m_gas_change) / m_gas_change < tol_m_star_tot)) {
				status = 1;
				amrex::Print() << "Test failed: Total mass of all stars does not match expectation\n";
			}
		}

		AMREX_ALWAYS_ASSERT_WITH_MESSAGE(status == 0, "Test failed at step 1");
	}
}

// Part 1: hydro-only star formation test. On a fresh run, particle_records is set to the particle records at the end (see
// getParticleRecords) and chk_name to the name of the last checkpoint (empty if no checkpoint was written).
auto runParticleSF(std::vector<std::vector<double>> &particle_records, std::string &chk_name) -> int
{
	// Problem initialization
	QuokkaSimulation<ParticleSFProblem> sim;

	sim.reconstructionOrder_ = 3; // 2=PLM, 3=PPM
	sim.cflNumber_ = 0.3;	      // *must* be less than 1/3 in 3D!
	sim.stopTime_ = 1.0e7 * year; // 10 Myr

	// set random state
	const int seed = 42;
	amrex::InitRandom(seed, 1); // all ranks should produce the same values

	// Real Tamb and n0 from the input file
	amrex::ParmParse const ppp("problem");
	ppp.query("Tamb", Tamb);
	ppp.query("n0", n0);
	ppp.query("validate_initial_imf_stats", validate_initial_imf_stats);
	bool verify_low_mass_cap_on_restart = false;
	ppp.query("verify_low_mass_cap_on_restart", verify_low_mass_cap_on_restart);

	sim.setInitialConditions();

	// get total gas mass
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx0 = sim.geom[0].CellSizeArray();
	const amrex::Real cell_volume = AMREX_D_TERM(dx0[0], *dx0[1], *dx0[2]);
	sim.userData_.m_gas_init = sim.state_new_cc_[0].sum(HydroSystem<ParticleSFProblem>::density_index) * cell_volume;

	sim.evolve();

	// We validate restarting from a checkpoint below when verify_low_mass_cap_on_restart is true.
	std::string restartfile;
	amrex::ParmParse const p3;
	p3.query("restartfile", restartfile);
	if (!restartfile.empty()) {
		if (!verify_low_mass_cap_on_restart) {
			return 0; // success
		}

		amrex::Real low_mass_cap = std::numeric_limits<amrex::Real>::max();
		amrex::ParmParse const p_particles("particles");
		p_particles.query("low_mass_composite_max_mass", low_mass_cap);

		const auto [real_data_restart, idata_restart] =
		    sim.particleRegister_.getParticleDescriptor(quokka::ParticleType::StochasticStellarPop)->getParticleDataAtLevel(0);
		const amrex::Real mass_tol = 1.0e-12 * std::max(low_mass_cap, static_cast<amrex::Real>(1.0));

		int num_low_mass_particles = 0;
		int num_cap_violations = 0;
		int restart_validation_status = 0;

		if (amrex::ParallelDescriptor::IOProcessor()) {
			for (std::size_t i = 0; i < real_data_restart.size(); ++i) {
				const bool is_low_mass_composite = (idata_restart[i][quokka::StochasticStellarPopParticleStageIdx] ==
								    static_cast<int>(quokka::StellarEvolutionStage::LowMassComposite));
				if (is_low_mass_composite) {
					num_low_mass_particles++;
					if (real_data_restart[i][quokka::StochasticStellarPopParticleMassIdx] > (low_mass_cap + mass_tol)) {
						num_cap_violations++;
					}
				}
			}

			amrex::Print() << "Restart low-mass cap validation: LowMassComposite particles = " << num_low_mass_particles
				       << ", cap violations = " << num_cap_violations << ", cap = " << low_mass_cap / C::M_solar << " Msun\n";
			if (num_low_mass_particles == 0 || num_cap_violations != 0) {
				restart_validation_status = 1;
			}
		}

		amrex::ParallelDescriptor::Bcast(&restart_validation_status, 1, amrex::ParallelDescriptor::IOProcessorNumber());
		AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
		    restart_validation_status == 0,
		    "Restart low-mass cap validation failed: either no LowMassComposite particles were found or at least one exceeded the mass cap.");
		return 0; // success
	}

	// If not restarting from a checkpoint, validate mass sonservation (roughly)

	const auto [real_data_final2, idata_final2] =
	    sim.particleRegister_.getParticleDescriptor(quokka::ParticleType::StochasticStellarPop)->getParticleDataAtLevel(0);
	amrex::ignore_unused(idata_final2);

	int status = 0;

	// total particle and gas masses (collective operations, so they must be called on all ranks)
	const double m_star_tot2 = sim.particleRegister_.getParticleDescriptor(quokka::ParticleType::StochasticStellarPop)->computeStellarMass();
	const double m_gas_final2 = sim.state_new_cc_[0].sum(HydroSystem<ParticleSFProblem>::density_index) * cell_volume;

	if (amrex::ParallelDescriptor::IOProcessor()) {
		const double m_gas_change2 = sim.userData_.m_gas_init - m_gas_final2;
		amrex::Print() << std::format("Mass of all stars [expected]   = {:.6e} [{:.6e}] M_sol \n", m_star_tot2 / C::M_solar,
					      m_gas_change2 / C::M_solar);

		const double tol_m_star_tot2 = 0.1;
		if (!((m_star_tot2 - m_gas_change2) / m_star_tot2 < tol_m_star_tot2)) {
			status = 1;
			amrex::Print() << "Test failed: Total mass of all stars does not match expectation\n";
		}
	}
	amrex::ParallelDescriptor::Bcast(&status, 1, amrex::ParallelDescriptor::IOProcessorNumber());

	particle_records = getParticleRecords(sim);
	if (sim.checkpointInterval_ > 0) {
		// the last checkpoint is written at the final step
		std::string chk_prefix = "chk";
		p3.query("checkpoint_prefix", chk_prefix);
		chk_name = amrex::Concatenate(chk_prefix, sim.istep[0], 7);
	}

	return status;
}

// Part 2: restart the hydro-only checkpoint written by part 1 with radiation enabled. Right after the restart, the radiation
// energy on the grid must equal the floor, the radiation flux and the particle luminosities must be zero, and all other
// particle data must equal the checkpoint.
template <typename problem_t> auto runRadRestart(const std::vector<std::vector<double>> &particle_records_chk, const std::string &chk_name) -> int
{
	using RadSys = RadSystem<problem_t>;
	constexpr int nGroups = Physics_Traits<problem_t>::nGroups;

	amrex::Print() << "\n=== Part 2: restart the hydro-only checkpoint with " << nGroups << "-group radiation ===\n";
	// Override runtime parameters in the global ParmParse table. add() appends a new value and query() returns the last one,
	// so these take precedence over the input file. They must be added before the simulation is constructed, because the
	// constructor reads the parameters and evolve() re-reads them; setting e.g. sim.maxTimesteps_ directly would be undone.
	amrex::ParmParse pp;
	pp.add("restartfile", chk_name);
	pp.add("max_timesteps", 12);
	pp.add("checkpoint_interval", -1);
	pp.add("plotfile_interval", -1);
	amrex::ParmParse pp_particles("particles");
	pp_particles.add("use_luminosity_table", false); // keep the luminosities at zero

	QuokkaSimulation<problem_t> sim;
	sim.reconstructionOrder_ = 3;
	sim.cflNumber_ = 0.3;
	sim.stopTime_ = 1.0e7 * year;
	sim.setInitialConditions();

	int status = 0;

	// radiation energy and flux of every group on every level
	for (int lev = 0; lev <= sim.finestLevel(); ++lev) {
		for (int g = 0; g < nGroups; ++g) {
			const int offset = Physics_NumVars::numRadVarsPerGroup * g;
			const int e_comp = RadSys::radEnergy_index + offset;
			const amrex::Real e_min = sim.state_new_cc_[lev].min(e_comp);
			const amrex::Real e_max = sim.state_new_cc_[lev].max(e_comp);
			if (e_min != RadSys::Erad_floor_ || e_max != RadSys::Erad_floor_) {
				status = 1;
				amrex::Print() << "Test failed: radiation energy of group " << g << " on level " << lev << " is in [" << e_min << ", " << e_max
					       << "], expected " << RadSys::Erad_floor_ << "\n";
			}
			for (const int comp : {RadSys::x1RadFlux_index, RadSys::x2RadFlux_index, RadSys::x3RadFlux_index}) {
				const amrex::Real max_abs = sim.state_new_cc_[lev].norm0(comp + offset);
				if (max_abs != 0.0) {
					status = 1;
					amrex::Print() << "Test failed: radiation flux component " << comp + offset << " on level " << lev
						       << " has max |value| = " << max_abs << "\n";
				}
			}
		}
	}

	// particle luminosities and all other particle data
	const auto [ids, real_data, int_data] =
	    sim.particleRegister_.getParticleDescriptor(quokka::ParticleType::StochasticStellarPop)->getParticleDataAtAllLevels();
	amrex::ignore_unused(ids, int_data);
	const auto particle_records = getParticleRecords(sim);
	if (amrex::ParallelDescriptor::IOProcessor()) {
		int n_nonzero_lum = 0;
		for (const auto &rdata : real_data) {
			for (int g = 0; g < nGroups; ++g) {
				// the first AMREX_SPACEDIM entries are the particle position
				if (rdata[AMREX_SPACEDIM + quokka::StochasticStellarPopParticleLumIdx + g] != 0.0) {
					n_nonzero_lum++;
				}
			}
		}
		const bool records_match = (particle_records == particle_records_chk);
		amrex::Print() << "Particles in checkpoint = " << particle_records_chk.size() << ", after restart = " << particle_records.size()
			       << ", nonzero luminosity components = " << n_nonzero_lum << ", other particle data identical = " << records_match << "\n";
		if (particle_records.empty() || !records_match || n_nonzero_lum != 0) {
			status = 1;
			amrex::Print() << "Test failed: particles were not restarted correctly\n";
		}
	}
	amrex::ParallelDescriptor::Bcast(&status, 1, amrex::ParallelDescriptor::IOProcessorNumber());

	// The radiation-hydro run must continue from the restarted state. No particle radiates (there is no luminosity table),
	// so the total gas + radiation energy must stay nearly constant, and stars formed after the restart must have zero
	// luminosity too.
	const auto total_energy = [&sim]() {
		amrex::Real e = sim.state_new_cc_[0].sum(HydroSystem<problem_t>::energy_index);
		for (int g = 0; g < nGroups; ++g) {
			e += sim.state_new_cc_[0].sum(RadSys::radEnergy_index + Physics_NumVars::numRadVarsPerGroup * g);
		}
		return e;
	};
	const amrex::Real energy_restart = total_energy();
	sim.evolve();
	const amrex::Real energy_final = total_energy();
	const auto [ids_final, real_data_final, int_data_final] =
	    sim.particleRegister_.getParticleDescriptor(quokka::ParticleType::StochasticStellarPop)->getParticleDataAtAllLevels();
	amrex::ignore_unused(ids_final, int_data_final);
	if (amrex::ParallelDescriptor::IOProcessor()) {
		int n_nonzero_lum_final = 0;
		for (const auto &rdata : real_data_final) {
			for (int g = 0; g < nGroups; ++g) {
				if (rdata[AMREX_SPACEDIM + quokka::StochasticStellarPopParticleLumIdx + g] != 0.0) {
					n_nonzero_lum_final++;
				}
			}
		}
		const amrex::Real rel_energy_change = std::abs(energy_final - energy_restart) / energy_restart;
		amrex::Print() << "After the radiation-hydro steps: particles = " << real_data_final.size()
			       << ", nonzero luminosity components = " << n_nonzero_lum_final << ", relative change of gas + radiation energy = " << rel_energy_change
			       << "\n";
		// written so that NaN or inf fails
		if (n_nonzero_lum_final != 0 || !(rel_energy_change < 0.1)) {
			status = 1;
			amrex::Print() << "Test failed: the radiation-hydro run after the restart is not consistent\n";
		}
	}
	amrex::ParallelDescriptor::Bcast(&status, 1, amrex::ParallelDescriptor::IOProcessorNumber());

	return status;
}

auto problem_main() -> int
{
	std::vector<std::vector<double>> particle_records;
	std::string chk_name;
	const int status = runParticleSF(particle_records, chk_name);

	// part 2 needs a checkpoint written by a fresh (not restarted) run of part 1
	if (status != 0 || chk_name.empty()) {
		return status;
	}

	// single-group radiation keeps the particle layout; two groups add a luminosity component to every particle
	if (runRadRestart<ParticleSFRadProblem>(particle_records, chk_name) != 0) {
		return 1;
	}
	return runRadRestart<ParticleSFRad2Problem>(particle_records, chk_name);
}
