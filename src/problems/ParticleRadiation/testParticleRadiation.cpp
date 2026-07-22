/// \file testParticleRadiation.cpp
/// \brief Defines a test problem for radiation from particles.
///

#include "AMReX_ParallelDescriptor.H"
#include "AMReX_ParmParse.H"
#include "AMReX_Print.H"

#include "QuokkaSimulation.hpp"
#include "fundamental_constants.H"
#include "hydro/hydro_system.hpp"
#include "particles/particle_update.hpp"
#include "radiation/radiation_system.hpp"
#include "util/BC.hpp"
#include "util/DataTable.hpp"

struct ParticleRadiationProblem {
};

constexpr double mu = 1.0 * C::m_p;
constexpr double gamma_ = 5. / 3.;
constexpr double rho0 = 1.0e-8 * C::m_p; // g cm^-3
constexpr double T0 = 10.0;		 // K
constexpr double CV = 1. / (gamma_ - 1.) / mu * C::k_B;
constexpr double initial_Erad = 1.0e-30 * CV * rho0 * T0;
constexpr double dt_ = 0.1 * quokka::seconds_per_year;
// constexpr double chat_over_c = 1.0e-5;
constexpr double chat_over_c = 1.0;
constexpr double formation_time = 1.5 * dt_;

template <> struct SimulationData<ParticleRadiationProblem> {
	std::string particles_filename = "../inputs/TestParticlesNoRad.txt";
};

template <> struct quokka::EOS_Traits<ParticleRadiationProblem> {
	static constexpr double gamma = gamma_;
	static constexpr double mean_molecular_weight = mu;
};

// Test enum to demonstrate type checking of particle_switch
enum class TestEnum : unsigned int {
	MISTAKE = 0b00000100U,
};

template <> struct Particle_Traits<ParticleRadiationProblem> : DefaultParticleTraits {
	// static constexpr ParticleSwitch particle_switch = ParticleSwitch::None;
	static constexpr ParticleSwitch particle_switch = ParticleSwitch::StochasticStellarPop;
};

template <> struct HydroSystem_Traits<ParticleRadiationProblem> {
	static constexpr bool reconstruct_eint = true; // need to reconstruct temperature
};

template <> struct Physics_Traits<ParticleRadiationProblem> : DefaultPhysicsTraits {
	static constexpr bool is_hydro_enabled = true;
	static constexpr bool is_radiation_enabled = true;
	static constexpr int nGroups = 2; // number of radiation groups
};

template <> struct RadSystem_Traits<ParticleRadiationProblem> {
	static constexpr double c_hat_over_c = chat_over_c;
	static constexpr double Erad_floor = initial_Erad;
	static constexpr int beta_order = 0;
	static constexpr double energy_unit = C::ev2erg; // set boundary unit to eV
	// Define radiation group boundaries for 2-group radiation
	// Group 0: 1 eV to 100 eV, Group 1: 100 eV to 10000 eV
	static constexpr amrex::GpuArray<double, Physics_Traits<ParticleRadiationProblem>::nGroups + 1> radBoundaries{1.0, 100.0, 10000.0};
	static constexpr OpacityModel opacity_model = OpacityModel::piecewise_constant_opacity;
};

template <>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto
RadSystem<ParticleRadiationProblem>::DefineOpacityExponentsAndLowerValues(amrex::GpuArray<double, nGroups_ + 1> /*rad_boundaries*/, const double /*rho*/,
									  const double /*Tgas*/) -> amrex::GpuArray<amrex::GpuArray<double, nGroups_ + 1>, 2>
{
	amrex::GpuArray<amrex::GpuArray<double, nGroups_ + 1>, 2> exponents_and_values{};
	for (int i = 0; i < nGroups_ + 1; ++i) {
		exponents_and_values[0][i] = 0.0;     // exponent (0 = constant opacity)
		exponents_and_values[1][i] = 1.0e-20; // opacity value (0 = optically thin)
	}
	return exponents_and_values;
}

template <> void QuokkaSimulation<ParticleRadiationProblem>::createInitialStochasticStellarPopParticles()
{
	// Read particles from ASCII file. Note that this only read real components and not integer components, therefore we need to use
	// InitSetPhyParticles to set the integer components
	// mass, vx, vy, vz, birth/death time, birth/death pos, death_density, mass_at_birth, lum[nGroups]
	const int nreal_extra = quokka::StochasticStellarPopParticleRealComps<ParticleRadiationProblem>;
	StochasticStellarPopParticles->SetVerbose(1);
	StochasticStellarPopParticles->InitFromAsciiFile(userData_.particles_filename, nreal_extra, nullptr);

	// Using a for loop from lev = 0 to StochasticStellarPopParticles->maxLevel() won't work because not all levels necessarily have particles, and when
	// some levels do not have particles, StochasticStellarPopParticles->GetParticles(lev) will result in a Segfault. Therefore, we loop over the actual
	// particle container.
	for (auto &kv : StochasticStellarPopParticles->GetParticles()) {
		for (auto &ikv : kv) {
			auto &particle_array = ikv.second.GetArrayOfStructs();
			const int np = particle_array.numParticles();

			if (np == 0) {
				continue;
			}

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

template <> void QuokkaSimulation<ParticleRadiationProblem>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const amrex::Array4<double> &state_cc = grid_elem.array_;

	const auto Erad0 = initial_Erad;

	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		const double rho = rho0;
		const double rho_e = CV * T0 * rho;

		// Set radiation variables
		for (int g = 0; g < Physics_Traits<ParticleRadiationProblem>::nGroups; ++g) {
			state_cc(i, j, k, RadSystem<ParticleRadiationProblem>::radEnergy_index + Physics_NumVars::numRadVarsPerGroup * g) = Erad0;
			state_cc(i, j, k, RadSystem<ParticleRadiationProblem>::x1RadFlux_index + Physics_NumVars::numRadVarsPerGroup * g) = 0;
			state_cc(i, j, k, RadSystem<ParticleRadiationProblem>::x2RadFlux_index + Physics_NumVars::numRadVarsPerGroup * g) = 0;
			state_cc(i, j, k, RadSystem<ParticleRadiationProblem>::x3RadFlux_index + Physics_NumVars::numRadVarsPerGroup * g) = 0;
		}
		state_cc(i, j, k, RadSystem<ParticleRadiationProblem>::gasEnergy_index) = rho_e;
		state_cc(i, j, k, RadSystem<ParticleRadiationProblem>::gasDensity_index) = rho;
		state_cc(i, j, k, RadSystem<ParticleRadiationProblem>::gasInternalEnergy_index) = rho_e;
		state_cc(i, j, k, RadSystem<ParticleRadiationProblem>::x1GasMomentum_index) = 0.0;
		state_cc(i, j, k, RadSystem<ParticleRadiationProblem>::x2GasMomentum_index) = 0.0;
		state_cc(i, j, k, RadSystem<ParticleRadiationProblem>::x3GasMomentum_index) = 0.0;
	});
}

auto problem_main() -> int
{
	// Problem initialization
	QuokkaSimulation<ParticleRadiationProblem> sim;
	sim.maxDt_ = dt_;

	// Read parameters from input file
	const amrex::ParmParse pp("problem");
	pp.query("particles_filename", sim.userData_.particles_filename);

	quokka::TransformType rad_table_output_transform = quokka::TransformType::linear;
	const amrex::ParmParse ppp("particles");
	ppp.query("rad_table_output_transform", rad_table_output_transform);

	// initialize (this will parse particle parameters and load luminosity table)
	sim.setInitialConditions();

	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx0 = sim.geom[0].CellSizeArray();
	amrex::Real const vol = AMREX_D_TERM(dx0[0], *dx0[1], *dx0[2]);
	// Total radiation energy in the field
	amrex::Real total_Erad_init = 0.0;
	for (int g = 0; g < Physics_Traits<ParticleRadiationProblem>::nGroups; ++g) {
		total_Erad_init +=
		    sim.state_new_cc_[0].sum(RadSystem<ParticleRadiationProblem>::radEnergy_index + Physics_NumVars::numRadVarsPerGroup * g) * vol;
	}

	// total gas energy
	const amrex::Real total_gas_energy_init = sim.state_new_cc_[0].sum(RadSystem<ParticleRadiationProblem>::gasEnergy_index) * vol;

	// set force finest level to true for test particles
	// sim.particleRegister_.getParticleDescriptor(quokka::ParticleType::StochasticStellarPop)->setForceFinestLevel(true);

	// evolve
	sim.evolve();

	// ----- Check Stochastic particles -----

	// Total radiation energy in the field
	amrex::Real total_Erad = 0.0;
	for (int g = 0; g < Physics_Traits<ParticleRadiationProblem>::nGroups; ++g) {
		total_Erad += sim.state_new_cc_[0].sum(RadSystem<ParticleRadiationProblem>::radEnergy_index + Physics_NumVars::numRadVarsPerGroup * g) * vol;
	}

	// total gas energy
	const amrex::Real total_gas_energy = sim.state_new_cc_[0].sum(RadSystem<ParticleRadiationProblem>::gasEnergy_index) * vol;

	int status = 0; // Initialize to success

	if (amrex::ParallelDescriptor::IOProcessor()) {

		// print total gas energy
		amrex::Print() << "Total gas energy (initial): " << total_gas_energy_init << "\n";
		amrex::Print() << "Total gas energy (final): " << total_gas_energy << "\n";
		amrex::Print() << "Total radiation energy (initial): " << total_Erad_init / chat_over_c << "\n";
		amrex::Print() << "Total radiation energy (final): " << total_Erad / chat_over_c << "\n";

		const double total_energy_init = total_Erad_init / chat_over_c + total_gas_energy_init;
		const double total_energy = total_Erad / chat_over_c + total_gas_energy;
		const double change_of_total_energy = total_energy - total_energy_init;
		amrex::Print() << "Change of total energy: " << change_of_total_energy << "\n";

		// Expected answer from table interpolation.
		// Radiation is deposited into cells after the first step.
		// The table gives luminosity values based on (mass, age) interpolation.
		// For this test with the current table, the expected luminosity per particle is determined by table interpolation.
		// The emission per particle from step 0 is 0.0;
		// The emission per particle from step 1 is 2.5e20 * dt_.
		// The emission per particle from step 2 is (2 * 2.5e20) * dt_.
		const int n_stars = 4;
		double L_star = NAN;
		double change_of_total_energy_expected = NAN;
		AMREX_ALWAYS_ASSERT_WITH_MESSAGE(sim.maxTimesteps_ == 3, "This test requires max_timesteps = 3");
		if (rad_table_output_transform == quokka::TransformType::log) {
			L_star = 3e40;
			change_of_total_energy_expected = (1.0 + 2.0) * L_star * dt_ * n_stars;
		} else if (rad_table_output_transform == quokka::TransformType::fast_log) {
			// The stellar age won't fall exactly onto the fastlog-sampled grids, so in order to test perfect accuracy, we have to set luminosity to
			// constant over time
			change_of_total_energy_expected = (1.0e+41 + 1.0e+41) * dt_ * n_stars;
		} else {
			L_star = 2.5e40;
			change_of_total_energy_expected = (1.0 + 2.0) * L_star * dt_ * n_stars;
		}
		const double change_of_total_energy_expected_group2 = change_of_total_energy_expected * 10.0;
		const double change_all_groups = change_of_total_energy_expected + change_of_total_energy_expected_group2;
		amrex::Print() << "Current time: " << sim.tNew_[0] << "\n";
		amrex::Print() << "Expected change of total energy: " << change_all_groups << "\n";

		const double error_rel_to_tot = std::abs(change_of_total_energy - change_all_groups) / total_energy;
		const double error_rel_to_rad = std::abs(change_of_total_energy - change_all_groups) / change_all_groups;
		amrex::Print() << "Relative error to total energy: " << error_rel_to_tot << "\n";
		amrex::Print() << "Relative error to radiation energy: " << error_rel_to_rad << "\n";

		// On CPUs, the error is 1e-15, close to machine accuracy. One GPUs, the error, caused by std::log or std::pow, is slight higher at 1e-14.
		const double tolerance = rad_table_output_transform == quokka::TransformType::fast_log ? 1.0e-11 : 1e-13; // Tolerance relative to total energy
		if (!(error_rel_to_tot < tolerance) || !(error_rel_to_rad < tolerance)) {
			status = 1;
			amrex::Print() << "Test failed: change of total energy mismatch.\n";
		}

		if (status == 0) {
			amrex::Print() << "Test passed: change of total energy within tolerance.\n";
		}
	}

	return status;
}
