/// \file testParticleRadiationSlug.cpp
/// \brief Validates a slug2-generated stellar luminosity table against direct slug2 output.
///
/// A single 120 Msun star with an age of 1 Myr radiates into two bands, FUV (6-13.6 eV) and
/// Lyman continuum (13.6-54.4 eV). Both the stellar mass and the stellar age sit exactly on a
/// node of the luminosity table, so the table interpolation is a lookup and the radiation energy
/// deposited on the grid can be compared with the luminosity that slug2 prints for that star.
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

struct ParticleRadiationSlugProblem {};

constexpr double mu = 1.0 * C::m_p;
constexpr double gamma_ = 5. / 3.;
constexpr double rho0 = 1.0e-8 * C::m_p; // g cm^-3
constexpr double T0 = 10.0;		 // K
constexpr double CV = 1. / (gamma_ - 1.) / mu * C::k_B;
constexpr double initial_Erad = 1.0e-30 * CV * rho0 * T0;
constexpr double dt_ = 0.1 * quokka::seconds_per_year;
constexpr double chat_over_c = 1.0;

// Stellar age and mass of the single test star. Both are nodes of the luminosity table
// ../inputs/slug_FUV_LyC.csv, whose age axis is log-spaced over [1e5, 1e8] yr with 31 points
// (node 10 is 1e6 yr) and whose mass axis is log-spaced over [2.1, 120] Msun with 21 points
// (node 20 is 120 Msun).
constexpr double stellar_age = 1.0e6 * quokka::seconds_per_year; // s
constexpr double stellar_mass = 120.0 * C::M_solar;		 // g

// Luminosities of a 120 Msun star at an age of 1 Myr, in erg/s, taken directly from slug2:
//   SLUG_DIR=<slug2> write_isochrone -m0 60 -m1 120 -nm 2 \
//       -tf 911.6485178911784 2066.403307220004 -tf 227.9121294727946 911.6485178911784 \
//       mist_2016_vvcrit_40 1e6
// The two top-hat filters are the FUV band (6-13.6 eV) and the Lyman continuum (13.6-54.4 eV).
constexpr double L_FUV = 2.44698e+39;
constexpr double L_LyC = 3.88642e+39;

template <> struct SimulationData<ParticleRadiationSlugProblem> {
	std::string particles_filename = "../inputs/TestParticleSlugStar.txt";
};

template <> struct quokka::EOS_Traits<ParticleRadiationSlugProblem> {
	static constexpr double gamma = gamma_;
	static constexpr double mean_molecular_weight = mu;
};

template <> struct Particle_Traits<ParticleRadiationSlugProblem> : DefaultParticleTraits {
	static constexpr ParticleSwitch particle_switch = ParticleSwitch::StochasticStellarPop;
};

template <> struct HydroSystem_Traits<ParticleRadiationSlugProblem> {
	static constexpr bool reconstruct_eint = true; // need to reconstruct temperature
};

template <> struct Physics_Traits<ParticleRadiationSlugProblem> : DefaultPhysicsTraits {
	static constexpr bool is_hydro_enabled = true;
	static constexpr bool is_radiation_enabled = true;
	static constexpr int nGroups = 2; // FUV and Lyman continuum
};

template <> struct RadSystem_Traits<ParticleRadiationSlugProblem> {
	static constexpr double c_hat_over_c = chat_over_c;
	static constexpr double Erad_floor = initial_Erad;
	static constexpr int beta_order = 0;
	static constexpr double energy_unit = C::ev2erg; // set boundary unit to eV
	// Group 0: FUV, 6 eV to 13.6 eV; group 1: Lyman continuum, 13.6 eV to 54.4 eV
	static constexpr amrex::GpuArray<double, Physics_Traits<ParticleRadiationSlugProblem>::nGroups + 1> radBoundaries{6.0, 13.6, 54.4};
	static constexpr OpacityModel opacity_model = OpacityModel::piecewise_constant_opacity;
};

template <>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto
RadSystem<ParticleRadiationSlugProblem>::DefineOpacityExponentsAndLowerValues(amrex::GpuArray<double, nGroups_ + 1> /*rad_boundaries*/, const double /*rho*/,
									      const double /*Tgas*/)
    -> amrex::GpuArray<amrex::GpuArray<double, nGroups_ + 1>, 2>
{
	amrex::GpuArray<amrex::GpuArray<double, nGroups_ + 1>, 2> exponents_and_values{};
	for (int i = 0; i < nGroups_ + 1; ++i) {
		exponents_and_values[0][i] = 0.0;     // exponent (0 = constant opacity)
		exponents_and_values[1][i] = 1.0e-20; // opacity value (0 = optically thin)
	}
	return exponents_and_values;
}

template <> void QuokkaSimulation<ParticleRadiationSlugProblem>::createInitialStochasticStellarPopParticles()
{
	// Read particles from ASCII file. This only reads real components, so the integer components
	// are set below.
	const int nreal_extra = quokka::StochasticStellarPopParticleRealComps<ParticleRadiationSlugProblem>;
	StochasticStellarPopParticles->SetVerbose(1);
	StochasticStellarPopParticles->InitFromAsciiFile(userData_.particles_filename, nreal_extra, nullptr);

	for (auto &kv : StochasticStellarPopParticles->GetParticles()) {
		for (auto &ikv : kv) {
			auto &particle_array = ikv.second.GetArrayOfStructs();
			const int np = particle_array.numParticles();

			if (np == 0) {
				continue;
			}

			auto *pdata = particle_array().data();

			amrex::ParallelFor(np, [=] AMREX_GPU_DEVICE(int i) {
				auto &p = pdata[i]; // NOLINT
				p.idata(0) = static_cast<int>(quokka::StellarEvolutionStage::SNProgenitor);
			});
		}
	}

	amrex::Gpu::streamSynchronize();
}

template <> void QuokkaSimulation<ParticleRadiationSlugProblem>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const amrex::Array4<double> &state_cc = grid_elem.array_;

	const auto Erad0 = initial_Erad;

	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		const double rho = rho0;
		const double rho_e = CV * T0 * rho;

		for (int g = 0; g < Physics_Traits<ParticleRadiationSlugProblem>::nGroups; ++g) {
			state_cc(i, j, k, RadSystem<ParticleRadiationSlugProblem>::radEnergy_index + Physics_NumVars::numRadVarsPerGroup * g) = Erad0;
			state_cc(i, j, k, RadSystem<ParticleRadiationSlugProblem>::x1RadFlux_index + Physics_NumVars::numRadVarsPerGroup * g) = 0;
			state_cc(i, j, k, RadSystem<ParticleRadiationSlugProblem>::x2RadFlux_index + Physics_NumVars::numRadVarsPerGroup * g) = 0;
			state_cc(i, j, k, RadSystem<ParticleRadiationSlugProblem>::x3RadFlux_index + Physics_NumVars::numRadVarsPerGroup * g) = 0;
		}
		state_cc(i, j, k, RadSystem<ParticleRadiationSlugProblem>::gasEnergy_index) = rho_e;
		state_cc(i, j, k, RadSystem<ParticleRadiationSlugProblem>::gasDensity_index) = rho;
		state_cc(i, j, k, RadSystem<ParticleRadiationSlugProblem>::gasInternalEnergy_index) = rho_e;
		state_cc(i, j, k, RadSystem<ParticleRadiationSlugProblem>::x1GasMomentum_index) = 0.0;
		state_cc(i, j, k, RadSystem<ParticleRadiationSlugProblem>::x2GasMomentum_index) = 0.0;
		state_cc(i, j, k, RadSystem<ParticleRadiationSlugProblem>::x3GasMomentum_index) = 0.0;
	});
}

auto problem_main() -> int
{
	QuokkaSimulation<ParticleRadiationSlugProblem> sim;
	sim.maxDt_ = dt_;

	const amrex::ParmParse pp("problem");
	pp.query("particles_filename", sim.userData_.particles_filename);

	// initialize (this will parse particle parameters and load the luminosity table)
	sim.setInitialConditions();

	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx0 = sim.geom[0].CellSizeArray();
	amrex::Real const vol = AMREX_D_TERM(dx0[0], *dx0[1], *dx0[2]);

	amrex::Real total_Erad_init = 0.0;
	for (int g = 0; g < Physics_Traits<ParticleRadiationSlugProblem>::nGroups; ++g) {
		total_Erad_init +=
		    sim.state_new_cc_[0].sum(RadSystem<ParticleRadiationSlugProblem>::radEnergy_index + Physics_NumVars::numRadVarsPerGroup * g) * vol;
	}
	const amrex::Real total_gas_energy_init = sim.state_new_cc_[0].sum(RadSystem<ParticleRadiationSlugProblem>::gasEnergy_index) * vol;

	sim.evolve();

	amrex::Real total_Erad = 0.0;
	for (int g = 0; g < Physics_Traits<ParticleRadiationSlugProblem>::nGroups; ++g) {
		total_Erad +=
		    sim.state_new_cc_[0].sum(RadSystem<ParticleRadiationSlugProblem>::radEnergy_index + Physics_NumVars::numRadVarsPerGroup * g) * vol;
	}
	const amrex::Real total_gas_energy = sim.state_new_cc_[0].sum(RadSystem<ParticleRadiationSlugProblem>::gasEnergy_index) * vol;

	int status = 0;

	if (amrex::ParallelDescriptor::IOProcessor()) {
		amrex::Print() << "Total gas energy (initial): " << total_gas_energy_init << "\n";
		amrex::Print() << "Total gas energy (final): " << total_gas_energy << "\n";
		amrex::Print() << "Total radiation energy (initial): " << total_Erad_init / chat_over_c << "\n";
		amrex::Print() << "Total radiation energy (final): " << total_Erad / chat_over_c << "\n";

		const double total_energy_init = total_Erad_init / chat_over_c + total_gas_energy_init;
		const double total_energy = total_Erad / chat_over_c + total_gas_energy;
		const double change_of_total_energy = total_energy - total_energy_init;
		amrex::Print() << "Change of total energy: " << change_of_total_energy << "\n";

		// A particle's luminosity is set at the beginning of a step and used by the radiation
		// deposition of the *following* step, so with two steps only the second one radiates,
		// and it does so with the luminosity of a star of age (0 - birth_time) = 1 Myr. The
		// particle file starts the star with zero luminosity, hence no emission in step 0.
		AMREX_ALWAYS_ASSERT_WITH_MESSAGE(sim.maxTimesteps_ == 2, "This test requires max_timesteps = 2");
		const double change_of_total_energy_expected = (L_FUV + L_LyC) * dt_;

		amrex::Print() << "Current time: " << sim.tNew_[0] << "\n";
		amrex::Print() << "Stellar age: " << stellar_age / quokka::seconds_per_year << " yr\n";
		amrex::Print() << "Stellar mass: " << stellar_mass / C::M_solar << " Msun\n";
		amrex::Print() << "Expected change of total energy: " << change_of_total_energy_expected << "\n";

		const double error_rel_to_tot = std::abs(change_of_total_energy - change_of_total_energy_expected) / total_energy;
		const double error_rel_to_rad = std::abs(change_of_total_energy - change_of_total_energy_expected) / change_of_total_energy_expected;
		amrex::Print() << "Relative error to total energy: " << error_rel_to_tot << "\n";
		amrex::Print() << "Relative error to radiation energy: " << error_rel_to_rad << "\n";

		const double tolerance = 1.0e-13;
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
