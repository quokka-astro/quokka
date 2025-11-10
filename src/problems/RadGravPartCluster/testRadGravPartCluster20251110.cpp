/// \file RadGravPartCluster.cpp
/// \brief Defines a test problem for radiation from particles.
///

#include "AMReX_ParallelDescriptor.H"
#include "AMReX_ParmParse.H"
#include "AMReX_Print.H"

#include "QuokkaSimulation.hpp"
#include "fundamental_constants.H"
#include "hydro/hydro_system.hpp"
#include "radiation/radiation_dust_system.hpp" // NOLINT
#include "util/BC.hpp"
#include "util/DataTable.hpp"

struct ParticleRadiationProblem {
};

// constexpr int ngroups_ = 4;
// constexpr amrex::GpuArray<double, ngroups_ + 1> radBoundaries_{1.e-04, 1.00778140e-01, 1.00778140e+00, 5.53817071e+00, 1.e+2};
constexpr int ngroups_ = 1;
constexpr amrex::GpuArray<double, ngroups_ + 1> radBoundaries_{1.0e-4, 1.0e+2};

constexpr double mu = 1.0 * C::m_p;
constexpr double gamma_ = 5. / 3.;
constexpr double rho0 = 1.0 * C::m_p; // g cm^-3
constexpr double T0 = 1.0e3;	      // K
constexpr double CV = 1. / (gamma_ - 1.) / mu * C::k_B;
// constexpr double initial_Erad = 1.0e-30 * CV * rho0 * T0;
// constexpr double dt_ = 0.1 * quokka::seconds_per_year;
// constexpr double chat_over_c = 1.0e-5;
constexpr double chat_over_c = 2000.0 * 1e5 / C::c_light; // 2000 km/s
// constexpr double formation_time = 1.5 * dt_;
constexpr Real arad = C::a_rad;
constexpr Real TCMB = 2.7; // K, CMB temperature
// constexpr Real floor_Erad = 1e-40 * arad * TCMB * TCMB * TCMB * TCMB;
constexpr Real floor_Erad = 1e-20 * arad * TCMB * TCMB * TCMB * TCMB;

template <> struct SimulationData<ParticleRadiationProblem> {
	std::string particles_filename = "../inputs/TestParticlesNoRad.txt";
};

template <> struct quokka::EOS_Traits<ParticleRadiationProblem> {
	static constexpr double gamma = gamma_;
	static constexpr double mean_molecular_weight = mu;
};

template <> struct Particle_Traits<ParticleRadiationProblem> {
	static constexpr ParticleSwitch particle_switch = ParticleSwitch::None;
	// static constexpr ParticleSwitch particle_switch = ParticleSwitch::StochasticStellarPop;
};

template <> struct HydroSystem_Traits<ParticleRadiationProblem> {
	static constexpr bool reconstruct_eint = true; // need to reconstruct temperature
};

template <> struct Physics_Traits<ParticleRadiationProblem> {
	static constexpr bool is_self_gravity_enabled = true;
	static constexpr bool is_hydro_enabled = true;
	static constexpr bool is_radiation_enabled = true;
	static constexpr bool is_mhd_enabled = false;
	static constexpr int numMassScalars = 0;		     // number of mass scalars
	static constexpr int numPassiveScalars = numMassScalars + 0; // number of passive scalars
	static constexpr int nGroups = ngroups_;
	static constexpr UnitSystem unit_system = UnitSystem::CGS;
};

template <> struct ISM_Traits<ParticleRadiationProblem> {
	static constexpr bool enable_dust_gas_thermal_coupling_model = true;
	static constexpr double gas_dust_coupling_threshold = 1.0e-4;
	static constexpr bool enable_photoelectric_heating = true;
};

template <> struct RadSystem_Traits<ParticleRadiationProblem> {
	static constexpr double c_hat_over_c = chat_over_c;
	static constexpr double Erad_floor = floor_Erad;
	static constexpr int beta_order = 0;
	static constexpr double energy_unit = C::ev2erg; // set boundary unit to eV
	// Define radiation group boundaries for 2-group radiation
	// Group 0: 1 eV to 100 eV, Group 1: 100 eV to 10000 eV
	static constexpr amrex::GpuArray<double, Physics_Traits<ParticleRadiationProblem>::nGroups + 1> radBoundaries = radBoundaries_;
	static constexpr OpacityModel opacity_model = OpacityModel::piecewise_constant_opacity;
};

template <>
AMREX_GPU_HOST_DEVICE auto RadSystem<ParticleRadiationProblem>::DefinePhotoelectricHeatingE1Derivative(amrex::Real const /*temperature*/,
												       amrex::Real const num_density) -> amrex::Real
{
	// Values in cgs units from Bate & Keto (2015), Eq. 26.
	const double epsilon = 0.05;	   // default efficiency factor for cold molecular clouds
	const double ref_J_ISR = 5.29e-14; // reference value for the ISR in erg cm^3
	const double coeff = 1.33e-24;
	return coeff * epsilon * num_density / ref_J_ISR; // s^-1

	// constant rate for testing
	// return PE_rate;
}

template <>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto
RadSystem<ParticleRadiationProblem>::DefineOpacityExponentsAndLowerValues(amrex::GpuArray<double, nGroups_ + 1> /*rad_boundaries*/, const double /*rho*/,
									  const double /*Tgas*/) -> amrex::GpuArray<amrex::GpuArray<double, nGroups_ + 1>, 2>
{
	constexpr double gas_to_dust_ratio = 1.0e-3;
	amrex::GpuArray<amrex::GpuArray<double, nGroups_ + 1>, 2> exponents_and_values{};
	for (int i = 0; i < nGroups_ + 1; ++i) {
		exponents_and_values[0][i] = 0.0; // power-law slopes
	}
	const amrex::GpuArray<double, nGroups_ + 1> dust_opacity{6e2, 1e3, 2e4, 1e5, 2e5}; // dust opacity, cm2/g. last element not used
	for (int i = 0; i < nGroups_ + 1; ++i) {
		exponents_and_values[1][i] = dust_opacity[i] * gas_to_dust_ratio;
	}
	return exponents_and_values;
}

template <>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto RadSystem<ParticleRadiationProblem>::ComputePlanckOpacity(const double rho, const double /*Tgas*/) -> amrex::Real
{
	return 1e4;
}

template <> AMREX_GPU_HOST_DEVICE auto RadSystem<ParticleRadiationProblem>::ComputeFluxMeanOpacity(const double rho, const double /*Tgas*/) -> amrex::Real
{
	return 1e4;
}

template <> void QuokkaSimulation<ParticleRadiationProblem>::createInitialStochasticStellarPopParticles()
{
	// Read particles from ASCII file. Note that this only read real components and not integer components, therefore we need to use
	// InitSetPhyParticles to set the integer components
	const int nreal_extra = 6 + Physics_Traits<ParticleRadiationProblem>::nGroups; // mass vx vy vz birth_time death_time lum
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
				const double death_time = p.rdata(quokka::StochasticStellarPopParticleDeathTimeIdx);
				if (death_time <= 0.0) {
					p.idata(0) = static_cast<int>(quokka::StellarEvolutionStage::SNRemnant);
				} else {
					p.idata(0) = static_cast<int>(quokka::StellarEvolutionStage::SNProgenitor);
				}
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
	const auto rad_boundary = RadSystem<ParticleRadiationProblem>::radBoundaries_;

	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		const double rho = rho0;
		const double rho_e = CV * T0 * rho;

		// compute energy fractions
		const auto Erad_g = RadSystem<ParticleRadiationProblem>::ComputeThermalRadiationMultiGroup(TCMB, rad_boundary);

		// Set radiation variables
		for (int g = 0; g < Physics_Traits<ParticleRadiationProblem>::nGroups; ++g) {
			state_cc(i, j, k, RadSystem<ParticleRadiationProblem>::radEnergy_index + Physics_NumVars::numRadVarsPerGroup * g) =
			    std::max(floor_Erad, Erad_g[g]);
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
	auto BCs_cc = quokka::BC<ParticleRadiationProblem>(quokka::BCType::int_dir); // periodic

	// Problem initialization
	QuokkaSimulation<ParticleRadiationProblem> sim(BCs_cc);

	// Read parameters from input file
	const amrex::ParmParse pp("problem");
	pp.query("particles_filename", sim.userData_.particles_filename);

	quokka::SpacingType rad_table_output_spacing = quokka::SpacingType::linear;
	const amrex::ParmParse ppp("particles");
	ppp.query("rad_table_output_spacing", rad_table_output_spacing);

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

		const double lum_mean = change_of_total_energy / sim.tNew_[0]; // mean luminosity, erg/s
		amrex::Print() << "Mean luminosity: " << lum_mean << " erg/s\n";
	}

	const int status = 0; // Initialize to success
	return status;
}
