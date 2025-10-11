/// \file random_blast_rad.cpp
/// \brief Implements the random blast problem with multigroup radiation transport and radiative cooling.
///
#include "AMReX.H"
#include "AMReX_BLProfiler.H"
#include "AMReX_BLassert.H"
#include "AMReX_Geometry.H"
#include "AMReX_MultiFab.H"
#include "AMReX_REAL.H"
#include "AMReX_SPACE.H"
#include "AMReX_TableData.H"
#include <fmt/format.h>

#include "QuokkaSimulation.hpp"
#include "fundamental_constants.H"
#include "hydro/hydro_system.hpp"
#include "math/quadrature.hpp"
#include "physics_info.hpp"
#include "util/BC.hpp"
#include "radiation/radiation_dust_system.hpp"

using amrex::Real;

constexpr Real chat_over_c = 1.0e-3;
constexpr Real mu = 1.0 * C::m_p;
constexpr Real gamma_ = 5. / 3.;
constexpr Real arad = C::a_rad;
constexpr Real TCMB = 2.7;		 // K, CMB temperature
constexpr Real floor_Erad = 1e-40 * arad * TCMB * TCMB * TCMB * TCMB;
constexpr Real Tgas0 = 1.0e4; // K
constexpr Real nH0 = 0.1;     // cm^-3
constexpr Real cloudy_H_mass_fraction = 1.0 / (1.0 + 0.1 * 3.971);
constexpr Real seconds_in_year = 3.1536e7;
constexpr Real parsec_in_cm = C::parsec; // cm == 1 pc
constexpr Real m_H = C::m_p + C::m_e;	   // mass of hydrogen atom
constexpr Real rho0 = nH0 * (m_H / cloudy_H_mass_fraction); // g cm^-3

struct TheProblem {
}; // dummy type to allow compile-type polymorphism via template specialization

template <> struct Particle_Traits<TheProblem> {
	static constexpr ParticleSwitch particle_switch = ParticleSwitch::None;
	// static constexpr ParticleSwitch particle_switch = ParticleSwitch::StochasticStellarPop;
};

template <> struct HydroSystem_Traits<TheProblem> {
	static constexpr bool reconstruct_eint = true; // need to reconstruct temperature
};

template <> struct quokka::EOS_Traits<TheProblem> {
	static constexpr double gamma = gamma_;
	static constexpr double mean_molecular_weight = mu;
};

template <> struct Physics_Traits<TheProblem> {
	static constexpr bool is_self_gravity_enabled = true;
	static constexpr bool is_hydro_enabled = true;
	static constexpr bool is_radiation_enabled = true;
	static constexpr bool is_mhd_enabled = false;
	static constexpr int numMassScalars = 0;
	static constexpr int numPassiveScalars = numMassScalars + 0;
	static constexpr int nGroups = 4; // number of radiation groups
	static constexpr UnitSystem unit_system = UnitSystem::CGS;
};

template <> struct RadSystem_Traits<TheProblem> {
	static constexpr double c_hat_over_c = chat_over_c;
	static constexpr double Erad_floor = floor_Erad;
	static constexpr int beta_order = 1;
	static constexpr double energy_unit = C::ev2erg; // set boundary unit to eV
	// groups: FIR, NIR, Optical, FUV
	static constexpr amrex::GpuArray<double, Physics_Traits<TheProblem>::nGroups + 1> radBoundaries{
		1.e-04, 1.00778140e-01, 1.00778140e+00, 5.53817071e+00, 1.e+2
	};
	static constexpr OpacityModel opacity_model = OpacityModel::piecewise_constant_opacity;
};

template <> struct ISM_Traits<TheProblem> {
	static constexpr bool enable_dust_gas_thermal_coupling_model = true;
	static constexpr double gas_dust_coupling_threshold = 1.0e-6;
	static constexpr bool enable_photoelectric_heating = false;
};

template <>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto
RadSystem<TheProblem>::DefineOpacityExponentsAndLowerValues(amrex::GpuArray<double, nGroups_ + 1> /*rad_boundaries*/, const double /*rho*/,
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

template <> struct SimulationData<TheProblem> {
	std::string stars_file = "../inputs/cluster_N500_r20.0_ng4.txt";

	Real refine_threshold = 1.0; // gradient refinement threshold
};

template <> void QuokkaSimulation<TheProblem>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	// set initial conditions
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const amrex::Array4<double> &state_cc = grid_elem.array_;

	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		Real const rho = rho0;
		Real const xmom = 0;
		Real const ymom = 0;
		Real const zmom = 0;
		Real const Eint = quokka::EOS<TheProblem>::ComputeEintFromTgas(rho, Tgas0);
		Real const Egas = Eint;
		Real const scalar_density = 0;

		state_cc(i, j, k, HydroSystem<TheProblem>::density_index) = rho;
		state_cc(i, j, k, HydroSystem<TheProblem>::x1Momentum_index) = xmom;
		state_cc(i, j, k, HydroSystem<TheProblem>::x2Momentum_index) = ymom;
		state_cc(i, j, k, HydroSystem<TheProblem>::x3Momentum_index) = zmom;
		state_cc(i, j, k, HydroSystem<TheProblem>::energy_index) = Egas;
		state_cc(i, j, k, HydroSystem<TheProblem>::internalEnergy_index) = Eint;
		state_cc(i, j, k, HydroSystem<TheProblem>::scalar0_index) = scalar_density;

		// compute energy fractions
		const auto Erad_g = RadSystem<TheProblem>::ComputeThermalRadiationMultiGroup(TCMB, RadSystem<TheProblem>::radBoundaries_);

		// Set radiation variables
		for (int g = 0; g < Physics_Traits<TheProblem>::nGroups; ++g) {
			state_cc(i, j, k, RadSystem<TheProblem>::radEnergy_index + Physics_NumVars::numRadVarsPerGroup * g) = Erad_g[g];
			state_cc(i, j, k, RadSystem<TheProblem>::x1RadFlux_index + Physics_NumVars::numRadVarsPerGroup * g) = 0;
			state_cc(i, j, k, RadSystem<TheProblem>::x2RadFlux_index + Physics_NumVars::numRadVarsPerGroup * g) = 0;
			state_cc(i, j, k, RadSystem<TheProblem>::x3RadFlux_index + Physics_NumVars::numRadVarsPerGroup * g) = 0;
		}
	});
}

auto problem_main() -> int
{
	// This problem is only implemented in CGS units because the cooling tables are provided in CGS units.
	static_assert(Physics_Traits<TheProblem>::unit_system == UnitSystem::CGS);

	// read parameters
	amrex::ParmParse const pp;

	// // read in refinement threshold (relative gradient in density)
	// Real refine_threshold = 0.1;
	// pp.query("refine_threshold", refine_threshold); // dimensionless

	// Problem initialization
	auto BCs_cc = quokka::BC<TheProblem>(quokka::BCType::int_dir); // periodic

	QuokkaSimulation<TheProblem> sim(BCs_cc);
	sim.densityFloor_ = 1.0e-5 * rho0; // density floor (to prevent vacuum)
	sim.tempFloor_ = 10.0;

	// Set initial conditions
	sim.setInitialConditions();

	// set random state
	const int seed = 42;
	amrex::InitRandom(seed, 1); // all ranks should produce the same values

	// run simulation
	sim.evolve();

	// Cleanup and exit
	const int status = 0;
	return status;
}
