//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file testRandomBlast.cpp
/// \brief Implements the random blast problem with particles, self-gravity, and Grackle cooling.
///
#include "AMReX_BLassert.H"
#include "AMReX_Geometry.H"
#include "AMReX_MultiFab.H"
#include <fmt/format.h>

#include "QuokkaSimulation.hpp"
#include "fundamental_constants.H"
#include "hydro/hydro_system.hpp"
#include "particles/particle_IO.hpp"
#include "physics_info.hpp"

struct RandomBlast {
}; // dummy type to allow compile-type polymorphism via template specialization

constexpr double m_H = C::m_p + C::m_e; // mass of hydrogen atom

template <> struct Physics_Traits<RandomBlast> {
	static constexpr bool is_self_gravity_enabled = true;
	static constexpr bool is_hydro_enabled = true;
	static constexpr bool is_radiation_enabled = false;
	static constexpr bool is_dust_enabled = false;
	static constexpr int nDustGroups = 1; // number of dust groups
	static constexpr bool is_mhd_enabled = false;
	static constexpr int numMassScalars = 0;
	static constexpr int numPassiveScalars = numMassScalars + 1;
	static constexpr int nGroups = 1; // number of radiation groups
	static constexpr UnitSystem unit_system = UnitSystem::CGS;
};

template <> struct quokka::EOS_Traits<RandomBlast> {
	static constexpr double gamma = 5. / 3.;
	static constexpr double mean_molecular_weight = C::m_u;
};

template <> struct Particle_Traits<RandomBlast> {
	static constexpr ParticleSwitch particle_switch = ParticleSwitch::StochasticStellarPop;
};

constexpr Real Tgas0 = 1.0e4; // K
constexpr Real nH0 = 0.1;     // cm^-3
constexpr Real cloudy_H_mass_fraction = 1.0 / (1.0 + 0.1 * 3.971);
constexpr Real rho0 = nH0 * (m_H / cloudy_H_mass_fraction); // g cm^-3

template <> struct SimulationData<RandomBlast> {
	int SN_counter_cumulative = 0;	 // Track cumulative number of SNe at current time
	std::vector<int> SN_counter_arr; // Track cumulative number of SNe at all time

	Real refine_threshold = 1.0; // gradient refinement threshold
	std::string part_fn = "../inputs/particles_stochastic_n100.txt";
};

template <> void QuokkaSimulation<RandomBlast>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	// set initial conditions
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const amrex::Array4<double> &state_cc = grid_elem.array_;

	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		Real const rho = rho0;
		Real const xmom = 0;
		Real const ymom = 0;
		Real const zmom = 0;
		Real const Eint = quokka::EOS<RandomBlast>::ComputeEintFromTgas(rho, Tgas0);
		Real const Egas = Eint;
		Real const scalar_density = 0;

		state_cc(i, j, k, HydroSystem<RandomBlast>::density_index) = rho;
		state_cc(i, j, k, HydroSystem<RandomBlast>::x1Momentum_index) = xmom;
		state_cc(i, j, k, HydroSystem<RandomBlast>::x2Momentum_index) = ymom;
		state_cc(i, j, k, HydroSystem<RandomBlast>::x3Momentum_index) = zmom;
		state_cc(i, j, k, HydroSystem<RandomBlast>::energy_index) = Egas;
		state_cc(i, j, k, HydroSystem<RandomBlast>::internalEnergy_index) = Eint;
		state_cc(i, j, k, HydroSystem<RandomBlast>::scalar0_index) = scalar_density;
	});
}

template <> void QuokkaSimulation<RandomBlast>::createInitialStochasticStellarPopParticles()
{
	// Read particles from ASCII file. Note that this only reads real components and not integer components, therefore we need to use
	// InitSetPhyParticles to set the integer components
	const int nreal_extra = 7 + Physics_Traits<RandomBlast>::nGroups; // mass vx vy vz birth_time death_time mass_at_birth lum[nGroups]
	StochasticStellarPopParticles->SetVerbose(1);
	quokka::particle_io::initParticlesFromAscii(StochasticStellarPopParticles.get(), userData_.part_fn, nreal_extra);

	// Set integer components (evolution stage) - initialize all as SNProgenitor
	for (auto &kv : StochasticStellarPopParticles->GetParticles()) {
		for (auto &ikv : kv) {
			auto &particle_tile = ikv.second;
			const int np = particle_tile.numParticles();

			if (np == 0) {
				continue;
			}

			auto ptd = particle_tile.getParticleTileData();
			auto *runtime_idata = ptd.m_runtime_idata;

			// Launch GPU kernel to set integer components
			amrex::ParallelFor(np, [=] AMREX_GPU_DEVICE(int i) {
				runtime_idata[quokka::StochasticStellarPopParticleStageIdx][i] = static_cast<int>(quokka::StellarEvolutionStage::SNProgenitor);
			});
		}
	}
}

template <> void QuokkaSimulation<RandomBlast>::computeAfterTimestep()
{
	// Count how many SN went off in this timestep
	userData_.SN_counter_cumulative += sn_count_;
	userData_.SN_counter_arr.push_back(userData_.SN_counter_cumulative);
}

template <> void QuokkaSimulation<RandomBlast>::ComputeDerivedVar(int lev, std::string const &dname, amrex::MultiFab &mf, const int ncomp_cc_in) const
{
	// compute derived variables and save in 'mf'
	if (dname == "temperature") {
		AMREX_ALWAYS_ASSERT_WITH_MESSAGE(coolingTableType_ == "resampled", "RandomBlast diagnostics require resampled cooling tables.");
		const int ncomp = ncomp_cc_in;
		auto tables = resampledTables_.const_tables();

		for (amrex::MFIter iter(mf); iter.isValid(); ++iter) {
			const amrex::Box &indexRange = iter.validbox();
			auto const &output = mf.array(iter);
			auto const &state = state_new_cc_[lev].const_array(iter);

			amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
				Real const rho = state(i, j, k, HydroSystem<RandomBlast>::density_index);
				Real const x1Mom = state(i, j, k, HydroSystem<RandomBlast>::x1Momentum_index);
				Real const x2Mom = state(i, j, k, HydroSystem<RandomBlast>::x2Momentum_index);
				Real const x3Mom = state(i, j, k, HydroSystem<RandomBlast>::x3Momentum_index);
				Real const Egas = state(i, j, k, HydroSystem<RandomBlast>::energy_index);
				Real const Eint = RadSystem<RandomBlast>::ComputeEintFromEgas(rho, x1Mom, x2Mom, x3Mom, Egas);
				Real const Tgas = quokka::ResampledCooling::ComputeTgasFromEgas(rho, Eint, tables);

				output(i, j, k, ncomp) = Tgas;
			});
		}
	}
}

auto problem_main() -> int
{
	// This problem is only implemented in CGS units because the cooling tables are provided in CGS units.
	static_assert(Physics_Traits<RandomBlast>::unit_system == UnitSystem::CGS);

	QuokkaSimulation<RandomBlast> sim;

	// read parameters
	amrex::ParmParse const pp("problem");
	pp.query("refine_threshold", sim.userData_.refine_threshold); // dimensionless
	pp.query("part_fn", sim.userData_.part_fn);

	// Set initial conditions
	sim.setInitialConditions();

	sim.particleRegister_.getParticleDescriptor(quokka::ParticleType::StochasticStellarPop)->setForceFinestLevel(true);

	// run simulation
	sim.evolve();

	if (amrex::ParallelDescriptor::IOProcessor()) {
		amrex::Print() << "\nCumulative N_sn = [";
		for (auto const &i : sim.userData_.SN_counter_arr) {
			amrex::Print() << i << ", ";
		}
		amrex::Print() << "]\n";
	}

	return 0;
}
