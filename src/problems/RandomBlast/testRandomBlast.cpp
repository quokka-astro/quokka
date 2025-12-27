//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file testRandomBlast.cpp
/// \brief Implements the random blast problem with radiative cooling.
///
#include "AMReX.H"
#include "AMReX_BLProfiler.H"
#include "AMReX_BLassert.H"
#include "AMReX_Geometry.H"
#include "AMReX_MultiFab.H"
#include "AMReX_ParallelDescriptor.H"
#include "AMReX_REAL.H"
#include <fmt/format.h>

#include "QuokkaSimulation.hpp"
#include "fundamental_constants.H"
#include "hydro/hydro_system.hpp"
#include "physics_info.hpp"
#include "util/BC.hpp"

using amrex::Real;

struct RandomBlast {
}; // dummy type to allow compile-type polymorphism via template specialization

constexpr double seconds_in_year = 3.1536e7;
constexpr double parsec_in_cm = C::parsec; // cm == 1 pc
constexpr double m_H = C::m_p + C::m_e;	   // mass of hydrogen atom

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
	int SN_counter_cumulative = 0; // Track total number of SNe

	Real SN_rate_per_vol = NAN; // rate per unit time per unit volume
	Real E_blast = 1.0e51;	    // ergs
	Real M_ejecta = 0;	    // 10.0 * Msun; // g

	Real refine_threshold = 1.0; // gradient refinement threshold
	int use_periodic_bc = 1;     // default is periodic
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
	StochasticStellarPopParticles->InitFromAsciiFile("../inputs/particles_stochastic_n100.txt", nreal_extra, nullptr);

	// Set integer components (evolution stage) - initialize all as SNProgenitor
	for (auto &kv : StochasticStellarPopParticles->GetParticles()) {
		for (auto &ikv : kv) {
			auto &particle_array = ikv.second.GetArrayOfStructs();
			const int np = particle_array.numParticles();

			if (np == 0) {
				continue;
			}

			auto *idata = particle_array().data();

			// Launch GPU kernel to set integer components
			amrex::ParallelFor(np, [=] AMREX_GPU_DEVICE(int i) {
				idata[i].m_idata[quokka::StochasticStellarPopParticleStageIdx] = static_cast<int>(quokka::StellarEvolutionStage::SNProgenitor);
			});
		}
	}
}

template <> void QuokkaSimulation<RandomBlast>::computeBeforeTimestep()
{
	// compute how many SNe will go off on this coarse timestep
	// sample from Poisson distribution
	const Real dt_coarse = dt_[0];
	const Real domain_vol = geom[0].ProbSize();
	const Real expectation_value = userData_.SN_rate_per_vol * domain_vol * dt_coarse;

	const int count = static_cast<int>(amrex::RandomPoisson(expectation_value));
	if (count > 0) {
		amrex::Print() << "\t" << count << " SNe to be exploded.\n";
	}

	userData_.SN_counter_cumulative += count;

	// SN feedback is handled automatically by the StochasticStellarPop particles
	// TODO(ben): need to force refinement to highest level for cells near particles
}

template <> void QuokkaSimulation<RandomBlast>::computeAfterTimestep()
{
	// With SN feedback, mass is not conserved due to mass injection
	// Instead, we track the cumulative injected mass and energy in problem_main
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

template <> void QuokkaSimulation<RandomBlast>::refineGrid(int lev, amrex::TagBoxArray &tags, Real /*time*/, int /*ngrow*/)
{
	// tag cells for refinement
	const Real q_min = 1e-5 * rho0; // minimum density for refinement
	const Real eta_threshold = userData_.refine_threshold;

	for (amrex::MFIter mfi(state_new_cc_[lev]); mfi.isValid(); ++mfi) {
		const amrex::Box &box = mfi.validbox();
		const auto state = state_new_cc_[lev].const_array(mfi);
		const auto tag = tags.array(mfi);
		const int nidx = HydroSystem<RandomBlast>::density_index;

		amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
			Real const q = state(i, j, k, nidx);
			Real const q_xplus = state(i + 1, j, k, nidx);
			Real const q_xminus = state(i - 1, j, k, nidx);
			Real const q_yplus = state(i, j + 1, k, nidx);
			Real const q_yminus = state(i, j - 1, k, nidx);
			Real const q_zplus = state(i, j, k + 1, nidx);
			Real const q_zminus = state(i, j, k - 1, nidx);

			Real const del_x = 0.5 * (q_xplus - q_xminus);
			Real const del_y = 0.5 * (q_yplus - q_yminus);
			Real const del_z = 0.5 * (q_zplus - q_zminus);
			Real const gradient_indicator = std::sqrt(del_x * del_x + del_y * del_y + del_z * del_z) / q;

			if ((gradient_indicator > eta_threshold) && (q > q_min)) {
				tag(i, j, k) = amrex::TagBox::SET;
			}
		});
	}
}

auto problem_main() -> int
{
	// This problem is only implemented in CGS units because the cooling tables are provided in CGS units.
	static_assert(Physics_Traits<RandomBlast>::unit_system == UnitSystem::CGS);

	// read parameters
	amrex::ParmParse const pp;

	// read in SN rate
	Real SN_rate_per_vol = NAN;
	pp.query("SN_rate_per_volume", SN_rate_per_vol); // yr^-1 kpc^-3
	SN_rate_per_vol /= seconds_in_year;
	SN_rate_per_vol /= std::pow(1.0e3 * parsec_in_cm, 3);
	AMREX_ALWAYS_ASSERT(!std::isnan(SN_rate_per_vol));

	// read in refinement threshold (relative gradient in density)
	Real refine_threshold = 0.1;
	pp.query("refine_threshold", refine_threshold); // dimensionless

	// use periodic boundary conditions or not
	int use_periodic_bc = 0;
	pp.query("use_periodic_bc", use_periodic_bc);

	// Problem initialization
	auto BCs_cc = (use_periodic_bc == 1) ? quokka::BC<RandomBlast>(quokka::BCType::int_dir) : quokka::BC<RandomBlast>(quokka::BCType::reflecting);

	QuokkaSimulation<RandomBlast> sim(BCs_cc);
	sim.densityFloor_ = 1.0e-5 * rho0; // density floor (to prevent vacuum)
	sim.userData_.SN_rate_per_vol = SN_rate_per_vol;
	sim.userData_.refine_threshold = refine_threshold;
	sim.userData_.use_periodic_bc = use_periodic_bc;

	// Set initial conditions
	sim.setInitialConditions();

	// set random state
	const int seed = 42;
	amrex::InitRandom(seed, 1); // all ranks should produce the same values

	// run simulation
	sim.evolve();

	// print injected energy, injected mass
	const Real E_in_cumulative = static_cast<Real>(sim.userData_.SN_counter_cumulative) * sim.userData_.E_blast;
	const Real M_in_cumulative = static_cast<Real>(sim.userData_.SN_counter_cumulative) * sim.userData_.M_ejecta;
	amrex::Print() << "Cumulative injected energy = " << E_in_cumulative << "\n";
	amrex::Print() << "Cumulative injected mass = " << M_in_cumulative << "\n";

	// Cleanup and exit
	const int status = 0;
	return status;
}
