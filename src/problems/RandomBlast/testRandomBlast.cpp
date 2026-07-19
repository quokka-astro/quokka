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
#include <format>

#include "QuokkaSimulation.hpp"
#include "fundamental_constants.H"
#include "hydro/hydro_system.hpp"
#include "physics_info.hpp"
#include "util/fextract.hpp"

#ifdef HAVE_PYTHON
#include "util/matplotlibcpp.h"
#endif

struct RandomBlast {
}; // dummy type to allow compile-type polymorphism via template specialization

constexpr double m_H = C::m_p + C::m_e;	     // mass of hydrogen atom
constexpr double seconds_per_year = 3.154e7; // seconds per year

template <> struct Physics_Traits<RandomBlast> : DefaultPhysicsTraits {
	static constexpr bool is_self_gravity_enabled = true;
	static constexpr bool is_hydro_enabled = true;
	static constexpr int numPassiveScalars = numMassScalars + 1;
};

template <> struct quokka::EOS_Traits<RandomBlast> {
	static constexpr double gamma = 5. / 3.;
	static constexpr double mean_molecular_weight = C::m_u;
	// EOSTabulated: ComputeEintFromTgas root-finds through the resampled table,
	// so ICs differ from the ideal-gas inverse by the table-vs-fixed-mu ratio.
	// This is intentional — the table EOS is needed for consistent cooling.
	// Temperatures below the table minimum are silently clamped to eint_min.
	using EOSBackend = quokka::EOSTabulated<RandomBlast>;
};

template <> struct Particle_Traits<RandomBlast> : DefaultParticleTraits {
	static constexpr ParticleSwitch particle_switch = ParticleSwitch::StochasticStellarPop;
};

constexpr Real cloudy_H_mass_fraction = 1.0 / (1.0 + 0.1 * 3.971);

template <> struct SimulationData<RandomBlast> {
	std::vector<int> SN_counter_arr; // Track cumulative number of SNe at all time

	Real n_amb = 0.1;   // ambient density (cm^-3)
	Real T_amb = 1.0e4; // ambient temperature (K)
	std::string part_fn = "../inputs/particles_stochastic_n100.txt";

	std::vector<Real> boost_velocity{0.0, 0.0, 0.0}; // NOLINT
};

template <> void QuokkaSimulation<RandomBlast>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	// set initial conditions
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const amrex::Array4<double> &state_cc = grid_elem.array_;

	const Real rho0 = userData_.n_amb * (m_H / cloudy_H_mass_fraction); // g cm^-3
	const Real Tgas0 = userData_.T_amb;

	const Real vx = userData_.boost_velocity[0];
	const Real vy = userData_.boost_velocity[1];
	const Real vz = userData_.boost_velocity[2];

	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		Real const rho = rho0;
		Real const xmom = rho * vx;
		Real const ymom = rho * vy;
		Real const zmom = rho * vz;
		Real const Eint = quokka::EOS<RandomBlast>::ComputeEintFromTgas(rho, Tgas0);
		Real const Egas = Eint + 0.5 * (xmom * xmom + ymom * ymom + zmom * zmom) / rho;
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
	// mass, vx, vy, vz, birth/death time, birth/death pos, death_density, mass_at_birth, lum[nGroups]
	const int nreal_extra = quokka::StochasticStellarPopParticleRealComps<RandomBlast>;
	StochasticStellarPopParticles->SetVerbose(0);
	StochasticStellarPopParticles->InitFromAsciiFile(userData_.part_fn, nreal_extra, nullptr);

	const Real vx = userData_.boost_velocity[0];
	const Real vy = userData_.boost_velocity[1];
	const Real vz = userData_.boost_velocity[2];

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
				idata[i].m_rdata[quokka::StochasticStellarPopParticleVxIdx] += vx;
				idata[i].m_rdata[quokka::StochasticStellarPopParticleVyIdx] += vy;
				idata[i].m_rdata[quokka::StochasticStellarPopParticleVzIdx] += vz;
			});
		}
	}
}

template <> void QuokkaSimulation<RandomBlast>::computeAfterTimestep()
{
	// Count how many SN went off in this timestep
	userData_.SN_counter_arr.push_back(sn_count_cumulative_); // cumulative number of SNe at current time
}

template <>
void QuokkaSimulation<RandomBlast>::ComputeDerivedVar(int /*lev*/, std::string const &dname, amrex::MultiFab &mf, const int ncomp_cc_in,
						      amrex::MultiFab const &state_cc, amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> const & /*state_fc*/) const
{
	// compute derived variables and save in 'mf'
	if (dname == "temperature") {
		AMREX_ALWAYS_ASSERT_WITH_MESSAGE(coolingTableType_ == "resampled", "RandomBlast diagnostics require resampled cooling tables.");
		const int ncomp = ncomp_cc_in;

		for (amrex::MFIter iter(mf); iter.isValid(); ++iter) {
			const amrex::Box &indexRange = iter.validbox();
			auto const &output = mf.array(iter);
			auto const &state = state_cc.const_array(iter);

			amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
				Real const rho = state(i, j, k, HydroSystem<RandomBlast>::density_index);
				Real const x1Mom = state(i, j, k, HydroSystem<RandomBlast>::x1Momentum_index);
				Real const x2Mom = state(i, j, k, HydroSystem<RandomBlast>::x2Momentum_index);
				Real const x3Mom = state(i, j, k, HydroSystem<RandomBlast>::x3Momentum_index);
				Real const Egas = state(i, j, k, HydroSystem<RandomBlast>::energy_index);
				static_assert(!Physics_Traits<RandomBlast>::is_mhd_enabled, "MHD is enabled; pass magnetic_energy instead of 0.0");
				Real const Eint = quokka::EOS<RandomBlast>::ComputeEintFromEgas(rho, x1Mom, x2Mom, x3Mom, Egas, 0.0);
				Real const Tgas = quokka::EOS<RandomBlast>::ComputeTgasFromEint(rho, Eint);

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
	pp.query("n_amb", sim.userData_.n_amb);
	pp.query("T_amb", sim.userData_.T_amb);
	pp.query("part_fn", sim.userData_.part_fn);

	if (pp.queryarr("boost_velocity", sim.userData_.boost_velocity) == 0) {
		amrex::Abort("boost_velocity must be specified in the input file.");
	} else {
		amrex::Print() << "boost_velocity: " << sim.userData_.boost_velocity[0] << ", " << sim.userData_.boost_velocity[1] << ", "
			       << sim.userData_.boost_velocity[2] << "\n";
	}

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

	// Extract and plot temperature along z-axis
	auto [position, values] = fextract(sim.state_new_cc_[0], sim.Geom(0), 2, 0.0, true);
	const int nz = static_cast<int>(position.size());

	if (amrex::ParallelDescriptor::IOProcessor()) {
		// Compute temperature from extracted state data
		std::vector<double> zs(nz);
		std::vector<double> temperature(nz);

		AMREX_ALWAYS_ASSERT_WITH_MESSAGE(sim.coolingTableType_ == "resampled", "RandomBlast temperature extraction requires resampled cooling tables.");

		for (int i = 0; i < nz; ++i) {
			zs[i] = position[i];
			Real const rho = values.at(HydroSystem<RandomBlast>::density_index)[i];
			Real const x1Mom = values.at(HydroSystem<RandomBlast>::x1Momentum_index)[i];
			Real const x2Mom = values.at(HydroSystem<RandomBlast>::x2Momentum_index)[i];
			Real const x3Mom = values.at(HydroSystem<RandomBlast>::x3Momentum_index)[i];
			Real const Egas = values.at(HydroSystem<RandomBlast>::energy_index)[i];
			static_assert(!Physics_Traits<RandomBlast>::is_mhd_enabled, "MHD is enabled; pass magnetic_energy instead of 0.0");
			Real const Eint = quokka::EOS<RandomBlast>::ComputeEintFromEgas(rho, x1Mom, x2Mom, x3Mom, Egas, 0.0);
			Real const Tgas = quokka::EOS<RandomBlast>::ComputeTgasFromEint(rho, Eint);
			temperature[i] = Tgas;
		}

#ifdef HAVE_PYTHON
		matplotlibcpp::clf();
		matplotlibcpp::plot(zs, temperature, {{"label", "temperature"}, {"color", "blue"}});
		matplotlibcpp::xlabel("z (cm)");
		matplotlibcpp::ylabel("Temperature (K)");
		matplotlibcpp::legend();
		matplotlibcpp::title(std::format("time t = {:.1g} yr", sim.tNew_[0] / seconds_per_year));
		matplotlibcpp::tight_layout();
		matplotlibcpp::save("./RandomBlast_temperature_z.png");
		amrex::Print() << "\nTemperature plot saved to RandomBlast_temperature_z.png\n";
#endif
	}

	return 0;
}
