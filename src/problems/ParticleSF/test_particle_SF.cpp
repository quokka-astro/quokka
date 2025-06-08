/// \file test_particle_SF.cpp
/// \brief Defines a test problem for stochastic star formation.
///

#include "AMReX.H"
#include "AMReX_BC_TYPES.H"
#include "AMReX_MultiFab.H"
#include "AMReX_ParmParse.H"
#include "AMReX_Print.H"
#include "AMReX_SPACE.H"
#include "util/fextract.hpp"

#include "QuokkaSimulation.hpp"
#include "fundamental_constants.H"
#include "hydro/hydro_system.hpp"
#include "particles/particle_types.hpp"
#include "test_particle_SF.hpp"

#ifdef HAVE_PYTHON
#include "util/matplotlibcpp.h"
#endif

struct ParticleSFProblem {
};

constexpr double M_sol = C::M_solar;
constexpr double mu = 1.0 * C::m_p;
constexpr double gamma_ = 5. / 3.;
constexpr double year = 3.15576e+07; // in seconds
AMREX_GPU_MANAGED Real n0 = 1.0e4;   // NOLINT
AMREX_GPU_MANAGED Real Tamb = 10.0;  // NOLINT

template <> struct Particle_Traits<ParticleSFProblem> {
	static constexpr ParticleSwitch particle_switch = ParticleSwitch::None;
	// static constexpr ParticleSwitch particle_switch = ParticleSwitch::StochasticStellarPop;
};

template <> struct quokka::EOS_Traits<ParticleSFProblem> {
	static constexpr double gamma = gamma_;
	static constexpr double mean_molecular_weight = mu;
};

template <> struct HydroSystem_Traits<ParticleSFProblem> {
	static constexpr bool reconstruct_eint = true; // need to reconstruct temperature
};

template <> struct Physics_Traits<ParticleSFProblem> {
	// cell-centred
	static constexpr bool is_hydro_enabled = true;
	static constexpr int numMassScalars = 0;		     // number of mass scalars
	static constexpr int numPassiveScalars = numMassScalars + 0; // number of passive scalars
	static constexpr bool is_radiation_enabled = false;
	// face-centred
	static constexpr bool is_mhd_enabled = false;
	static constexpr int nGroups = 1; // number of radiation groups
	static constexpr UnitSystem unit_system = UnitSystem::CGS;
};

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

template <> void QuokkaSimulation<ParticleSFProblem>::ErrorEst(int lev, amrex::TagBoxArray &tags, amrex::Real /*time*/, int /*ngrow*/)
{
	// tag cells for refinement: static mesh refinement for the whole domain

	for (amrex::MFIter mfi(state_new_cc_[lev]); mfi.isValid(); ++mfi) {
		const amrex::Box &box = mfi.validbox();
		const auto tag = tags.array(mfi);

		amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept { tag(i, j, k) = amrex::TagBox::SET; });
	}
}

auto problem_main() -> int
{

	const int ncomp_cc = Physics_Indices<ParticleSFProblem>::nvarTotal_cc;
	amrex::Vector<amrex::BCRec> BCs_cc(ncomp_cc);
	for (int n = 0; n < ncomp_cc; ++n) {
		for (int i = 0; i < AMREX_SPACEDIM; ++i) {
			// periodic boundaries
			BCs_cc[n].setLo(i, amrex::BCType::int_dir);
			BCs_cc[n].setHi(i, amrex::BCType::int_dir);
		}
	}

	// Problem initialization
	QuokkaSimulation<ParticleSFProblem> sim(BCs_cc);

	sim.reconstructionOrder_ = 3; // 2=PLM, 3=PPM
	sim.cflNumber_ = 0.3;	      // *must* be less than 1/3 in 3D!
	sim.stopTime_ = 1.0e6 * year; // 1 Myr
	sim.initDt_ = 1.0e5 * year;   // 0.1 Myr

	// Real Tamb and n0 from the input file
	amrex::ParmParse const ppp("problem");
	ppp.query("Tamb", Tamb);
	ppp.query("n0", n0);
	int max_timesteps = 10;
	ppp.query("stage_2_max_timesteps", max_timesteps);

	// set random state
	const int seed = 42;
	amrex::InitRandom(seed, 1); // all ranks should produce the same values

	// initialize
	sim.maxTimesteps_ = 1;
	sim.setInitialConditions();
	sim.evolve();
	return 0;
}
