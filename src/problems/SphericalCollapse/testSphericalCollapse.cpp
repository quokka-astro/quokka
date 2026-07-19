//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file testSphericalCollapse.cpp
/// \brief Defines a test problem for pressureless spherical collapse.
///
#include "hydro/hydro_system.hpp"

#include "AMReX_BLassert.H"
#include "AMReX_MultiFab.H"
#include "AMReX_ParmParse.H"
#include "QuokkaSimulation.hpp"
#include "util/BC.hpp"

struct GlobalConfig {
	static int num_particles;
	static int seed;
};

// Initialize static members with default values
int GlobalConfig::num_particles = 1000;
int GlobalConfig::seed = 42;

struct CollapseProblem {
};

template <> struct quokka::EOS_Traits<CollapseProblem> {
	static constexpr double gamma = 5. / 3.;
	static constexpr double mean_molecular_weight = C::m_u;
};

template <> struct Particle_Traits<CollapseProblem> : DefaultParticleTraits {
	static constexpr ParticleSwitch particle_switch = ParticleSwitch::CIC;
};

template <> struct HydroSystem_Traits<CollapseProblem> {
	static constexpr bool reconstruct_eint = false;
};

template <> struct Physics_Traits<CollapseProblem> : DefaultPhysicsTraits {
	// cell-centred
	static constexpr bool is_hydro_enabled = true;
	static constexpr bool is_self_gravity_enabled = true;
	// face-centred
	static constexpr UnitSystem unit_system = UnitSystem::CONSTANTS;
	static constexpr double boltzmann_constant = C::k_B;
	static constexpr double gravitational_constant = 1.0;
};

template <> void QuokkaSimulation<CollapseProblem>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	// set initial conditions
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const dx = grid_elem.dx_;
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = grid_elem.prob_lo_;
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_hi = grid_elem.prob_hi_;
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const amrex::Array4<double> &state_cc = grid_elem.array_;

	amrex::Real const x0 = prob_lo[0] + 0.5 * (prob_hi[0] - prob_lo[0]);
	amrex::Real const y0 = prob_lo[1] + 0.5 * (prob_hi[1] - prob_lo[1]);
	amrex::Real const z0 = prob_lo[2] + 0.5 * (prob_hi[2] - prob_lo[2]);

	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		amrex::Real const x = prob_lo[0] + (i + static_cast<amrex::Real>(0.5)) * dx[0];
		amrex::Real const y = prob_lo[1] + (j + static_cast<amrex::Real>(0.5)) * dx[1];
		amrex::Real const z = prob_lo[2] + (k + static_cast<amrex::Real>(0.5)) * dx[2];
		amrex::Real const r = std::sqrt(std::pow(x - x0, 2) + std::pow(y - y0, 2) + std::pow(z - z0, 2));

		double const rho_min = 1.0e-5;
		double const rho_max = 10.0;
		double const R_sphere = 0.5;
		double const R_smooth = 0.025;
		double const rho = std::max(rho_min, rho_max * ((std::tanh((R_sphere - r) / R_smooth) + 1.0) / 2.0));
		double const P = 1.0e-1;

		AMREX_ASSERT(!std::isnan(rho));
		AMREX_ASSERT(!std::isnan(P));

		state_cc(i, j, k, HydroSystem<CollapseProblem>::density_index) = rho;
		state_cc(i, j, k, HydroSystem<CollapseProblem>::x1Momentum_index) = 0;
		state_cc(i, j, k, HydroSystem<CollapseProblem>::x2Momentum_index) = 0;
		state_cc(i, j, k, HydroSystem<CollapseProblem>::x3Momentum_index) = 0;
		state_cc(i, j, k, HydroSystem<CollapseProblem>::energy_index) = quokka::EOS<CollapseProblem>::ComputeEintFromPres(rho, P);
		state_cc(i, j, k, HydroSystem<CollapseProblem>::internalEnergy_index) = quokka::EOS<CollapseProblem>::ComputeEintFromPres(rho, P);
	});
}

template <> void QuokkaSimulation<CollapseProblem>::createInitialCICParticles()
{
	// add particles at random positions in the box
	const bool generate_on_root_rank = true;
	const int iseed = GlobalConfig::seed;
	const int num_particles = GlobalConfig::num_particles;
	const double total_particle_mass = 0.5; // about 0.1 of the total fluid mass
	const double particle_mass = total_particle_mass / static_cast<double>(num_particles);

	const quokka::CICParticleContainer::ParticleInitData pdata = {{particle_mass, 0, 0, 0}, {}, {}, {}}; // {mass vx vy vz}, empty, empty, empty
	CICParticles->InitRandom(num_particles, iseed, pdata, generate_on_root_rank);
}

template <> void QuokkaSimulation<CollapseProblem>::refineGrid(int lev, amrex::TagBoxArray &tags, amrex::Real /*time*/, int /*ngrow*/)
{
	// tag cells for refinement
	const Real q_min = 5.0; // minimum density for refinement

	for (amrex::MFIter mfi(state_new_cc_[lev]); mfi.isValid(); ++mfi) {
		const amrex::Box &box = mfi.validbox();
		const auto state = state_new_cc_[lev].const_array(mfi);
		const auto tag = tags.array(mfi);
		const int nidx = HydroSystem<CollapseProblem>::density_index;

		amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
			Real const q = state(i, j, k, nidx);
			if (q > q_min) {
				tag(i, j, k) = amrex::TagBox::SET;
			}
		});
	}
}

template <>
void QuokkaSimulation<CollapseProblem>::ComputeDerivedVar(int lev, std::string const &dname, amrex::MultiFab &mf, const int ncomp_cc_in,
							  amrex::MultiFab const & /*state_cc*/,
							  amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> const & /*state_fc*/) const
{
	// compute derived variables and save in 'mf'
	if (dname == "gpot") {
		const int ncomp = ncomp_cc_in;
		auto const &phi_arr = phi[lev].const_arrays();
		auto output = mf.arrays();
		amrex::ParallelFor(mf, [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept { output[bx](i, j, k, ncomp) = phi_arr[bx](i, j, k); });
	}
}

auto problem_main() -> int
{
	amrex::ParmParse const pp("problem");
	pp.query("num_particles", GlobalConfig::num_particles);
	pp.query("seed", GlobalConfig::seed);

	// Problem initialization
	QuokkaSimulation<CollapseProblem> sim;

	// initialize
	sim.setInitialConditions();

	// evolve
	sim.evolve();

	int const status = 0;
	return status;
}
