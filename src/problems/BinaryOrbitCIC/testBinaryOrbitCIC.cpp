//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file testBinaryOrbitCIC.cpp
/// \brief Defines a test problem for a binary orbit.
///

#include "AMReX_BLassert.H"
#include "hydro/hydro_system.hpp"
#include "math/interpolate.hpp"
#include <algorithm>
#include <fstream>
#include <limits>
#include <utility>

#include "AMReX.H"
#include "AMReX_BC_TYPES.H"
#include "AMReX_DistributionMapping.H"
#include "AMReX_Geometry.H"
#include "AMReX_GpuContainers.H"
#include "AMReX_MultiFab.H"
#include "AMReX_ParallelDescriptor.H"
#include "AMReX_ParmParse.H"
#include "AMReX_Print.H"
#include "util/BC.hpp"

#include "AMReX_REAL.H"
#include "QuokkaSimulation.hpp"

struct BinaryOrbit {
};

static bool do_split_particles = false;	    // NOLINT
static int split_factor = 8;		    // NOLINT
static bool verify_particle_layout = false; // NOLINT

template <> struct quokka::EOS_Traits<BinaryOrbit> {
	static constexpr double gamma = 1.0;	       // isothermal
	static constexpr double cs_isothermal = 1.3e7; // cm s^{-1}
	static constexpr double mean_molecular_weight = C::m_u;
};

template <> struct Particle_Traits<BinaryOrbit> : DefaultParticleTraits {
	static constexpr ParticleSwitch particle_switch = ParticleSwitch::CIC;
};

template <> struct HydroSystem_Traits<BinaryOrbit> {
	static constexpr bool reconstruct_eint = false;
};

template <> struct Physics_Traits<BinaryOrbit> : DefaultPhysicsTraits {
	static constexpr bool is_hydro_enabled = true;
	static constexpr bool is_self_gravity_enabled = true;
};

template <> struct SimulationData<BinaryOrbit> {
	std::vector<amrex::ParticleReal> time;
	std::vector<amrex::ParticleReal> dist;
};

template <> void QuokkaSimulation<BinaryOrbit>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const amrex::Array4<double> &state_cc = grid_elem.array_;

	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		double const rho = 1.0e-22; // g cm^{-3}
		state_cc(i, j, k, HydroSystem<BinaryOrbit>::density_index) = rho;
		state_cc(i, j, k, HydroSystem<BinaryOrbit>::x1Momentum_index) = 0;
		state_cc(i, j, k, HydroSystem<BinaryOrbit>::x2Momentum_index) = 0;
		state_cc(i, j, k, HydroSystem<BinaryOrbit>::x3Momentum_index) = 0;
		state_cc(i, j, k, HydroSystem<BinaryOrbit>::energy_index) = 0;
		state_cc(i, j, k, HydroSystem<BinaryOrbit>::internalEnergy_index) = 0;
	});
}

template <> void QuokkaSimulation<BinaryOrbit>::createInitialCICParticles()
{
	// read particles from ASCII file
	const int nreal_extra = 4; // mass vx vy vz
	CICParticles->SetVerbose(1);
	CICParticles->InitFromAsciiFile("../inputs/BinaryOrbit_particles.txt", nreal_extra, nullptr);

	// test particle splitting
	// (this is intended to only be used when restarting at a higher resolution)
	if (do_split_particles) {
		amrex::Print() << "Splitting CICParticles using split_factor = " << split_factor << "\n";
		int const lev = 0; // all CICParticles are on level 0
		particleRegister_.getParticleDescriptor(quokka::ParticleType::CIC)->splitParticles(lev, split_factor);
	}
}

template <>
void QuokkaSimulation<BinaryOrbit>::ComputeDerivedVar(int lev, std::string const &dname, amrex::MultiFab &mf, const int ncomp_cc_in,
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

template <> void QuokkaSimulation<BinaryOrbit>::computeAfterTimestep()
{
	if (verify_particle_layout) {
		bool layout_matches = CICParticles->GetParGDB() == GetParGDB();
		for (int lev = 0; lev <= finestLevel(); ++lev) {
			layout_matches = layout_matches && CICParticles->ParticleBoxArray(lev).CellEqual(boxArray(lev)) &&
					 CICParticles->ParticleDistributionMap(lev) == DistributionMap(lev);
		}
		AMREX_ALWAYS_ASSERT_WITH_MESSAGE(layout_matches, "Particle container does not track the AMR hierarchy after restart refinement");
	}

	// every N cycles, save particle statistics at the finest level
	static int cycle = 1;
	if (cycle % 10 == 0) {
		// get the finest level
		const int finest_level = finestLevel();

		// Get particle data using the physics particle descriptor
		const auto [real_data, int_data] = particleRegister_.getParticleDescriptor(quokka::ParticleType::CIC)->getParticleDataAtLevel(finest_level);

		if (amrex::ParallelDescriptor::IOProcessor()) {
			if (real_data.size() >= 2) {
				amrex::Print() << "Computing particle statistics...\n";

				// compute orbital elements
				double dist = 0.0;

				// Loop over all pairs of particles
				for (size_t i = 0; i < real_data.size(); ++i) {
					for (size_t j = i + 1; j < real_data.size(); ++j) {
						const auto &p1 = real_data[i];
						const auto &p2 = real_data[j];
						const double dx = p1[0] - p2[0]; // position x
						const double dy = p1[1] - p2[1]; // position y
						const double dz = p1[2] - p2[2]; // position z
						const double pair_dist = std::sqrt((dx * dx) + (dy * dy) + (dz * dz));
						dist = std::max(dist, pair_dist);
					}
				}

				const double dist0 = 6.25e12; // cm
				const amrex::Real cell_dx0 = this->geom[0].CellSize(0);

				// save statistics
				userData_.time.push_back(tNew_[finest_level]);
				userData_.dist.push_back((dist - dist0) / cell_dx0);
				amrex::Print() << "Maximum particle separation: " << dist << " cm, initial separation is " << dist0 << " cm.\n";
			}
		}
	}
	++cycle;
}

auto problem_main() -> int
{
	// read in runtime parameters for this test problem
	amrex::ParmParse const pp("problem");
	pp.query("do_split_particles", do_split_particles);
	pp.query("split_factor", split_factor);
	pp.query("verify_particle_layout", verify_particle_layout);

	// Problem initialization
	QuokkaSimulation<BinaryOrbit> sim;

	// initialize
	sim.setInitialConditions();

	sim.particleRegister_.getParticleDescriptor(quokka::ParticleType::CIC)->setForceFinestLevel(true);

	// evolve
	sim.evolve();

	std::array<int, AMREX_SPACEDIM> n_cell{};
	amrex::ParmParse const amr_pp("amr");
	amr_pp.query("n_cell", n_cell);
	const bool is_refactor = n_cell[0] == 64;
	const bool is_refactor_splitparticle = is_refactor && sim.splitParticlesOnRestartRefine_;

	// get the number of particles
	const int n_particles = sim.particleRegister_.getParticleDescriptor(quokka::ParticleType::CIC)->getNumParticles();

	const auto &real_data = sim.particleRegister_.getParticleDescriptor(quokka::ParticleType::CIC)->getParticleDataAtLevel(0).first;

	int status = 0;
	// check max abs particle distance
	double max_deviation = 0.0;
	if (amrex::ParallelDescriptor::IOProcessor()) {
		amrex::Print() << "Number of particles: " << n_particles << "\n";

		// compute total particle mass and error
		double max_part_mass = 0.0;
		for (const auto &p : real_data) {
			max_part_mass = std::max(max_part_mass, p[3]);
		}
		amrex::Print() << "Maximum particle mass: " << max_part_mass << "\n";

		if (!sim.userData_.dist.empty()) {
			auto result = std::max_element(sim.userData_.dist.begin(), sim.userData_.dist.end(),
						       [](amrex::ParticleReal a, amrex::ParticleReal b) { return std::abs(a) < std::abs(b); });
			max_deviation = std::abs(*result);
			amrex::Print() << "max deviation from initial particle separation = " << max_deviation << " cell widths.\n";
		} else {
			max_deviation = 1.0;
			amrex::Print() << "No particles in userData_.dist.\n";
		}

		if (do_split_particles) {
			AMREX_ALWAYS_ASSERT_WITH_MESSAGE(sim.maxTimesteps_ <= 20, "maxTimesteps_ must be <= 20 when do_split_particles is true");
			// compute velocity dispersion in particle splitting
			// compute dx
			const Real cell_dx0 = sim.geom[0].CellSize(0);
			// compute vdisp
			const Real orig_particle_mass = split_factor * max_part_mass;
			const Real vdisp = std::sqrt(C::Gconst * orig_particle_mass / cell_dx0);
			// expected maximum drift
			const Real l_drift_over_dx = vdisp * sim.tNew_[0] / cell_dx0;
			const double split_max_deviation = 2.0 * l_drift_over_dx * 1.01; // max deviation from the initial separation

			if (max_deviation < split_max_deviation) {
				amrex::Print() << "    Expected maximum deviation = " << split_max_deviation << " cell widths.\n";
			} else {
				status += 1;
				amrex::Print() << "    Expected maximum deviation = " << split_max_deviation << " cell widths.\n";
			}
			if (n_particles != 2 * split_factor) {
				status += 1;
				amrex::Print() << "Test failed (split particles, number of particles is not 2 * split_factor)\n";
			}
		} else if (!is_refactor) {
			const double max_err_tol = 0.18; // max error tol in cell widths
			if (max_deviation < max_err_tol) {
				amrex::Print() << "Test passed\n";
			} else {
				amrex::Print() << "Test failed\n";
			}
		} else {
			double max_err_tol = 0.05;
			int n_particles_expected = 2;
			if (is_refactor_splitparticle) {
				max_err_tol = 2.0;
				n_particles_expected = 2 * 8; // 8 = 2^3 is the split factor and is hard-coded
			}
			if (max_deviation < max_err_tol) {
				amrex::Print() << "Test passed\n";
			} else {
				amrex::Print() << "Test failed\n";
			}
			if (n_particles != n_particles_expected) {
				status += 1;
				amrex::Print() << "Test failed (refactor, number of particles is not " << n_particles_expected << ")\n";
			}
		}

		if (status > 0) {
			amrex::Print() << "Test failed\n";
		} else {
			amrex::Print() << "Test passed\n";
		}
	}

	return status;
}
