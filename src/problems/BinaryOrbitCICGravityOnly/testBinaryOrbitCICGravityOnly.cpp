//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file testBinaryOrbitCICGravityOnly.cpp
/// \brief Defines a test problem for a binary orbit with only self-gravity enabled.
///
/// This is a copy of the BinaryOrbitCIC test problem with hydro, MHD and radiation
/// switched off. The gas in BinaryOrbitCIC is dynamically irrelevant (its total mass
/// is ~1e-15 of the particle mass), so removing it must not change the orbit: this
/// test therefore uses the same particles, the same grid and the same tolerance as
/// BinaryOrbitCIC.
///

#include <algorithm>
#include <cmath>
#include <cstddef>

#include "AMReX.H"
#include "AMReX_MultiFab.H"
#include "AMReX_ParallelDescriptor.H"
#include "AMReX_ParmParse.H"
#include "AMReX_Print.H"
#include "AMReX_REAL.H"

#include "QuokkaSimulation.hpp"

struct BinaryOrbitGravityOnly {
};

template <> struct Particle_Traits<BinaryOrbitGravityOnly> : DefaultParticleTraits {
	static constexpr ParticleSwitch particle_switch = ParticleSwitch::CIC;
};

template <> struct Physics_Traits<BinaryOrbitGravityOnly> : DefaultPhysicsTraits {
	// hydro, MHD and radiation are all disabled; only self-gravity acts on the particles
	static constexpr bool is_self_gravity_enabled = true;
};

template <> struct SimulationData<BinaryOrbitGravityOnly> {
	std::vector<amrex::ParticleReal> time;
	std::vector<amrex::ParticleReal> dist;
};

template <> void QuokkaSimulation<BinaryOrbitGravityOnly>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	// with no hyperbolic state, the cell-centred state holds a single unused placeholder component
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const amrex::Array4<double> &state_cc = grid_elem.array_;

	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) { state_cc(i, j, k, 0) = 0; });
}

template <> void QuokkaSimulation<BinaryOrbitGravityOnly>::createInitialCICParticles()
{
	// read particles from ASCII file
	const int nreal_extra = 4; // mass vx vy vz
	CICParticles->SetVerbose(1);
	CICParticles->InitFromAsciiFile("../inputs/BinaryOrbit_particles.txt", nreal_extra, nullptr);
}

template <>
void QuokkaSimulation<BinaryOrbitGravityOnly>::ComputeDerivedVar(int lev, std::string const &dname, amrex::MultiFab &mf, const int ncomp_cc_in,
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

template <> void QuokkaSimulation<BinaryOrbitGravityOnly>::computeAfterTimestep()
{
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

				// compute the maximum pairwise separation
				double dist = 0.0;
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
	// Problem initialization
	QuokkaSimulation<BinaryOrbitGravityOnly> sim;

	// initialize
	sim.setInitialConditions();

	sim.particleRegister_.getParticleDescriptor(quokka::ParticleType::CIC)->setForceFinestLevel(true);

	// evolve
	sim.evolve();

	// get the number of particles
	const int n_particles = sim.particleRegister_.getParticleDescriptor(quokka::ParticleType::CIC)->getNumParticles();

	int status = 0;
	if (amrex::ParallelDescriptor::IOProcessor()) {
		amrex::Print() << "Number of particles: " << n_particles << "\n";

		if (n_particles != 2) {
			status += 1;
			amrex::Print() << "Test failed (number of particles is not 2)\n";
		}

		double max_deviation = 1.0;
		if (!sim.userData_.dist.empty()) {
			auto result = std::max_element(sim.userData_.dist.begin(), sim.userData_.dist.end(),
						       [](amrex::ParticleReal a, amrex::ParticleReal b) { return std::abs(a) < std::abs(b); });
			max_deviation = std::abs(*result);
			amrex::Print() << "max deviation from initial particle separation = " << max_deviation << " cell widths.\n";
		} else {
			amrex::Print() << "No particles in userData_.dist.\n";
		}

		// same tolerance as the hydro-enabled BinaryOrbitCIC test
		const double max_err_tol = 0.18; // max error tol in cell widths
		if (max_deviation >= max_err_tol) {
			status += 1;
			amrex::Print() << "Test failed (max deviation exceeds " << max_err_tol << " cell widths)\n";
		}

		if (status > 0) {
			amrex::Print() << "Test failed\n";
		} else {
			amrex::Print() << "Test passed\n";
		}
	}

	return status;
}
