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

static bool do_split_particles = false; // NOLINT
static int split_factor = 8;		// NOLINT

template <> struct quokka::EOS_Traits<BinaryOrbit> {
	static constexpr double gamma = 1.0;	       // isothermal
	static constexpr double cs_isothermal = 1.3e7; // cm s^{-1}
	static constexpr double mean_molecular_weight = C::m_u;
};

template <> struct Particle_Traits<BinaryOrbit> {
	static constexpr ParticleSwitch particle_switch = ParticleSwitch::CIC;
};

template <> struct HydroSystem_Traits<BinaryOrbit> {
	static constexpr bool reconstruct_eint = false;
};

template <> struct Physics_Traits<BinaryOrbit> {
	static constexpr bool is_hydro_enabled = true;
	static constexpr bool is_self_gravity_enabled = true;
	static constexpr bool is_radiation_enabled = false;
	static constexpr bool is_dust_enabled = false;
	static constexpr int nDustGroups = 1; // number of dust groups
	static constexpr bool is_mhd_enabled = false;
	static constexpr int numMassScalars = 0;		     // number of mass scalars
	static constexpr int numPassiveScalars = numMassScalars + 0; // number of passive scalars
	static constexpr int nGroups = 1;			     // number of radiation groups
	static constexpr UnitSystem unit_system = UnitSystem::CGS;
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

template <> void QuokkaSimulation<BinaryOrbit>::ComputeDerivedVar(int lev, std::string const &dname, amrex::MultiFab &mf, const int ncomp_cc_in) const
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

				if (do_split_particles) {
					// For split particles, partition into two groups of equal size and use group COM separation.
					const size_t n_particles = real_data.size();
					const size_t group_size = static_cast<size_t>(split_factor);
					AMREX_ALWAYS_ASSERT_WITH_MESSAGE(n_particles == 2 * group_size, "Expected exactly two split groups.");
					AMREX_ALWAYS_ASSERT_WITH_MESSAGE(n_particles <= 63, "Split-particle group search supports up to 63 particles.");

					const int mass_comp = 3; // x,y,z,mass,vx,vy,vz
					double best_sse = std::numeric_limits<double>::infinity();
					std::array<double, 3> best_com0{0.0, 0.0, 0.0};
					std::array<double, 3> best_com1{0.0, 0.0, 0.0};

					const uint64_t nmask = (uint64_t{1} << n_particles);
					for (uint64_t mask = 0; mask < nmask; ++mask) {
						if (((mask & 1U) == 0U) || (static_cast<size_t>(__builtin_popcountll(mask)) != group_size)) {
							continue;
						}

						double msum0 = 0.0;
						double msum1 = 0.0;
						std::array<double, 3> com0{0.0, 0.0, 0.0};
						std::array<double, 3> com1{0.0, 0.0, 0.0};
						for (size_t i = 0; i < n_particles; ++i) {
							const auto &p = real_data[i];
							const double m = p[mass_comp];
							if (((mask >> i) & 1U) != 0U) {
								msum0 += m;
								com0[0] += m * p[0];
								com0[1] += m * p[1];
								com0[2] += m * p[2];
							} else {
								msum1 += m;
								com1[0] += m * p[0];
								com1[1] += m * p[1];
								com1[2] += m * p[2];
							}
						}
						com0[0] /= msum0;
						com0[1] /= msum0;
						com0[2] /= msum0;
						com1[0] /= msum1;
						com1[1] /= msum1;
						com1[2] /= msum1;

						double sse = 0.0;
						for (size_t i = 0; i < n_particles; ++i) {
							const auto &p = real_data[i];
							const bool in_group0 = ((mask >> i) & 1U) != 0U;
							const auto &com = in_group0 ? com0 : com1;
							const double dx = p[0] - com[0];
							const double dy = p[1] - com[1];
							const double dz = p[2] - com[2];
							sse += (dx * dx) + (dy * dy) + (dz * dz);
						}

						if (sse < best_sse) {
							best_sse = sse;
							best_com0 = com0;
							best_com1 = com1;
						}
					}

					const double dx = best_com0[0] - best_com1[0];
					const double dy = best_com0[1] - best_com1[1];
					const double dz = best_com0[2] - best_com1[2];
					dist = std::sqrt((dx * dx) + (dy * dy) + (dz * dz));
					amrex::Print() << "COM separation of split-particle groups: " << dist << " cm.\n";
				} else {
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

	int status = 0;

	// check max abs particle distance
	double max_err = 0.0;
	if (amrex::ParallelDescriptor::IOProcessor()) {
		amrex::Print() << "Number of particles: " << n_particles << "\n";

		if (!sim.userData_.dist.empty()) {
			auto result = std::max_element(sim.userData_.dist.begin(), sim.userData_.dist.end(),
						       [](amrex::ParticleReal a, amrex::ParticleReal b) { return std::abs(a) < std::abs(b); });
			max_err = std::abs(*result);
			amrex::Print() << "max deviation from initial particle separation = " << max_err << " cell widths.\n";
		} else {
			max_err = 1.0;
			amrex::Print() << "No particles in userData_.dist.\n";
		}

		if (do_split_particles) {
			AMREX_ALWAYS_ASSERT_WITH_MESSAGE(sim.maxTimesteps_ <= 20, "maxTimesteps_ must be <= 20 when do_split_particles is true");
			const double split_max_err_tol = 1.2; // max error tol in cell widths
			if (max_err < split_max_err_tol) {
				amrex::Print() << "Test passed (split particles)\n";
			} else {
				status = 1;
				amrex::Print() << "Test failed (split particles)\n";
			}
			if (n_particles != 2 * split_factor) {
				status = 1;
				amrex::Print() << "Test failed (split particles, number of particles is not 2 * split_factor)\n";
			}
		} else if (!is_refactor) {
			const double max_err_tol = 0.18; // max error tol in cell widths
			if (max_err < max_err_tol) {
				status = 0;
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
			if (max_err < max_err_tol) {
				status = 0;
				amrex::Print() << "Test passed\n";
			} else {
				amrex::Print() << "Test failed\n";
			}
			if (n_particles != n_particles_expected) {
				status = 1;
				amrex::Print() << "Test failed (refactor, number of particles is not " << n_particles_expected << ")\n";
			}
		}
	}

	return status;
}
