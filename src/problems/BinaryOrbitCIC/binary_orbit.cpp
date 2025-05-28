//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file binary_orbit.cpp
/// \brief Defines a test problem for a binary orbit.
///

#include "AMReX_BLassert.H"
#include "hydro/hydro_system.hpp"
#include "math/interpolate.hpp"
#include <algorithm>
#include <fstream>

#include "AMReX.H"
#include "AMReX_BC_TYPES.H"
#include "AMReX_DistributionMapping.H"
#include "AMReX_Geometry.H"
#include "AMReX_GpuContainers.H"
#include "AMReX_MultiFab.H"
#include "AMReX_ParallelDescriptor.H"
#include "AMReX_ParmParse.H"
#include "AMReX_Print.H"

#include "AMReX_REAL.H"
#include "QuokkaSimulation.hpp"
#include "hydro/hydro_system.hpp"

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
	static constexpr bool is_radiation_enabled = false;
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
	CICParticles->InitFromAsciiFile("BinaryOrbit_particles.txt", nreal_extra, nullptr);

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
				const amrex::Real cell_dx0 = this->geom[finest_level].CellSize(0);

				// save statistics
				userData_.time.push_back(tNew_[finest_level]);
				userData_.dist.push_back((dist - dist0) / cell_dx0);
				amrex::Print() << "Maximum particle separation: " << dist << " cm, initial separation is " << dist0 << " cm.\n";
			}
		}
	}
	++cycle;
}

template <> void QuokkaSimulation<BinaryOrbit>::refineGrid(int lev, amrex::TagBoxArray &tags, amrex::Real /*time*/, int /*ngrow*/)
{
	for (amrex::MFIter mfi(state_new_cc_[lev]); mfi.isValid(); ++mfi) {
		const amrex::Box &box = mfi.validbox();
		const auto tag = tags.array(mfi);

		amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept { tag(i, j, k) = amrex::TagBox::SET; });
	}
}

auto problem_main() -> int
{
	auto isNormalComp = [=](int n, int dim) {
		if ((n == HydroSystem<BinaryOrbit>::x1Momentum_index) && (dim == 0)) {
			return true;
		}
		if ((n == HydroSystem<BinaryOrbit>::x2Momentum_index) && (dim == 1)) {
			return true;
		}
		if ((n == HydroSystem<BinaryOrbit>::x3Momentum_index) && (dim == 2)) {
			return true;
		}
		return false;
	};

	const int ncomp_cc = Physics_Indices<BinaryOrbit>::nvarTotal_cc;
	amrex::Vector<amrex::BCRec> BCs_cc(ncomp_cc);
	for (int n = 0; n < ncomp_cc; ++n) {
		for (int i = 0; i < AMREX_SPACEDIM; ++i) {
			if (isNormalComp(n, i)) {
				BCs_cc[n].setLo(i, amrex::BCType::reflect_odd);
				BCs_cc[n].setHi(i, amrex::BCType::reflect_odd);
			} else {
				BCs_cc[n].setLo(i, amrex::BCType::reflect_even);
				BCs_cc[n].setHi(i, amrex::BCType::reflect_even);
			}
		}
	}

	// read in runtime parameters for this test problem
	amrex::ParmParse const pp("problem");
	pp.query("do_split_particles", do_split_particles);
	pp.query("split_factor", split_factor);

	// Problem initialization
	QuokkaSimulation<BinaryOrbit> sim(BCs_cc);
	sim.doPoissonSolve_ = 1; // enable self-gravity

	// initialize
	sim.setInitialConditions();

	// evolve
	sim.evolve();

	// check max abs particle distance
	double max_err = 0.0;
	if (amrex::ParallelDescriptor::IOProcessor()) {
		if (!sim.userData_.dist.empty()) {
			auto result = std::max_element(sim.userData_.dist.begin(), sim.userData_.dist.end(),
						       [](amrex::ParticleReal a, amrex::ParticleReal b) { return std::abs(a) < std::abs(b); });
			max_err = std::abs(*result);
			amrex::Print() << "max deviation from initial particle separation = " << max_err << " cell widths.\n";
		} else {
			max_err = 1.0;
			amrex::Print() << "No particles in userData_.dist.\n";
		}
	}

	int status = 1;
	if (do_split_particles) {
		AMREX_ALWAYS_ASSERT_WITH_MESSAGE(sim.maxTimesteps_ <= 20, "maxTimesteps_ must be <= 20 when do_split_particles is true");
		const double split_max_err_tol = 0.6; // max error tol in cell widths
		if (max_err < split_max_err_tol) {
			status = 0;
			amrex::Print() << "Test passed (split particles)\n";
		} else {
			amrex::Print() << "Test failed (split particles)\n";
		}
	} else {
		const double max_err_tol = 0.18; // max error tol in cell widths
		if (max_err < max_err_tol) {
			status = 0;
			amrex::Print() << "Test passed\n";
		} else {
			amrex::Print() << "Test failed\n";
		}
	}
	return status;
}
