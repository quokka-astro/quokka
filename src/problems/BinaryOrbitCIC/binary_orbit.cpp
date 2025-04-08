//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file binary_orbit.cpp
/// \brief Defines a test problem for a binary orbit.
///

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
#include "AMReX_ccse-mpi.H"
#include "QuokkaSimulation.hpp"
#include "binary_orbit.hpp"
#include "hydro/hydro_system.hpp"
#include <algorithm>

struct BinaryOrbit {
};

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
	std::vector<amrex::ParticleReal> time{};
	std::vector<amrex::ParticleReal> dist{};
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
	// every N cycles, save particle statistics
	static int cycle = 1;
	if (cycle % 10 == 0) {
		// Get the finest level number
		const int finest_level = finestLevel();

		// create particle container for analysis using the same geometry as finest level
		amrex::ParticleContainer<quokka::CICParticleRealComps> analysisPC{};
		amrex::Box const box(amrex::IntVect{AMREX_D_DECL(0, 0, 0)}, amrex::IntVect{AMREX_D_DECL(1, 1, 1)});
		amrex::Geometry const geom(box);
		amrex::BoxArray const boxArray(box);
		amrex::Vector<int> const ranks({0}); // workaround nvcc bug
		amrex::DistributionMapping const dmap(ranks);
		analysisPC.Define(geom, dmap, boxArray);

		// Print number of particles in source container
		if (amrex::ParallelDescriptor::IOProcessor()) {
			amrex::Long total_particles = 0;
			const auto& particles = CICParticles->GetParticles(finest_level);
			for (const auto& kv : particles) {
				total_particles += kv.second.numParticles();
			}
			amrex::Print() << "Number of particles at finest level: " << total_particles << "\n";
		}

		// Create a single destination tile
		auto& dst_tile = analysisPC.DefineAndReturnParticleTile(0, 0, 0);

		// Copy particles from each source tile
		const auto& particles = CICParticles->GetParticles(finest_level);
		for (const auto& kv : particles) {
			const auto& src_tile = kv.second;
			const int np = src_tile.numParticles();
			if (np > 0) {
				// Get current size of destination tile
				const int old_size = dst_tile.numParticles();
				// Resize to accommodate new particles
				dst_tile.resize(old_size + np);
				// Get source and destination arrays
				auto& src_aos = src_tile.GetArrayOfStructs();
				auto& dst_aos = dst_tile.GetArrayOfStructs();
				// Copy particles
				amrex::Gpu::copy(amrex::Gpu::deviceToDevice,
							   src_aos.data(),
							   src_aos.data() + np,
							   dst_aos.data() + old_size);
			}
		}

		// Print number of particles in analysis container
		if (amrex::ParallelDescriptor::IOProcessor()) {
			amrex::Long total_particles_analysis = 0;
			const auto& analysis_particles = analysisPC.GetParticles(0);
			for (const auto& kv : analysis_particles) {
				total_particles_analysis += kv.second.numParticles();
			}
			amrex::Print() << "Number of particles in analysis container: " << total_particles_analysis << "\n";
		}

		if (amrex::ParallelDescriptor::IOProcessor()) {
			quokka::CICParticleIterator const pIter(analysisPC, 0);
			if (pIter.isValid()) {
				amrex::Print() << "Computing particle statistics...\n";
				const amrex::Long np = pIter.numParticles();
				amrex::Print() << "Number of particles in iterator: " << np << "\n";
				
				if (np >= 2) {  // Only proceed if we have at least 2 particles
					auto &particles = pIter.GetArrayOfStructs();

					// copy particles from device to host
					quokka::CICParticleContainer::ParticleType *pData = particles().data();
					amrex::Vector<quokka::CICParticleContainer::ParticleType> pData_h(np);
					amrex::Gpu::copy(amrex::Gpu::deviceToHost, pData, pData + np, pData_h.begin()); // NOLINT

					// compute orbital elements
					quokka::CICParticleContainer::ParticleType &p1 = pData_h[0];
					quokka::CICParticleContainer::ParticleType &p2 = pData_h[1];
					const amrex::ParticleReal dx = p1.pos(0) - p2.pos(0);
					const amrex::ParticleReal dy = p1.pos(1) - p2.pos(1);
					const amrex::ParticleReal dz = p1.pos(2) - p2.pos(2);
					const amrex::ParticleReal dist = std::sqrt(dx * dx + dy * dy + dz * dz);
					printf("dist = %e\n", dist);
					const amrex::ParticleReal dist0 = 6.25e12; // cm
					const amrex::Real cell_dx0 = this->geom[finest_level].CellSize(0); // Use finest level cell size

					// save statistics
					userData_.time.push_back(tNew_[finest_level]); // Use time from finest level
					userData_.dist.push_back((dist - dist0) / cell_dx0);
				} else {
					amrex::Print() << "Not enough particles for analysis (need at least 2)!\n";
				}
			} else {
				amrex::Print() << "Particle iterator is not valid!\n";
			}
		}
	}
	++cycle;
}

template <> void QuokkaSimulation<BinaryOrbit>::ErrorEst(int lev, amrex::TagBoxArray &tags, amrex::Real /*time*/, int /*ngrow*/)
{
	for (amrex::MFIter mfi(state_new_cc_[lev]); mfi.isValid(); ++mfi) {
		const amrex::Box &box = mfi.validbox();
		const auto prob_lo = geom[lev].ProbLoArray();
		const auto dx = geom[lev].CellSizeArray();
		const auto state = state_new_cc_[lev].const_array(mfi);
		const auto tag = tags.array(mfi);
		const int nidx = HydroSystem<BinaryOrbit>::density_index;

		amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
			tag(i, j, k) = amrex::TagBox::SET;
		});
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

	// Problem initialization
	QuokkaSimulation<BinaryOrbit> sim(BCs_cc);
	sim.doPoissonSolve_ = 1; // enable self-gravity
	sim.initDt_ = 1.0e3;	 // s

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
			amrex::Print() << "max particle separation = " << max_err << " cell widths.\n";
		} else {
			max_err = 1.0;
			amrex::Print() << "No particles in userData_.dist.\n";
		}
	}

	int status = 1;
	const double max_err_tol = 0.18; // max error tol in cell widths
	if (max_err < max_err_tol) {
		status = 0;
	}
	return status;
}
