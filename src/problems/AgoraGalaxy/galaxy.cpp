//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2024 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file galaxy.cpp
/// \brief Defines a simulation using the AGORA isolated galaxy initial conditions.
///

#include <cmath>

#include "AMReX_BC_TYPES.H"
#include "AMReX_BLassert.H"
#include "AMReX_GpuContainers.H"
#include "AMReX_REAL.H"

#include "QuokkaSimulation.hpp"
#include "SimulationData.hpp"
#include "fundamental_constants.H"
#include "galaxy.hpp"
#include "hydro/EOS.hpp"
#include "hydro/hydro_system.hpp"
#include "math/interpolate.hpp"
#include "physics_info.hpp"

struct AgoraGalaxy {
};

template <> struct quokka::EOS_Traits<AgoraGalaxy> {
	static constexpr double gamma = 5. / 3.;
	static constexpr double mean_molecular_weight = 0.6 * C::m_u;
	static constexpr double boltzmann_constant = C::k_B;
};

template <> struct HydroSystem_Traits<AgoraGalaxy> {
	static constexpr bool reconstruct_eint = true;
};

template <> struct Physics_Traits<AgoraGalaxy> {
	static constexpr UnitSystem unit_system = UnitSystem::CGS;
	static constexpr bool is_hydro_enabled = true;
	static constexpr bool is_radiation_enabled = false;
	static constexpr bool is_mhd_enabled = false;
	static constexpr int numMassScalars = 0;		     // number of mass scalars
	static constexpr int numPassiveScalars = numMassScalars + 0; // number of passive scalars
	static constexpr int nGroups = 1;			     // number of radiation groups
};

template <> struct Particle_Traits<AgoraGalaxy> {
	static constexpr ParticleSwitch particle_switch = ParticleSwitch::CIC;
};

template <> struct SimulationData<AgoraGalaxy> {
	amrex::Gpu::PinnedVector<amrex::Real> radius;
	amrex::Gpu::PinnedVector<amrex::Real> vcirc;
};

template <> void QuokkaSimulation<AgoraGalaxy>::preCalculateInitialConditions()
{
	// 1. read in circular velocity table "vcirc.dat"
	std::vector<amrex::Real> radius_h;
	std::vector<amrex::Real> vcirc_h;

	std::string filename = "../extern/agora_data/vcirc.dat";
	std::ifstream fstream(filename, std::ios::in);
	AMREX_ALWAYS_ASSERT(fstream.is_open());
	std::string header;
	std::getline(fstream, header);

	for (std::string line; std::getline(fstream, line);) {
		std::istringstream iss(line);
		std::vector<double> values;

		for (double value = NAN; iss >> value;) {
			values.push_back(value);
		}
		Real R_val = values.at(0);
		Real vcirc_val = values.at(1);

		radius_h.push_back(R_val);
		vcirc_h.push_back(vcirc_val);
	}

	// 2. copy data to simData_.radius and simData_.vcirc
	const size_t N = radius_h.size();
	userData_.radius.resize(N);
	userData_.vcirc.resize(N);

	for (int i = 0; i < N; ++i) {
		userData_.radius[i] = radius_h[i] * 1.0e3 * C::parsec; // kpc
		userData_.vcirc[i] = vcirc_h[i] * 1.0e5;	       // km/s
	}
}

template <> void QuokkaSimulation<AgoraGalaxy>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = grid_elem.dx_;
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = grid_elem.prob_lo_;
	const amrex::Array4<double> &state_cc = grid_elem.array_;

	double const *R_table = userData_.radius.dataPtr();
	double const *vcirc_table = userData_.vcirc.dataPtr();
	int const len_table = static_cast<int>(userData_.radius.size());

	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		// Cartesian coordinates
		amrex::Real const x = prob_lo[0] + ((i + static_cast<amrex::Real>(0.5)) * dx[0]);
		amrex::Real const y = prob_lo[1] + ((j + static_cast<amrex::Real>(0.5)) * dx[1]);
		amrex::Real const z = prob_lo[2] + ((j + static_cast<amrex::Real>(0.5)) * dx[2]);

		// cylindrical coordinates
		amrex::Real const R = std::sqrt(std::pow(x, 2) + std::pow(y, 2));
		amrex::Real const theta = std::atan2(x, y);

		// Disk mass: 8.59322e9 Msun  (i.e. 20% gas fraction)
		constexpr double M_GAS = 8.59322e9 * C::M_solar;
		// Disk scale length: 3.43218 kpc
		constexpr double r_d = 3.43218e3 * C::parsec;
		// Disk scale height: 0.343218 kpc (10% of scale length)
		constexpr double z_d = 0.343218e3 * C::parsec;

		// compute double exponential density profile
		const double rho_0 = M_GAS / 4. / M_PI / (r_d * r_d) / z_d;
		double rho = rho_0 * std::exp(-R / r_d) * std::exp(-std::abs(z) / z_d);

		// interpolate circular velocity based on radius of cell center R
		// std::cout << i << " " << j << " " << k << ": R = " << R << std::endl;
		double const vcirc = interpolate_value(R, R_table, vcirc_table, len_table);
		AMREX_ALWAYS_ASSERT(!std::isnan(vcirc));

		double const vx = vcirc * std::cos(theta);
		double const vy = vcirc * std::sin(theta);
		double const vz = 0;
		double const vsq = (vx * vx) + (vy * vy) + (vz * vz);

		// compute temperature
		double T = NAN;
		if ((R < 20.0e3 * C::parsec) && (std::abs(z) < 3.0e3 * C::parsec)) {
			T = 1.0e4; // K
		} else {
			T = 1.0e6; // K
			rho = 1.0e-6 * quokka::EOS_Traits<AgoraGalaxy>::mean_molecular_weight;
		}
		const double Eint = quokka::EOS<AgoraGalaxy>::ComputeEintFromTgas(rho, T);

		state_cc(i, j, k, HydroSystem<AgoraGalaxy>::density_index) = rho;
		state_cc(i, j, k, HydroSystem<AgoraGalaxy>::x1Momentum_index) = rho * vx;
		state_cc(i, j, k, HydroSystem<AgoraGalaxy>::x2Momentum_index) = rho * vy;
		state_cc(i, j, k, HydroSystem<AgoraGalaxy>::x3Momentum_index) = rho * vz;
		state_cc(i, j, k, HydroSystem<AgoraGalaxy>::energy_index) = Eint + 0.5 * rho * vsq;
		state_cc(i, j, k, HydroSystem<AgoraGalaxy>::internalEnergy_index) = Eint;
	});
}

template <> void QuokkaSimulation<AgoraGalaxy>::createInitialCICParticles()
{
	// read particles from ASCII file
	const int nreal_extra = 4; // mass vx vy vz
	CICParticles->SetVerbose(1);
	CICParticles->InitFromAsciiFile("AgoraGalaxy_particles.txt", nreal_extra, nullptr);
}

template <> void QuokkaSimulation<AgoraGalaxy>::ComputeDerivedVar(int lev, std::string const &dname, amrex::MultiFab &mf, const int ncomp_cc_in) const
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
	auto isNormalComp = [=](int n, int dim) {
		if ((n == HydroSystem<AgoraGalaxy>::x1Momentum_index) && (dim == 0)) {
			return true;
		}
		if ((n == HydroSystem<AgoraGalaxy>::x2Momentum_index) && (dim == 1)) {
			return true;
		}
		if ((n == HydroSystem<AgoraGalaxy>::x3Momentum_index) && (dim == 2)) {
			return true;
		}
		return false;
	};

	const int ncomp_cc = Physics_Indices<AgoraGalaxy>::nvarTotal_cc;
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
	QuokkaSimulation<AgoraGalaxy> sim(BCs_cc);
	sim.doPoissonSolve_ = 1; // enable self-gravity

	// initialize
	sim.setInitialConditions();

	// evolve
	sim.evolve();

	const int status = 0;
	return status;
}
