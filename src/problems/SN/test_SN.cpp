//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file test_SN.cpp
/// \brief Defines a test problem for supernova feedback.
///

#include "AMReX.H"
#include "AMReX_BC_TYPES.H"
#include "AMReX_BLassert.H"
#include "AMReX_MultiFab.H"
#include "AMReX_ParmParse.H"
#include "AMReX_Print.H"
#include "AMReX_SPACE.H"

#include "QuokkaSimulation.hpp"
#include "fundamental_constants.H"
#include "hydro/hydro_system.hpp"
#include "particles/particle_types.hpp"
#include "radiation/radiation_system.hpp"

struct SNProblem {
};

AMREX_GPU_MANAGED std::string SN_particles_file = "SN_particles.txt";

constexpr double mu = 1.0 * C::m_u;
// constexpr double mu = 1.295 * C::m_u; // neutral gas
constexpr double gamma_ = 5. / 3.;
const double CV = 1. / (gamma_ - 1.) / mu * C::k_B;
const double cloudy_H_mass_fraction = 1.0 / (1.0 + 0.1 * 3.971);
const double year = 3.15576e+07; // in seconds

AMREX_GPU_MANAGED double n_amb = 1.0;	 // ambient density (g cm^-3)
AMREX_GPU_MANAGED double T_amb = 100.0;	 // ambient temperature (K)
AMREX_GPU_MANAGED double t_stop = 3.0e5; // stop time (yr)

template <> struct Particle_Traits<SNProblem> {
	// static constexpr ParticleSwitch particle_switch = ParticleSwitch::None;
	static constexpr ParticleSwitch particle_switch = ParticleSwitch::StochasticStellarPop;
};

template <> struct quokka::EOS_Traits<SNProblem> {
	static constexpr double gamma = gamma_;
	static constexpr double mean_molecular_weight = mu;
};

template <> struct HydroSystem_Traits<SNProblem> {
	static constexpr bool reconstruct_eint = true; // need to reconstruct temperature
};

template <> struct Physics_Traits<SNProblem> {
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

template <> void QuokkaSimulation<SNProblem>::createInitialStochasticStellarPopParticles()
{
	// read particles from ASCII file
	const int nreal_extra = 7; // mass vx vy vz birth_time death_time lum
	StochasticStellarPopParticles->SetVerbose(1);
	StochasticStellarPopParticles->InitFromAsciiFile(SN_particles_file, nreal_extra, nullptr);

	// Loop over all particles and set first integer component to 0
	const int lev = 0;
	auto &particles = StochasticStellarPopParticles->GetParticles(lev);
	for (auto &kv : particles) {
		auto &particle_array = kv.second.GetArrayOfStructs();
		const int np = particle_array.numParticles();
		for (int i = 0; i < np; i++) {
			auto &p = particle_array[i];
			p.idata(0) = static_cast<int>(quokka::StellarEvolutionStage::SNProgenitor);
		}
	}
}

template <> void QuokkaSimulation<SNProblem>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const amrex::Array4<double> &state_cc = grid_elem.array_;
	const double rho_bg = n_amb * C::m_u / cloudy_H_mass_fraction;
	const double E0 = CV * T_amb * rho_bg;
	const double rho = rho_bg;
	const double rho_e = E0;

	// loop over the grid and set the initial condition
	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		state_cc(i, j, k, HydroSystem<SNProblem>::density_index) = rho;
		state_cc(i, j, k, HydroSystem<SNProblem>::x1Momentum_index) = 0.0;
		state_cc(i, j, k, HydroSystem<SNProblem>::x2Momentum_index) = 0.0;
		state_cc(i, j, k, HydroSystem<SNProblem>::x3Momentum_index) = 0.0;
		state_cc(i, j, k, HydroSystem<SNProblem>::energy_index) = rho_e;
		state_cc(i, j, k, HydroSystem<SNProblem>::internalEnergy_index) = rho_e;
	});
}

// template <> void QuokkaSimulation<SNProblem>::ErrorEst(int lev, amrex::TagBoxArray &tags, amrex::Real /*time*/, int /*ngrow*/)
// {
// 	// tag cells for refinement

// 	const amrex::Real eta_threshold = 0.1; // gradient refinement threshold
// 	const amrex::Real P_min = 1.0e-3;      // minimum pressure for refinement

// 	for (amrex::MFIter mfi(state_new_cc_[lev]); mfi.isValid(); ++mfi) {
// 		const amrex::Box &box = mfi.validbox();
// 		const auto state = state_new_cc_[lev].const_array(mfi);
// 		const auto tag = tags.array(mfi);

// 		amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
// 			amrex::Real const P = HydroSystem<SNProblem>::ComputePressure(state, i, j, k);

// 			amrex::Real const P_xplus = HydroSystem<SNProblem>::ComputePressure(state, i + 1, j, k);
// 			amrex::Real const P_xminus = HydroSystem<SNProblem>::ComputePressure(state, i - 1, j, k);
// 			amrex::Real const P_yplus = HydroSystem<SNProblem>::ComputePressure(state, i, j + 1, k);
// 			amrex::Real const P_yminus = HydroSystem<SNProblem>::ComputePressure(state, i, j - 1, k);
// 			amrex::Real const P_zplus = HydroSystem<SNProblem>::ComputePressure(state, i, j, k + 1);
// 			amrex::Real const P_zminus = HydroSystem<SNProblem>::ComputePressure(state, i, j, k - 1);

// 			amrex::Real const del_x = std::max(std::abs(P_xplus - P), std::abs(P - P_xminus));
// 			amrex::Real const del_y = std::max(std::abs(P_yplus - P), std::abs(P - P_yminus));
// 			amrex::Real const del_z = std::max(std::abs(P_zplus - P), std::abs(P - P_zminus));

// 			amrex::Real const gradient_indicator = std::max({del_x, del_y, del_z}) / P;

// 			if ((gradient_indicator > eta_threshold) && (P > P_min)) {
// 				tag(i, j, k) = amrex::TagBox::SET;
// 			}
// 		});
// 	}
// }

auto problem_main() -> int
{
	auto isNormalComp = [=](int n, int dim) {
		if ((n == HydroSystem<SNProblem>::x1Momentum_index) && (dim == 0)) {
			return true;
		}
		if ((n == HydroSystem<SNProblem>::x2Momentum_index) && (dim == 1)) {
			return true;
		}
		if ((n == HydroSystem<SNProblem>::x3Momentum_index) && (dim == 2)) {
			return true;
		}
		return false;
	};

	const int ncomp_cc = Physics_Indices<SNProblem>::nvarTotal_cc;
	amrex::Vector<amrex::BCRec> BCs_cc(ncomp_cc);
	for (int n = 0; n < ncomp_cc; ++n) {
		for (int i = 0; i < AMREX_SPACEDIM; ++i) {
			if constexpr (true) { // periodic boundaries
				BCs_cc[n].setLo(i, amrex::BCType::int_dir);
				BCs_cc[n].setHi(i, amrex::BCType::int_dir);
			} else { // octant symmetry
				if (isNormalComp(n, i)) {
					BCs_cc[n].setLo(i, amrex::BCType::reflect_odd);
					BCs_cc[n].setHi(i, amrex::BCType::reflect_odd);
				} else {
					BCs_cc[n].setLo(i, amrex::BCType::reflect_even);
					BCs_cc[n].setHi(i, amrex::BCType::reflect_even);
				}
			}
		}
	}

	// get n_amb from the input file
	amrex::ParmParse const pp("problem");
	pp.query("n_amb", n_amb);
	pp.query("T_amb", T_amb);
	pp.query("t_stop", t_stop);
	pp.query("SN_particles_file", SN_particles_file);

	// Problem initialization
	QuokkaSimulation<SNProblem> sim(BCs_cc);

	sim.reconstructionOrder_ = 3; // 2=PLM, 3=PPM
	sim.stopTime_ = t_stop * year;
	sim.cflNumber_ = 0.3; // *must* be less than 1/3 in 3D!
	sim.initDt_ = 0.001 * year;

	// initialize
	sim.setInitialConditions();

	// evolve
	sim.evolve();

	// find the maximum internal energy in the state_new_cc_[0]
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx0 = sim.geom[0].CellSizeArray();
	const amrex::Real vol = AMREX_D_TERM(dx0[0], *dx0[1], *dx0[2]);
	const amrex::Real max_internal_energy_density = sim.state_new_cc_[0].max(HydroSystem<SNProblem>::internalEnergy_index);
	const amrex::Real max_internal_energy = max_internal_energy_density * vol;
	const amrex::Real expected_minimum_max_internal_energy = 1.0e51 / (7 * 7 * 7); // 1e51 erg energy into (2 * 3 + 1)^3 cells
	int status = 1;
	if (max_internal_energy > expected_minimum_max_internal_energy) {
		status = 0;
		amrex::Print() << "Test passed. Max internal energy in cells: " << max_internal_energy << "\n";
	} else {
		status = 1;
		amrex::Print() << "Test failed. Max internal energy in cells too low: " << max_internal_energy << "\n";
		amrex::Print() << "Expected minimum max internal energy: " << expected_minimum_max_internal_energy << "\n";
	}

	return status;
}
