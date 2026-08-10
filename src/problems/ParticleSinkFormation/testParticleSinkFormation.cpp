/// \file testParticleSinkFormation.cpp
/// \brief Defines a test problem for sink particle formation.
///

#include "AMReX.H"
#include "AMReX_BC_TYPES.H"
#include "AMReX_MultiFab.H"
#include "AMReX_ParmParse.H"
#include "AMReX_Print.H"
#include "AMReX_SPACE.H"
#include "util/fextract.hpp"
#include <exception>
#include <format>

#include "QuokkaSimulation.hpp"
#include "fundamental_constants.H"
#include "hydro/hydro_system.hpp"
#include "particles/particle_types.hpp"
#include "util/BC.hpp"

#ifdef HAVE_PYTHON
#include "util/matplotlibcpp.h"
#endif

struct SinkProblem {
};

constexpr double M_sol = C::M_solar;
constexpr double mu = 1.0 * C::m_p;
constexpr double gamma_ = 5. / 3.;
const double rho0 = 1.0 * C::m_p; // g cm^-3
const double T0 = 10.0;		  // K
const double CV = 1. / (gamma_ - 1.) / mu * C::k_B;
const double year = 3.15576e+07;			// in seconds
const double cs = std::sqrt(gamma_ * C::k_B * T0 / mu); // NOLINT
constexpr double B0 = 1.0e-7;				// uniform background field

template <> struct Particle_Traits<SinkProblem> : DefaultParticleTraits {
	// static constexpr ParticleSwitch particle_switch = ParticleSwitch::None;
	static constexpr ParticleSwitch particle_switch = ParticleSwitch::Sink;
};

template <> struct quokka::EOS_Traits<SinkProblem> {
	static constexpr double gamma = gamma_;
	static constexpr double mean_molecular_weight = mu;
};

template <> struct HydroSystem_Traits<SinkProblem> {
	static constexpr bool reconstruct_eint = true; // need to reconstruct temperature
};

template <> struct Physics_Traits<SinkProblem> : DefaultPhysicsTraits {
	// cell-centred
	static constexpr bool is_hydro_enabled = true;
	static constexpr bool is_self_gravity_enabled = true;
	// face-centred
	static constexpr bool is_mhd_enabled = true;
};

template <> void QuokkaSimulation<SinkProblem>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const amrex::Array4<double> &state_cc = grid_elem.array_;
	const auto prob_lo = geom[0].ProbLoArray();
	const auto dx = geom[0].CellSizeArray();

	const double jeans_density = quokka::ParticleUtils::computeJeansDensity(cs, dx[0]);
	const double Emag = 0.5 * B0 * B0;

	// loop over the grid and set the initial condition
	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		const double x = prob_lo[0] + (i * dx[0]);
		const double y = prob_lo[1] + (j * dx[1]);
		const double z = prob_lo[2] + (k * dx[2]);
		double rho = rho0;
		const double sf_cell_loc = 0.1 * dx[0]; // The cell right next to the origin
		if (x <= sf_cell_loc && x + dx[0] > sf_cell_loc && y <= sf_cell_loc && y + dx[1] > sf_cell_loc && z <= sf_cell_loc && z + dx[2] > sf_cell_loc) {
			// this is the cell with peak density
			rho = 1.2 * jeans_density;
		} else if (x - 2 * dx[0] <= sf_cell_loc && x - dx[0] > sf_cell_loc && y <= sf_cell_loc && y + dx[1] > sf_cell_loc && z <= sf_cell_loc &&
			   z + dx[2] > sf_cell_loc) {
			// this is the cell that is 2 cells left of the peak density cell
			rho = 1.1 * jeans_density;
		}
		const double rho_e = CV * T0 * rho;
		state_cc(i, j, k, HydroSystem<SinkProblem>::density_index) = rho;
		state_cc(i, j, k, HydroSystem<SinkProblem>::x1Momentum_index) = 0.0;
		state_cc(i, j, k, HydroSystem<SinkProblem>::x2Momentum_index) = 0.0;
		state_cc(i, j, k, HydroSystem<SinkProblem>::x3Momentum_index) = 0.0;
		state_cc(i, j, k, HydroSystem<SinkProblem>::energy_index) = rho_e + Emag;
		state_cc(i, j, k, HydroSystem<SinkProblem>::internalEnergy_index) = rho_e;
	});
}

template <> void QuokkaSimulation<SinkProblem>::setInitialConditionsOnGridFaceVars(quokka::grid const &grid_elem)
{
	const amrex::Array4<double> &state_fc = grid_elem.array_;
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const quokka::direction dir = grid_elem.dir_;
	const double B_val = (dir == quokka::direction::x) ? B0 : 0.0;

	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) { state_fc(i, j, k, Physics_Indices<SinkProblem>::mhdFirstIndex) = B_val; });
}

template <> void QuokkaSimulation<SinkProblem>::refineGrid(int lev, amrex::TagBoxArray &tags, amrex::Real /*time*/, int /*ngrow*/)
{
	// tag cells for refinement: static mesh refinement for the whole domain

	// auto const &dx = geom[lev].CellSizeArray();
	// auto const &plo = geom[lev].ProbLoArray();
	// auto const &phi = geom[lev].ProbHiArray();

	for (amrex::MFIter mfi(state_new_cc_[lev]); mfi.isValid(); ++mfi) {
		const amrex::Box &box = mfi.validbox();
		const auto tag = tags.array(mfi);

		amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept { tag(i, j, k) = amrex::TagBox::SET; });
	}
}

auto problem_main() -> int
{
	// Problem initialization
	QuokkaSimulation<SinkProblem> sim;

	sim.reconstructionOrder_ = 3; // 2=PLM, 3=PPM
	sim.cflNumber_ = 0.3;	      // *must* be less than 1/3 in 3D!
	sim.stopTime_ = 1.0e7 * year; // 1 Myr

	// initialize
	sim.setInitialConditions();

	sim.evolve();

	return 0;
}
