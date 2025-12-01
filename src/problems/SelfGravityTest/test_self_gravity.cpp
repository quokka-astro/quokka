/// \file test_self_gravity.cpp
/// \brief Defines a test problem for self-gravity with AMR subcycling.
/// This test implements a simple self-gravitating gas cloud to demonstrate
/// the PoissonGravity class functionality using Approach 1 (Enzo method).
///

#include <algorithm>
#include <cmath>
#include <cstdlib>

#include "AMReX.H"
#include "AMReX_BCRec.H"
#include "AMReX_BC_TYPES.H"
#include "AMReX_Box.H"
#include "AMReX_Vector.H"
#include "QuokkaSimulation.hpp"
#include "hydro/EOS.hpp"
#include "hydro/hydro_system.hpp"

struct SelfGravityProblem {
};

constexpr double rho_ambient = 1.0e-3;	    // ambient density
constexpr double rho_center = 1.0;	    // central density
constexpr double cloud_radius = 0.1;	    // radius of the dense cloud
constexpr double pressure_ambient = 1.0e-3; // ambient pressure
constexpr double pressure_center = 1.0e-2;  // central pressure

template <> struct quokka::EOS_Traits<SelfGravityProblem> {
	static constexpr double mean_molecular_weight = 1.0;
	static constexpr double gamma = 5.0 / 3.0;
};

template <> struct Physics_Traits<SelfGravityProblem> {
	// cell-centred
	static constexpr bool is_hydro_enabled = true;
	static constexpr int numMassScalars = 0;		     // number of mass scalars
	static constexpr int numPassiveScalars = numMassScalars + 0; // number of passive scalars
	static constexpr bool is_radiation_enabled = false;
	static constexpr bool is_self_gravity_enabled = true;

	// face-centred
	static constexpr bool is_mhd_enabled = false;

	// unit system
	static constexpr UnitSystem unit_system = UnitSystem::CONSTANTS;
	static constexpr double gravitational_constant = 1.0; // normalized units
};

template <> struct SimulationData<SelfGravityProblem> {
	// runtime parameters
	amrex::Real cloud_center_x = 0.5;
	amrex::Real cloud_center_y = 0.5;
	amrex::Real cloud_center_z = 0.5;
};

template <> AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void QuokkaSimulation<SelfGravityProblem>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	// Extract simulation data
	const amrex::Real cloud_center_x = userData_.cloud_center_x;
	const amrex::Real cloud_center_y = userData_.cloud_center_y;
	const amrex::Real cloud_center_z = userData_.cloud_center_z;

	// Extract grid properties
	amrex::Array4<amrex::Real> const &state = grid_elem.array;
	const auto &box = grid_elem.indexRange;
	const auto &dx = grid_elem.dx;
	const auto &prob_lo = grid_elem.prob_lo;

	// Set initial conditions: spherical gas cloud
	amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		amrex::Real const x = prob_lo[0] + (i + static_cast<amrex::Real>(0.5)) * dx[0];
		amrex::Real const y = prob_lo[1] + (j + static_cast<amrex::Real>(0.5)) * dx[1];
		amrex::Real const z = prob_lo[2] + (k + static_cast<amrex::Real>(0.5)) * dx[2];

		const amrex::Real distance = std::sqrt((x - cloud_center_x) * (x - cloud_center_x) + (y - cloud_center_y) * (y - cloud_center_y) +
						       (z - cloud_center_z) * (z - cloud_center_z));

		amrex::Real rho, pressure;
		if (distance <= cloud_radius) {
			// Inside the cloud: higher density and pressure
			const amrex::Real r_norm = distance / cloud_radius;
			rho = rho_center * (1.0 - 0.5 * r_norm * r_norm); // parabolic density profile
			pressure = pressure_center * (1.0 - 0.3 * r_norm * r_norm);
		} else {
			// Outside the cloud: ambient medium
			rho = rho_ambient;
			pressure = pressure_ambient;
		}

		// Set state: [rho, rho*vx, rho*vy, rho*vz, E]
		state(i, j, k, HydroSystem<SelfGravityProblem>::density_index) = rho;
		state(i, j, k, HydroSystem<SelfGravityProblem>::x1Momentum_index) = 0.0; // zero initial velocity
		state(i, j, k, HydroSystem<SelfGravityProblem>::x2Momentum_index) = 0.0;
		state(i, j, k, HydroSystem<SelfGravityProblem>::x3Momentum_index) = 0.0;

		// Total energy = internal energy (no kinetic energy initially)
		const amrex::Real gamma = quokka::EOS_Traits<SelfGravityProblem>::gamma;
		const amrex::Real internal_energy = pressure / ((gamma - 1.0) * rho);
		state(i, j, k, HydroSystem<SelfGravityProblem>::energy_index) = rho * internal_energy;
	});
}

template <>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void QuokkaSimulation<SelfGravityProblem>::setInitialConditionsOnGridFaceVars(quokka::grid const & /*grid_elem*/)
{
	// No face-centered variables for this problem
}

template <>
void QuokkaSimulation<SelfGravityProblem>::computeReferenceSolution(amrex::MultiFab &ref, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx,
								    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_lo)
{
	// No analytical reference solution for this test
	// (In practice, one could compare to spherically symmetric collapse solutions)
}

template <> void QuokkaSimulation<SelfGravityProblem>::ErrorEst(int lev, amrex::TagBoxArray &tags, amrex::Real /*time*/, int /*ngrow*/)
{
	// Refine on density gradients
	const amrex::Real density_threshold = 0.1; // refine where density > threshold
	const auto &prob_lo = geom[lev].ProbLoArray();
	const auto &dx = geom[lev].CellSizeArray();
	const auto &state = state_new_cc_[lev];

	for (amrex::MFIter mfi(tags); mfi.isValid(); ++mfi) {
		const amrex::Box &box = mfi.validbox();
		const auto &state_fab = state.const_array(mfi);
		const auto &tag = tags.array(mfi);

		amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
			const amrex::Real rho = state_fab(i, j, k, HydroSystem<SelfGravityProblem>::density_index);
			if (rho > density_threshold) {
				tag(i, j, k) = amrex::TagBox::SET;
			}
		});
	}
}

auto problem_main() -> int
{
	// Read problem parameters
	amrex::ParmParse pp("problem");
	SimulationData<SelfGravityProblem> sim_data{};
	pp.query("cloud_center_x", sim_data.cloud_center_x);
	pp.query("cloud_center_y", sim_data.cloud_center_y);
	pp.query("cloud_center_z", sim_data.cloud_center_z);

	// Boundary conditions: outflow on all boundaries
	constexpr int nvars = HydroSystem<SelfGravityProblem>::nvar_;
	amrex::Vector<amrex::BCRec> BCs_cc(nvars);
	for (int n = 0; n < nvars; ++n) {
		BCs_cc[n].setLo(0, amrex::BCType::foextrap); // x-lo
		BCs_cc[n].setHi(0, amrex::BCType::foextrap); // x-hi
		BCs_cc[n].setLo(1, amrex::BCType::foextrap); // y-lo
		BCs_cc[n].setHi(1, amrex::BCType::foextrap); // y-hi
		BCs_cc[n].setLo(2, amrex::BCType::foextrap); // z-lo
		BCs_cc[n].setHi(2, amrex::BCType::foextrap); // z-hi
	}

	// Create simulation
	QuokkaSimulation<SelfGravityProblem> sim(BCs_cc);
	sim.userData_ = sim_data;

	// Initialize and evolve
	sim.setInitialConditions();
	sim.evolve();

	// Check if simulation completed successfully
	const int status = 0;
	return status;
}