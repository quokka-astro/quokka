/// \file mass_conserv.cpp
/// \brief Defines a test problem for mass conservation.
///

#include "AMReX_BC_TYPES.H"
#include "AMReX_ParmParse.H"
#include "AMReX_Print.H"
#include "util/fextract.hpp"

#include "AMReX_REAL.H"
#include "QuokkaSimulation.hpp"
#include "hydro/hydro_system.hpp"
#include "mass_conserv.hpp"

struct TheProblem {
};

constexpr double initial_Tgas = 1.0;
constexpr double CV = 1.5;
constexpr double initial_rho = 1.0;
constexpr int nx_lev0 = 32;
constexpr double Lx = 1.0;
constexpr double dx_lev0 = Lx / nx_lev0;
constexpr double mass_in_a_cell = initial_rho * dx_lev0 * dx_lev0 * dx_lev0;
constexpr double mass_in_central_cell = 1.0 + mass_in_a_cell;
constexpr double rho_in_central_cell = mass_in_central_cell / (dx_lev0 * dx_lev0 * dx_lev0);
constexpr double expected_total_mass = 2.0;

template <> struct quokka::EOS_Traits<TheProblem> {
	static constexpr double mean_molecular_weight = 1.0;
	static constexpr double gamma = 5. / 3.;
};

template <> struct Physics_Traits<TheProblem> {
	// cell-centred
	static constexpr bool is_hydro_enabled = true;
	static constexpr int numMassScalars = 0;		     // number of mass scalars
	static constexpr int numPassiveScalars = numMassScalars + 0; // number of passive scalars
	static constexpr bool is_radiation_enabled = false;
	// face-centred
	static constexpr bool is_mhd_enabled = false;
	static constexpr int nGroups = 1; // number of radiation groups
	static constexpr UnitSystem unit_system = UnitSystem::CONSTANTS;
	static constexpr double boltzmann_constant = 1.0;
	static constexpr double gravitational_constant = 1.0;
	static constexpr double c_light = 1.0;
	static constexpr double radiation_constant = 1.0;
};

template <> void QuokkaSimulation<TheProblem>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const amrex::Array4<double> &state_cc = grid_elem.array_;

	// find the index of the central cell
	const int nx_central_cell = nx_lev0 / 2;

	// loop over the grid and set the initial condition
	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		double rho = initial_rho;
		if (i == nx_central_cell && j == nx_central_cell && k == nx_central_cell) {
			rho = rho_in_central_cell;
		}
		const double Egas = rho * CV * initial_Tgas;
		state_cc(i, j, k, RadSystem<TheProblem>::gasDensity_index) = rho;
		state_cc(i, j, k, RadSystem<TheProblem>::x1GasMomentum_index) = 0.;
		state_cc(i, j, k, RadSystem<TheProblem>::x2GasMomentum_index) = 0.;
		state_cc(i, j, k, RadSystem<TheProblem>::x3GasMomentum_index) = 0.;
		state_cc(i, j, k, RadSystem<TheProblem>::gasInternalEnergy_index) = Egas;
		state_cc(i, j, k, RadSystem<TheProblem>::gasEnergy_index) = Egas;
	});
}

auto problem_main() -> int
{
	// Problem parameters
	const double tmax = 1.0;
	const int max_timesteps = 10;

	// Boundary conditions
	constexpr int nvars = RadSystem<TheProblem>::nvar_;
	amrex::Vector<amrex::BCRec> BCs_cc(nvars);
	for (int n = 0; n < nvars; ++n) {
		for (int i = 0; i < AMREX_SPACEDIM; ++i) {
			BCs_cc[n].setLo(i, amrex::BCType::int_dir); // periodic
			BCs_cc[n].setHi(i, amrex::BCType::int_dir);
		}
	}

	// Problem initialization
	QuokkaSimulation<TheProblem> sim(BCs_cc);

	sim.radiationReconstructionOrder_ = 3; // PPM
	sim.stopTime_ = tmax;
	sim.maxTimesteps_ = max_timesteps;

	// initialize
	sim.setInitialConditions();

	// get total mass
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx0 = sim.geom[0].CellSizeArray();
	amrex::Real const vol = AMREX_D_TERM(dx0[0], *dx0[1], *dx0[2]);
	amrex::Real const total_mass = sim.state_new_cc_[0].sum(HydroSystem<TheProblem>::density_index) * vol;
	amrex::Print() << "Initial total mass: " << total_mass << "\n";

	// evolve
	sim.evolve();

	// get total mass
	amrex::Real const total_mass_final = sim.state_new_cc_[0].sum(HydroSystem<TheProblem>::density_index) * vol;
	amrex::Print() << "Final total mass: " << total_mass_final << "\n";

	const double rel_err = std::abs(total_mass_final - total_mass) / total_mass;
	amrex::Print() << "Relative error: " << rel_err << "\n";

	// check if mass is conserved
	if (rel_err > 1.0e-10) {
		amrex::Print() << "Mass is not conserved!\n";
		return 1;
	}

	return 0;
}
