/// \file testRadCoveredCell.cpp
/// \brief Matter-radiation coupling on a coarse level that is fully covered by a fine level whose velocity alternates in sign cell to cell.
///
/// The fine level is cold (T = 1) but moves supersonically with alternating sign, so the average-down of the conserved variables
/// gives the covered coarse cells a near-zero mean momentum and an internal energy (Etot - |p|^2/(2 rho)) that is dominated by the
/// unresolved velocity dispersion. If the stiff matter-radiation source solve is applied to those covered cells, it reads that
/// kinetic energy as thermal energy (T_coarse >> T_fine) and the Newton iteration fails to converge. The covered cells are
/// overwritten by average-down, so the source solve must be skipped there.
///

#include "AMReX_Print.H"
#include "QuokkaSimulation.hpp"
#include "physics_info.hpp"
#include "radiation/radiation_system.hpp"
#include "util/BC.hpp"

struct CoveredCellProblem {}; // dummy type to allow compile-type polymorphism via template specialization

constexpr double c = 1.0e6;
constexpr double v0 = 1.0e4; // Mach ~ 8000 at T0
constexpr double kappa0 = 1.0e2;
constexpr double T0 = 1.0;
constexpr double rho0 = 1.0;
constexpr double a_rad = 1.0;
constexpr double mu = 1.0;
constexpr double k_B = 1.0;

template <> struct quokka::EOS_Traits<CoveredCellProblem> {
	static constexpr double mean_molecular_weight = mu;
	static constexpr double gamma = 5. / 3.;
};

template <> struct RadSystem_Traits<CoveredCellProblem> {
	static constexpr double c_hat_over_c = 0.01;
	static constexpr double Erad_floor = 0.0;
	static constexpr int beta_order = 1;
};

template <> struct Physics_Traits<CoveredCellProblem> : DefaultPhysicsTraits {
	static constexpr bool is_hydro_enabled = true;
	static constexpr bool is_radiation_enabled = true;
	static constexpr UnitSystem unit_system = UnitSystem::CONSTANTS;
	static constexpr double boltzmann_constant = k_B;
	static constexpr double c_light = c;
	static constexpr double gravitational_constant = 1.0;
	static constexpr double radiation_constant = a_rad;
};

template <> AMREX_GPU_HOST_DEVICE auto RadSystem<CoveredCellProblem>::ComputePlanckOpacity(const double /*rho*/, const double /*Tgas*/) -> amrex::Real
{
	return kappa0;
}

template <> AMREX_GPU_HOST_DEVICE auto RadSystem<CoveredCellProblem>::ComputeFluxMeanOpacity(const double rho, const double Tgas) -> amrex::Real
{
	return ComputePlanckOpacity(rho, Tgas);
}

template <> void QuokkaSimulation<CoveredCellProblem>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const amrex::Array4<double> &state_cc = grid_elem.array_;
	const double Egas = quokka::EOS<CoveredCellProblem>::ComputeEintFromTgas(rho0, T0);
	const double Erad0 = a_rad * T0 * T0 * T0 * T0;

	// the velocity alternates in sign on every cell of the finest level; coarse levels are overwritten by average-down
	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		const double v = (i % 2 == 0) ? v0 : -v0;
		state_cc(i, j, k, RadSystem<CoveredCellProblem>::radEnergy_index) = Erad0;
		state_cc(i, j, k, RadSystem<CoveredCellProblem>::x1RadFlux_index) = 0;
		state_cc(i, j, k, RadSystem<CoveredCellProblem>::x2RadFlux_index) = 0;
		state_cc(i, j, k, RadSystem<CoveredCellProblem>::x3RadFlux_index) = 0;
		state_cc(i, j, k, RadSystem<CoveredCellProblem>::gasEnergy_index) = Egas + 0.5 * rho0 * v * v;
		state_cc(i, j, k, RadSystem<CoveredCellProblem>::gasDensity_index) = rho0;
		state_cc(i, j, k, RadSystem<CoveredCellProblem>::gasInternalEnergy_index) = Egas;
		state_cc(i, j, k, RadSystem<CoveredCellProblem>::x1GasMomentum_index) = v * rho0;
		state_cc(i, j, k, RadSystem<CoveredCellProblem>::x2GasMomentum_index) = 0.;
		state_cc(i, j, k, RadSystem<CoveredCellProblem>::x3GasMomentum_index) = 0.;
	});
}

// refine the entire domain, so that every level-0 cell is covered by level 1
template <> void QuokkaSimulation<CoveredCellProblem>::refineGrid(int lev, amrex::TagBoxArray &tags, amrex::Real /*time*/, int /*ngrow*/)
{
	if (lev == 0) {
		tags.setVal(amrex::TagBox::SET);
	}
}

auto problem_main() -> int
{
	QuokkaSimulation<CoveredCellProblem> sim;
	sim.radiationReconstructionOrder_ = 3;
	sim.radiationCflNumber_ = 0.3;
	sim.cflNumber_ = 0.3;
	sim.maxTimesteps_ = 5;
	sim.plotfileInterval_ = -1;

	sim.setInitialConditions();
	// the run aborts if the Newton iteration fails on a covered coarse cell
	sim.evolve();

	amrex::Print() << "Finished." << '\n';
	return 0;
}
