#include "QuokkaSimulation.hpp"

struct RemakeProblem {};
template <> struct quokka::EOS_Traits<RemakeProblem> {
	static constexpr double gamma = 1.4;
	static constexpr double mean_molecular_weight = C::m_u;
};
template <> struct Physics_Traits<RemakeProblem> : DefaultPhysicsTraits {
	static constexpr bool is_hydro_enabled = true;
	static constexpr UnitSystem unit_system = UnitSystem::CONSTANTS;
};

template <> void QuokkaSimulation<RemakeProblem>::setInitialConditionsOnGrid(quokka::grid const &grid)
{
	const auto a = grid.array_;
	amrex::ParallelFor(grid.indexRange_, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		for (int n = 0; n < Physics_Indices<RemakeProblem>::nvarTotal_cc; ++n) {
			a(i, j, k, n) = 0.0;
		}
		a(i, j, k, HydroSystem<RemakeProblem>::density_index) = 2.0;
		a(i, j, k, HydroSystem<RemakeProblem>::energy_index) = 4.0;
		a(i, j, k, HydroSystem<RemakeProblem>::internalEnergy_index) = 4.0;
	});
}

class RemakeSimulation : public QuokkaSimulation<RemakeProblem>
{
      public:
	auto checkRemakes() -> int
	{
		// Create a real fine level before testing both root and fine-level remakes.
		amrex::BoxArray fine(Geom(1).Domain());
		const amrex::DistributionMapping fine_dm(fine);
		MakeNewLevelFromCoarse(1, 0.0, fine, fine_dm);
		SetBoxArray(1, fine);
		SetDistributionMap(1, fine_dm);
		SetFinestLevel(1);
		int errors = 0;
		for (int lev = 0; lev <= 1; ++lev) {
			amrex::BoxArray ba(Geom(lev).Domain());
			ba.maxSize(8);
			const amrex::DistributionMapping dm(ba);
			RemakeLevel(lev, 0.0, ba, dm);
			SetBoxArray(lev, ba);
			SetDistributionMap(lev, dm);
			const auto &current = state_new_cc_[lev];
			const auto &previous = state_old_cc_[lev];
			const int ncomp = current.nComp();
			const int ngrow = current.nGrow();
			amrex::MultiFab difference(ba, dm, ncomp, ngrow);
			amrex::MultiFab::Copy(difference, previous, 0, 0, ncomp, ngrow);
			amrex::MultiFab::Subtract(difference, current, 0, 0, ncomp, ngrow);
			bool equal = !difference.contains_nan(0, ncomp, ngrow);
			for (int n = 0; n < ncomp; ++n) {
				equal = equal && difference.norm0(n, ngrow) == 0.0;
			}
			if (!equal) {
				++errors;
				amrex::Print() << "Old cell buffer differs after remake at level " << lev << '\n';
			}
			AMREX_ALWAYS_ASSERT(tNew_[lev] == 0.0 && tOld_[lev] < -1.e100);
		}
		return errors;
	}
	auto checkConstantState() -> int
	{
		int errors = 0;
		for (int lev = 0; lev <= finestLevel(); ++lev) {
			const auto &state = state_new_cc_[lev];
			if (state.contains_nan()) {
				++errors;
			}
			for (int n = 0; n < state.nComp(); ++n) {
				const double expected =
				    n == HydroSystem<RemakeProblem>::density_index
					? 2.0
					: (n == HydroSystem<RemakeProblem>::energy_index || n == HydroSystem<RemakeProblem>::internalEnergy_index ? 4.0 : 0.0);
				if (std::abs(state.min(n) - expected) > 1.e-12 || std::abs(state.max(n) - expected) > 1.e-12) {
					++errors;
				}
			}
		}
		return errors;
	}
};

auto problem_main() -> int
{
	RemakeSimulation sim;
	sim.stopTime_ = 0.01;
	sim.setInitialConditions();
	const int errors = sim.checkRemakes();
	if (errors != 0) {
		return 1;
	}
	sim.evolve();
	return sim.checkConstantState() == 0 ? 0 : 1;
}
