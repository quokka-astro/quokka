/// \file testMHDRemakeFaceField.cpp
/// \brief Regression test verifying RemakeLevel preserves a level's fine mhd face
///        field on regrid instead of discarding it.
///

#include "AMReX_MultiFab.H"
#include "AMReX_ParmParse.H"
#include "AMReX_Print.H"

#include "QuokkaSimulation.hpp"
#include "hydro/hydro_system.hpp"

struct MHDRemakeFaceField {};

namespace
{
constexpr double B0 = 1.0;
constexpr double delta = 0.3;
} // namespace

template <> struct quokka::EOS_Traits<MHDRemakeFaceField> {
	static constexpr double gamma = 5. / 3.;
	static constexpr double mean_molecular_weight = C::m_u;
};

template <> struct HydroSystem_Traits<MHDRemakeFaceField> {
	static constexpr bool reconstruct_eint = false;
};

template <> struct Physics_Traits<MHDRemakeFaceField> : DefaultPhysicsTraits {
	static constexpr UnitSystem unit_system = UnitSystem::CONSTANTS;
	static constexpr bool is_hydro_enabled = true;
	static constexpr bool is_mhd_enabled = true;
};

template <> void QuokkaSimulation<MHDRemakeFaceField>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const amrex::Array4<double> &state_cc = grid_elem.array_;
	constexpr double rho = 1.0;
	constexpr double P = 1.0;
	constexpr double Emag = 0.5 * B0 * B0;

	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		state_cc(i, j, k, HydroSystem<MHDRemakeFaceField>::density_index) = rho;
		state_cc(i, j, k, HydroSystem<MHDRemakeFaceField>::x1Momentum_index) = 0.0;
		state_cc(i, j, k, HydroSystem<MHDRemakeFaceField>::x2Momentum_index) = 0.0;
		state_cc(i, j, k, HydroSystem<MHDRemakeFaceField>::x3Momentum_index) = 0.0;
		state_cc(i, j, k, HydroSystem<MHDRemakeFaceField>::energy_index) = P / (quokka::EOS_Traits<MHDRemakeFaceField>::gamma - 1.) + Emag;
		state_cc(i, j, k, HydroSystem<MHDRemakeFaceField>::internalEnergy_index) = P / (quokka::EOS_Traits<MHDRemakeFaceField>::gamma - 1.);
	});
}

// Bx alternates by fine y-face index j: divergence-free (Bx depends only on j, By=Bz=0),
// but averages to B0 on any coarse restriction, discarding delta.
template <> void QuokkaSimulation<MHDRemakeFaceField>::setInitialConditionsOnGridFaceVars(quokka::grid const &grid_elem)
{
	const amrex::Array4<double> &state_fc = grid_elem.array_;
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const quokka::direction dir = grid_elem.dir_;

	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		if (dir == quokka::direction::x) {
			const double sign = (j % 2 == 0) ? 1.0 : -1.0;
			state_fc(i, j, k, Physics_Indices<MHDRemakeFaceField>::mhdFirstIndex) = B0 + delta * sign;
		} else {
			state_fc(i, j, k, Physics_Indices<MHDRemakeFaceField>::mhdFirstIndex) = 0.0;
		}
	});
}

// tag the entire level-0 domain unconditionally, so level 1 always covers the full
// domain regardless of how SetMaxGridSize() re-chops it into boxes
template <> void QuokkaSimulation<MHDRemakeFaceField>::refineGrid(int lev, amrex::TagBoxArray &tags, amrex::Real /*time*/, int /*ngrow*/)
{
	if (lev > 0) {
		return;
	}
	auto tag = tags.arrays();
	amrex::ParallelFor(tags, [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept { tag[bx](i, j, k) = amrex::TagBox::SET; });
}

auto problem_main() -> int
{
	QuokkaSimulation<MHDRemakeFaceField> sim;
	sim.setInitialConditions();
	AMREX_ALWAYS_ASSERT(sim.finestLevel() == 1);

	const auto &Bx_new_before = sim.getNewMF_fc()[1][0];
	const int ncomp = Bx_new_before.nComp();
	amrex::MultiFab Bx_before(Bx_new_before.boxArray(), Bx_new_before.DistributionMap(), ncomp, 0);
	amrex::MultiFab::Copy(Bx_before, Bx_new_before, 0, 0, ncomp, 0);
	const amrex::Real energy_before = amrex::MultiFab::Dot(Bx_before, 0, ncomp, 0);
	const auto old_num_boxes = static_cast<long>(Bx_before.boxArray().size());

	// force a decomposition-only remake of level 1: identical tags, different box chopping
	sim.SetMaxGridSize(8);
	sim.regrid(0, 0.0, false);
	AMREX_ALWAYS_ASSERT(sim.finestLevel() == 1);

	const auto &Bx_new_after = sim.getNewMF_fc()[1][0];
	const auto new_num_boxes = static_cast<long>(Bx_new_after.boxArray().size());
	amrex::Print() << "level-1 box count: " << old_num_boxes << " -> " << new_num_boxes << "\n";
	AMREX_ALWAYS_ASSERT(new_num_boxes != old_num_boxes); // sanity check: the remake actually re-chopped the level

	amrex::MultiFab Bx_after_remapped(Bx_before.boxArray(), Bx_before.DistributionMap(), ncomp, 0);
	Bx_after_remapped.ParallelCopy(Bx_new_after, 0, 0, ncomp, 0, 0);
	const amrex::Real energy_after = amrex::MultiFab::Dot(Bx_after_remapped, 0, ncomp, 0);

	amrex::MultiFab::Subtract(Bx_after_remapped, Bx_before, 0, 0, ncomp, 0);
	const amrex::Real max_diff = Bx_after_remapped.norminf(0, 0);
	const amrex::Real energy_rel_diff = std::abs(energy_after - energy_before) / energy_before;

	amrex::Print() << "max |Bx_after - Bx_before| on retained overlap = " << max_diff << "\n";
	amrex::Print() << "magnetic energy: before=" << energy_before << " after=" << energy_after << " rel_diff=" << energy_rel_diff << "\n";

	constexpr amrex::Real tol = 1.0e-12;
	if (max_diff > tol * delta || energy_rel_diff > tol) {
		amrex::Print() << "ISSUE #2210 REPRODUCED: RemakeLevel discarded the level's existing fine mhd "
				  "face field instead of preserving it across the remake.\n";
		return 1;
	}
	amrex::Print() << "Test passed: fine mhd face field preserved across RemakeLevel.\n";
	return 0;
}
