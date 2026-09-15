#include "QuokkaSimulation.hpp"

struct FaceProblem {};
template <> struct Physics_Traits<FaceProblem> : DefaultPhysicsTraits {
	static constexpr bool is_hydro_enabled = true;
	static constexpr bool is_mhd_enabled = true;
	static constexpr UnitSystem unit_system = UnitSystem::CONSTANTS;
};

template <> void QuokkaSimulation<FaceProblem>::setInitialConditionsOnGrid(quokka::grid const &grid)
{
	const auto a = grid.array_;
	amrex::ParallelFor(grid.indexRange_, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		for (int n = 0; n < Physics_Indices<FaceProblem>::nvarTotal_cc; ++n) {
			a(i, j, k, n) = 0.0;
		}
		a(i, j, k, HydroSystem<FaceProblem>::density_index) = 1.0;
		a(i, j, k, HydroSystem<FaceProblem>::energy_index) = 100.0;
		a(i, j, k, HydroSystem<FaceProblem>::internalEnergy_index) = 100.0;
	});
}

template <> void QuokkaSimulation<FaceProblem>::setInitialConditionsOnGridFaceVars(quokka::grid const &grid)
{
	const auto a = grid.array_;
	const int axis = static_cast<int>(grid.dir_);
	const auto dx = grid.dx_;
	amrex::ParallelFor(grid.indexRange_, [=] AMREX_GPU_DEVICE(int i, int j, int k) { a(i, j, k, 0) = axis == 2 ? 3.0 + (i + 0.5) * dx[0] : axis + 1.0; });
}

class FaceSimulation : public QuokkaSimulation<FaceProblem>
{
      public:
	using QuokkaSimulation<FaceProblem>::QuokkaSimulation;
	auto check(bool remake) -> int
	{
		amrex::BoxArray ba(Geom(1).Domain());
		if (remake) {
			ba.maxSize(8);
		}
		const amrex::DistributionMapping dm(ba);
		if (remake) {
			RemakeLevel(1, 0.0, ba, dm);
		} else {
			MakeNewLevelFromCoarse(1, 0.0, ba, dm);
		}
		SetBoxArray(1, ba);
		SetDistributionMap(1, dm);
		SetFinestLevel(1);
		int errors = 0;
		for (int axis = 0; axis < 3; ++axis) {
			for (int old = 0; old < 2; ++old) {
				auto &mf = old ? state_old_fc_[1][axis] : state_new_fc_[1][axis];
				amrex::Gpu::DeviceScalar<int> failed(0);
				auto *error = failed.dataPtr();
				for (amrex::MFIter it(mf); it.isValid(); ++it) {
					const auto a = mf.const_array(it);
					// Interior transverse indices avoid edge/corner boundary conventions.
					amrex::ParallelFor(2, [=] AMREX_GPU_DEVICE(int side) {
						const int i = side == 0 ? -2 : 17;
						if (a.contains(i, 3, 3)) {
							const int interior = axis == 0 ? (side == 0 ? 0 : 16) : (side == 0 ? 1 : 14);
							const double expected = (axis == 1 ? -1.0 : 1.0) * a(interior, 3, 3, 0);
							if (!std::isfinite(expected) || !std::isfinite(a(i, 3, 3, 0)) ||
							    std::abs(a(i, 3, 3, 0) - expected) > 1.e-12) {
								amrex::Gpu::Atomic::Add(error, 1);
							}
						}
					});
				}
				const int count = failed.dataValue();
				errors += count;
				if (count) {
					amrex::Print() << "Face BC mismatch: remake=" << remake << " axis=" << axis << " old=" << old << '\n';
				}
			}
		}
		amrex::ParallelDescriptor::ReduceIntSum(errors);
		return errors;
	}
};

auto problem_main() -> int
{
	constexpr int nc = Physics_Indices<FaceProblem>::nvarTotal_cc;
	amrex::Vector<amrex::BCRec> cc(nc), fc(3);
	for (auto &bc : cc) {
		for (int d = 0; d < 3; ++d) {
			bc.setLo(d, amrex::BCType::foextrap);
			bc.setHi(d, amrex::BCType::foextrap);
		}
	}
	for (auto &bc : fc) {
		for (int d = 0; d < 3; ++d) {
			bc.setLo(d, amrex::BCType::foextrap);
			bc.setHi(d, amrex::BCType::foextrap);
		}
	}
	fc[1].setLo(0, amrex::BCType::reflect_odd);
	fc[1].setHi(0, amrex::BCType::reflect_odd);
	fc[2].setLo(0, amrex::BCType::reflect_even);
	fc[2].setHi(0, amrex::BCType::reflect_even);
	FaceSimulation sim(cc, fc);
	sim.setInitialConditions();
	int errors = sim.check(false);
	errors += sim.check(true);
	return errors == 0 ? 0 : 1;
}
