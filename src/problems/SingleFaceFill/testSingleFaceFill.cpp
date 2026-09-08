#include "QuokkaSimulation.hpp"

struct SingleFaceProblem {};
template <> struct Physics_Traits<SingleFaceProblem> : DefaultPhysicsTraits {
	static constexpr bool is_hydro_enabled = true;
	static constexpr bool is_mhd_enabled = true;
	static constexpr UnitSystem unit_system = UnitSystem::CONSTANTS;
};

template <> void QuokkaSimulation<SingleFaceProblem>::setInitialConditionsOnGrid(quokka::grid const &grid)
{
	const auto a = grid.array_;
	amrex::ParallelFor(grid.indexRange_, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		for (int n = 0; n < Physics_Indices<SingleFaceProblem>::nvarTotal_cc; ++n) {
			a(i, j, k, n) = 0.0;
		}
		a(i, j, k, HydroSystem<SingleFaceProblem>::density_index) = 1.0;
		a(i, j, k, HydroSystem<SingleFaceProblem>::energy_index) = 100.0;
		a(i, j, k, HydroSystem<SingleFaceProblem>::internalEnergy_index) = 100.0;
	});
}

template <> void QuokkaSimulation<SingleFaceProblem>::setInitialConditionsOnGridFaceVars(quokka::grid const &grid)
{
	const auto a = grid.array_;
	const int axis = static_cast<int>(grid.dir_);
	amrex::ParallelFor(grid.indexRange_, [=] AMREX_GPU_DEVICE(int i, int j, int k) { a(i, j, k, 0) = axis + 1.0; });
}

template <>
template <quokka::direction dir>
AMREX_GPU_DEVICE void AMRSimulation<SingleFaceProblem>::setCustomBoundaryConditionsFaceVar(const amrex::IntVect &iv, amrex::Array4<amrex::Real> const &dest,
											   int dcomp, int numcomp, amrex::GeometryData const &geom,
											   amrex::Real /*time*/, const amrex::BCRec * /*bcr*/, int /*bcomp*/,
											   int /*orig_comp*/)
{
	constexpr int axis = static_cast<int>(dir);
	const auto domain = amrex::convert(geom.Domain(), amrex::IntVect::TheDimensionVector(axis));
	if (!domain.contains(iv)) {
		for (int n = 0; n < numcomp; ++n) {
			dest(iv, dcomp + n) = 100.0 + axis;
		}
	}
}

class SingleFaceSimulation : public QuokkaSimulation<SingleFaceProblem>
{
      public:
	using QuokkaSimulation<SingleFaceProblem>::QuokkaSimulation;
	auto check() -> int
	{
		int errors = 0;
		for (int axis = 0; axis < 3; ++axis) {
			const auto box = amrex::convert(Geom(1).Domain(), amrex::IntVect::TheDimensionVector(axis));
			amrex::BoxArray ba(box);
			const amrex::DistributionMapping dm(ba);
			amrex::MultiFab mf(ba, dm, 3, 2);
			mf.setVal(-999.0);
			amrex::Vector<amrex::BCRec> bcs(1);
			for (int d = 0; d < 3; ++d) {
				bcs[0].setLo(d, amrex::BCType::ext_dir);
				bcs[0].setHi(d, amrex::BCType::ext_dir);
			}
			FillCoarsePatch(1, 0.0, mf, 1, 1, bcs, quokka::centering::fc, static_cast<quokka::direction>(axis));
			amrex::Gpu::DeviceScalar<int> failure(0);
			auto *failed = failure.dataPtr();
			for (amrex::MFIter it(mf); it.isValid(); ++it) {
				const auto a = mf.const_array(it);
				amrex::ParallelFor(3, [=] AMREX_GPU_DEVICE(int point) {
					amrex::IntVect iv(4);
					iv[axis] = point == 0 ? -1 : (point == 1 ? 17 : 4);
					const double expected = point == 2 ? axis + 1.0 : 100.0 + axis;
					if (a(iv, 0) != -999.0 || a(iv, 2) != -999.0 || !std::isfinite(a(iv, 1)) || std::abs(a(iv, 1) - expected) > 1.e-12) {
						amrex::Gpu::Atomic::Add(failed, 1);
					}
				});
			}
			const int count = failure.dataValue();
			errors += count;
			if (count) {
				amrex::Print() << "Single-face fill failed for axis " << axis << '\n';
			}
		}
		amrex::ParallelDescriptor::ReduceIntSum(errors);
		return errors;
	}
};

auto problem_main() -> int
{
	amrex::Vector<amrex::BCRec> cc(Physics_Indices<SingleFaceProblem>::nvarTotal_cc), fc(3);
	for (auto &bc : cc) {
		for (int d = 0; d < 3; ++d) {
			bc.setLo(d, amrex::BCType::foextrap);
			bc.setHi(d, amrex::BCType::foextrap);
		}
	}
	for (auto &bc : fc) {
		for (int d = 0; d < 3; ++d) {
			bc.setLo(d, amrex::BCType::ext_dir);
			bc.setHi(d, amrex::BCType::ext_dir);
		}
	}
	SingleFaceSimulation sim(cc, fc);
	sim.setInitialConditions();
	return sim.check() == 0 ? 0 : 1;
}
